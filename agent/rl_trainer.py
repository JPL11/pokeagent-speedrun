"""
Real Reinforcement Learning Trainer for Pokemon Emerald
Inspired by PokemonRedExperiments with hybrid VLM guidance

Architecture:
- PPO (Proximal Policy Optimization) for learning
- SNN Encoder for temporal feature extraction
- VLM for high-level reasoning and goal-setting
- Reward shaping based on coordinate novelty, events, dialogue progress
"""

import os
import sys
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Gymnasium for environment interface
try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYM = True
except ImportError:
    HAS_GYM = False
    print("⚠️  gymnasium not installed. Install with: pip install gymnasium")

# Stable Baselines3 for RL algorithms
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("⚠️  stable-baselines3 not installed. Install with: pip install stable-baselines3")

from utils.snn_encoder import create_snn_encoder
from agent.vlm_unstuck import VLMUnstuckDetector, VLMGuidedRewardShaper

logger = logging.getLogger(__name__)


@dataclass
class RLConfig:
    """Configuration for RL training"""
    # Environment
    max_steps_per_episode: int = 10000
    coordinate_history_size: int = 200
    frame_stack_size: int = 4
    
    # Rewards
    coordinate_novelty_weight: float = 1.0
    event_progress_weight: float = 10.0
    dialogue_progress_weight: float = 5.0
    loop_penalty: float = -1.0
    sign_spam_penalty: float = -2.0
    animation_farm_penalty: float = -1.5
    
    # Training
    total_timesteps: int = 10_000_000
    learning_rate: float = 2.5e-4
    n_steps: int = 512
    batch_size: int = 64
    n_epochs: int = 4
    gamma: float = 0.995  # Discount factor
    
    # VLM guidance
    use_vlm_guidance: bool = True
    vlm_guidance_frequency: int = 100  # Ask VLM every N steps for high-level goals
    use_vlm_unstuck: bool = True  # Use VLM for stuck detection instead of heuristics
    vlm_unstuck_frequency: int = 30  # Check for stuck every N steps
    
    # SNN features
    use_snn_features: bool = True
    snn_feature_dim: int = 5  # SNN encoder outputs 5 features (spike_rate, activity_level, temporal_contrast, change_detected, fill_ratio)
    
    # Checkpointing
    save_freq: int = 50000  # Save every 50k steps
    checkpoint_dir: str = ".rl_checkpoints"


class SNNFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor that uses SNN encoder for temporal features
    Combined with CNN for spatial features
    
    Inspired by PokemonRedExperiments but with SNN enhancement
    """
    
    def __init__(self, observation_space, features_dim: int = 512, use_snn: bool = True):
        super().__init__(observation_space, features_dim)
        
        self.use_snn = use_snn
        
        # SNN encoder for temporal dynamics
        if self.use_snn:
            try:
                self.snn_encoder = create_snn_encoder()
                # SNN encoder returns 5 features: spike_rate, activity_level, temporal_contrast, change_detected, fill_ratio
                self.snn_feature_dim = 5
            except Exception as e:
                logger.warning(f"Failed to create SNN encoder: {e}. Using CNN only.")
                self.use_snn = False
                self.snn_feature_dim = 0
        else:
            self.snn_feature_dim = 0
        
        # Standard CNN for spatial features (like PokemonRedExperiments)
        # Input is (C, H, W) = (4, 160, 240) for GBA frame stack
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate CNN output size dynamically for GBA resolution
        with torch.no_grad():
            sample = torch.zeros(1, 4, 160, 240)  # GBA: height=160, width=240
            cnn_output_size = self.cnn(sample).shape[1]
        
        # Combine CNN + SNN features
        total_input_size = cnn_output_size + self.snn_feature_dim
        
        # Final fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(total_input_size, features_dim),
            nn.ReLU(),
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract features from observations
        
        Args:
            observations: (batch, channels, height, width)
        
        Returns:
            features: (batch, features_dim)
        """
        # CNN features (spatial)
        cnn_features = self.cnn(observations)
        
        # SNN features (temporal) - process most recent frame
        if self.use_snn and self.snn_encoder is not None:
            try:
                # Get most recent frame (last channel)
                recent_frame = observations[:, -1:, :, :].cpu().numpy()
                
                snn_features_list = []
                for i in range(recent_frame.shape[0]):
                    frame = recent_frame[i, 0, :, :]  # (H, W)
                    # Convert to uint8 for SNN encoder
                    frame_uint8 = (frame * 255).astype(np.uint8)
                    
                    # Encode frame
                    snn_result = self.snn_encoder.encode_frame(frame_uint8)
                    snn_feature_vec = self.snn_encoder.get_feature_vector()
                    snn_features_list.append(snn_feature_vec)
                
                snn_features = torch.tensor(
                    np.array(snn_features_list), 
                    dtype=torch.float32, 
                    device=observations.device
                )
            except Exception as e:
                logger.debug(f"SNN encoding failed: {e}")
                # Fallback to zeros if SNN fails
                snn_features = torch.zeros(
                    observations.shape[0], 
                    self.snn_feature_dim, 
                    device=observations.device
                )
        else:
            snn_features = torch.zeros(
                observations.shape[0], 
                self.snn_feature_dim, 
                device=observations.device
            )
        
        # Concatenate CNN + SNN features
        if self.snn_feature_dim > 0:
            combined_features = torch.cat([cnn_features, snn_features], dim=1)
        else:
            combined_features = cnn_features
        
        # Final FC layer
        return self.fc(combined_features)


class PokemonEmeraldRLEnv(gym.Env):
    """
    RL Environment for Pokemon Emerald (Gymnasium interface)
    Inspired by PokemonRedExperiments but adapted for Emerald
    
    Provides:
    - Coordinate-based exploration rewards
    - Event/milestone rewards
    - Dialogue progress rewards
    - Loop/spam penalties
    """
    
    metadata = {"render_modes": []}
    
    def __init__(self, emulator, config: RLConfig, vlm=None, show_display=False):
        super().__init__()
        self.emulator = emulator
        self.config = config
        self.vlm = vlm
        self.show_display = show_display
        
        # Initialize display if requested
        self.display = None
        if self.show_display:
            try:
                from utils.display_helper import GameDisplay
                self.display = GameDisplay(title="Pokemon Emerald - RL Training", scale=3)
                logger.info("🎬 Display window initialized!")
            except Exception as e:
                logger.warning(f"Could not initialize display: {e}")
                self.show_display = False
        
        # State tracking
        self.step_count = 0
        self.episode_step = 0
        self.total_reward = 0.0
        
        # Coordinate-based exploration (like PokemonRedExperiments)
        self.seen_coordinates = set()
        self.coordinate_history = deque(maxlen=config.coordinate_history_size)
        
        # Event tracking
        self.seen_events = set()
        self.last_event_count = 0
        
        # Dialogue tracking
        self.seen_dialogues = set()
        self.last_dialogue = None
        self.dialogue_repeat_count = 0
        self.dialogue_present_steps = 0  # How many steps dialogue has been present
        self.last_dialogue_state = False  # Was dialogue present last step?
        
        # Frame stacking for temporal context
        self.frame_stack = deque(maxlen=config.frame_stack_size)
        
        # VLM guidance
        self.current_vlm_goal = None
        self.steps_since_vlm_guidance = 0
        
        # VLM unstuck detection (replaces hard-coded heuristics!)
        self.vlm_unstuck = None
        self.vlm_reward_shaper = None
        if config.use_vlm_unstuck and vlm:
            logger.info("🧠 Initializing VLM-based unstuck detection")
            self.vlm_unstuck = VLMUnstuckDetector(
                vlm=vlm,
                check_frequency=config.vlm_unstuck_frequency,
                enable_suggestions=True,
                verbose=True
            )
            self.vlm_reward_shaper = VLMGuidedRewardShaper(vlm, check_frequency=100)
            logger.info("✅ VLM unstuck detection enabled (heuristics minimized!)")
        
        # Action space (9 discrete actions)
        self.action_list = ['A', 'B', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'START', 'L', 'R']
        self.action_space = spaces.Discrete(len(self.action_list))
        
        # Observation space (single RGB frame) - matches train_rl_event_flags.py
        # GBA screen resized to 84x84 for RL
        # Frame stacking is maintained internally but not exposed in obs space
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(84, 84, 3), 
            dtype=np.uint8
        )
        
        logger.info(f"🎮 Pokemon Emerald RL Environment initialized")
        logger.info(f"   Config: {config.total_timesteps:,} timesteps, LR={config.learning_rate}")
        logger.info(f"   SNN Features: {'ENABLED' if config.use_snn_features else 'DISABLED'}")
        logger.info(f"   VLM Guidance: {'ENABLED' if config.use_vlm_guidance else 'DISABLED'}")
        logger.info(f"   Action space: {self.action_space}")
        logger.info(f"   Observation space: {self.observation_space}")
    
    def reset(self, seed=None, options=None):
        """Reset environment for new episode (optimized for speed)"""
        super().reset(seed=seed)
        
        # Clear episode state (fast)
        self.episode_step = 0
        self.coordinate_history.clear()
        self.dialogue_repeat_count = 0
        self.last_dialogue = None
        
        # Initialize emulator if needed (only if not already initialized)
        if not hasattr(self.emulator, 'core') or self.emulator.core is None:
            self.emulator.initialize()
        
        # Get initial frame (fast - no savestate loading!)
        frame = self.emulator.get_screenshot()
        
        # Show initial frame in display window immediately (non-blocking)
        if self.show_display and self.display and frame:
            self.display.show_frame(frame)
        
        # Initialize frame stack (fast)
        self.frame_stack.clear()
        processed_frame = self._preprocess_frame(frame)
        for _ in range(self.config.frame_stack_size):
            self.frame_stack.append(processed_frame)
        
        observation = self._get_observation()
        info = {}
        return observation, info
    
    def step(self, action_idx: int):
        """
        Execute action and return (observation, reward, done, info)
        
        Args:
            action_idx: Index into action_space
        
        Returns:
            observation: Stacked frames
            reward: Shaped reward signal
            done: Episode termination flag
            info: Additional information
        """
        self.step_count += 1
        self.episode_step += 1
        
        # Execute action
        action = self.action_list[action_idx]
        self.emulator.press_key(action.lower(), frames=2)  # Press button for 2 frames (tick is internal)
        
        # Get new state (press_key already advanced the emulator!)
        frame = self.emulator.get_screenshot()
        game_state = self._get_game_state()
        
        # Show frame in display window if enabled
        if self.show_display and self.display and frame:
            self.display.show_frame(frame)
        
        # Update frame stack
        self.frame_stack.append(self._preprocess_frame(frame))
        
        # VLM-based unstuck detection (replaces hard heuristics!)
        vlm_stuck_pattern = None
        if self.vlm_unstuck:
            # Update VLM unstuck detector with current state
            self.vlm_unstuck.update_state(game_state, action)
            
            # Check if stuck (VLM analyzes pattern)
            vlm_stuck_pattern = self.vlm_unstuck.check_stuck(frame, game_state)
            
            # If VLM detected stuck, include context in game_state for reward calculation
            if vlm_stuck_pattern:
                game_state['vlm_stuck_pattern'] = vlm_stuck_pattern
        
        # Calculate reward (now with VLM context!)
        reward, reward_info = self._calculate_reward(game_state, action)
        
        # Add VLM unstuck bonus if agent followed good advice
        if self.vlm_unstuck and vlm_stuck_pattern:
            was_unstuck = self.vlm_unstuck.was_unstuck()
            unstuck_bonus = self.vlm_unstuck.get_suggestion_reward_bonus(action, was_unstuck)
            reward += unstuck_bonus
            reward_info['vlm_unstuck_bonus'] = unstuck_bonus
        
        self.total_reward += reward
        
        # Check if done (Gymnasium requires terminated and truncated separately)
        terminated = False  # Episode ended naturally (e.g., game over, won)
        truncated = self.episode_step >= self.config.max_steps_per_episode  # Episode cut off by step limit
        
        # VLM guidance (high-level goals) and reward priority updates
        if self.config.use_vlm_guidance and self.vlm is not None:
            self.steps_since_vlm_guidance += 1
            if self.steps_since_vlm_guidance >= self.config.vlm_guidance_frequency:
                self._update_vlm_guidance(frame, game_state)
                self.steps_since_vlm_guidance = 0
            
            # Update VLM reward priorities
            if self.vlm_reward_shaper:
                self.vlm_reward_shaper.update_priorities(frame, game_state)
        
        # Build info dict
        info = {
            'reward_breakdown': reward_info,
            'total_reward': self.total_reward,
            'episode_step': self.episode_step,
            'coordinates': game_state.get('coordinates'),
            'vlm_goal': self.current_vlm_goal,
            **game_state
        }
        
        # Return 5 values for Gymnasium compatibility
        return self._get_observation(), reward, terminated, truncated, info
    
    def _get_observation(self):
        """Get current observation (latest RGB frame)"""
        # Return only the latest frame - frame stacking maintained internally
        if len(self.frame_stack) > 0:
            return self.frame_stack[-1]
        return np.zeros((84, 84, 3), dtype=np.uint8)
    
    def _preprocess_frame(self, frame):
        """Preprocess frame for RL (matches train_rl_event_flags.py)"""
        # Convert to RGB and resize to standard RL size
        from PIL import Image
        if isinstance(frame, np.ndarray):
            img = Image.fromarray(frame)
        else:
            img = frame
        
        # Resize to 84x84 (standard RL)
        img = img.resize((84, 84))
        
        # Convert to RGB
        img = img.convert('RGB')
        
        # Return as uint8 [0, 255]
        arr = np.array(img, dtype=np.uint8)
        
        return arr
    
    def _get_game_state(self) -> Dict[str, Any]:
        """Read current game state from memory"""
        try:
            memory_reader = self.emulator.memory_reader
            
            coords = memory_reader.read_coordinates() if hasattr(memory_reader, 'read_coordinates') else (0, 0)
            map_id = memory_reader.read_map_id() if hasattr(memory_reader, 'read_map_id') else 0
            dialogue = memory_reader.read_dialogue() if hasattr(memory_reader, 'read_dialogue') else ""
            location = memory_reader.read_location() if hasattr(memory_reader, 'read_location') else "UNKNOWN"
            
            # Get event flags
            events = []
            if hasattr(memory_reader, 'milestones'):
                events = list(memory_reader.milestones.keys())
            
            return {
                'coordinates': coords,
                'map_id': map_id,
                'dialogue': dialogue,
                'location': location,
                'events': events,
            }
        except Exception as e:
            logger.debug(f"Error reading game state: {e}")
            return {
                'coordinates': (0, 0),
                'map_id': 0,
                'dialogue': "",
                'location': "UNKNOWN",
                'events': [],
            }
    
    def _calculate_reward(self, game_state: Dict[str, Any], action: str) -> Tuple[float, Dict[str, float]]:
        """
        Calculate shaped reward based on PokemonRedExperiments principles
        
        Rewards:
        - Coordinate novelty (exploring new areas)
        - Event progress (achieving milestones)
        - Dialogue progress (advancing story)
        
        Penalties:
        - Loops (repeating same coordinates)
        - Sign spam (repeated dialogue)
        - Animation farming (staying in one spot)
        """
        reward = 0.0
        reward_info = {}
        
        # 1. COORDINATE NOVELTY REWARD (like PokemonRedExperiments)
        coords = game_state.get('coordinates', (0, 0))
        map_id = game_state.get('map_id', 0)
        coord_tuple = (coords[0], coords[1], map_id)
        
        if coord_tuple not in self.seen_coordinates:
            # NEW COORDINATE! Big reward
            base_novelty = self.config.coordinate_novelty_weight
            # Apply VLM-guided reward shaping if available
            if self.vlm_reward_shaper:
                novelty_reward = self.vlm_reward_shaper.get_shaped_reward(base_novelty, 'exploration')
            else:
                novelty_reward = base_novelty
            self.seen_coordinates.add(coord_tuple)
            reward += novelty_reward
            reward_info['coordinate_novelty'] = novelty_reward
        else:
            reward_info['coordinate_novelty'] = 0.0
        
        # Track recent coordinates for loop detection
        self.coordinate_history.append(coord_tuple)
        
        # 2. EVENT/MILESTONE PROGRESS REWARD
        events = set(game_state.get('events', []))
        new_events = events - self.seen_events
        if new_events:
            # Achieved new milestone!
            event_reward = self.config.event_progress_weight * len(new_events)
            self.seen_events.update(new_events)
            reward += event_reward
            reward_info['event_progress'] = event_reward
            logger.info(f"🎯 New milestone achieved! Reward: +{event_reward:.1f}")
        else:
            reward_info['event_progress'] = 0.0
        
        # 3. DIALOGUE PROGRESS REWARD
        dialogue = game_state.get('dialogue', '').strip()
        dialogue_active = dialogue and len(dialogue) > 3
        
        # Check if dialogue is actually present (using memory reader if available)
        try:
            if hasattr(self.emulator, 'memory_reader') and self.emulator.memory_reader:
                is_in_dialog = self.emulator.memory_reader.is_in_dialog()
                dialogue_active = dialogue_active or is_in_dialog
        except:
            pass  # Fall back to dialogue text detection
        
        if dialogue_active:
            # Dialogue is present - reward pressing 'A' to advance it!
            if action.upper() == 'A':
                # BIG reward for pressing A when dialogue is present
                dialogue_advance_reward = 0.5  # Significant reward for advancing dialogue
                reward += dialogue_advance_reward
                reward_info['dialogue_advance'] = dialogue_advance_reward
                self.dialogue_present_steps = 0  # Reset counter
                logger.debug(f"✅ Pressed A during dialogue! Reward: +{dialogue_advance_reward:.2f}")
            else:
                # Penalty for NOT pressing A when dialogue is present
                self.dialogue_present_steps += 1
                if self.dialogue_present_steps > 3:
                    # Dialogue has been present for 3+ steps without pressing A
                    dialogue_stall_penalty = -0.1 * (self.dialogue_present_steps - 3)
                    reward += dialogue_stall_penalty
                    reward_info['dialogue_stall'] = dialogue_stall_penalty
                    logger.debug(f"⚠️  Dialogue present for {self.dialogue_present_steps} steps without A press. Penalty: {dialogue_stall_penalty:.2f}")
                else:
                    reward_info['dialogue_stall'] = 0.0
                reward_info['dialogue_advance'] = 0.0
            
            # Track new dialogue content
            if dialogue and len(dialogue) > 3:
                if dialogue not in self.seen_dialogues:
                    # NEW dialogue
                    base_dialogue = self.config.dialogue_progress_weight
                    # Apply VLM-guided reward shaping if available
                    if self.vlm_reward_shaper:
                        dialogue_reward = self.vlm_reward_shaper.get_shaped_reward(base_dialogue, 'dialogue')
                    else:
                        dialogue_reward = base_dialogue
                    self.seen_dialogues.add(dialogue)
                    reward += dialogue_reward
                    reward_info['dialogue_progress'] = dialogue_reward
                    self.dialogue_repeat_count = 0
                else:
                    # Repeated dialogue
                    reward_info['dialogue_progress'] = 0.0
                    if dialogue == self.last_dialogue:
                        self.dialogue_repeat_count += 1
                    else:
                        self.dialogue_repeat_count = 0
                
                self.last_dialogue = dialogue
            else:
                reward_info['dialogue_progress'] = 0.0
        else:
            # No dialogue present
            self.dialogue_present_steps = 0
            reward_info['dialogue_progress'] = 0.0
            reward_info['dialogue_advance'] = 0.0
            reward_info['dialogue_stall'] = 0.0
        
        # Update dialogue state tracking
        self.last_dialogue_state = dialogue_active
        
        # 4. VLM-GUIDED LOOP DETECTION (replaces hard-coded heuristic!)
        if 'vlm_stuck_pattern' in game_state:
            # VLM detected stuck - apply its analyzed penalty
            vlm_pattern = game_state['vlm_stuck_pattern']
            reward += vlm_pattern.reward_modifier
            reward_info['vlm_stuck_penalty'] = vlm_pattern.reward_modifier
            reward_info['vlm_stuck_type'] = vlm_pattern.pattern_type
        else:
            # Fallback: simple loop detection (if VLM not available)
            if len(self.coordinate_history) >= 20:
                recent_coords = list(self.coordinate_history)[-20:]
                unique_coords = len(set(recent_coords))
                if unique_coords < 5:
                    # Stuck in loop!
                    loop_penalty = self.config.loop_penalty
                    reward += loop_penalty
                    reward_info['loop_penalty'] = loop_penalty
                else:
                    reward_info['loop_penalty'] = 0.0
            else:
                reward_info['loop_penalty'] = 0.0
        
        # 5. SIGN SPAM PENALTY (reading same sign repeatedly)
        if self.dialogue_repeat_count >= 2:
            # Spamming same dialogue
            sign_spam_penalty = self.config.sign_spam_penalty
            reward += sign_spam_penalty
            reward_info['sign_spam_penalty'] = sign_spam_penalty
        else:
            reward_info['sign_spam_penalty'] = 0.0
        
        # 6. ANIMATION FARMING PENALTY (staying in one spot with pixel changes)
        if len(self.coordinate_history) >= 10:
            recent_coords = list(self.coordinate_history)[-10:]
            unique_coords = len(set(recent_coords))
            if unique_coords <= 2:
                # Standing still (animation farming)
                animation_penalty = self.config.animation_farm_penalty
                reward += animation_penalty
                reward_info['animation_farm_penalty'] = animation_penalty
            else:
                reward_info['animation_farm_penalty'] = 0.0
        
        return reward, reward_info
    
    def _update_vlm_guidance(self, frame, game_state: Dict[str, Any]):
        """
        Ask VLM for high-level guidance (what should the agent focus on?)
        This is NOT direct control - just setting goals for RL to learn
        """
        try:
            prompt = f"""You are advising an RL agent playing Pokemon Emerald.
Current State:
- Location: {game_state.get('location')}
- Coordinates: {game_state.get('coordinates')}
- Recent Events: {len(self.seen_events)}
- Unique Coordinates Explored: {len(self.seen_coordinates)}

What high-level goal should the agent focus on? (1-2 sentences)
Example: "Find the exit to the next town" or "Talk to NPCs to progress story"
"""
            
            # Get VLM guidance
            response = self.vlm.get_query(frame, prompt, "rl_guidance")
            
            # Extract goal from response
            if response and 'reasoning' in response:
                self.current_vlm_goal = response['reasoning'][:200]  # Truncate
                logger.info(f"🧠 VLM Guidance: {self.current_vlm_goal}")
            
        except Exception as e:
            logger.debug(f"Error getting VLM guidance: {e}")


class VLMGuidanceCallback(BaseCallback):
    """
    Callback to periodically log VLM guidance and RL progress
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        # Log episode statistics
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])
                    
                    if len(self.episode_rewards) % 10 == 0:
                        avg_reward = np.mean(self.episode_rewards[-10:])
                        avg_length = np.mean(self.episode_lengths[-10:])
                        logger.info(f"📊 Episodes: {len(self.episode_rewards)} | "
                                  f"Avg Reward: {avg_reward:.2f} | "
                                  f"Avg Length: {avg_length:.0f}")
        
        return True


def train_rl_agent(emulator, config: RLConfig, vlm=None, show_display=False):
    """
    Train RL agent with PPO
    
    Args:
        emulator: Pokemon Emerald emulator instance
        config: RL configuration
        vlm: Optional VLM for guidance
        show_display: Whether to show OpenCV display window
    
    Returns:
        Trained PPO model
    """
    if not HAS_SB3:
        raise ImportError("stable-baselines3 required for RL training. Install with: pip install stable-baselines3")
    
    logger.info("🚀 Starting RL training with PPO + SNN features")
    
    # Create environment (with display support)
    base_env = PokemonEmeraldRLEnv(emulator, config, vlm, show_display=show_display)
    
    # Wrap in vectorized environment
    env = DummyVecEnv([lambda: base_env])
    
    # Create custom policy with SNN feature extractor
    policy_kwargs = dict(
        features_extractor_class=SNNFeatureExtractor,
        features_extractor_kwargs=dict(
            features_dim=512,
            use_snn=config.use_snn_features
        ),
    )
    
    # Create PPO agent
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        verbose=1,
        policy_kwargs=policy_kwargs,
        tensorboard_log=f"{config.checkpoint_dir}/tensorboard/"
    )
    
    logger.info(f"✅ PPO agent created with SNN feature extractor")
    logger.info(f"   Policy: {model.policy}")
    logger.info(f"   Training for {config.total_timesteps:,} timesteps")
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=config.save_freq,
        save_path=config.checkpoint_dir,
        name_prefix="pokemon_emerald_rl"
    )
    
    vlm_callback = VLMGuidanceCallback()
    
    # Train!
    logger.info("🎓 Starting training...")
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=[checkpoint_callback, vlm_callback],
        progress_bar=True
    )
    
    logger.info("✅ Training complete!")
    
    # Save final model
    final_path = os.path.join(config.checkpoint_dir, "pokemon_emerald_final.zip")
    model.save(final_path)
    logger.info(f"💾 Final model saved to: {final_path}")
    
    return model


def load_trained_agent(checkpoint_path: str, device: str = None):
    """Load a trained RL agent
    
    Args:
        checkpoint_path: Path to the saved model
        device: Device to load on ('cuda', 'cpu', or None for auto-detect)
    """
    if not HAS_SB3:
        raise ImportError("stable-baselines3 required")
    
    # Auto-detect device if not specified
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"📂 Loading model from: {checkpoint_path}")
    logger.info(f"   Target device: {device}")
    
    # Load model and explicitly set device
    model = PPO.load(checkpoint_path, device=device)
    
    # Verify device
    actual_device = next(model.policy.parameters()).device
    logger.info(f"✅ Loaded trained agent from: {checkpoint_path}")
    logger.info(f"   Model is on device: {actual_device}")
    
    if device == "cuda" and actual_device.type == "cpu":
        logger.warning("⚠️  WARNING: Model loaded on CPU but CUDA was requested!")
        logger.warning("   This will be VERY slow. Check CUDA installation.")
    
    return model


if __name__ == "__main__":
    print("""
🎮 Pokemon Emerald RL Trainer
=============================

This module provides real reinforcement learning for Pokemon Emerald.

Key Features:
- PPO (Proximal Policy Optimization) with neural network policy
- SNN encoder for temporal feature extraction  
- VLM guidance for high-level goals
- Reward shaping based on PokemonRedExperiments
- Actually learns from mistakes (gradient descent on policy)

Usage:
    from agent.rl_trainer import train_rl_agent, RLConfig
    
    config = RLConfig(
        total_timesteps=10_000_000,
        use_snn_features=True,
        use_vlm_guidance=True
    )
    
    model = train_rl_agent(emulator, config, vlm)

The trained agent will learn to:
- Explore new coordinates (reward)
- Achieve milestones (big reward)
- Avoid loops and repetition (penalty)
- Use VLM hints to set high-level goals

This is REAL learning - the policy network updates via gradient descent!
    """)

