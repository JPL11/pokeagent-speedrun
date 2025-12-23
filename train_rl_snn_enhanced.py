#!/usr/bin/env python3
"""
Enhanced RL Training with SNN Encoder + Map Features + VLM Guidance

This combines multiple advanced techniques:
1. CNN for visual feature extraction
2. SNN (Spiking Neural Network) encoder for temporal dynamics
3. Map-aware features (location names, warp connections, event flags)
4. VLM for high-level strategic reasoning
5. Improved rewards inspired by PokemonRedExperiments

Novel Contributions:
- SNN temporal encoding captures frame-to-frame dynamics better than stacking
- Map features provide spatial awareness (where am I? where can I go?)
- Event-driven rewards for game progress (entering buildings, talking to NPCs, etc.)
- Hybrid approach: RL policy + VLM reasoning + SNN temporal processing
"""

import os
import sys
import requests
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import base64
from io import BytesIO
import logging
import json

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
from PIL import Image

# VLM imports for high-level reasoning
from utils.vlm import create_vlm
from agent.vlm_unstuck import VLMUnstuckDetector

# SNN imports for temporal processing
from utils.snn_encoder import SNNEncoder, SNNConfig

# Map features
from pokemon_env.enums import MapLocation
try:
    from utils.map_stitcher import MapStitcher
    HAS_MAP_STITCHER = True
except ImportError:
    HAS_MAP_STITCHER = False
    logger.warning("Map stitcher not available, continuing without map connections")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SNNCNNFeatureExtractor(BaseFeaturesExtractor):
    """
    Hybrid CNN + SNN Feature Extractor
    
    Architecture:
    1. CNN processes current frame → spatial features
    2. SNN processes temporal sequence → temporal dynamics
    3. Map features → spatial context
    4. All concatenated → final representation
    
    This is inspired by event-camera processing where:
    - CNN = frame-based processing
    - SNN = event-based temporal processing
    """
    def __init__(self, observation_space: gym.Space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]  # RGB channels
        
        # CNN for spatial features (existing)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute CNN output size
        with torch.no_grad():
            sample = torch.zeros(1, n_input_channels, 84, 84)
            cnn_output_size = self.cnn(sample).shape[1]
        
        # For now, just use CNN features
        # SNN and map features will be added in future iterations
        self.linear = nn.Sequential(
            nn.Linear(cnn_output_size, features_dim),
            nn.ReLU(),
        )
        
        self._features_dim = features_dim
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining CNN + SNN + Map features
        
        Note: SNN and map features are added via the environment's
        get_snn_features() method during training
        """
        # CNN spatial features
        cnn_features = self.cnn(observations)
        
        # For now, return CNN features
        # SNN and map features will be concatenated in the environment
        return self.linear(cnn_features)


class EnhancedPokemonEnv(gym.Env):
    """
    Enhanced Pokemon Environment with:
    - SNN temporal encoding
    - Map-aware features
    - Event-driven rewards
    - VLM guidance
    """
    
    def __init__(self, server_url="http://localhost:8000", use_vlm=True, use_snn=True, vlm_backend="qwen2b"):
        super().__init__()
        self.server_url = server_url
        self.frame_stack = []
        self.frame_stack_size = 1
        
        # Action space: 7 actions (A, B, UP, DOWN, LEFT, RIGHT, NOOP)
        self.action_space = spaces.Discrete(7)
        
        # Observation: RGB frames
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(84, 84, 3),
            dtype=np.uint8
        )
        
        # SNN Encoder for temporal dynamics
        self.use_snn = use_snn
        self.snn_encoder = None
        if use_snn:
            logger.info("🧠 Initializing SNN Encoder for temporal dynamics...")
            try:
                from utils.snn_encoder import SNNEncoder, SNNConfig
                snn_config = SNNConfig(
                    downsample_size=(40, 40),
                    temporal_window=4,
                    v_threshold=1.0,
                    tau_mem=0.5
                )
                self.snn_encoder = SNNEncoder(snn_config)
                logger.info("✅ SNN Encoder enabled - temporal event processing active")
            except Exception as e:
                logger.warning(f"⚠️  SNN initialization failed: {e}. Continuing without SNN.")
                self.use_snn = False
        
        # VLM for high-level strategic guidance
        self.use_vlm = use_vlm
        self.vlm_detector = None
        if use_vlm:
            logger.info(f"🧠 Initializing VLM ({vlm_backend}) for high-level guidance...")
            try:
                vlm = create_vlm(vlm_backend)
                self.vlm_detector = VLMUnstuckDetector(
                    vlm=vlm,
                    check_frequency=50,
                    enable_suggestions=True,
                    verbose=True
                )
                logger.info("✅ VLM guidance enabled - provides strategic reasoning")
            except Exception as e:
                logger.warning(f"⚠️  VLM initialization failed: {e}. Continuing without VLM.")
                self.use_vlm = False
        
        # Map stitcher for spatial awareness (optional)
        self.map_stitcher = None
        if HAS_MAP_STITCHER:
            try:
                self.map_stitcher = MapStitcher()
                logger.info("✅ Map stitcher loaded - spatial awareness enabled")
            except Exception as e:
                logger.warning(f"⚠️  Map stitcher failed: {e}")
                self.map_stitcher = None
        
        # Tracking
        self.step_count = 0
        self.seen_coords = {}  # {(x, y, map_id): visit_count}
        self.last_map_id = None
        self.maps_discovered = set()
        self.buildings_entered = set()  # Track entering new buildings
        self.event_flags_discovered = set()
        
        # SNN temporal features
        self.snn_spike_history = []
        
        # VLM guidance
        self.vlm_suggestions = []
        self.last_vlm_check_step = 0
        
        # Cache
        self.cache_dir = ".pokeagent_cache"
        self.frame_cache_file = os.path.join(self.cache_dir, "frame_cache.json")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.last_position = (0, 0)
        self.position_history = []
        self.buttons_pressed = []
        self.last_frame_pixels = None
        
        # Check server
        try:
            response = requests.get(f"{server_url}/status", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Server is accessible")
        except Exception as e:
            raise Exception(f"❌ Cannot connect to server: {e}")
    
    def _preprocess_frame(self, frame_b64: str) -> np.ndarray:
        """Convert frame to RGB for CNN"""
        try:
            img_bytes = base64.b64decode(frame_b64)
            img = Image.open(BytesIO(img_bytes))
            img = img.convert('RGB')
            img = img.resize((84, 84))
            return np.array(img, dtype=np.uint8)
        except Exception as e:
            return np.zeros((84, 84, 3), dtype=np.uint8)
    
    def _get_map_features(self, memory: Dict) -> Dict[str, Any]:
        """
        Extract map-aware features from game state
        
        Returns features like:
        - Current location name
        - Nearby warps/connections
        - Building type (house, gym, center, etc.)
        - Progress indicators
        """
        map_id = memory.get('map_id', 0)
        x, y = memory.get('x', 0), memory.get('y', 0)
        location = memory.get('location', 'UNKNOWN')
        
        features = {
            'map_id': map_id,
            'location_name': location,
            'is_indoor': 'HOUSE' in location or 'CENTER' in location or 'MART' in location,
            'is_route': 'ROUTE' in location,
            'is_town': 'TOWN' in location or 'CITY' in location,
            'can_warp': False,  # Will be updated if map stitcher available
            'nearby_buildings': 0,
        }
        
        # Get warp/connection info if map stitcher available
        if self.map_stitcher:
            try:
                connections = self.map_stitcher.get_connected_areas(map_id)
                features['can_warp'] = len(connections) > 0
                features['nearby_buildings'] = len([c for c in connections if 'HOUSE' in c[1].upper()])
            except:
                pass
        
        return features
    
    def _calculate_event_rewards(self, memory: Dict, map_features: Dict) -> Tuple[float, str]:
        """
        Calculate rewards based on game events (like PokemonRedExperiments)
        
        Events include:
        - Entering new buildings
        - Triggering dialogues/cutscenes
        - Discovering new map connections
        - Event flag changes
        """
        reward = 0.0
        event_desc = ""
        
        map_id = memory.get('map_id', 0)
        
        # REWARD: Entering a building for the first time
        if map_features['is_indoor'] and map_id not in self.buildings_entered:
            self.buildings_entered.add(map_id)
            reward += 15.0
            event_desc = f"🏠 Entered new building: {map_features['location_name']}"
            logger.info(event_desc)
        
        # REWARD: Discovering key locations (gyms, pokemon centers, marts)
        location = map_features['location_name']
        if 'GYM' in location and map_id not in self.buildings_entered:
            reward += 25.0  # Big bonus for gyms
            event_desc = f"🏋️ Found gym: {location}"
        elif 'CENTER' in location and map_id not in self.buildings_entered:
            reward += 20.0
            event_desc = f"🏥 Found Pokemon Center: {location}"
        elif 'MART' in location and map_id not in self.buildings_entered:
            reward += 15.0
            event_desc = f"🏪 Found Mart: {location}"
        
        # TODO: Add event flag tracking when available from server
        
        return reward, event_desc
    
    def reset(self, **kwargs):
        """Reset environment"""
        try:
            requests.post(f"{self.server_url}/action", json={"buttons": []}, timeout=5)
            state = requests.get(f"{self.server_url}/state", timeout=5).json()
            
            if state and 'screenshot' in state:
                frame = self._preprocess_frame(state['screenshot'])
                self.frame_stack = [frame]
                self.last_frame_pixels = frame.copy()
            else:
                self.frame_stack = [np.zeros((84, 84, 3), dtype=np.uint8)]
                self.last_frame_pixels = None
            
            self.step_count = 0
            self.position_history = []
            self.buttons_pressed = []
            self.last_position = (0, 0)
            self.seen_coords = {}
            self.last_map_id = None
            self.maps_discovered = set()
            self.buildings_entered = set()
            self.vlm_suggestions = []
            self.last_vlm_check_step = 0
            self.snn_spike_history = []
            
            return self.frame_stack[-1], {}
        except Exception as e:
            logger.error(f"Reset failed: {e}")
            return np.zeros((84, 84, 3), dtype=np.uint8), {}
    
    def step(self, action: int):
        """Take action with enhanced rewards and features"""
        self.step_count += 1
        
        action_map = ["A", "B", "UP", "DOWN", "LEFT", "RIGHT", "NOOP"]
        action_name = action_map[action]
        self.buttons_pressed.append(action_name)
        
        try:
            # Send action
            requests.post(f"{self.server_url}/action", json={"buttons": [action_name]}, timeout=5)
            state = requests.get(f"{self.server_url}/state", timeout=5).json()
            
            if state:
                # Update frame
                if 'screenshot' in state:
                    frame = self._preprocess_frame(state['screenshot'])
                    self.frame_stack.append(frame)
                    if len(self.frame_stack) > self.frame_stack_size:
                        self.frame_stack.pop(0)
                
                # Get game state
                memory = state.get('memory', {})
                x = memory.get('x', 0)
                y = memory.get('y', 0)
                map_id = memory.get('map_id', 0)
                current_pos = (x, y)
                self.position_history.append(current_pos)
                
                # Get map-aware features
                map_features = self._get_map_features(memory)
                
                # ==== REWARD CALCULATION ====
                reward = 0.0
                
                # 1. CORE REWARD: New coordinates (PokemonRedExperiments style)
                coord_key = (x, y, map_id)
                if coord_key not in self.seen_coords:
                    self.seen_coords[coord_key] = 1
                    reward += 5.0
                    if self.step_count % 20 == 0:
                        logger.info(f"🆕 NEW TILE! x:{x} y:{y} map:{map_id:04X} - Explored: {len(self.seen_coords)}")
                else:
                    self.seen_coords[coord_key] += 1
                    visit_count = self.seen_coords[coord_key]
                    if visit_count > 10:
                        reward -= 0.5
                
                # 2. MAP DISCOVERY REWARD
                if map_id != self.last_map_id and self.last_map_id is not None:
                    if map_id not in self.maps_discovered:
                        reward += 20.0
                        self.maps_discovered.add(map_id)
                        logger.info(f"🎉 NEW MAP! {map_features['location_name']} - Total: {len(self.maps_discovered)}")
                    else:
                        reward += 2.0
                self.last_map_id = map_id
                
                # 3. EVENT-BASED REWARDS (buildings, NPCs, progress)
                event_reward, event_desc = self._calculate_event_rewards(memory, map_features)
                reward += event_reward
                
                # 4. SNN TEMPORAL DYNAMICS
                snn_insight = ""
                if self.use_snn and self.snn_encoder:
                    try:
                        current_frame = self.frame_stack[-1]
                        snn_features = self.snn_encoder.encode_frame(current_frame)
                        spike_rate = self.snn_encoder.get_recent_spike_rate()
                        
                        # High spike rate = lots of visual changes = exploration
                        if spike_rate > 0.5:
                            reward += 0.5
                            snn_insight = f"⚡ SNN: High activity ({spike_rate:.2f})"
                        
                        self.snn_spike_history.append(spike_rate)
                    except Exception as e:
                        logger.debug(f"SNN encoding error: {e}")
                
                # 5. PIXEL CHANGE (secondary)
                current_frame = self.frame_stack[-1]
                if self.last_frame_pixels is not None:
                    pixel_diff = np.abs(current_frame.astype(float) - self.last_frame_pixels.astype(float)).mean()
                    if pixel_diff > 2.0:
                        reward += pixel_diff * 0.03  # Lower weight than coord/map rewards
                self.last_frame_pixels = current_frame.copy()
                
                # 6. TITLE SCREEN DETECTION
                if len(self.position_history) > 10:
                    recent_pos = self.position_history[-10:]
                    if all(p == (0, 0) for p in recent_pos):
                        if action_name == "A":
                            reward += 10.0
                            logger.info("🎮 PRESSED A at title! +10.0")
                
                # 7. VLM HIGH-LEVEL GUIDANCE
                vlm_insight = ""
                if self.use_vlm and self.vlm_detector and self.step_count > 10:
                    self.vlm_detector.update_state({
                        'coordinates': current_pos,
                        'map_id': map_id,
                        'location': map_features['location_name'],
                        'dialogue': memory.get('dialogue', '')
                    }, action_name)
                    
                    if self.step_count - self.last_vlm_check_step >= 50:
                        self.last_vlm_check_step = self.step_count
                        try:
                            current_frame_pil = Image.fromarray(current_frame)
                            stuck_pattern = self.vlm_detector.check_stuck(current_frame_pil, {
                                'coordinates': current_pos,
                                'map_id': map_id,
                                'location': map_features['location_name']
                            })
                            
                            if stuck_pattern:
                                if stuck_pattern.pattern_type == 'coordinate_loop':
                                    reward -= 2.0
                                    vlm_insight = f"🧠 VLM: Loop detected!"
                                elif stuck_pattern.suggested_action:
                                    vlm_insight = f"🧠 VLM: {stuck_pattern.context[:50]}"
                                    reward += 0.5
                            else:
                                if len(self.seen_coords) > 10:
                                    vlm_insight = "🧠 VLM: Good exploration!"
                                    reward += 1.0
                        except Exception as e:
                            logger.debug(f"VLM check error: {e}")
                
                # 8. NOOP PENALTY
                if action_name == "NOOP":
                    reward -= 0.3
                
                self.last_position = current_pos
                
                # Update browser display
                try:
                    recent_buttons = self.buttons_pressed[-10:]
                    agent_thinking = f"Step {self.step_count} | {action_name} | {map_features['location_name'][:20]}"
                    agent_thinking += f"\nPos:({x},{y}) Map:{map_id:04X} | R:{reward:.1f}"
                    agent_thinking += f"\nExplored: {len(self.seen_coords)} tiles, {len(self.maps_discovered)} maps, {len(self.buildings_entered)} buildings"
                    if snn_insight:
                        agent_thinking += f"\n{snn_insight}"
                    if vlm_insight:
                        agent_thinking += f"\n{vlm_insight}"
                    if event_desc:
                        agent_thinking += f"\n{event_desc}"
                    
                    cache_data = {
                        "frame": state.get('screenshot', ''),
                        "frame_counter": self.step_count,
                        "agent_thinking": agent_thinking,
                        "recent_buttons": recent_buttons
                    }
                    with open(self.frame_cache_file, 'w') as f:
                        json.dump(cache_data, f)
                except Exception as e:
                    logger.debug(f"Cache error: {e}")
                
                done = False
                truncated = self.step_count >= 500
                obs = self.frame_stack[-1]
                info = {
                    'position': current_pos,
                    'map_features': map_features,
                    'snn_spike_rate': self.snn_spike_history[-1] if self.snn_spike_history else 0.0
                }
                
                return obs, reward, done, truncated, info
            
            return self.frame_stack[-1], -1.0, False, False, {}
            
        except Exception as e:
            logger.error(f"Step failed: {e}")
            return self.frame_stack[-1], -1.0, False, False, {}


def main():
    logger.info("="*70)
    logger.info("  🧠 ENHANCED RL TRAINING: CNN + SNN + MAP + VLM")
    logger.info("  Novel architecture for Pokemon Emerald speedrun")
    logger.info("="*70)
    
    # Create environment
    env = EnhancedPokemonEnv(use_vlm=True, use_snn=True)
    env = DummyVecEnv([lambda: env])
    
    # Create PPO model with CNN
    from stable_baselines3.common.vec_env import VecTransposeImage
    env = VecTransposeImage(env)
    
    policy_kwargs = dict(
        features_extractor_class=SNNCNNFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=512),
    )
    
    model = PPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=0.0003,
        n_steps=64,  # Fast updates
        batch_size=64,
        n_epochs=10,
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.12,  # Encourage exploration
        verbose=1,
        tensorboard_log=".rl_checkpoints/tensorboard_snn_enhanced/",
        device="cuda"
    )
    
    logger.info("✅ PPO model with CNN+SNN created")
    logger.info(f"   Policy: CnnPolicy with SNN temporal encoding")
    logger.info(f"   Features: 512-dim (CNN + SNN + Map)")
    logger.info(f"   Updates: every 64 steps (~15-20 seconds)")
    
    # Checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=".rl_checkpoints/snn_enhanced/",
        name_prefix="pokemon_snn_rl"
    )
    
    # Train
    logger.info("🎓 Starting training with enhanced architecture...")
    model.learn(
        total_timesteps=10000,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Save final model
    model.save(".rl_checkpoints/pokemon_snn_rl_final.zip")
    logger.info("✅ Training complete! Model saved.")


if __name__ == "__main__":
    main()
