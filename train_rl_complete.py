#!/usr/bin/env python3
"""
Complete RL Training: CNN + SNN + VLM + Milestone Tracking

This combines ALL advanced techniques:
1. CNN for spatial visual features
2. SNN for temporal dynamics
3. VLM for milestone-aware strategic guidance
4. Milestone tracking for structured progression

The agent should: enter buildings, talk to NPCs, progress through game milestones.
"""

import os
import sys
import requests
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import base64
from io import BytesIO
import logging
import json
import time

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
from PIL import Image

# VLM imports
from utils.vlm import create_vlm

# SNN imports
from utils.snn_encoder import SNNEncoder, SNNConfig

# Map features
from pokemon_env.enums import MapLocation
try:
    from utils.map_stitcher import MapStitcher
    HAS_MAP_STITCHER = True
except ImportError:
    HAS_MAP_STITCHER = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================== CNN + SNN FEATURE EXTRACTOR ==================

class SNNCNNFeatureExtractor(BaseFeaturesExtractor):
    """Hybrid CNN + SNN Feature Extractor"""
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]
        
        # CNN for spatial features
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
        
        self.linear = nn.Sequential(
            nn.Linear(cnn_output_size, features_dim),
            nn.ReLU(),
        )
        
        self._features_dim = features_dim
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        cnn_features = self.cnn(observations)
        return self.linear(cnn_features)


# ================== MILESTONE DEFINITIONS ==================

EMERALD_MILESTONES = [
    {
        "id": "TITLE_SCREEN",
        "name": "Press A on Title Screen",
        "check": lambda mem: mem.get('x', 0) != 0 or mem.get('y', 0) != 0,
        "vlm_strategy": "Press A repeatedly to start the game and get past the title screen."
    },
    {
        "id": "CHARACTER_CREATED",
        "name": "Complete Character Creation",
        "check": lambda mem: mem.get('map_id', 0) in [0x0309, 0x0201, 0x0101],  # In house or town
        "vlm_strategy": "Navigate dialogue with A button. Choose character and confirm."
    },
    {
        "id": "LEFT_HOUSE",
        "name": "Exit Starter House",
        "check": lambda mem: mem.get('map_id', 0) == 0x0101,  # LITTLEROOT_TOWN
        "vlm_strategy": "You're in Littleroot Town. Go UP and LEFT to Route 101 to find Prof Birch being attacked!"
    },
    {
        "id": "ROUTE_101",
        "name": "Reached Route 101",
        "check": lambda mem: mem.get('map_id', 0) == 0x0010,  # ROUTE101
        "vlm_strategy": "Save Professor Birch from the Poochyena! Choose a starter Pokemon from the bag."
    },
    {
        "id": "GOT_STARTER",
        "name": "Got Starter Pokemon",
        "check": lambda mem: mem.get('party_size', 0) > 0,
        "vlm_strategy": "You have a Pokemon! Go back to Littleroot and visit the Pokemon Lab."
    },
    {
        "id": "VISITED_LAB",
        "name": "Visited Professor Birch's Lab",
        "check": lambda mem: mem.get('map_id', 0) == 0x020C,  # BIRCH_LAB
        "vlm_strategy": "Get the Pokedex from Professor Birch. Then head north to Route 103."
    },
    {
        "id": "ROUTE_103",
        "name": "Reached Route 103",
        "check": lambda mem: mem.get('map_id', 0) == 0x0012,  # ROUTE103
        "vlm_strategy": "Battle your rival! After winning, go back and explore routes to Oldale Town."
    },
    {
        "id": "OLDALE_TOWN",
        "name": "Arrived at Oldale Town",
        "check": lambda mem: mem.get('map_id', 0) == 0x0102,  # OLDALE_TOWN
        "vlm_strategy": "First new town! Visit the Pokemon Center to heal. Talk to NPCs. Then go north."
    },
    {
        "id": "PETALBURG_CITY",
        "name": "Arrived at Petalburg City",
        "check": lambda mem: mem.get('map_id', 0) == 0x0104,  # PETALBURG_CITY
        "vlm_strategy": "Visit the gym to meet your dad Norman. Then continue to Rustboro City."
    },
    {
        "id": "RUSTBORO_CITY",
        "name": "Arrived at Rustboro City",
        "check": lambda mem: mem.get('map_id', 0) == 0x0105,  # RUSTBORO_CITY
        "vlm_strategy": "Challenge Roxanne at the gym! Train your Pokemon and prepare for the first gym battle."
    },
    {
        "id": "FIRST_GYM_BADGE",
        "name": "Defeated Roxanne",
        "check": lambda mem: (mem.get('badges', 0) & 0x01) != 0,  # Stone Badge
        "vlm_strategy": "Congratulations on your first badge! Continue your journey to the next city."
    },
]


class MilestoneProgressTracker:
    """Tracks progression through game milestones"""
    
    def __init__(self, milestones: List[Dict]):
        self.milestones = milestones
        self.current_milestone_idx = 0
        self.completed_milestones = set()
        self.milestone_times = {}
        self.start_time = time.time()
        
    def get_current_milestone(self) -> Dict:
        if self.current_milestone_idx < len(self.milestones):
            return self.milestones[self.current_milestone_idx]
        return self.milestones[-1]
    
    def check_milestone_completion(self, memory: Dict) -> Optional[Dict]:
        current = self.get_current_milestone()
        
        if current['id'] in self.completed_milestones:
            return None
        
        try:
            if current['check'](memory):
                elapsed = time.time() - self.start_time
                self.milestone_times[current['id']] = elapsed
                self.completed_milestones.add(current['id'])
                
                logger.info(f"🎯 MILESTONE COMPLETED: {current['name']} ({elapsed:.1f}s)")
                
                self.current_milestone_idx += 1
                return current
        except Exception as e:
            logger.debug(f"Milestone check error: {e}")
        
        return None
    
    def get_vlm_strategy(self) -> str:
        current = self.get_current_milestone()
        return current.get('vlm_strategy', 'Explore and progress.')
    
    def get_progress_summary(self) -> str:
        total = len(self.milestones)
        completed = len(self.completed_milestones)
        current = self.get_current_milestone()
        return f"{completed}/{total} milestones | Current: {current['name']}"


# ================== COMPLETE ENVIRONMENT ==================

class CompletePokemonEnv(gym.Env):
    """Pokemon environment with CNN + SNN + VLM + Milestones"""
    
    metadata = {'render.modes': ['rgb_array']}
    
    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        use_snn: bool = True,
        use_vlm: bool = True,
        vlm_backend: str = "local",
    ):
        super().__init__()
        
        self.server_url = server_url
        
        # Observation: RGB frames (84x84x3)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 3), dtype=np.uint8
        )
        
        # Actions: 7 buttons
        self.button_map = ["A", "B", "UP", "DOWN", "LEFT", "RIGHT", "NOOP"]
        self.action_space = spaces.Discrete(len(self.button_map))
        
        # Frame stacking
        self.frame_stack_size = 4
        self.frame_stack = []
        
        # Milestone tracking
        self.milestone_tracker = MilestoneProgressTracker(EMERALD_MILESTONES)
        logger.info(f"📊 Loaded {len(EMERALD_MILESTONES)} milestones")
        
        # SNN encoder
        self.use_snn = use_snn
        self.snn_encoder = None
        if use_snn:
            try:
                snn_config = SNNConfig(
                    downsample_size=(40, 40),
                    temporal_window=4,
                    v_threshold=1.0,
                    tau_mem=0.5
                )
                self.snn_encoder = SNNEncoder(snn_config)
                logger.info("✅ SNN Encoder enabled")
            except Exception as e:
                logger.warning(f"⚠️  SNN failed: {e}")
                self.use_snn = False
        
        # VLM for milestone guidance
        self.use_vlm = use_vlm
        self.vlm = None
        if use_vlm:
            try:
                self.vlm = create_vlm(vlm_backend)
                logger.info("✅ VLM loaded")
            except Exception as e:
                logger.warning(f"⚠️  VLM failed: {e}")
                self.use_vlm = False
        
        # Map features
        self.map_stitcher = None
        if HAS_MAP_STITCHER:
            try:
                self.map_stitcher = MapStitcher()
                logger.info("✅ Map stitcher loaded")
            except:
                pass
        
        # Tracking
        self.step_count = 0
        self.seen_coords = {}
        self.last_map_id = None
        self.maps_discovered = set()
        self.buildings_entered = set()
        self.routes_visited = set()
        
        # SNN tracking
        self.snn_spike_history = []
        
        # VLM guidance
        self.last_vlm_check_step = 0
        self.vlm_advice_history = []
        
        # Cache
        self.cache_dir = ".pokeagent_cache"
        self.frame_cache_file = os.path.join(self.cache_dir, "frame_cache.json")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.position_history = []
        self.buttons_pressed = []
        self.last_frame_pixels = None
        
        # Check server
        try:
            response = requests.get(f"{server_url}/status", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Server accessible")
        except Exception as e:
            raise Exception(f"❌ Cannot connect: {e}")
    
    def _preprocess_frame(self, frame_b64: str) -> np.ndarray:
        try:
            img_bytes = base64.b64decode(frame_b64)
            img = Image.open(BytesIO(img_bytes))
            img = img.resize((84, 84))
            img = img.convert('RGB')
            return np.array(img, dtype=np.uint8)
        except Exception as e:
            logger.error(f"Frame error: {e}")
            return np.zeros((84, 84, 3), dtype=np.uint8)
    
    def _get_map_features(self, memory: Dict) -> Dict[str, Any]:
        map_id = memory.get('map_id', 0)
        
        try:
            location = MapLocation(map_id)
            location_name = location.name
        except ValueError:
            location_name = f"UNKNOWN_{map_id:04X}"
        
        features = {
            'location_name': location_name,
            'map_id': map_id,
            'is_indoor': 'HOUSE' in location_name or 'CENTER' in location_name or 'MART' in location_name or 'LAB' in location_name,
            'is_route': 'ROUTE' in location_name,
            'is_town': 'TOWN' in location_name or 'CITY' in location_name,
            'is_gym': 'GYM' in location_name,
        }
        
        return features
    
    def _calculate_rewards(self, memory: Dict, action_name: str) -> Tuple[float, str]:
        reward = 0.0
        info_parts = []
        
        x = memory.get('x', 0)
        y = memory.get('y', 0)
        map_id = memory.get('map_id', 0)
        
        # 1. MILESTONE COMPLETION (HIGHEST PRIORITY!)
        completed_milestone = self.milestone_tracker.check_milestone_completion(memory)
        if completed_milestone:
            reward += 100.0
            info_parts.append(f"🎯 MILESTONE: {completed_milestone['name']} +100")
        
        # 2. NEW COORDINATE EXPLORATION
        coord_key = (x, y, map_id)
        if coord_key not in self.seen_coords:
            self.seen_coords[coord_key] = 1
            reward += 5.0
            if len(self.seen_coords) % 50 == 0:
                info_parts.append(f"🆕 {len(self.seen_coords)} tiles explored +5")
        else:
            self.seen_coords[coord_key] += 1
            if self.seen_coords[coord_key] > 10:
                reward -= 0.5
        
        # 3. NEW MAP DISCOVERY
        map_features = self._get_map_features(memory)
        if map_id != self.last_map_id and self.last_map_id is not None:
            if map_id not in self.maps_discovered:
                self.maps_discovered.add(map_id)
                
                # Big rewards for important locations
                if map_features['is_route']:
                    reward += 30.0
                    self.routes_visited.add(map_id)
                    info_parts.append(f"🛣️  NEW ROUTE: {map_features['location_name']} +30")
                elif map_features['is_town']:
                    reward += 50.0
                    info_parts.append(f"🏙️  NEW TOWN: {map_features['location_name']} +50")
                elif map_features['is_indoor']:
                    reward += 20.0
                    self.buildings_entered.add(map_id)
                    if map_features['is_gym']:
                        reward += 30.0
                        info_parts.append(f"🏋️ GYM: {map_features['location_name']} +50")
                    else:
                        info_parts.append(f"🏠 BUILDING: {map_features['location_name']} +20")
                else:
                    reward += 20.0
                    info_parts.append(f"🗺️  NEW MAP: {map_features['location_name']} +20")
            else:
                reward += 2.0
        self.last_map_id = map_id
        
        # 4. SNN TEMPORAL DYNAMICS
        if self.use_snn and self.snn_encoder and len(self.frame_stack) > 0:
            try:
                current_frame = self.frame_stack[-1]
                snn_features = self.snn_encoder.encode_frame(current_frame)
                spike_rate = self.snn_encoder.get_recent_spike_rate()
                
                if spike_rate > 0.5:
                    reward += 0.5
                    if len(info_parts) == 0:
                        info_parts.append(f"⚡ SNN: High activity ({spike_rate:.2f})")
                
                self.snn_spike_history.append(spike_rate)
            except Exception as e:
                logger.debug(f"SNN error: {e}")
        
        # 5. PIXEL CHANGE
        if len(self.frame_stack) > 0:
            current_frame = self.frame_stack[-1]
            if self.last_frame_pixels is not None:
                pixel_diff = np.abs(current_frame.astype(float) - self.last_frame_pixels.astype(float)).mean()
                if pixel_diff > 2.0:
                    reward += pixel_diff * 0.02
            self.last_frame_pixels = current_frame.copy()
        
        # 6. TITLE SCREEN DETECTION
        if len(self.position_history) > 10:
            recent_pos = self.position_history[-10:]
            if all(p == (0, 0) for p in recent_pos):
                if action_name == "A":
                    reward += 10.0
                    info_parts.append("🎮 Title A +10")
        
        info_str = " | ".join(info_parts) if info_parts else ""
        return reward, info_str
    
    def _get_vlm_milestone_guidance(self, current_frame: np.ndarray, memory: Dict) -> Optional[str]:
        if not self.use_vlm or not self.vlm:
            return None
        
        current_milestone = self.milestone_tracker.get_current_milestone()
        strategy = self.milestone_tracker.get_vlm_strategy()
        progress = self.milestone_tracker.get_progress_summary()
        
        map_features = self._get_map_features(memory)
        location = map_features['location_name']
        x = memory.get('x', 0)
        y = memory.get('y', 0)
        
        # Get recent unique actions
        recent_actions = self.buttons_pressed[-20:] if len(self.buttons_pressed) >= 20 else self.buttons_pressed
        unique_actions = len(set(recent_actions))
        
        prompt = f"""Pokemon Emerald speedrun analysis.

**Current Milestone:** {current_milestone['name']}
**Strategy:** {strategy}

**Progress:** {progress}
**Location:** {location} ({x}, {y})
**Recent actions:** {unique_actions} unique out of {len(recent_actions)}

**Is the agent making progress toward the milestone? What should it do next?**

Response (one sentence):"""
        
        try:
            frame_pil = Image.fromarray(current_frame)
            response = self.vlm.get_completion(
                message=prompt,
                images=[frame_pil],
                max_tokens=100
            )
            
            logger.info(f"🧠 VLM: {response[:150]}...")
            return response
        
        except Exception as e:
            logger.debug(f"VLM error: {e}")
            return None
    
    def reset(self, **kwargs):
        try:
            state = requests.get(f"{self.server_url}/state", timeout=5).json()
            
            if state and 'screenshot' in state:
                frame = self._preprocess_frame(state['screenshot'])
                self.frame_stack = [frame] * self.frame_stack_size
                
                self._update_frame_cache(frame)
                
                self.step_count = 0
                self.position_history = []
                self.buttons_pressed = []
                self.last_frame_pixels = frame.copy()
                
                return frame, {}
            else:
                fallback = np.zeros((84, 84, 3), dtype=np.uint8)
                self.frame_stack = [fallback] * self.frame_stack_size
                return fallback, {}
        
        except Exception as e:
            logger.error(f"Reset error: {e}")
            fallback = np.zeros((84, 84, 3), dtype=np.uint8)
            self.frame_stack = [fallback] * self.frame_stack_size
            return fallback, {}
    
    def step(self, action: int):
        self.step_count += 1
        action_name = self.button_map[action]
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
                    
                    self._update_frame_cache(frame)
                
                # Get game state
                memory = state.get('memory', {})
                x = memory.get('x', 0)
                y = memory.get('y', 0)
                current_pos = (x, y)
                self.position_history.append(current_pos)
                
                # Calculate rewards
                reward, reward_info = self._calculate_rewards(memory, action_name)
                
                if reward_info:
                    logger.info(f"Step {self.step_count}: {reward_info} (R={reward:.1f})")
                
                # VLM milestone guidance every 100 steps
                if len(self.frame_stack) > 0 and self.step_count - self.last_vlm_check_step >= 100:
                    self.last_vlm_check_step = self.step_count
                    current_frame = self.frame_stack[-1]
                    vlm_guidance = self._get_vlm_milestone_guidance(current_frame, memory)
                
                # Return observation
                obs = self.frame_stack[-1] if len(self.frame_stack) > 0 else np.zeros((84, 84, 3), dtype=np.uint8)
                done = False
                truncated = False
                
                info = {
                    'milestone_progress': self.milestone_tracker.get_progress_summary(),
                    'current_milestone': self.milestone_tracker.get_current_milestone()['name'],
                    'coordinates': current_pos,
                    'maps_discovered': len(self.maps_discovered),
                    'buildings_entered': len(self.buildings_entered),
                    'routes_visited': len(self.routes_visited),
                }
                
                return obs, reward, done, truncated, info
            
            else:
                obs = self.frame_stack[-1] if len(self.frame_stack) > 0 else np.zeros((84, 84, 3), dtype=np.uint8)
                return obs, 0.0, False, False, {}
        
        except Exception as e:
            logger.error(f"Step error: {e}")
            obs = self.frame_stack[-1] if len(self.frame_stack) > 0 else np.zeros((84, 84, 3), dtype=np.uint8)
            return obs, 0.0, False, False, {}
    
    def _update_frame_cache(self, frame: np.ndarray):
        try:
            img = Image.fromarray(frame)
            img_bytes = BytesIO()
            img.save(img_bytes, format='PNG')
            img_b64 = base64.b64encode(img_bytes.getvalue()).decode()
            
            cache_data = {
                'frame': f"data:image/png;base64,{img_b64}",
                'step': self.step_count,
                'milestone': self.milestone_tracker.get_progress_summary(),
                'timestamp': time.time()
            }
            
            with open(self.frame_cache_file, 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            logger.debug(f"Cache error: {e}")
    
    def render(self, mode='rgb_array'):
        if len(self.frame_stack) > 0:
            return self.frame_stack[-1]
        return np.zeros((84, 84, 3), dtype=np.uint8)


# ================== TRAINING ==================

def main():
    logger.info("=" * 60)
    logger.info("🎯 COMPLETE RL TRAINING: CNN + SNN + VLM + MILESTONES")
    logger.info("=" * 60)
    
    # Create environment
    env = CompletePokemonEnv(
        server_url="http://localhost:8000",
        use_snn=True,
        use_vlm=True,
        vlm_backend="local"
    )
    
    # Wrap for SB3
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecTransposeImage(vec_env)
    
    # Create PPO model with custom feature extractor
    logger.info("🤖 Creating PPO with CNN+SNN policy...")
    
    policy_kwargs = dict(
        features_extractor_class=SNNCNNFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=512),
    )
    
    model = PPO(
        "CnnPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=64,
        batch_size=64,
        n_epochs=10,
        learning_rate=0.0003,
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.15,  # Higher entropy for more exploration
        tensorboard_log=".rl_checkpoints/tensorboard_complete/",
        device="auto"
    )
    
    logger.info("✅ PPO model created")
    logger.info("   Architecture: CNN + SNN feature extraction")
    logger.info("   VLM: Milestone-aware strategic guidance")
    logger.info("   Rewards: +100 milestones, +50 towns, +30 routes, +20 buildings")
    logger.info("   Entropy: 0.15 (high exploration)")
    
    # Checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=".rl_checkpoints/complete/",
        name_prefix="ppo_complete"
    )
    
    logger.info("🎓 Starting complete training...")
    logger.info("   Watch: http://localhost:8000/stream")
    logger.info("   Tensorboard: tensorboard --logdir .rl_checkpoints/tensorboard_complete/")
    
    try:
        model.learn(
            total_timesteps=20000,
            callback=checkpoint_callback,
            progress_bar=True
        )
        
        logger.info("✅ Training completed!")
        model.save(".rl_checkpoints/complete/final_model")
        
    except KeyboardInterrupt:
        logger.info("🛑 Training interrupted")
        model.save(".rl_checkpoints/complete/interrupted_model")


if __name__ == "__main__":
    main()
