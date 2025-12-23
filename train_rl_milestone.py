#!/usr/bin/env python3
"""
Milestone-Driven RL Training with Enhanced VLM Guidance

This version focuses on game progression through major milestones:
1. Title screen → New game
2. Character creation
3. Exit starter house
4. Get Pokeball from Dad
5. Route 101 → Route 103 
6. Back to Lab → Get starter
7. Route 101 → Oldale Town
8. First Pokecenter heal
... and so on

The VLM provides strategic reasoning for each milestone phase.
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
from agent.vlm_unstuck import VLMUnstuckDetector

# Map features
from pokemon_env.enums import MapLocation
try:
    from utils.map_stitcher import MapStitcher
    HAS_MAP_STITCHER = True
except ImportError:
    HAS_MAP_STITCHER = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================== MILESTONE DEFINITIONS ==================

EMERALD_MILESTONES = [
    {
        "id": "TITLE_SCREEN",
        "name": "Press A on Title Screen",
        "description": "Start the game by pressing A",
        "check": lambda mem: mem.get('x', 0) != 0 or mem.get('y', 0) != 0,
        "vlm_strategy": "You need to press A to start the game. Keep pressing A until you see character selection."
    },
    {
        "id": "CHARACTER_CREATED",
        "name": "Complete Character Creation",
        "description": "Choose gender, name, rival name",
        "check": lambda mem: mem.get('map_id', 0) == 0x0309,  # PLAYER_HOUSE_2F
        "vlm_strategy": "Navigate the dialogue by pressing A. Choose your character and confirm selections."
    },
    {
        "id": "ENTERED_ROOM",
        "name": "Wake Up in Bedroom",
        "description": "Initial spawn in player's room",
        "check": lambda mem: mem.get('map_id', 0) == 0x0309 and mem.get('x', 0) > 0,
        "vlm_strategy": "You're in your bedroom. Go downstairs (DOWN then exit door) to meet your mom."
    },
    {
        "id": "DOWNSTAIRS",
        "name": "Reach Ground Floor",
        "description": "Exit bedroom to downstairs",
        "check": lambda mem: mem.get('map_id', 0) == 0x0201,  # PLAYER_HOUSE_1F
        "vlm_strategy": "Go DOWN to talk to your mom, then exit the house through the bottom door."
    },
    {
        "id": "LEFT_HOUSE",
        "name": "Exit Starter House",
        "description": "First time outside in Littleroot Town",
        "check": lambda mem: mem.get('map_id', 0) == 0x0101,  # LITTLEROOT_TOWN
        "vlm_strategy": "Explore Littleroot Town. Go UP and LEFT to find Professor Birch being attacked. Save him!"
    },
    {
        "id": "SAVED_BIRCH",
        "name": "Saved Prof Birch from Poochyena",
        "description": "Chose starter and defeated wild Pokemon",
        "check": lambda mem: mem.get('party_size', 0) > 0,  # Has a Pokemon
        "vlm_strategy": "You have a Pokemon now! Go to Route 103 (UP from Littleroot) to find your rival May/Brendan."
    },
    {
        "id": "ROUTE_103",
        "name": "Reached Route 103",
        "description": "First route area",
        "check": lambda mem: mem.get('map_id', 0) == 0x0201,  # ROUTE103
        "vlm_strategy": "Battle your rival, then head back to Littleroot to visit the Pokemon Lab."
    },
    {
        "id": "GOT_POKEDEX",
        "name": "Received Pokedex",
        "description": "Got Pokedex from Birch",
        "check": lambda mem: mem.get('has_pokedex', False),
        "vlm_strategy": "Now you can travel! Go through Route 101 to reach Oldale Town (your first new town)."
    },
    {
        "id": "OLDALE_TOWN",
        "name": "Arrived at Oldale Town",
        "description": "First town visited",
        "check": lambda mem: mem.get('map_id', 0) == 0x0201,  # OLDALE_TOWN
        "vlm_strategy": "Visit the Pokemon Center to heal. Talk to NPCs. Then continue NORTH to Route 103."
    },
    {
        "id": "FIRST_POKECENTER",
        "name": "First Pokemon Center Heal",
        "description": "Healed at Pokemon Center",
        "check": lambda mem: mem.get('healed_at_center', False),
        "vlm_strategy": "Great! You know how to heal. Continue exploring routes and towns."
    },
    {
        "id": "ROUTE_102",
        "name": "Reached Route 102",
        "description": "Route to Petalburg",
        "check": lambda mem: mem.get('map_id', 0) == 0x0102,
        "vlm_strategy": "Battle trainers for experience. Continue WEST to reach Petalburg City."
    },
    {
        "id": "PETALBURG_CITY",
        "name": "Arrived at Petalburg City",
        "description": "First gym location",
        "check": lambda mem: mem.get('map_id', 0) == 0x0102,
        "vlm_strategy": "Visit Petalburg Gym but you can't battle yet. Talk to Norman (your dad). Then head to Route 104."
    },
    {
        "id": "FIRST_GYM_BADGE",
        "name": "Defeated First Gym Leader",
        "description": "Got Stone Badge from Roxanne",
        "check": lambda mem: mem.get('badges', 0) > 0,
        "vlm_strategy": "Congratulations! You're making progress. Continue to the next city and gym."
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
        """Get the current active milestone"""
        if self.current_milestone_idx < len(self.milestones):
            return self.milestones[self.current_milestone_idx]
        return self.milestones[-1]  # Stay on last milestone
    
    def check_milestone_completion(self, memory: Dict) -> Optional[Dict]:
        """Check if current milestone is completed"""
        current = self.get_current_milestone()
        
        if current['id'] in self.completed_milestones:
            return None
        
        if current['check'](memory):
            # Milestone completed!
            elapsed = time.time() - self.start_time
            self.milestone_times[current['id']] = elapsed
            self.completed_milestones.add(current['id'])
            
            logger.info(f"🎯 MILESTONE COMPLETED: {current['name']} ({elapsed:.1f}s)")
            
            # Move to next milestone
            self.current_milestone_idx += 1
            return current
        
        return None
    
    def get_vlm_strategy(self) -> str:
        """Get VLM strategic guidance for current milestone"""
        current = self.get_current_milestone()
        return current.get('vlm_strategy', 'Explore and progress through the game.')
    
    def get_progress_summary(self) -> str:
        """Get human-readable progress summary"""
        total = len(self.milestones)
        completed = len(self.completed_milestones)
        current = self.get_current_milestone()
        return f"{completed}/{total} milestones | Current: {current['name']}"


# ================== ENVIRONMENT ==================

class MilestonePokemonEnv(gym.Env):
    """Pokemon environment with milestone-driven progression"""
    
    metadata = {'render.modes': ['rgb_array']}
    
    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        use_vlm: bool = True,
        vlm_backend: str = "local",
    ):
        super().__init__()
        
        self.server_url = server_url
        
        # Observation: RGB frames (84x84x3)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 3), dtype=np.uint8
        )
        
        # Actions: 7 buttons (removed START/SELECT to avoid menu traps)
        self.button_map = ["A", "B", "UP", "DOWN", "LEFT", "RIGHT", "NOOP"]
        self.action_space = spaces.Discrete(len(self.button_map))
        
        # Frame stacking for temporal context
        self.frame_stack_size = 4
        self.frame_stack = []
        
        # Milestone tracking
        self.milestone_tracker = MilestoneProgressTracker(EMERALD_MILESTONES)
        logger.info(f"📊 Loaded {len(EMERALD_MILESTONES)} milestones for progression tracking")
        
        # VLM for milestone-aware strategic guidance
        self.use_vlm = use_vlm
        self.vlm = None
        if use_vlm:
            logger.info(f"🧠 Initializing VLM ({vlm_backend}) for milestone-aware guidance...")
            try:
                self.vlm = create_vlm(vlm_backend)
                logger.info("✅ VLM loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️  VLM failed: {e}")
                self.use_vlm = False
        
        # Map features
        self.map_stitcher = None
        if HAS_MAP_STITCHER:
            try:
                self.map_stitcher = MapStitcher()
                logger.info("✅ Map stitcher loaded")
            except Exception as e:
                logger.warning(f"⚠️  Map stitcher failed: {e}")
        
        # Tracking
        self.step_count = 0
        self.seen_coords = {}
        self.last_map_id = None
        self.maps_discovered = set()
        self.buildings_entered = set()
        
        # VLM guidance
        self.last_vlm_check_step = 0
        self.vlm_advice_history = []
        
        # Cache for browser display
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
                logger.info("✅ Server is accessible")
        except Exception as e:
            raise Exception(f"❌ Cannot connect to server: {e}")
    
    def _preprocess_frame(self, frame_b64: str) -> np.ndarray:
        """Convert frame to RGB for CNN"""
        try:
            img_bytes = base64.b64decode(frame_b64)
            img = Image.open(BytesIO(img_bytes))
            img = img.resize((84, 84))
            img = img.convert('RGB')
            return np.array(img, dtype=np.uint8)
        except Exception as e:
            logger.error(f"Frame preprocessing error: {e}")
            return np.zeros((84, 84, 3), dtype=np.uint8)
    
    def _get_map_features(self, memory: Dict) -> Dict[str, Any]:
        """Extract map-aware features"""
        map_id = memory.get('map_id', 0)
        
        # Get location name
        try:
            location = MapLocation(map_id)
            location_name = location.name
        except ValueError:
            location_name = f"UNKNOWN_{map_id:04X}"
        
        features = {
            'location_name': location_name,
            'map_id': map_id,
            'is_indoor': 'HOUSE' in location_name or 'CENTER' in location_name or 'MART' in location_name,
            'is_route': 'ROUTE' in location_name,
            'is_town': 'TOWN' in location_name or 'CITY' in location_name,
            'is_gym': 'GYM' in location_name,
        }
        
        return features
    
    def _calculate_rewards(self, memory: Dict, action_name: str) -> Tuple[float, str]:
        """Calculate rewards with milestone awareness"""
        reward = 0.0
        info_parts = []
        
        x = memory.get('x', 0)
        y = memory.get('y', 0)
        map_id = memory.get('map_id', 0)
        
        # 1. MILESTONE COMPLETION REWARD (HIGHEST PRIORITY)
        completed_milestone = self.milestone_tracker.check_milestone_completion(memory)
        if completed_milestone:
            reward += 100.0  # HUGE reward for milestone completion
            info_parts.append(f"🎯 MILESTONE: {completed_milestone['name']} +100")
        
        # 2. NEW COORDINATE EXPLORATION
        coord_key = (x, y, map_id)
        if coord_key not in self.seen_coords:
            self.seen_coords[coord_key] = 1
            reward += 5.0
            info_parts.append(f"🆕 New tile +5")
        else:
            self.seen_coords[coord_key] += 1
            if self.seen_coords[coord_key] > 10:
                reward -= 0.5  # Penalty for revisiting
        
        # 3. NEW MAP DISCOVERY
        if map_id != self.last_map_id and self.last_map_id is not None:
            if map_id not in self.maps_discovered:
                reward += 20.0
                self.maps_discovered.add(map_id)
                info_parts.append(f"🗺️  New map +20")
            else:
                reward += 2.0  # Smaller reward for map transitions
        self.last_map_id = map_id
        
        # 4. BUILDING ENTRY REWARD
        map_features = self._get_map_features(memory)
        if map_features['is_indoor']:
            building_key = map_id
            if building_key not in self.buildings_entered:
                self.buildings_entered.add(building_key)
                if map_features['is_gym']:
                    reward += 50.0  # GYM is super important!
                    info_parts.append(f"🏋️ GYM ENTERED +50")
                else:
                    reward += 15.0
                    info_parts.append(f"🏠 Building +15")
        
        # 5. TITLE SCREEN BYPASS
        if len(self.position_history) > 10:
            recent_pos = self.position_history[-10:]
            if all(p == (0, 0) for p in recent_pos):
                if action_name == "A":
                    reward += 10.0
                    info_parts.append("🎮 Title A +10")
        
        # 6. PIXEL CHANGE (lowest priority)
        if len(self.frame_stack) > 0:
            current_frame = self.frame_stack[-1]
            if self.last_frame_pixels is not None:
                pixel_diff = np.abs(current_frame.astype(float) - self.last_frame_pixels.astype(float)).mean()
                if pixel_diff > 2.0:
                    reward += pixel_diff * 0.02
            self.last_frame_pixels = current_frame.copy()
        
        info_str = " | ".join(info_parts) if info_parts else ""
        return reward, info_str
    
    def _get_vlm_milestone_guidance(self, current_frame: np.ndarray, memory: Dict) -> Optional[str]:
        """Get VLM strategic advice for current milestone"""
        if not self.use_vlm or not self.vlm:
            return None
        
        current_milestone = self.milestone_tracker.get_current_milestone()
        strategy = self.milestone_tracker.get_vlm_strategy()
        progress = self.milestone_tracker.get_progress_summary()
        
        # Get recent actions
        recent_actions = self.buttons_pressed[-10:] if len(self.buttons_pressed) >= 10 else self.buttons_pressed
        action_str = ", ".join(recent_actions[-10:])
        
        # Get location
        map_features = self._get_map_features(memory)
        location = map_features['location_name']
        
        x = memory.get('x', 0)
        y = memory.get('y', 0)
        
        prompt = f"""You are a Pokemon Emerald speedrunning coach analyzing the agent's progress.

**Current Milestone:**
{current_milestone['name']} - {current_milestone['description']}

**Strategy for this milestone:**
{strategy}

**Current State:**
- Progress: {progress}
- Location: {location}
- Coordinates: ({x}, {y})
- Recent actions: {action_str}

**Questions:**
1. Is the agent making progress towards the current milestone?
2. Is the agent stuck in a loop or confused?
3. What specific action should the agent take next?

**Response format (JSON-like):**
{{
    "making_progress": true/false,
    "is_stuck": true/false,
    "reasoning": "...",
    "suggested_action": "A/B/UP/DOWN/LEFT/RIGHT",
    "strategic_advice": "..."
}}

Respond with analysis:"""
        
        try:
            frame_pil = Image.fromarray(current_frame)
            response = self.vlm.get_completion(
                message=prompt,
                images=[frame_pil],
                max_tokens=300
            )
            
            logger.info(f"🧠 VLM Milestone Guidance: {response[:200]}...")
            self.vlm_advice_history.append({
                'step': self.step_count,
                'milestone': current_milestone['id'],
                'advice': response
            })
            
            return response
        
        except Exception as e:
            logger.warning(f"VLM guidance error: {e}")
            return None
    
    def reset(self, **kwargs):
        """Reset environment"""
        try:
            # Get initial state
            state = requests.get(f"{self.server_url}/state", timeout=5).json()
            
            if state and 'screenshot' in state:
                frame = self._preprocess_frame(state['screenshot'])
                self.frame_stack = [frame] * self.frame_stack_size
                
                # Update frame cache for browser
                self._update_frame_cache(frame)
                
                self.step_count = 0
                self.position_history = []
                self.buttons_pressed = []
                self.last_frame_pixels = None
                
                return frame, {}
            else:
                return np.zeros((84, 84, 3), dtype=np.uint8), {}
        
        except Exception as e:
            logger.error(f"Reset error: {e}")
            return np.zeros((84, 84, 3), dtype=np.uint8), {}
    
    def step(self, action: int):
        """Execute action and return observation"""
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
                    
                    # Update cache
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
                    logger.info(f"Step {self.step_count}: {reward_info} (Total: {reward:.1f})")
                
                # VLM milestone guidance every 50 steps
                if len(self.frame_stack) > 0 and self.step_count - self.last_vlm_check_step >= 50:
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
                }
                
                return obs, reward, done, truncated, info
            
            else:
                return self.frame_stack[-1], 0.0, False, False, {}
        
        except Exception as e:
            logger.error(f"Step error: {e}")
            return self.frame_stack[-1], 0.0, False, False, {}
    
    def _update_frame_cache(self, frame: np.ndarray):
        """Update frame cache for browser display"""
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
            logger.debug(f"Cache update error: {e}")
    
    def render(self, mode='rgb_array'):
        if len(self.frame_stack) > 0:
            return self.frame_stack[-1]
        return np.zeros((84, 84, 3), dtype=np.uint8)


# ================== TRAINING ==================

def main():
    logger.info("=" * 60)
    logger.info("🎯 MILESTONE-DRIVEN RL TRAINING")
    logger.info("=" * 60)
    
    # Create environment
    env = MilestonePokemonEnv(
        server_url="http://localhost:8000",
        use_vlm=True,
        vlm_backend="local"
    )
    
    # Wrap for SB3
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecTransposeImage(vec_env)
    
    # Create PPO model
    logger.info("🤖 Creating PPO model with CnnPolicy...")
    model = PPO(
        "CnnPolicy",
        vec_env,
        verbose=1,
        n_steps=64,
        batch_size=64,
        n_epochs=10,
        learning_rate=0.0003,
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.12,
        tensorboard_log=".rl_checkpoints/tensorboard_milestone/",
        device="auto"
    )
    
    logger.info("✅ PPO model created")
    logger.info("   Policy: CnnPolicy")
    logger.info("   Updates: every 64 steps")
    logger.info("   Milestone-driven rewards: +100 per milestone")
    
    # Checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=500,
        save_path=".rl_checkpoints/milestone/",
        name_prefix="ppo_milestone"
    )
    
    logger.info("🎓 Starting milestone-driven training...")
    logger.info("   Watch at: http://localhost:8000/stream")
    logger.info("   Tensorboard: tensorboard --logdir .rl_checkpoints/tensorboard_milestone/")
    
    try:
        model.learn(
            total_timesteps=10000,
            callback=checkpoint_callback,
            progress_bar=True
        )
        
        logger.info("✅ Training completed!")
        model.save(".rl_checkpoints/milestone/final_model")
        
    except KeyboardInterrupt:
        logger.info("🛑 Training interrupted by user")
        model.save(".rl_checkpoints/milestone/interrupted_model")


if __name__ == "__main__":
    main()
