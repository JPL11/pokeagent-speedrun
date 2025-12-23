#!/usr/bin/env python3
"""
Fast RL Training for Title Screen - From Scratch
Learns to press START/A buttons to get past title screen quickly
"""

import os
import sys
import os
import requests
import numpy as np
from pathlib import Path
from typing import Dict, Any
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
import torch
import torch.nn as nn
from PIL import Image

# VLM imports for high-level reasoning
from utils.vlm import create_vlm
from agent.vlm_unstuck import VLMUnstuckDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleCNN(nn.Module):
    """Lightweight CNN for title screen detection"""
    def __init__(self, observation_space, features_dim=256):
        super().__init__()
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        with torch.no_grad():
            sample = torch.zeros(1, n_input_channels, 160, 240)
            n_flatten = self.cnn(sample).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
        
        self.output_dim = features_dim
    
    def forward(self, observations):
        return self.linear(self.cnn(observations))


class TitleScreenEnv(gym.Env):
    """Environment with VLM guidance for high-level reasoning + RL for actions"""
    
    def __init__(self, server_url="http://localhost:8000", use_vlm=True, vlm_backend="qwen2b"):
        super().__init__()
        self.server_url = server_url
        self.frame_stack = []
        self.frame_stack_size = 1  # Single frame for CNN
        
        # VLM for high-level strategic guidance
        self.use_vlm = use_vlm
        self.vlm_detector = None
        if use_vlm:
            logger.info(f"🧠 Initializing VLM ({vlm_backend}) for high-level guidance...")
            try:
                vlm = create_vlm(vlm_backend)
                self.vlm_detector = VLMUnstuckDetector(
                    vlm=vlm,
                    check_frequency=50,  # Check every 50 steps (strategic, not every step)
                    enable_suggestions=True,
                    verbose=True
                )
                logger.info("✅ VLM guidance enabled - provides strategic reasoning")
            except Exception as e:
                logger.warning(f"⚠️  VLM initialization failed: {e}. Continuing without VLM.")
                self.use_vlm = False
                self.vlm_detector = None
        
        # 7 actions: A, B, UP, DOWN, LEFT, RIGHT, NOOP (removed START and SELECT!)
        self.action_space = spaces.Discrete(7)
        # Shape: (height, width, channels) - 3 channels for RGB
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(84, 84, 3),  # Standard size for CNN, RGB
            dtype=np.uint8
        )
        
        # Track pixels for exploration reward
        self.last_frame_pixels = None
        self.total_pixel_changes = 0
        
        # Frame cache for agent thinking display
        self.cache_dir = ".pokeagent_cache"
        self.frame_cache_file = os.path.join(self.cache_dir, "frame_cache.json")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.step_count = 0
        self.position_history = []
        self.buttons_pressed = []
        self.last_position = (0, 0)
        
        # EXPLORATION TRACKING (PokemonRedExperiments style)
        self.seen_coords = {}  # {(x, y, map_id): visit_count}
        self.last_map_id = None
        self.maps_discovered = set()
        
        # VLM guidance tracking
        self.vlm_suggestions = []
        self.vlm_guidance_bonus = 0.0
        self.last_vlm_check_step = 0
        
        # Check server
        try:
            response = requests.get(f"{server_url}/status", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Server is accessible")
        except Exception as e:
            raise Exception(f"❌ Cannot connect to server: {e}")
    
    def _preprocess_frame(self, frame_b64: str) -> np.ndarray:
        """Convert frame to RGB for CNN (like PokemonRedExperiments)"""
        try:
            img_bytes = base64.b64decode(frame_b64)
            img = Image.open(BytesIO(img_bytes))
            img = img.convert('RGB')  # Keep RGB for CNN
            img = img.resize((84, 84))  # Standard CNN size
            return np.array(img, dtype=np.uint8)
        except Exception as e:
            return np.zeros((84, 84, 3), dtype=np.uint8)
    
    def _get_stacked_observation(self):
        """Return current frame for CNN"""
        if len(self.frame_stack) > 0:
            return self.frame_stack[-1]
        else:
            return np.zeros((84, 84, 3), dtype=np.uint8)
    
    def reset(self, **kwargs):
        """Reset environment"""
        try:
            # Server expects {"buttons": [...]} as a list
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
            self.total_pixel_changes = 0
            self.seen_coords = {}
            self.last_map_id = None
            self.maps_discovered = set()
            
            return self._get_stacked_observation(), {}
        except Exception as e:
            logger.error(f"Reset failed: {e}")
            return np.zeros((84, 84, 3), dtype=np.uint8), {}
    
    def step(self, action: int):
        """Take action"""
        self.step_count += 1
        
        # Removed START and SELECT - focus on core movement and A!
        action_map = ["A", "B", "UP", "DOWN", "LEFT", "RIGHT", "NOOP"]
        action_name = action_map[action]
        self.buttons_pressed.append(action_name)
        
        try:
            # Send action - server expects {"buttons": [...]} as a list
            action_response = requests.post(f"{self.server_url}/action", json={"buttons": [action_name]}, timeout=5)
            if self.step_count % 50 == 0:
                logger.info(f"🔍 Action sent: {action_name}, Response: {action_response.status_code}")
            state = requests.get(f"{self.server_url}/state", timeout=5).json()
            
            if state:
                # Update frames
                if 'screenshot' in state:
                    frame = self._preprocess_frame(state['screenshot'])
                    self.frame_stack.append(frame)
                    if len(self.frame_stack) > self.frame_stack_size:
                        self.frame_stack.pop(0)
                
                # Get position and map_id (like PokemonRedExperiments)
                memory = state.get('memory', {})
                x = memory.get('x', 0)
                y = memory.get('y', 0)
                map_id = memory.get('map_id', 0)
                current_pos = (x, y)
                self.position_history.append(current_pos)
                
                # REWARD CALCULATION - PokemonRedExperiments style: NEW LOCATIONS!
                reward = 0.0
                
                # CORE REWARD: Visiting NEW coordinates (most important!)
                coord_key = (x, y, map_id)
                if coord_key not in self.seen_coords:
                    # First time visiting this tile!
                    self.seen_coords[coord_key] = 1
                    reward += 5.0  # BIG reward for new tile
                    logger.info(f"🆕 NEW TILE! x:{x} y:{y} map:{map_id:04X} - Total explored: {len(self.seen_coords)}")
                else:
                    # Already visited - small penalty to discourage loops
                    self.seen_coords[coord_key] += 1
                    visit_count = self.seen_coords[coord_key]
                    if visit_count > 10:  # Too many visits to same tile
                        reward -= 0.5  # Penalty for looping
                        if self.step_count % 20 == 0:
                            logger.info(f"🔄 Loop detected at x:{x} y:{y} (visited {visit_count}x)")
                
                # BONUS: Discovering new map (exiting house, entering new area!)
                if map_id != self.last_map_id and self.last_map_id is not None:
                    if map_id not in self.maps_discovered:
                        reward += 20.0  # HUGE bonus for new map
                        self.maps_discovered.add(map_id)
                        logger.info(f"🎉 NEW MAP DISCOVERED! map_id: {map_id:04X} - Total maps: {len(self.maps_discovered)}")
                    else:
                        reward += 2.0  # Small bonus for changing maps
                        logger.info(f"🚪 Changed map to: {map_id:04X}")
                self.last_map_id = map_id
                
                # SECONDARY: Pixel change reward (lower weight than coord rewards)
                current_frame = self.frame_stack[-1]
                if self.last_frame_pixels is not None:
                    pixel_diff = np.abs(current_frame.astype(float) - self.last_frame_pixels.astype(float)).mean()
                    if pixel_diff > 2.0:  # Only significant changes
                        reward += pixel_diff * 0.05  # Lower weight than coord rewards
                        self.total_pixel_changes += pixel_diff
                
                self.last_frame_pixels = current_frame.copy()
                
                # Title screen: reward A button to progress (if at (0,0))
                if len(self.position_history) > 10:
                    recent_pos = self.position_history[-10:]
                    if all(p == (0, 0) for p in recent_pos):
                        if action_name == "A":
                            reward += 10.0
                            logger.info(f"🎮 PRESSED A at title! +10.0")
                
                # NOOP penalty (encourage action)
                if action_name == "NOOP":
                    reward -= 0.3
                
                self.last_position = current_pos
                
                # VLM HIGH-LEVEL GUIDANCE (strategic reasoning every 50 steps)
                vlm_insight = ""
                if self.use_vlm and self.vlm_detector and self.step_count > 10:
                    # Update VLM with current state
                    self.vlm_detector.update_state({
                        'coordinates': current_pos,
                        'map_id': map_id,
                        'location': memory.get('location', 'UNKNOWN'),
                        'dialogue': memory.get('dialogue', '')
                    }, action_name)
                    
                    # Periodically ask VLM for strategic analysis (not every step!)
                    if self.step_count - self.last_vlm_check_step >= 50:
                        self.last_vlm_check_step = self.step_count
                        try:
                            current_frame_pil = Image.fromarray(current_frame)
                            
                            stuck_pattern = self.vlm_detector.check_stuck(current_frame_pil, {
                                'coordinates': current_pos,
                                'map_id': map_id,
                                'location': memory.get('location', 'UNKNOWN')
                            })
                            
                            if stuck_pattern:
                                # VLM detected stuck - provide guidance
                                if stuck_pattern.pattern_type == 'coordinate_loop':
                                    reward -= 2.0  # Extra penalty for confirmed loops
                                    vlm_insight = f"🧠 VLM: Loop detected! Explore new areas"
                                    logger.info(f"🧠 VLM GUIDANCE: {stuck_pattern.context}")
                                elif stuck_pattern.suggested_action:
                                    self.vlm_suggestions.append(stuck_pattern.suggested_action)
                                    vlm_insight = f"🧠 VLM: {stuck_pattern.context[:60]}"
                                    reward += 0.5  # Small bonus for VLM providing insight
                            else:
                                # VLM confirms good exploration
                                if len(self.seen_coords) > 10:
                                    vlm_insight = "🧠 VLM: Good exploration progress!"
                                    reward += 1.0  # Bonus for confirmed good behavior
                        except Exception as e:
                            logger.debug(f"VLM check error: {e}")
                
                # Update frame cache for browser display
                try:
                    recent_buttons = self.buttons_pressed[-10:] if len(self.buttons_pressed) >= 10 else self.buttons_pressed
                    agent_thinking = f"Step {self.step_count} | {action_name} | x:{x} y:{y} map:{map_id:04X}"
                    agent_thinking += f"\nReward: {reward:.1f} | Explored: {len(self.seen_coords)} tiles, {len(self.maps_discovered)} maps"
                    if vlm_insight:
                        agent_thinking += f"\n{vlm_insight}"
                    agent_thinking += f"\nRecent: {', '.join(recent_buttons[-5:])}"
                    
                    cache_data = {
                        "frame": state.get('screenshot', ''),
                        "frame_counter": self.step_count,
                        "agent_thinking": agent_thinking,
                        "recent_buttons": recent_buttons
                    }
                    
                    with open(self.frame_cache_file, 'w') as f:
                        json.dump(cache_data, f)
                except Exception as e:
                    logger.debug(f"Error updating frame cache: {e}")
                
                # Episode ends after 500 steps
                done = False
                truncated = self.step_count >= 500
                
                obs = self._get_stacked_observation()
                info = {'position': current_pos, 'buttons': self.buttons_pressed[-10:]}
                
                return obs, reward, done, truncated, info
            
            return self._get_stacked_observation(), -1.0, False, False, {}
            
        except Exception as e:
            logger.error(f"Step failed: {e}")
            return self._get_stacked_observation(), -1.0, False, False, {}


class StatsCallback(BaseCallback):
    """Log statistics"""
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self):
        if self.n_calls % 100 == 0:
            infos = self.locals.get('infos', [])
            if infos:
                info = infos[0]
                if 'buttons' in info:
                    logger.info(f"📊 Recent buttons: {info['buttons']}")
        return True


def train_title_screen():
    """Train agent to get past title screen"""
    
    logger.info("🚀 Training agent to press START at title screen")
    logger.info("   ULTRA FAST learning mode - updates every 64 steps")
    
    # Create environment
    env = TitleScreenEnv(server_url="http://localhost:8000")
    vec_env = DummyVecEnv([lambda: env])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"   Device: {device}")
    
    # Create PPO with FAST learning and CNN
    policy_kwargs = dict(
        features_extractor_kwargs=dict(features_dim=512)
    )
    
    model = PPO(
        "CnnPolicy",  # Use CNN for image observations!
        vec_env,
        learning_rate=0.001,  # VERY high learning rate
        n_steps=64,  # Update every 64 steps for ultra-fast learning!
        batch_size=64,
        n_epochs=10,  # Learn intensively from each batch
        gamma=0.98,  # Shorter-term focus
        ent_coef=0.15,  # VERY HIGH exploration
        clip_range=0.2,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
        tensorboard_log=".rl_checkpoints/tensorboard_title/",
    )
    
    logger.info("✅ PPO model created for title screen learning")
    logger.info(f"   Updates every {model.n_steps} steps (~15-20 seconds!)")
    
    # Setup callbacks
    os.makedirs(".rl_checkpoints", exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=".rl_checkpoints",
        name_prefix="title_screen"
    )
    stats_callback = StatsCallback()
    
    # Train!
    logger.info("🎓 Starting fast training...")
    model.learn(
        total_timesteps=10000,  # Just 10k steps to learn title screen
        callback=[checkpoint_callback, stats_callback],
        progress_bar=True
    )
    
    # Save
    final_path = ".rl_checkpoints/title_screen_final.zip"
    model.save(final_path)
    logger.info(f"✅ Training complete! Saved to {final_path}")
    
    return model


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     Title Screen Speedrun - From Scratch Training           ║
║     Updates every 128 steps for fast learning!              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    train_title_screen()
