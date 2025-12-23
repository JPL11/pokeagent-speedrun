#!/usr/bin/env python3
"""
RL Training with VLM Guidance using Server Architecture
Combines working server architecture with VLM-based stuck detection
"""

import os
import sys
import time
import logging
import argparse
import requests
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess
import signal
import base64
from io import BytesIO

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.logger import configure
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("❌ stable-baselines3 not installed")

import torch
import torch.nn as nn
from PIL import Image

# Import VLM
from utils.vlm import create_vlm
from agent.vlm_unstuck import VLMUnstuckDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CNNFeatureExtractor(BaseFeaturesExtractor):
    """Custom CNN for stacked frame observations"""
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]  # Should be 4 (stacked frames)
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample = torch.zeros(1, n_input_channels, 160, 240)
            n_flatten = self.cnn(sample).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


class ServerRLEnvWithVLM(gym.Env):
    """RL Environment that connects to game server AND uses VLM for stuck detection"""
    
    def __init__(self, server_url: str = "http://localhost:8000", vlm_backend: str = "qwen2b"):
        super().__init__()
        
        self.server_url = server_url
        self.frame_stack = []
        self.frame_stack_size = 4
        
        # Initialize VLM unstuck detector
        logger.info(f"🧠 Initializing VLM with {vlm_backend}...")
        vlm = create_vlm(vlm_backend)
        logger.info("✅ VLM loaded")
        
        logger.info("🧠 Initializing VLM unstuck detector...")
        self.vlm_detector = VLMUnstuckDetector(
            vlm=vlm,
            check_frequency=20,  # Check every 20 steps (more frequent for stuck detection)
            enable_suggestions=True,
            verbose=True
        )
        logger.info("✅ VLM unstuck detector ready")
        
        # Track title screen detection
        self.at_title_screen = True
        self.title_screen_steps = 0
        
        # RL spaces - match the checkpoint's observation space
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(4, 160, 240),  # 4 stacked grayscale frames
            dtype=np.float32
        )
        
        # Tracking
        self.step_count = 0
        self.last_position = (0, 0)
        self.position_history = []
        self.vlm_interventions = 0
        self.vlm_rewards = 0.0
        
        # Verify server is running
        try:
            response = requests.get(f"{server_url}/status", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Server is running and accessible")
            else:
                raise Exception(f"Server returned status {response.status_code}")
        except Exception as e:
            raise Exception(f"❌ Cannot connect to server at {server_url}: {e}")
    
    def _get_state_from_server(self):
        """Get current game state from server"""
        try:
            response = requests.get(f"{self.server_url}/state", timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"Error getting state: {e}")
            return None
    
    def _preprocess_frame(self, frame_b64: str) -> np.ndarray:
        """Convert base64 frame to normalized grayscale array"""
        try:
            img_bytes = base64.b64decode(frame_b64)
            img = Image.open(BytesIO(img_bytes))
            img = img.convert('L')  # Grayscale
            img = img.resize((240, 160))  # GBA resolution
            arr = np.array(img, dtype=np.float32) / 255.0  # Normalize to 0-1
            return arr
        except Exception as e:
            logger.error(f"Error preprocessing frame: {e}")
            return np.zeros((160, 240), dtype=np.float32)
    
    def _get_stacked_observation(self):
        """Get stacked frames as observation"""
        if len(self.frame_stack) < self.frame_stack_size:
            # Pad with zeros if not enough frames
            padding = [np.zeros((160, 240), dtype=np.float32)] * (self.frame_stack_size - len(self.frame_stack))
            frames = padding + self.frame_stack
        else:
            frames = self.frame_stack[-self.frame_stack_size:]
        
        return np.stack(frames, axis=0).astype(np.float32)
    
    def reset(self, **kwargs):
        """Reset environment"""
        try:
            # Reset via server
            response = requests.post(f"{self.server_url}/action", json={"action": "RESET"}, timeout=5)
            
            # Get initial state
            state = self._get_state_from_server()
            if state and 'screenshot' in state:
                frame = self._preprocess_frame(state['screenshot'])
                self.frame_stack = [frame] * self.frame_stack_size
            else:
                self.frame_stack = [np.zeros((160, 240), dtype=np.float32)] * self.frame_stack_size
            
            self.step_count = 0
            self.position_history = []
            self.vlm_detector.reset()
            
            logger.info("✅ Environment reset via server")
            return self._get_stacked_observation(), {}
            
        except Exception as e:
            logger.error(f"Reset failed: {e}")
            return np.zeros((4, 160, 240), dtype=np.float32), {}
    
    def step(self, action: int):
        """Take action in environment"""
        self.step_count += 1
        
        # Action mapping
        action_map = ["A", "B", "START", "SELECT", "UP", "DOWN", "LEFT", "RIGHT", "NOOP"]
        action_name = action_map[action]
        
        try:
            # Send action to server
            response = requests.post(
                f"{self.server_url}/action",
                json={"action": action_name},
                timeout=5
            )
            
            # Get new state
            state = self._get_state_from_server()
            
            if state:
                # Update frame stack
                if 'screenshot' in state:
                    frame = self._preprocess_frame(state['screenshot'])
                    self.frame_stack.append(frame)
                    if len(self.frame_stack) > self.frame_stack_size:
                        self.frame_stack.pop(0)
                
                # Extract position
                memory = state.get('memory', {})
                x = memory.get('x', 0)
                y = memory.get('y', 0)
                current_pos = (x, y)
                self.position_history.append(current_pos)
                
                # Calculate base reward
                reward = 0.0
                
                # Special handling for title screen / intro
                # If position is (0,0) for extended period, likely stuck at title
                if current_pos == (0, 0) and len(self.position_history) > 10:
                    recent_pos = self.position_history[-10:]
                    if all(p == (0, 0) for p in recent_pos):
                        # Stuck at title screen - strongly reward START/A action
                        if action_name in ["START", "A"]:
                            reward += 10.0  # HUGE reward for pressing START or A
                            logger.info(f"🎮 Pressed {action_name} at title screen! +10.0 reward")
                        else:
                            reward -= 2.0  # Strong penalty for wrong actions at title screen
                
                # Movement reward
                if current_pos != self.last_position:
                    reward += 0.1
                
                self.last_position = current_pos
                
                # Title screen detection
                if current_pos == (0, 0):
                    self.title_screen_steps += 1
                    if self.title_screen_steps > 30:
                        self.at_title_screen = True
                else:
                    self.at_title_screen = False
                    self.title_screen_steps = 0
                
                # VLM-guided stuck detection and reward shaping
                if state.get('screenshot'):
                    # Build game_state dict for VLM
                    game_state_info = {
                        'coordinates': current_pos,
                        'map_id': memory.get('map_id', 0),
                        'location': memory.get('location', 'UNKNOWN'),
                        'dialogue': memory.get('dialogue', '')
                    }
                    
                    # Update VLM tracker
                    self.vlm_detector.update_state(game_state_info, action_name)
                    
                    # For title screen, check more frequently
                    check_now = self.at_title_screen or self.step_count % 20 == 0
                    
                    if check_now:
                        # Get frame as PIL Image for VLM
                        from PIL import Image
                        from io import BytesIO
                        img_bytes = base64.b64decode(state['screenshot'])
                        frame_img = Image.open(BytesIO(img_bytes))
                        
                        vlm_result = self.vlm_detector.check_stuck(
                            frame=frame_img,
                            game_state=game_state_info
                        )
                        
                        if vlm_result:
                            # VLM detected stuck - give penalty
                            reward -= 1.0
                            self.vlm_interventions += 1
                            
                            logger.info(f"🧠 VLM: Agent stuck! Type: {vlm_result.pattern_type}")
                            logger.info(f"   Reasoning: {vlm_result.context}")
                            
                            # Apply VLM's suggested action as reward shaping
                            if vlm_result.suggested_action:
                                if action_name == vlm_result.suggested_action:
                                    reward += 1.0  # Reward for following VLM advice
                                    self.vlm_rewards += 1.0
                                    logger.info(f"✅ Agent followed VLM suggestion: {action_name}")
                
                # Check if done (can add termination conditions)
                done = False
                truncated = self.step_count >= 1000
                
                obs = self._get_stacked_observation()
                info = {
                    'position': current_pos,
                    'vlm_interventions': self.vlm_interventions,
                    'vlm_rewards': self.vlm_rewards
                }
                
                return obs, reward, done, truncated, info
            
            else:
                # Server error
                return self._get_stacked_observation(), -1.0, False, False, {}
                
        except Exception as e:
            logger.error(f"Step failed: {e}")
            return self._get_stacked_observation(), -1.0, False, False, {}


class VLMLoggingCallback(BaseCallback):
    """Callback to log VLM intervention stats"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.vlm_interventions = 0
        self.vlm_rewards = 0.0
    
    def _on_step(self) -> bool:
        # Log VLM stats every 100 steps
        if self.n_calls % 100 == 0:
            infos = self.locals.get('infos', [])
            if infos:
                info = infos[0]
                if 'vlm_interventions' in info:
                    logger.info(f"📊 VLM Stats - Interventions: {info['vlm_interventions']}, "
                              f"Rewards from VLM: {info['vlm_rewards']:.2f}")
        return True


def train_rl_with_vlm(
    server_url: str = "http://localhost:8000",
    vlm_backend: str = "qwen2b",
    total_timesteps: int = 50000,
    resume_from: Optional[str] = None
):
    """Train RL agent with VLM guidance"""
    
    if not HAS_SB3:
        raise ImportError("stable-baselines3 required")
    
    logger.info("🚀 Starting RL training with VLM GUIDANCE")
    logger.info(f"   Server: {server_url}")
    logger.info(f"   VLM Backend: {vlm_backend}")
    logger.info(f"   Timesteps: {total_timesteps:,}")
    
    # Create environment with VLM
    env = ServerRLEnvWithVLM(server_url=server_url, vlm_backend=vlm_backend)
    
    # Wrap in vectorized environment
    vec_env = DummyVecEnv([lambda: env])
    
    # Create or load PPO model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"   Using device: {device}")
    
    if resume_from and os.path.exists(resume_from):
        logger.info(f"📂 Loading checkpoint from: {resume_from}")
        model = PPO.load(resume_from, env=vec_env, device=device)
        logger.info("✅ Checkpoint loaded - now training with VLM guidance!")
    else:
        model = PPO(
            "CnnPolicy",
            vec_env,
            learning_rate=0.00025,
            n_steps=2048,
            batch_size=512,
            n_epochs=4,
            gamma=0.99,
            verbose=1,
            device=device,
            tensorboard_log=".rl_checkpoints/tensorboard_vlm/",
            policy_kwargs=dict(
                features_extractor_class=CNNFeatureExtractor,
                features_extractor_kwargs=dict(features_dim=512)
            )
        )
        logger.info("✅ PPO model created from scratch with VLM")
    
    logger.info(f"   Policy device: {next(model.policy.parameters()).device}")
    
    # Setup callbacks
    os.makedirs(".rl_checkpoints", exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=".rl_checkpoints",
        name_prefix="pokemon_emerald_rl_vlm"
    )
    vlm_callback = VLMLoggingCallback()
    
    # Train!
    logger.info("🎓 Starting training with VLM guidance...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, vlm_callback],
        progress_bar=True
    )
    
    # Save final model
    final_path = ".rl_checkpoints/pokemon_emerald_rl_vlm_final.zip"
    model.save(final_path)
    logger.info(f"✅ Training complete! VLM-guided model saved to {final_path}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="RL Training with VLM Guidance")
    parser.add_argument("--server-url", type=str, default="http://localhost:8000",
                       help="Server URL")
    parser.add_argument("--vlm-backend", type=str, default="qwen2b",
                       choices=["qwen2b", "qwen", "local", "gpt", "claude", "gemini"],
                       help="VLM backend to use")
    parser.add_argument("--timesteps", type=int, default=50000,
                       help="Total training timesteps")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--start-server", action="store_true",
                       help="Start server automatically")
    
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     Pokemon Emerald RL Training - VLM GUIDED MODE            ║
║     (Qwen 2B Vision helps agent learn naturally!)            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Start server if requested
    server_process = None
    if args.start_server:
        logger.info("🚀 Starting server...")
        server_process = subprocess.Popen(
            ["python", "-m", "server.app", "--port", "8000"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        time.sleep(5)
        logger.info("✅ Server started")
    
    try:
        # Train with VLM
        model = train_rl_with_vlm(
            server_url=args.server_url,
            vlm_backend=args.vlm_backend,
            total_timesteps=args.timesteps,
            resume_from=args.resume
        )
        
        print("\n✅ VLM-guided training complete!")
        print("   Agent now learns to unstuck itself naturally with VLM reasoning!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if server_process:
            logger.info("🛑 Stopping server...")
            server_process.terminate()
            server_process.wait()


if __name__ == "__main__":
    main()
