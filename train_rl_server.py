#!/usr/bin/env python3
"""
RL Training using the WORKING Server Architecture
This uses the proven server/client pattern (like VLM) instead of direct emulator access.

Benefits:
- Uses the working server architecture
- Proper GPU utilization (model on GPU, emulator on CPU)
- More stable and proven
- Can integrate with VLM later
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

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.logger import configure
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("❌ stable-baselines3 not installed. Install with: pip install stable-baselines3")

import torch
import torch.nn as nn
from PIL import Image
import base64
from io import BytesIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ServerRLEnv(gym.Env):
    """
    RL Environment that uses the WORKING server architecture
    (like the VLM code does)
    """
    
    metadata = {"render_modes": []}
    
    def __init__(self, server_url: str = "http://localhost:8000", frame_stack_size: int = 4):
        super().__init__()
        self.server_url = server_url
        self.frame_stack_size = frame_stack_size
        
        # Action space: same as VLM agent
        self.action_list = ['A', 'B', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'START', 'L', 'R']
        self.action_space = spaces.Discrete(len(self.action_list))
        
        # Observation space: channel-first format for stable-baselines3
        # (channels, height, width) = (stack_size, 160, 240)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(frame_stack_size, 160, 240),  # (channels, height, width) - channel-first!
            dtype=np.float32
        )
        
        # State tracking
        self.frame_stack = []
        self.step_count = 0
        self.episode_step = 0
        self.total_reward = 0.0
        
        # Verify server is running
        self._check_server()
    
    def _check_server(self):
        """Verify server is accessible"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=2)
            if response.status_code == 200:
                logger.info("✅ Server is running and accessible")
            else:
                raise RuntimeError(f"Server returned status {response.status_code}")
        except Exception as e:
            raise RuntimeError(f"❌ Cannot connect to server at {self.server_url}. "
                             f"Start server with: python -m server.app --port 8000\n"
                             f"Error: {e}")
    
    def _preprocess_frame(self, frame: Image.Image) -> np.ndarray:
        """Preprocess frame for RL (grayscale, resize, normalize)"""
        # Convert to grayscale
        img = frame.convert('L')
        
        # Resize to GBA resolution
        img = img.resize((240, 160))
        
        # Normalize to [0, 1]
        arr = np.array(img, dtype=np.float32) / 255.0
        
        return arr
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation (stacked frames)"""
        # Ensure we have exactly frame_stack_size frames
        while len(self.frame_stack) > self.frame_stack_size:
            self.frame_stack.pop(0)  # Remove oldest frame
        while len(self.frame_stack) < self.frame_stack_size:
            # Pad with last frame if needed
            if self.frame_stack:
                self.frame_stack.append(self.frame_stack[-1])
            else:
                # Create blank frame if stack is empty
                blank = np.zeros((160, 240), dtype=np.float32)
                self.frame_stack.append(blank)
        
        stacked = np.array(list(self.frame_stack))
        return stacked
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)
        
        self.episode_step = 0
        self.total_reward = 0.0
        
        # Get initial state from server
        try:
            response = requests.get(f"{self.server_url}/state", timeout=5)
            if response.status_code != 200:
                raise RuntimeError(f"Failed to get state: {response.status_code}")
            
            state_data = response.json()
            
            # Try to get screenshot from state (visual.screenshot_base64)
            screenshot_data = None
            if 'visual' in state_data:
                screenshot_data = state_data['visual'].get('screenshot_base64')
            
            if not screenshot_data:
                # Fallback: request screenshot directly
                screenshot_response = requests.get(f"{self.server_url}/screenshot", timeout=5)
                if screenshot_response.status_code == 200:
                    screenshot_json = screenshot_response.json()
                    screenshot_data = screenshot_json.get('screenshot_base64')
            
            if screenshot_data:
                # Decode base64 screenshot
                img_data = base64.b64decode(screenshot_data)
                frame = Image.open(BytesIO(img_data))
            else:
                raise RuntimeError("Could not get screenshot from server")
            
            # Initialize frame stack
            self.frame_stack.clear()
            processed = self._preprocess_frame(frame)
            for _ in range(self.frame_stack_size):
                self.frame_stack.append(processed)
            
            logger.info("✅ Environment reset via server")
            return self._get_observation(), {}
            
        except Exception as e:
            logger.error(f"❌ Reset failed: {e}")
            raise
    
    def step(self, action_idx: int):
        """Execute action and return (obs, reward, done, truncated, info)"""
        self.step_count += 1
        self.episode_step += 1
        
        # Get action
        action = self.action_list[action_idx]
        
        # Send action to server (like VLM does)
        # Add small delay to avoid overwhelming server
        time.sleep(0.1)  # 100ms delay between actions
        
        # Retry logic for connection errors
        max_retries = 3
        retry_delay = 0.5  # Start with 0.5s delay
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.server_url}/action",
                    json={"buttons": [action]},
                    timeout=10  # Increased timeout
                )
                break  # Success, exit retry loop
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Connection error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Connection failed after {max_retries} attempts: {e}")
                    # Return zero reward and continue
                    reward = 0.0
                    info = {"error": "connection_failed", "error_detail": str(e)}
                    terminated = False
                    truncated = self.episode_step >= 10000
                    return self._get_observation(), reward, terminated, truncated, info
            except Exception as e:
                logger.error(f"Unexpected error sending action: {e}")
                reward = 0.0
                info = {"error": "unexpected_error", "error_detail": str(e)}
                terminated = False
                truncated = self.episode_step >= 10000
                return self._get_observation(), reward, terminated, truncated, info
        
        try:
            
            if response.status_code != 200:
                error_detail = ""
                try:
                    error_json = response.json()
                    error_detail = error_json.get("detail", str(error_json))
                except:
                    error_detail = response.text[:100] if response.text else "Unknown error"
                
                logger.warning(f"Action failed: {response.status_code} - {error_detail}")
                # Return zero reward and continue (training is resilient)
                reward = 0.0
                info = {"error": "action_failed", "error_detail": error_detail}
            else:
                # Get new state (with retry logic)
                time.sleep(0.05)  # Small delay for server to process
                
                state_response = None
                for state_attempt in range(max_retries):
                    try:
                        state_response = requests.get(f"{self.server_url}/state", timeout=10)
                        break  # Success
                    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                        if state_attempt < max_retries - 1:
                            logger.warning(f"State request failed (attempt {state_attempt+1}/{max_retries}): {e}. Retrying...")
                            time.sleep(retry_delay)
                        else:
                            logger.error(f"State request failed after {max_retries} attempts: {e}")
                            # Use previous frame if available
                            if self.frame_stack:
                                processed = self.frame_stack[-1]
                                self.frame_stack.append(processed)
                            reward = 0.0
                            info = {"error": "state_fetch_failed"}
                            terminated = False
                            truncated = self.episode_step >= 10000
                            return self._get_observation(), reward, terminated, truncated, info
                
                if state_response is None:
                    # Fallback: use previous frame
                    if self.frame_stack:
                        processed = self.frame_stack[-1]
                        self.frame_stack.append(processed)
                    reward = 0.0
                    info = {"error": "state_fetch_failed"}
                    terminated = False
                    truncated = self.episode_step >= 10000
                    return self._get_observation(), reward, terminated, truncated, info
                
                if state_response.status_code == 200:
                    state_data = state_response.json()
                    
                    # Get screenshot from state
                    screenshot_data = None
                    if 'visual' in state_data:
                        screenshot_data = state_data['visual'].get('screenshot_base64')
                    
                    if not screenshot_data:
                        # Fallback: request screenshot directly
                        screenshot_response = requests.get(f"{self.server_url}/screenshot", timeout=5)
                        if screenshot_response.status_code == 200:
                            screenshot_json = screenshot_response.json()
                            screenshot_data = screenshot_json.get('screenshot_base64')
                    
                    if screenshot_data:
                        img_data = base64.b64decode(screenshot_data)
                        frame = Image.open(BytesIO(img_data))
                        # Update frame stack (maintain exact size)
                        processed = self._preprocess_frame(frame)
                        self.frame_stack.append(processed)
                        # Remove oldest frame if stack is too large
                        if len(self.frame_stack) > self.frame_stack_size:
                            self.frame_stack.pop(0)
                    else:
                        logger.warning("Could not get screenshot, using previous frame")
                        # Use previous frame if available (don't add if stack is full)
                        if self.frame_stack and len(self.frame_stack) < self.frame_stack_size:
                            processed = self.frame_stack[-1]
                            self.frame_stack.append(processed)  # Duplicate last frame
                        elif not self.frame_stack:
                            # Create blank frame as fallback
                            frame = Image.new('RGB', (240, 160), color='black')
                            processed = self._preprocess_frame(frame)
                            self.frame_stack.append(processed)
                    
                    # Calculate reward (simple for now)
                    reward = self._calculate_reward(state_data)
                    self.total_reward += reward
                    
                    info = {
                        "episode_step": self.episode_step,
                        "total_reward": self.total_reward,
                        "coordinates": state_data.get('game_state', {}).get('player', {}).get('position'),
                    }
                else:
                    reward = 0.0
                    info = {"error": "state_fetch_failed"}
        
        except Exception as e:
            logger.warning(f"Step error: {e}")
            reward = 0.0
            info = {"error": str(e)}
        
        # Check if done
        terminated = False
        truncated = self.episode_step >= 10000  # Max episode length
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _calculate_reward(self, state_data: Dict[str, Any]) -> float:
        """Calculate reward from game state"""
        # Simple reward: pixel change + progress
        reward = 0.0
        
        # Small reward for any step (encourage exploration)
        reward += 0.01
        
        # Check for progress (coordinates changed, new location, etc.)
        game_state = state_data.get('game_state', {})
        player = game_state.get('player', {})
        
        # Reward for being in a new location
        location = player.get('location', 'Unknown')
        if location != 'Unknown' and location != 'TITLE_SEQUENCE':
            reward += 0.1
        
        return reward


class CNNFeatureExtractor(BaseFeaturesExtractor):
    """Custom CNN feature extractor for stacked frames (like PokemonRedExperiments)"""
    
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        
        # Input: (channels, H, W) = (4, 160, 240) - channel-first format
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate output size dynamically
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            cnn_output_size = self.cnn(sample).shape[1]
        
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_size, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.fc(self.cnn(observations))


def train_rl_with_server(server_url: str = "http://localhost:8000", total_timesteps: int = 50000, resume_from: Optional[str] = None):
    """Train RL agent using server architecture"""
    
    if not HAS_SB3:
        raise ImportError("stable-baselines3 required")
    
    logger.info("🚀 Starting RL training with SERVER architecture")
    logger.info(f"   Server: {server_url}")
    logger.info(f"   Timesteps: {total_timesteps:,}")
    
    # Create environment
    env = ServerRLEnv(server_url=server_url)
    
    # Wrap in vectorized environment
    vec_env = DummyVecEnv([lambda: env])
    
    # Create or load PPO model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"   Using device: {device}")
    
    if resume_from and os.path.exists(resume_from):
        logger.info(f"📂 Loading checkpoint from: {resume_from}")
        model = PPO.load(resume_from, env=vec_env, device=device)
        logger.info("✅ Checkpoint loaded successfully")
    else:
        # Use custom feature extractor for stacked frames
        model = PPO(
            "CnnPolicy",
            vec_env,
            learning_rate=0.00025,
            n_steps=2048,
            batch_size=512,
            n_epochs=4,
            gamma=0.99,
            verbose=1,
            device=device,  # Explicitly set device
            tensorboard_log=".rl_checkpoints/tensorboard/",
            policy_kwargs=dict(
                features_extractor_class=CNNFeatureExtractor,
                features_extractor_kwargs=dict(features_dim=512)
            )
        )
        logger.info("✅ PPO model created from scratch")
    
    logger.info(f"   Policy device: {next(model.policy.parameters()).device}")
    
    # Setup checkpoint callback
    os.makedirs(".rl_checkpoints", exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=".rl_checkpoints",
        name_prefix="pokemon_emerald_rl_server"
    )
    
    # Train!
    logger.info("🎓 Starting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Save final model
    final_path = ".rl_checkpoints/pokemon_emerald_rl_server_final.zip"
    model.save(final_path)
    logger.info(f"✅ Training complete! Model saved to {final_path}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="RL Training with Server Architecture")
    parser.add_argument("--server-url", type=str, default="http://localhost:8000",
                       help="Server URL")
    parser.add_argument("--timesteps", type=int, default=50000,
                       help="Total training timesteps")
    parser.add_argument("--start-server", action="store_true",
                       help="Start server automatically")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     Pokemon Emerald RL Training - SERVER MODE                ║
║     (Uses Working Server Architecture!)                       ║
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
        time.sleep(5)  # Wait for server to start
        logger.info("✅ Server started")
    
    try:
        # Train
        model = train_rl_with_server(
            server_url=args.server_url,
            total_timesteps=args.timesteps,
            resume_from=args.resume
        )
        
        print("\n✅ Training complete!")
        
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

