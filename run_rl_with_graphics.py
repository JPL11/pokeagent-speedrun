#!/usr/bin/env python3
"""
Run trained RL agent WITH GRAPHICS so you can watch it play!
"""

import argparse
import os
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pokemon_env.emulator import EmeraldEmulator
from agent.rl_trainer import load_trained_agent, RLConfig, PokemonEmeraldRLEnv

def main():
    parser = argparse.ArgumentParser(description="Run trained RL agent with graphics")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained model')
    parser.add_argument('--rom', type=str, default='Emerald-GBAdvance/rom.gba')
    parser.add_argument('--max_steps', type=int, default=10000)
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║  🎮 Running Trained RL Agent with Graphics                   ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Load trained model
    print(f"📂 Loading trained model: {args.checkpoint}")
    model = load_trained_agent(args.checkpoint)
    print("✅ Model loaded!")
    
    # Initialize emulator with graphics
    print(f"🎮 Initializing emulator: {args.rom}")
    emulator = EmeraldEmulator(args.rom, headless=False)  # headless=False for graphics!
    emulator.initialize()
    print("✅ Emulator initialized with graphics!")
    
    # Create environment
    config = RLConfig()
    env = PokemonEmeraldRLEnv(emulator, config, vlm=None)
    
    # Reset environment
    observation, info = env.reset()
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║  🚀 Agent is running!                                        ║
║  Watch the emulator window to see it play!                   ║
║  Press Ctrl+C to stop                                        ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    step = 0
    episode_reward = 0.0
    
    try:
        while step < args.max_steps:
            step += 1
            
            # Get action from trained model
            action_idx, _states = model.predict(observation, deterministic=True)
            action = env.action_list[action_idx]
            
            # Take step
            observation, reward, done, truncated, info = env.step(action_idx)
            episode_reward += reward
            
            # Log progress
            if step % 100 == 0:
                coords = info.get('coordinates', (0, 0))
                location = info.get('location', 'UNKNOWN')
                print(f"Step {step:6d} | Reward: {episode_reward:7.2f} | Location: {location} | Coords: {coords} | Action: {action}")
            
            # Small delay so you can see what's happening
            time.sleep(0.05)  # 20 FPS
            
            # Reset if done
            if done:
                print(f"Episode complete! Total reward: {episode_reward:.2f}")
                observation, info = env.reset()
                episode_reward = 0.0
        
    except KeyboardInterrupt:
        print("\n⏸️  Stopped by user")
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  ✅ Session complete!                                        ║
║  Total Steps: {step:,}                                       
║  Final Reward: {episode_reward:.2f}                          ║
╚══════════════════════════════════════════════════════════════╝
    """)

if __name__ == "__main__":
    main()

