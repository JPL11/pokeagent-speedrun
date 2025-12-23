#!/usr/bin/env python3
"""
Interactive RL Training with Live Display
Inspired by PokemonRedExperiments/v2/run_pretrained_interactive.py

Shows the game screen while training so you can see what the agent is doing!
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Import our RL components
from pokemon_env.emulator import EmeraldEmulator
from agent.rl_trainer import PokemonEmeraldRLEnv, RLConfig, train_rl_agent
from stable_baselines3 import PPO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Interactive RL Training with Live Display")
    parser.add_argument("--rom", type=str, default="Emerald-GBAdvance/rom.gba", help="Path to ROM")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--display", action="store_true", default=True, help="Show live display (disable with --no-display)")
    parser.add_argument("--no-display", action="store_false", dest="display", help="Disable live display")
    parser.add_argument("--load-checkpoint", type=str, help="Load checkpoint and continue training")
    parser.add_argument("--speed", type=int, default=0, help="Emulation speed (0=unlimited, 1=1x, 2=2x, etc.)")
    
    args = parser.parse_args()
    
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                                                              ║")
    print("║     Pokemon Emerald RL Training - INTERACTIVE MODE          ║")
    print("║     (Like PokemonRedExperiments!)                            ║")
    print("║                                                              ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    
    # Create emulator (ALWAYS headless - we use GameDisplay for visualization!)
    logger.info(f"🎮 Initializing emulator with ROM: {args.rom}")
    emulator = EmeraldEmulator(
        rom_path=args.rom,
        headless=True,  # Always headless! GameDisplay handles visualization
        sound=False  # No sound for training
    )
    emulator.initialize()
    logger.info("✅ Emulator initialized (headless mode - fast!)")
    
    if args.display:
        logger.info("🎬 Live display enabled - OpenCV window will show the agent play!")
    else:
        logger.info("🔇 No display - training only")
    
    # Create RL config
    config = RLConfig(
        total_timesteps=args.timesteps,
        learning_rate=0.00025,
        use_snn_features=False,  # Start without SNN for speed
        use_vlm_guidance=False,  # Pure RL
        n_steps=2048,
        batch_size=512,
        n_epochs=4,
    )
    
    logger.info("📋 Training Configuration:")
    logger.info(f"   Total Timesteps: {config.total_timesteps:,}")
    logger.info(f"   Learning Rate: {config.learning_rate}")
    logger.info(f"   Display: {'ON' if args.display else 'OFF'}")
    logger.info(f"   Speed: {'Unlimited' if args.speed == 0 else f'{args.speed}x'}")
    
    # Train or load model
    if args.load_checkpoint:
        logger.info(f"📂 Loading checkpoint: {args.load_checkpoint}")
        env = PokemonEmeraldRLEnv(emulator, config, vlm=None, show_display=args.display)
        model = PPO.load(args.load_checkpoint, env=env)
        logger.info("✅ Checkpoint loaded, continuing training...")
    else:
        logger.info("🚀 Starting training from scratch...")
        model = train_rl_agent(emulator, config, vlm=None, show_display=args.display)
    
    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════════╗")
    logger.info("║  🎮 TRAINING STARTED!                                        ║")
    logger.info("║                                                              ║")
    logger.info("║  Watch the mgba window to see the agent play!               ║")
    logger.info("║  Progress is logged to tensorboard:                         ║")
    logger.info("║  tensorboard --logdir .rl_checkpoints/tensorboard           ║")
    logger.info("║                                                              ║")
    logger.info("║  Press Ctrl+C to stop training                              ║")
    logger.info("╚══════════════════════════════════════════════════════════════╝")
    logger.info("")
    
    return model

if __name__ == "__main__":
    try:
        model = main()
        print("\n✅ Training complete!")
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

