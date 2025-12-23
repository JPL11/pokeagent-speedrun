#!/usr/bin/env python3
"""
Train RL Agent with Display (No VLM overhead)
Fast training you can watch in real-time!
"""

import argparse
import os
import sys
import logging
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pokemon_env.emulator import EmeraldEmulator
from agent.rl_trainer import train_rl_agent, RLConfig, load_trained_agent, PokemonEmeraldRLEnv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train RL with display (no VLM)")
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Training timesteps')
    parser.add_argument('--rom', type=str, default='emerald.gba',
                       help='Path to ROM')
    args = parser.parse_args()
    
    print("""
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║     Watch Your RL Agent Train (Fast Mode - No VLM)        ║
║                                                            ║
║     Display enabled - see the agent learn in real-time!   ║
║     No VLM overhead - fast and lightweight                ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize emulator (headless - mGBA's SDL window will show automatically)
    logger.info(f"🎮 Initializing emulator: {args.rom}")
    logger.info("   Note: mGBA window should appear automatically")
    emulator = EmeraldEmulator(args.rom, headless=False)  # headless=False shows mGBA window
    emulator.initialize()
    logger.info("✅ Emulator initialized (mGBA window should be visible)")
    
    # Config without VLM
    config = RLConfig(
        total_timesteps=args.timesteps,
        use_vlm_guidance=False,  # No VLM overhead
        use_vlm_unstuck=False,
        use_snn_features=True,   # Keep SNN for temporal features
    )
    
    logger.info("📋 Configuration:")
    logger.info(f"   Timesteps: {config.total_timesteps:,}")
    logger.info(f"   Display: ENABLED 🎬")
    logger.info(f"   VLM: DISABLED (for speed)")
    logger.info(f"   SNN Features: ENABLED")
    
    if args.resume:
        logger.info(f"🔄 Resuming from: {args.resume}")
        
        # Create environment (no separate OpenCV display, use mGBA's window)
        logger.info("🏗️  Creating environment...")
        env = PokemonEmeraldRLEnv(emulator, config, vlm=None, show_display=False)
        
        # Load and set environment
        logger.info("📂 Loading checkpoint...")
        model = load_trained_agent(args.resume)
        model.set_env(env)
        
        logger.info("✅ Ready to train!")
        logger.info("🎬 Watch the mGBA window to see the agent play!")
        logger.info("   (The mGBA window should already be open)")
        logger.info(f"🚀 Training for {config.total_timesteps:,} more steps...")
        logger.info("")
        
        # Train
        model.learn(
            total_timesteps=config.total_timesteps,
            reset_num_timesteps=False,
            progress_bar=True
        )
    else:
        logger.info("🚀 Training from scratch...")
        model = train_rl_agent(emulator, config, vlm=None, show_display=False)
    
    logger.info("✅ Training complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
