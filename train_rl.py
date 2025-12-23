#!/usr/bin/env python3
"""
Training script for RL agent with real learning

Usage:
    # Train from scratch
    python train_rl.py --timesteps 10000000
    
    # Resume training
    python train_rl.py --resume .rl_checkpoints/pokemon_emerald_rl_500000_steps.zip
    
    # Train with VLM guidance
    python train_rl.py --vlm_backend qwen --vlm_guidance
"""

import argparse
import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pokemon_env.emulator import EmeraldEmulator
from agent.rl_trainer import train_rl_agent, RLConfig, load_trained_agent

# VLM is optional - only import if needed
create_vlm = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train RL agent for Pokemon Emerald")
    
    # Training parameters
    parser.add_argument('--timesteps', type=int, default=10_000_000,
                       help='Total timesteps to train (default: 10M)')
    parser.add_argument('--learning_rate', type=float, default=2.5e-4,
                       help='Learning rate for PPO')
    parser.add_argument('--save_freq', type=int, default=50000,
                       help='Save checkpoint every N steps')
    parser.add_argument('--checkpoint_dir', type=str, default='.rl_checkpoints',
                       help='Directory to save checkpoints')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    
    # Features
    parser.add_argument('--use_snn', action='store_true', default=True,
                       help='Use SNN encoder for temporal features (default: True)')
    parser.add_argument('--no_snn', action='store_false', dest='use_snn',
                       help='Disable SNN encoder')
    
    parser.add_argument('--vlm_guidance', action='store_true', default=False,
                       help='Use VLM for high-level guidance (slower but smarter)')
    parser.add_argument('--vlm_backend', type=str, default='qwen',
                       choices=['qwen', 'gemini', 'claude', 'gpt'],
                       help='VLM backend if using guidance')
    parser.add_argument('--vlm_guidance_freq', type=int, default=100,
                       help='Ask VLM every N steps')
    
    # Game
    parser.add_argument('--rom', type=str, default='emerald.gbc',
                       help='Path to Pokemon Emerald ROM')
    parser.add_argument('--max_episode_steps', type=int, default=10000,
                       help='Max steps per episode')
    
    # Reward shaping
    parser.add_argument('--coord_novelty_weight', type=float, default=1.0,
                       help='Reward weight for exploring new coordinates')
    parser.add_argument('--event_weight', type=float, default=10.0,
                       help='Reward weight for achieving milestones')
    parser.add_argument('--dialogue_weight', type=float, default=5.0,
                       help='Reward weight for new dialogue')
    parser.add_argument('--loop_penalty', type=float, default=-1.0,
                       help='Penalty for getting stuck in loops')
    
    args = parser.parse_args()
    
    # Print banner
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║     Pokemon Emerald RL Training                          ║
║     Real Learning with PPO + SNN + VLM                   ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # Initialize emulator
    logger.info(f"🎮 Initializing emulator with ROM: {args.rom}")
    if not os.path.exists(args.rom):
        logger.error(f"❌ ROM not found: {args.rom}")
        logger.error("   Please place your Pokemon Emerald ROM in the project directory")
        return 1
    
    emulator = EmeraldEmulator(args.rom)
    logger.info("✅ Emerald Emulator initialized")
    
    # Initialize VLM if requested
    vlm = None
    if args.vlm_guidance:
        logger.info(f"🧠 Initializing VLM for guidance: {args.vlm_backend}")
        try:
            # Import VLM only when needed
            try:
                from vlm.vlm_factory import create_vlm as _create_vlm
            except ImportError:
                # Try alternative import
                import sys
                sys.path.insert(0, str(Path(__file__).parent))
                from agent import create_vlm as _create_vlm
            
            vlm = _create_vlm(args.vlm_backend)
            logger.info(f"✅ VLM initialized: {args.vlm_backend}")
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize VLM: {e}")
            logger.warning("   Continuing without VLM guidance")
            vlm = None
    else:
        logger.info("VLM guidance disabled (faster training)")
    
    # Create config
    config = RLConfig(
        total_timesteps=args.timesteps,
        learning_rate=args.learning_rate,
        save_freq=args.save_freq,
        checkpoint_dir=args.checkpoint_dir,
        max_steps_per_episode=args.max_episode_steps,
        
        # Features
        use_snn_features=args.use_snn,
        use_vlm_guidance=args.vlm_guidance and vlm is not None,
        vlm_guidance_frequency=args.vlm_guidance_freq,
        
        # Reward shaping
        coordinate_novelty_weight=args.coord_novelty_weight,
        event_progress_weight=args.event_weight,
        dialogue_progress_weight=args.dialogue_weight,
        loop_penalty=args.loop_penalty,
    )
    
    # Print config
    logger.info("📋 Training Configuration:")
    logger.info(f"   Total Timesteps: {config.total_timesteps:,}")
    logger.info(f"   Learning Rate: {config.learning_rate}")
    logger.info(f"   SNN Features: {'ENABLED' if config.use_snn_features else 'DISABLED'}")
    logger.info(f"   VLM Guidance: {'ENABLED' if config.use_vlm_guidance else 'DISABLED'}")
    logger.info(f"   Coordinate Novelty Weight: {config.coordinate_novelty_weight}")
    logger.info(f"   Event Progress Weight: {config.event_progress_weight}")
    logger.info(f"   Loop Penalty: {config.loop_penalty}")
    
    # Resume or train from scratch
    if args.resume:
        logger.info(f"🔄 Resuming training from: {args.resume}")
        model = load_trained_agent(args.resume)
        logger.info("✅ Checkpoint loaded, resuming training...")
        # Continue training
        model.learn(
            total_timesteps=config.total_timesteps,
            reset_num_timesteps=False,  # Continue from where we left off
            progress_bar=True
        )
    else:
        logger.info("🚀 Training from scratch...")
        model = train_rl_agent(emulator, config, vlm)
    
    logger.info("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║     ✅ Training Complete!                                ║
║                                                          ║
║     Your agent has LEARNED through real RL!              ║
║     The policy network now knows:                        ║
║     - How to explore efficiently                         ║
║     - Which actions lead to rewards                      ║
║     - How to avoid loops and mistakes                    ║
║                                                          ║
║     Next: Run trained agent with run_rl.py               ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

