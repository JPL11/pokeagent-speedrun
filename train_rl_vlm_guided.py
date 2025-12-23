#!/usr/bin/env python3
"""
Train RL Agent with VLM Unstuck Guidance
=========================================

This training approach uses VLM reasoning to:
1. Detect stuck patterns (instead of hard-coded heuristics)
2. Guide exploration (what to try next)
3. Shape rewards dynamically (what's important now)
4. Teach RL to handle dialogue naturally (no forced A-button spam)

Key Differences from Heuristic Approach:
- VLM provides CONTEXT and REASONING
- RL learns to USE this context
- Minimal hard-coded overrides
- Agent learns true behavior, not shortcuts

Inspired by PokemonRedExperiments but with enhanced VLM integration.
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
from utils.vlm import create_vlm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train RL agent with VLM unstuck guidance (minimizes heuristics)"
    )
    
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
    
    # VLM configuration (REQUIRED for this training mode!)
    parser.add_argument('--vlm_backend', type=str, default='qwen',
                       choices=['qwen', 'qwen2b', 'local', 'gemini', 'claude', 'gpt'],
                       help='VLM backend for unstuck detection and guidance (qwen2b/local = Qwen 2B locally)')
    parser.add_argument('--vlm_guidance_freq', type=int, default=100,
                       help='Update high-level goals every N steps')
    parser.add_argument('--vlm_unstuck_freq', type=int, default=30,
                       help='Check for stuck patterns every N steps')
    
    # Features
    parser.add_argument('--use_snn', action='store_true', default=True,
                       help='Use SNN encoder for temporal features (default: True)')
    parser.add_argument('--no_snn', action='store_false', dest='use_snn',
                       help='Disable SNN encoder')
    
    # Game
    parser.add_argument('--rom', type=str, default='emerald.gba',
                       help='Path to Pokemon Emerald ROM')
    parser.add_argument('--max_episode_steps', type=int, default=10000,
                       help='Max steps per episode')
    
    # Display (watch agent learn!)
    parser.add_argument('--display', action='store_true', default=False,
                       help='Show live display window to watch agent play during training')
    parser.add_argument('--watch', action='store_true', dest='display',
                       help='Alias for --display (watch the agent learn!)')
    
    # Reward shaping
    parser.add_argument('--coord_novelty_weight', type=float, default=1.0,
                       help='Base reward for exploring new coordinates')
    parser.add_argument('--event_weight', type=float, default=10.0,
                       help='Base reward for achieving milestones')
    parser.add_argument('--dialogue_weight', type=float, default=5.0,
                       help='Base reward for new dialogue (encourages learning A-press)')
    parser.add_argument('--loop_penalty', type=float, default=-1.0,
                       help='Base penalty for loops (VLM will refine this)')
    
    args = parser.parse_args()
    
    # Print banner
    print("""
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║     Pokemon Emerald - VLM-Guided RL Training               ║
║                                                            ║
║     ✓ VLM analyzes stuck patterns                          ║
║     ✓ VLM guides exploration strategies                    ║
║     ✓ VLM shapes rewards dynamically                       ║
║     ✓ Minimal hard-coded heuristics                        ║
║     ✓ RL learns TRUE behavior (not shortcuts)              ║
║                                                            ║
║     This agent will LEARN to handle dialogue,              ║
║     navigate menus, and explore - not just spam A!         ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    # Check ROM exists
    if not os.path.exists(args.rom):
        logger.error(f"❌ ROM not found: {args.rom}")
        logger.error("   Please place your Pokemon Emerald ROM in the project directory")
        return 1
    
    # Initialize emulator
    logger.info(f"🎮 Initializing emulator with ROM: {args.rom}")
    emulator = EmeraldEmulator(args.rom)
    logger.info("✅ Emerald Emulator initialized")
    
    # Initialize VLM (REQUIRED for this training mode!)
    logger.info(f"🧠 Initializing VLM for unstuck guidance: {args.vlm_backend}")
    try:
        vlm = create_vlm(args.vlm_backend)
        logger.info(f"✅ VLM initialized: {args.vlm_backend}")
        logger.info("   VLM will:")
        logger.info("     - Detect stuck patterns with reasoning")
        logger.info("     - Suggest exploration strategies")
        logger.info("     - Shape rewards based on game context")
        logger.info("     - Teach RL to handle dialogue naturally")
    except Exception as e:
        logger.error(f"❌ Failed to initialize VLM: {e}")
        logger.error("   VLM is REQUIRED for this training mode!")
        logger.error("   Set appropriate API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)")
        return 1
    
    # Create config with VLM features enabled
    config = RLConfig(
        total_timesteps=args.timesteps,
        learning_rate=args.learning_rate,
        save_freq=args.save_freq,
        checkpoint_dir=args.checkpoint_dir,
        max_steps_per_episode=args.max_episode_steps,
        
        # Enable VLM features
        use_snn_features=args.use_snn,
        use_vlm_guidance=True,  # Always enabled
        vlm_guidance_frequency=args.vlm_guidance_freq,
        use_vlm_unstuck=True,  # Core feature!
        vlm_unstuck_frequency=args.vlm_unstuck_freq,
        
        # Reward shaping (VLM will tune these dynamically!)
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
    logger.info(f"   VLM Guidance: ENABLED (freq={config.vlm_guidance_frequency})")
    logger.info(f"   VLM Unstuck Detection: ENABLED (freq={config.vlm_unstuck_frequency})")
    logger.info(f"   Live Display: {'ENABLED 🎬 (watch agent play!)' if args.display else 'DISABLED (faster training)'}")
    logger.info("")
    logger.info("🎯 Reward Structure (VLM-guided):")
    logger.info(f"   Coordinate Novelty: {config.coordinate_novelty_weight} (base)")
    logger.info(f"   Event Progress: {config.event_progress_weight} (base)")
    logger.info(f"   Dialogue Progress: {config.dialogue_progress_weight} (base)")
    logger.info(f"   Loop Penalty: {config.loop_penalty} (base)")
    logger.info("   NOTE: VLM will adjust these dynamically based on game context!")
    
    # Resume or train from scratch
    if args.resume:
        logger.info(f"🔄 Resuming training from: {args.resume}")
        
        # Initialize emulator first
        logger.info("🎮 Initializing emulator for resumed training...")
        emulator.initialize()
        
        # Create environment with display
        from agent.rl_trainer import PokemonEmeraldRLEnv
        logger.info(f"🏗️  Creating RL environment (display={'enabled' if args.display else 'disabled'})...")
        env = PokemonEmeraldRLEnv(emulator, config, vlm, show_display=args.display)
        
        # Load model and set environment
        logger.info("📂 Loading checkpoint...")
        model = load_trained_agent(args.resume)
        model.set_env(env)  # Set the environment for resumed training
        
        logger.info("✅ Checkpoint loaded, environment ready!")
        if args.display:
            logger.info("🎬 Display window should be visible now")
            logger.info("   (If not visible, check if window opened in background)")
        
        # Continue training
        logger.info(f"🚀 Resuming training for {config.total_timesteps:,} more timesteps...")
        model.learn(
            total_timesteps=config.total_timesteps,
            reset_num_timesteps=False,
            progress_bar=True
        )
    else:
        logger.info("🚀 Training from scratch with VLM guidance...")
        if args.display:
            logger.info("🎬 Opening display window - you can watch the agent learn!")
        model = train_rl_agent(emulator, config, vlm, show_display=args.display)
    
    logger.info("""
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║     ✅ VLM-Guided RL Training Complete!                    ║
║                                                            ║
║     Your agent has LEARNED:                                ║
║     ✓ How to recognize and escape stuck patterns           ║
║     ✓ When to press A in dialogue (not just spam!)         ║
║     ✓ Strategic exploration (not just random walk)         ║
║     ✓ Context-aware decision making                        ║
║                                                            ║
║     This is TRUE learning, not heuristic shortcuts!        ║
║                                                            ║
║     Next Steps:                                            ║
║     1. Run agent: python run_rl.py --checkpoint <path>     ║
║     2. Add VLM overlay for even smarter play:              ║
║        python run_rl.py --checkpoint <path> --vlm_backend qwen
║                                                            ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
