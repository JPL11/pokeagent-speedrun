#!/usr/bin/env python3
"""
Run trained RL agent in HYBRID mode with VLM reasoning

This combines:
- Trained RL policy (learned from experience)
- VLM high-level reasoning (context and goals)

The RL policy makes action decisions, but VLM provides strategic guidance.
This is much smarter than pure VLM (which doesn't learn) or pure RL (which lacks reasoning).

Usage:
    # Run with trained agent
    python run_rl.py --checkpoint .rl_checkpoints/pokemon_emerald_final.zip
    
    # Run with VLM reasoning overlay
    python run_rl.py --checkpoint .rl_checkpoints/pokemon_emerald_final.zip --vlm_backend qwen
    
    # Visualize
    python run_rl.py --checkpoint .rl_checkpoints/pokemon_emerald_final.zip --render
"""

import argparse
import os
import sys
import time
import logging
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pokemon_env.emulator import EmeraldEmulator
from agent.rl_trainer import load_trained_agent, RLConfig, PokemonEmeraldRLEnv
from utils.display_helper import GameDisplay

# VLM is optional - only needed if --vlm_backend is specified
create_vlm = None
try:
    from vlm.vlm_factory import create_vlm
except ImportError:
    try:
        from agent import create_vlm
    except ImportError:
        # VLM not available - that's fine for pure RL
        def create_vlm(backend):
            raise ImportError("VLM not available. Install VLM dependencies or run without --vlm_backend")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HybridRLVLMAgent:
    """
    Hybrid agent that combines:
    - RL policy for action selection (LEARNED behavior)
    - VLM for strategic reasoning and goal-setting
    
    This is the best of both worlds:
    - RL provides fast, learned action selection
    - VLM provides context, reasoning, and high-level strategy
    """
    
    def __init__(self, rl_model, vlm=None, vlm_frequency: int = 50, emulator=None):
        self.rl_model = rl_model
        self.vlm = vlm
        self.vlm_frequency = vlm_frequency
        self.emulator = emulator  # Store emulator reference for dialogue detection
        
        self.step_count = 0
        self.current_goal = None
        self.vlm_suggestions = []
        self.dialogue_steps_without_a = 0  # Track dialogue without A press
        
        logger.info("🤖 Hybrid RL+VLM Agent initialized")
        logger.info(f"   RL Policy: {type(rl_model).__name__}")
        logger.info(f"   VLM: {'ENABLED' if vlm else 'DISABLED'}")
    
    def get_action(self, observation, game_state, frame=None):
        """
        Get action from hybrid policy
        
        1. RL policy predicts action (fast, learned)
        2. VLM provides strategic context (slow, smart)
        3. Dialogue detection override (if dialogue present, press A)
        4. Combine both for best decision
        """
        self.step_count += 1
        
        # Check if dialogue is present (heuristic fallback for inference)
        dialogue = game_state.get('dialogue', '').strip()
        dialogue_active = dialogue and len(dialogue) > 3
        
        # Also check using memory reader if available
        try:
            if hasattr(self, 'emulator') and hasattr(self.emulator, 'memory_reader') and self.emulator.memory_reader:
                is_in_dialog = self.emulator.memory_reader.is_in_dialog()
                dialogue_active = dialogue_active or is_in_dialog
        except:
            pass
        
        # Track how long dialogue has been present without A press
        if dialogue_active:
            if not hasattr(self, 'dialogue_steps_without_a'):
                self.dialogue_steps_without_a = 0
            self.dialogue_steps_without_a += 1
        else:
            if hasattr(self, 'dialogue_steps_without_a'):
                self.dialogue_steps_without_a = 0
        
        # Get RL policy prediction
        action_idx, _states = self.rl_model.predict(observation, deterministic=True)
        rl_action = ['A', 'B', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'START', 'L', 'R'][action_idx]
        
        # REDUCED HEURISTIC: Only override if dialogue stuck for VERY long (let RL learn!)
        # Previously: override after 3 steps
        # Now: only override after 10+ steps (give RL more chances to learn)
        if dialogue_active and self.dialogue_steps_without_a >= 10:
            if rl_action != 'A':
                logger.info(f"💬 Dialogue STUCK detected! Emergency override '{rl_action}' → 'A' (dialogue present for {self.dialogue_steps_without_a} steps)")
                logger.info(f"   NOTE: This is a fallback. RL should learn to press A in dialogue!")
                rl_action = 'A'
                self.dialogue_steps_without_a = 0  # Reset counter
        
        # If agent pressed A, reset dialogue counter
        if rl_action == 'A' and dialogue_active:
            if hasattr(self, 'dialogue_steps_without_a'):
                self.dialogue_steps_without_a = 0
        
        # Periodically get VLM guidance
        if self.vlm and self.step_count % self.vlm_frequency == 0:
            self._update_vlm_strategy(frame, game_state)
        
        # Log reasoning
        reasoning = f"RL Policy: {rl_action}"
        if dialogue_active:
            reasoning += f" | Dialogue: YES"
        if self.current_goal:
            reasoning += f" | VLM Goal: {self.current_goal}"
        
        return {
            'action': rl_action,
            'reasoning': reasoning,
            'rl_action': rl_action,
            'vlm_goal': self.current_goal
        }
    
    def _update_vlm_strategy(self, frame, game_state):
        """Ask VLM for strategic advice"""
        try:
            prompt = f"""You are advising a trained RL agent playing Pokemon Emerald.

Current State:
- Location: {game_state.get('location', 'UNKNOWN')}
- Coordinates: {game_state.get('coordinates', (0, 0))}
- Dialogue: {game_state.get('dialogue', '')[:100]}

The RL agent has learned efficient exploration and loop avoidance.
What high-level strategy should guide its next actions? (1 sentence)
"""
            
            response = self.vlm.get_query(frame, prompt, "hybrid_strategy")
            
            if response and 'reasoning' in response:
                self.current_goal = response['reasoning'][:200]
                logger.info(f"🧠 VLM Strategy: {self.current_goal}")
        
        except Exception as e:
            logger.debug(f"VLM strategy update failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run trained RL agent")
    
    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    
    # VLM overlay
    parser.add_argument('--vlm_backend', type=str, default=None,
                       choices=['qwen', 'gemini', 'claude', 'gpt'],
                       help='Add VLM reasoning overlay (optional)')
    parser.add_argument('--vlm_frequency', type=int, default=50,
                       help='Get VLM guidance every N steps')
    
    # Game
    parser.add_argument('--rom', type=str, default='Emerald-GBAdvance/rom.gba',
                       help='Path to Pokemon Emerald ROM')
    parser.add_argument('--render', action='store_true',
                       help='Show game screen while playing')
    parser.add_argument('--max_steps', type=int, default=100000,
                       help='Max steps to run')
    
    # Checkpoint management
    parser.add_argument('--load_checkpoint', action='store_true',
                       help='Load game checkpoint from previous session')
    parser.add_argument('--save_freq', type=int, default=1000,
                       help='Save game checkpoint every N steps')
    
    args = parser.parse_args()
    
    # Print banner
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║     Pokemon Emerald - Hybrid RL+VLM Agent                ║
║                                                          ║
║     Trained Policy + Strategic Reasoning                 ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        logger.error(f"❌ Checkpoint not found: {args.checkpoint}")
        logger.error("   Train a model first with: python train_rl_server.py")
        return 1
    
    # Initialize VLM if requested
    vlm = None
    if args.vlm_backend:
        logger.info(f"🧠 Initializing VLM overlay: {args.vlm_backend}")
        try:
            vlm = create_vlm(args.vlm_backend)
            logger.info(f"✅ VLM overlay enabled")
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize VLM: {e}")
            vlm = None
    
    # Initialize display FIRST (before any slow operations) so it shows up immediately
    display = None
    if args.render:
        try:
            logger.info("🎬 Initializing display window (early, before slow operations)...")
            display = GameDisplay(title="Pokemon Emerald - RL Agent", scale=3)
            logger.info("✅ Display window initialized!")
            # Give display thread a moment to start
            import time as time_module
            time_module.sleep(0.2)
        except Exception as e:
            logger.warning(f"⚠️  Could not initialize display: {e}")
            logger.warning("   Continuing without display (agent will still run!)")
            args.render = False
            display = None
    
    # Initialize emulator
    logger.info(f"🎮 Initializing emulator: {args.rom}")
    emulator = EmeraldEmulator(args.rom)
    emulator.initialize()  # Initialize the emulator core
    
    # Try to load savestate past title screen first (fastest option)
    savestate_path = ".pokeagent_cache/past_title_screen.state"
    savestate_loaded = False
    if os.path.exists(savestate_path):
        logger.info(f"📂 Loading savestate past title screen: {savestate_path}")
        try:
            emulator.load_state(savestate_path)
            logger.info("✅ Loaded savestate past title screen!")
            savestate_loaded = True
            # Show frame immediately
            if display:
                frame = emulator.get_screenshot()
                if frame:
                    display.show_frame(frame)
                    logger.info("🎬 Frame shown from savestate")
            # Tick a few times to ensure state is fully loaded
            for _ in range(10):
                emulator.tick(1)
        except Exception as e:
            logger.warning(f"⚠️  Failed to load savestate: {e}")
            logger.warning("   Will skip title screen manually...")
            savestate_loaded = False
    else:
        logger.info("ℹ️  No savestate found, will skip title screen manually...")
        logger.info("   💡 Tip: Run 'python create_start_savestate.py' to create one!")
    
    # Only do manual title screen skip if savestate wasn't loaded
    if not savestate_loaded:
        # IMPORTANT: Tick emulator to generate initial frames (like test_display.py)
        # Without this, frames will be blank (min=255, max=255)
        logger.info("🎮 Generating initial frames (ticking emulator 50 times)...")
        for i in range(50):
            emulator.tick(1)
            # Show frames frequently so user sees progress
            if display and i % 10 == 0:
                frame = emulator.get_screenshot()
                if frame:
                    display.show_frame(frame)
                    time_module.sleep(0.01)
        
        # Skip title screen (press A multiple times) - very aggressive
        logger.info("🎮 Skipping title screen (pressing A 50 times with ticks)...")
        for i in range(50):
            emulator.press_key('a', frames=2)
            # Tick between presses to let game process
            for _ in range(8):
                emulator.tick(1)
            # Show frames during title screen skip
            if display and i % 5 == 0:
                frame = emulator.get_screenshot()
                if frame:
                    display.show_frame(frame)
                    # Check if we're past title screen (frame has content)
                    if isinstance(frame, np.ndarray):
                        frame_min = frame.min()
                        frame_max = frame.max()
                        if frame_min < 200 or frame_max < 200:  # Has actual game content
                            logger.info(f"✅ Past title screen! (frame min={frame_min}, max={frame_max})")
                            break
                    time_module.sleep(0.01)
        
        # Extra ticks to ensure we're fully past title screen
        logger.info("🎮 Finalizing title screen skip (50 more ticks)...")
        for i in range(50):
            emulator.tick(1)
            if display and i % 10 == 0:
                frame = emulator.get_screenshot()
                if frame:
                    display.show_frame(frame)
                    time_module.sleep(0.01)
    
    # Show final frame and verify it has content
    logger.info("🎮 Verifying frame has content...")
    frame = emulator.get_screenshot()
    if frame:
        # Check if frame has actual content (not all white)
        if isinstance(frame, np.ndarray):
            frame_min = frame.min()
            frame_max = frame.max()
            if frame_min < 255 or frame_max < 255:  # Has actual content
                if display:
                    display.show_frame(frame)
                logger.info(f"✅ Frame has content (min={frame_min}, max={frame_max})")
            else:
                logger.warning(f"⚠️  Frame is blank (all white: min={frame_min}, max={frame_max})")
                logger.warning("   This might indicate emulator initialization issue")
        else:
            if display:
                display.show_frame(frame)
            logger.info("✅ Frame obtained")
    else:
        logger.warning("⚠️  No frame obtained from emulator!")
    
    # Load checkpoint if requested (this might be slow, but display is already showing)
    if args.load_checkpoint:
        checkpoint_path = ".pokeagent_cache/checkpoint.state"
        if os.path.exists(checkpoint_path):
            logger.info(f"📂 Loading game checkpoint (this may take a moment)...")
            emulator.load_state(checkpoint_path)
            logger.info(f"✅ Loaded game checkpoint from: {checkpoint_path}")
            # Show frame after checkpoint load
            if display:
                frame = emulator.get_screenshot()
                if frame:
                    display.show_frame(frame)
        else:
            logger.warning(f"⚠️  No checkpoint found at: {checkpoint_path}")
    
    logger.info("✅ Emulator initialized")
    
    # Create environment for observation preprocessing (needed for model loading)
    config = RLConfig()
    env = PokemonEmeraldRLEnv(emulator, config, vlm)
    
    # Load trained model (after environment is created for compatibility)
    # Ensure GPU is used if available
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"📂 Loading trained model: {args.checkpoint}")
    logger.info(f"   Using device: {device}")
    rl_model = load_trained_agent(args.checkpoint, device=device)
    logger.info("✅ Trained model loaded")
    
    # Verify GPU usage
    if torch.cuda.is_available():
        actual_device = next(rl_model.policy.parameters()).device
        if actual_device.type == "cuda":
            logger.info(f"✅ Model is using GPU: {actual_device}")
        else:
            logger.warning(f"⚠️  Model is on CPU but GPU is available! This will be slow.")
    else:
        logger.info("ℹ️  CUDA not available, using CPU")
    
    # Create hybrid agent (pass emulator for dialogue detection)
    agent = HybridRLVLMAgent(rl_model, vlm, args.vlm_frequency, emulator=emulator)
    
    # Reset environment (Gymnasium returns tuple: observation, info)
    # This is now fast - no blocking savestate loading!
    logger.info("🔄 Resetting environment...")
    observation, info = env.reset()
    logger.info("✅ Environment reset complete!")
    
    # Show frame after reset (display is already initialized and showing)
    if display:
        frame = emulator.get_screenshot()
        if frame:
            display.show_frame(frame)
            logger.info("🎬 Frame displayed after reset!")
    
    logger.info("""
╔══════════════════════════════════════════════════════════╗
║  🚀 Starting agent!                                      ║
║                                                          ║
║  The RL policy has LEARNED from experience.              ║
║  It knows how to explore, avoid loops, and progress.     ║
║                                                          ║
║  Press Ctrl+C to stop.                                   ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    step = 0
    episode_reward = 0.0
    last_log_step = 0
    
    # Performance monitoring
    import time as time_module
    step_times = []
    last_perf_log = time_module.time()
    
    logger.info("🎮 Agent loop starting...")
    
    try:
        while step < args.max_steps:
            step_start = time_module.time()
            step += 1
            
            # Get action from hybrid agent (using current observation)
            # Ensure observation is numpy array for fast GPU inference
            if not isinstance(observation, np.ndarray):
                observation = np.array(observation, dtype=np.float32)
            
            # Get current frame and game state for agent decision (before action)
            current_frame = emulator.get_screenshot()
            game_state = env._get_game_state()
            
            # Check if we're stuck at title screen and auto-skip it
            location = game_state.get('location', '')
            is_in_title = False
            try:
                if hasattr(emulator, 'memory_reader') and emulator.memory_reader:
                    is_in_title = emulator.memory_reader.is_in_title_sequence()
            except:
                pass
            
            if location == 'TITLE_SEQUENCE' or is_in_title:
                # We're at title screen - auto-skip it!
                if not hasattr(agent, '_title_skip_count'):
                    agent._title_skip_count = 0
                
                agent._title_skip_count += 1
                
                # Title screen strategy: Press START first (to start game), then A to skip cutscenes
                if agent._title_skip_count <= 10:
                    # First 10 steps: Press START to start the game
                    logger.info(f"🎮 Title screen detected! Pressing START (step {agent._title_skip_count}/10)...")
                    action = 'START'
                    emulator.tick(8)  # Give more time for START to register
                elif agent._title_skip_count <= 60:
                    # Next 50 steps: Press A to skip cutscenes and intro
                    logger.info(f"🎮 Title screen detected! Pressing A to skip (step {agent._title_skip_count}/60)...")
                    action = 'A'
                    emulator.tick(5)
                else:
                    # If still stuck after 60 steps, alternate A and START
                    if agent._title_skip_count % 2 == 0:
                        logger.info(f"🎮 Still at title screen, trying START (step {agent._title_skip_count})...")
                        action = 'START'
                    else:
                        logger.info(f"🎮 Still at title screen, trying A (step {agent._title_skip_count})...")
                        action = 'A'
                    emulator.tick(5)
            else:
                # Not at title screen - reset counter and use normal agent
                if hasattr(agent, '_title_skip_count'):
                    if agent._title_skip_count > 0:
                        logger.info(f"✅ Past title screen! (was stuck for {agent._title_skip_count} steps)")
                    agent._title_skip_count = 0
                
                # Get action from agent
                action_result = agent.get_action(observation, game_state, current_frame)
                action = action_result['action']
            
            # Convert action string to index (e.g., 'A' -> 0, 'UP' -> 2)
            action_idx = env.action_list.index(action)
            
            # Take step in environment (this will press key and advance emulator)
            observation, reward, terminated, truncated, info = env.step(action_idx)
            done = terminated or truncated
            episode_reward += reward
            
            # Get NEW frame after step (this is the frame that shows the result of the action)
            # This matches test_display.py pattern: tick/step first, then get frame
            frame = emulator.get_screenshot()
            
            # Show frame in display if enabled (fully non-blocking now)
            if display and frame:
                display.show_frame(frame)  # This is now thread-safe and non-blocking
            
            # Track step time for performance monitoring
            step_time = time_module.time() - step_start
            step_times.append(step_time)
            if len(step_times) > 100:
                step_times.pop(0)  # Keep last 100 steps
            
            # Log progress more frequently (every 10 steps) so user knows it's working
            if step % 10 == 0:
                coords = game_state.get('coordinates', (0, 0))
                location = game_state.get('location', 'UNKNOWN')
                
                # Calculate performance metrics
                avg_step_time = sum(step_times) / len(step_times) if step_times else 0
                fps = 1.0 / avg_step_time if avg_step_time > 0 else 0
                
                logger.info(
                    f"Step {step:6d} | "
                    f"Reward: {episode_reward:7.2f} | "
                    f"Location: {location} | "
                    f"Coords: {coords} | "
                    f"Action: {action} | "
                    f"FPS: {fps:.1f}"
                )
                last_log_step = step
            
            # Log performance summary every 5 seconds
            current_time = time_module.time()
            if current_time - last_perf_log >= 5.0:
                if step_times:
                    avg_step_time = sum(step_times) / len(step_times)
                    fps = 1.0 / avg_step_time if avg_step_time > 0 else 0
                    logger.info(f"📊 Performance: {fps:.1f} FPS (avg step time: {avg_step_time*1000:.1f}ms)")
                last_perf_log = current_time
            
            # Save checkpoint periodically
            if step % args.save_freq == 0:
                os.makedirs(".pokeagent_cache", exist_ok=True)
                emulator.save_state(".pokeagent_cache/checkpoint.state")
                logger.info(f"💾 Checkpoint saved at step {step}")
            
            # Reset if episode done
            if done:
                logger.info(f"Episode complete! Total reward: {episode_reward:.2f}")
                observation, info = env.reset()
                episode_reward = 0.0
            
            # Small delay for rendering (allows display to update)
            # Reduced delay to avoid bottleneck - display is non-blocking now
            if args.render:
                time.sleep(0.016)  # ~60 FPS for smooth display (reduced from 0.033)
            # No delay when headless - run as fast as possible
        
    except KeyboardInterrupt:
        logger.info("\n⏸️  Interrupted by user")
    finally:
        # Close display if it was opened
        if display:
            display.close()
            logger.info("🎬 Display window closed")
    
    # Save final checkpoint
    os.makedirs(".pokeagent_cache", exist_ok=True)
    emulator.save_state(".pokeagent_cache/checkpoint.state")
    logger.info(f"💾 Final checkpoint saved")
    
    logger.info(f"""
╔══════════════════════════════════════════════════════════╗
║  ✅ Session complete!                                    ║
║                                                          ║
║  Total Steps: {step:,}                                       
║  Final Reward: {episode_reward:.2f}                      ║
║                                                          ║
║  The agent used LEARNED behavior from RL training.       ║
║  Much smarter than hard-coded heuristics!                ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

