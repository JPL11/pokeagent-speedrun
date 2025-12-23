#!/usr/bin/env python3
"""
Test RL environment in isolation to find bottleneck
"""

import time
import logging
from pokemon_env.emulator import EmeraldEmulator
from agent.rl_trainer import PokemonEmeraldRLEnv, RLConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_env():
    print("=" * 60)
    print("Testing RL Environment Performance")
    print("=" * 60)
    
    # 1. Test emulator initialization
    print("\n1️⃣ Testing emulator initialization...")
    start = time.time()
    emulator = EmeraldEmulator(
        rom_path="/home/jpli/pokeagent-speedrun/Emerald-GBAdvance/rom.gba",
        headless=True
    )
    emulator.initialize()
    print(f"   ✅ Emulator initialized in {time.time()-start:.2f}s")
    
    # 2. Test screenshot
    print("\n2️⃣ Testing screenshot capture...")
    start = time.time()
    frame = emulator.get_screenshot()
    print(f"   ✅ Screenshot captured in {time.time()-start:.4f}s")
    print(f"   Frame size: {frame.size if frame else 'None'}")
    
    # 3. Test environment creation
    print("\n3️⃣ Testing environment creation...")
    start = time.time()
    config = RLConfig(
        total_timesteps=1000,
        use_snn_features=False,  # Start without SNN
        use_vlm_guidance=False
    )
    env = PokemonEmeraldRLEnv(emulator, config, vlm=None)
    print(f"   ✅ Environment created in {time.time()-start:.2f}s")
    
    # 4. Test reset (THIS IS LIKELY WHERE IT HANGS)
    print("\n4️⃣ Testing environment reset...")
    print("   (This might take a while if it's the bottleneck)")
    start = time.time()
    obs, info = env.reset()
    elapsed = time.time()-start
    print(f"   ✅ Environment reset in {elapsed:.2f}s")
    print(f"   Observation shape: {obs.shape}")
    
    if elapsed > 10:
        print(f"   ⚠️  SLOW! Reset took {elapsed:.2f}s - this is the bottleneck!")
    
    # 5. Test steps
    print("\n5️⃣ Testing environment steps...")
    step_times = []
    for i in range(10):
        start = time.time()
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        step_time = time.time()-start
        step_times.append(step_time)
        print(f"   Step {i+1}: {step_time:.4f}s, reward: {reward:.3f}")
        
        if step_time > 1.0:
            print(f"   ⚠️  SLOW STEP! This is a problem.")
    
    avg_step_time = sum(step_times) / len(step_times)
    print(f"\n   Average step time: {avg_step_time:.4f}s")
    
    # 6. Calculate expected training time
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    steps_needed = 2048  # PPO rollout buffer
    estimated_rollout_time = steps_needed * avg_step_time
    
    print(f"Average step time: {avg_step_time:.4f}s")
    print(f"Steps per rollout: {steps_needed}")
    print(f"Estimated time per rollout: {estimated_rollout_time/60:.1f} minutes")
    print(f"Estimated FPS: {1/avg_step_time:.1f}")
    
    if estimated_rollout_time > 600:  # 10 minutes
        print("\n⚠️  PROBLEM: Training is TOO SLOW!")
        print("   Each rollout would take over 10 minutes.")
        print("   For 500k timesteps, that's over 40 HOURS!")
    elif estimated_rollout_time > 300:  # 5 minutes
        print("\n⚠️  WARNING: Training is slow")
        print(f"   For 500k timesteps: ~{(500000/steps_needed * estimated_rollout_time/3600):.1f} hours")
    else:
        print("\n✅ Performance acceptable!")
        print(f"   For 500k timesteps: ~{(500000/steps_needed * estimated_rollout_time/3600):.1f} hours")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        test_env()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

