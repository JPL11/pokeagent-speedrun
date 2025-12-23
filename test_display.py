#!/usr/bin/env python3
"""
Quick test to verify display works
"""

from pokemon_env.emulator import EmeraldEmulator
from utils.display_helper import GameDisplay
import time

print("Testing display...")
print("=" * 60)

# Create emulator (headless=False for display test)
emulator = EmeraldEmulator(
    rom_path="Emerald-GBAdvance/rom.gba",
    headless=True  # Still headless for emulator, we add display layer
)
emulator.initialize()

# Create display
display = GameDisplay(title="Display Test", scale=3)

print("✅ Display created")
print("🎮 Running emulator to generate frames...")

# IMPORTANT: Need to run emulator frames first!
print("   Ticking emulator 100 times to initialize...")
for _ in range(100):
    emulator.tick(1)

print("   Pressing A to skip title screen...")
for _ in range(20):
    emulator.press_key('a', frames=3)
    
print("   Ticking more frames...")
for _ in range(50):
    emulator.tick(1)

print("\n✅ Emulator initialized, now showing display...")
print("🎮 Showing 100 frames...")
print("   (Press Ctrl+C to stop)")

try:
    for i in range(100):
        # Tick emulator to generate new frame
        emulator.tick(1)
        
        # Get frame
        frame = emulator.get_screenshot()
        
        # Show in window
        display.show_frame(frame)
        
        # Press random button occasionally
        if i % 20 == 0:
            emulator.press_key('a', frames=2)
            print(f"   Frame {i}: Pressed A")
        
        time.sleep(0.05)  # ~20 FPS
    
    print("\n✅ Display test complete!")
    print("   If you saw a window with the game, display is working!")
    
except KeyboardInterrupt:
    print("\n\n⚠️  Test interrupted")

display.close()
print("Window closed")

