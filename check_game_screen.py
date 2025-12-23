#!/usr/bin/env python3
"""
Check what the current game screen looks like
"""

from pokemon_env.emulator import EmeraldEmulator
import time

print("Checking game screen state...")
print("=" * 60)

emulator = EmeraldEmulator(
    rom_path="/home/jpli/pokeagent-speedrun/Emerald-GBAdvance/rom.gba",
    headless=True
)
emulator.initialize()

# Get screenshot
screenshot = emulator.get_screenshot()
screenshot.save("/tmp/current_screen.png")
print(f"✅ Screenshot saved to: /tmp/current_screen.png")

# Try to read game state
try:
    if hasattr(emulator, 'memory_reader'):
        coords = emulator.memory_reader.read_coordinates()
        location = emulator.memory_reader.read_location()
        dialogue = emulator.memory_reader.read_dialogue()
        
        print(f"\n📍 Coordinates: {coords}")
        print(f"🗺️  Location: {location}")
        print(f"💬 Dialogue: {dialogue[:100] if dialogue else 'None'}")
        
        if coords == (0, 0) and not location:
            print("\n⚠️  LIKELY AT TITLE SCREEN!")
            print("   Coordinates (0,0) and no location = title screen")
    else:
        print("\n⚠️  No memory reader available")
except Exception as e:
    print(f"\n⚠️  Error reading game state: {e}")
    print("   This might indicate title screen or uninitialized game")

# Test: Press START a few times and see if state changes
print("\n🎮 Testing button presses...")
print("   Initial state read...")
initial_screenshot = emulator.get_screenshot()

print("   Pressing START...")
emulator.press_key('start', frames=5)
time.sleep(0.1)

print("   Pressing A...")
emulator.press_key('a', frames=5)
time.sleep(0.1)

final_screenshot = emulator.get_screenshot()
final_screenshot.save("/tmp/after_buttons.png")

# Compare screenshots
import numpy as np
initial_arr = np.array(initial_screenshot)
final_arr = np.array(final_screenshot)
diff = np.abs(initial_arr.astype(float) - final_arr.astype(float)).mean()

print(f"   Screenshot difference: {diff:.2f}")
if diff < 1.0:
    print("   ⚠️  NO CHANGE! Likely stuck at title screen or frozen")
else:
    print("   ✅ Screen changed!")

print("\n" + "=" * 60)
print("Screenshots saved:")
print("  - /tmp/current_screen.png")
print("  - /tmp/after_buttons.png")
print("\nView with: display /tmp/current_screen.png")

