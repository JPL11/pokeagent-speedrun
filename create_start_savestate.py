#!/usr/bin/env python3
"""
Create a savestate past the title screen for RL training
"""

from pokemon_env.emulator import EmeraldEmulator
import time
import os

print("Creating savestate past title screen...")
print("=" * 60)

emulator = EmeraldEmulator(
    rom_path="/home/jpli/pokeagent-speedrun/Emerald-GBAdvance/rom.gba",
    headless=True
)
emulator.initialize()

print("✅ Emulator initialized")
print("🎮 Skipping title screen and cutscenes...")
print("   Strategy: Multiple A presses (like PokemonRedExperiments)")

# Strategy: Just spam A to skip everything!
# This works for: title screen → intro cutscene → new game setup
for i in range(50):
    if i % 10 == 0:
        print(f"   Press {i+1}-{i+10}: A (spam to skip cutscenes)")
    emulator.press_key('a', frames=3)
    time.sleep(0.05)  # Very fast

print("\n✅ Button presses complete!")

# Check if we're past title screen
try:
    coords = emulator.memory_reader.read_coordinates()
    location = emulator.memory_reader.read_location()
    print(f"\n📍 Current state:")
    print(f"   Coordinates: {coords}")
    print(f"   Location: {location}")
    
    if coords != (0, 0) or location:
        print("   ✅ Successfully past title screen!")
    else:
        print("   ⚠️  Still at title screen (0,0)")
except Exception as e:
    print(f"   ⚠️  Could not read state: {e}")

# Save the state
os.makedirs(".pokeagent_cache", exist_ok=True)
savestate_path = ".pokeagent_cache/past_title_screen.state"

emulator.save_state(savestate_path)
print(f"\n💾 Savestate saved to: {savestate_path}")

# Also save screenshot
screenshot = emulator.get_screenshot()
screenshot.save("/tmp/past_title_screen.png")
print(f"📸 Screenshot saved to: /tmp/past_title_screen.png")

print("\n" + "=" * 60)
print("✅ Done! Use this savestate for RL training.")
print(f"\nTo use: emulator.load_state('{savestate_path}')")

