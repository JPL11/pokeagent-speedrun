#!/usr/bin/env python3
"""
Manual break out of stair loop
Use if agent is currently going up/down stairs repeatedly
"""

import requests
import time

SERVER_URL = "http://localhost:8000"

print("="*80)
print("🪜 BREAK STAIR LOOP - FORCE EXIT SEARCH")
print("="*80)

try:
    response = requests.get(f"{SERVER_URL}/status", timeout=2)
    print("✅ Server is running")
except:
    print("❌ Cannot connect to server")
    exit(1)

# Get position
try:
    response = requests.get(f"{SERVER_URL}/state", timeout=5)
    if response.status_code == 200:
        state = response.json()
        player_pos = state.get("player", {}).get("position", {})
        px = player_pos.get("x", 0)
        py = player_pos.get("y", 0)
        location = state.get("player", {}).get("location", "Unknown")
        print(f"📍 Position: ({px}, {py}) at {location}")
except:
    px, py = None, None

print(f"\n🎯 Strategy: Move AWAY from stairs, search for house exit")
print(f"   Stairs are usually at edges, exit is at BOTTOM")

# Force movement away from stairs and toward exit
actions = [
    # Move left away from stairs
    'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT',
    # Move down toward exit
    'DOWN', 'DOWN', 'DOWN',
    # Try right side
    'RIGHT', 'RIGHT', 'RIGHT',
    # More down (exit is at bottom)
    'DOWN', 'DOWN', 'DOWN', 'DOWN', 'DOWN'
]

print(f"\n🚀 Sending {len(actions)} actions to break loop:")
print(f"   {' → '.join(actions[:15])}...")

success = 0
for i, action in enumerate(actions):
    try:
        response = requests.post(
            f"{SERVER_URL}/action",
            json={"buttons": [action]},
            timeout=5
        )
        if response.status_code == 200:
            success += 1
            print(f"   [{i+1}/{len(actions)}] {action} ✓", end='\r')
        time.sleep(0.1)
    except Exception as e:
        print(f"\n   Error: {e}")
        break

print(f"\n\n✅ Sent {success}/{len(actions)} actions")

if success == len(actions):
    print("\n🎉 Stair loop should be broken!")
    print("   Agent moved away from stairs")
    print("   Should find exit door now")
else:
    print("\n⚠️  Try running again")

print("="*80)

