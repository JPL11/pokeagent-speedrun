#!/usr/bin/env python3
"""
Manual corner unstuck - force agent out of corner collision
Use when stuck at corner/obstacle intersection
"""

import requests
import time

SERVER_URL = "http://localhost:8000"

print("="*80)
print("🔺 UNSTUCK FROM CORNER - AGGRESSIVE EXPLORATION")
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
    pass

print(f"\n🎯 Strategy: Try all 4 directions aggressively")
print(f"   Corner = obstacle intersection, need systematic exploration")

# Aggressive corner escape: try all directions with A presses
actions = [
    'UP', 'UP', 'A',        # Try up and interact
    'RIGHT', 'RIGHT', 'A',  # Try right and interact
    'DOWN', 'DOWN', 'A',    # Try down and interact
    'LEFT', 'LEFT', 'A',    # Try left and interact
    'UP', 'RIGHT', 'A',     # Diagonal attempts
    'DOWN', 'RIGHT', 'A',
    'UP', 'LEFT', 'A',
    'DOWN', 'LEFT', 'A'
]

print(f"\n🚀 Sending {len(actions)} actions to break out of corner:")
print(f"   Trying all 4 directions + diagonals with interactions")

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
    print("\n🎉 Corner unstuck sequence complete!")
    print("   Agent should have found a way out")
    print("   If still stuck, check for NPCs blocking path")
else:
    print("\n⚠️  Some actions failed")

print("="*80)

