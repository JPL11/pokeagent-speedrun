#!/usr/bin/env python3
"""
Force agent to find and use the exit door to leave the house
Use this when stuck on ground floor
"""

import requests
import time

SERVER_URL = "http://localhost:8000"

print("="*80)
print("🚪 FORCE EXIT HOUSE - GO OUTSIDE TO TOWN")
print("="*80)

# Check server
try:
    response = requests.get(f"{SERVER_URL}/status", timeout=2)
    print("✅ Server is running")
except Exception as e:
    print(f"❌ Cannot connect to server")
    exit(1)

# Get current position
try:
    response = requests.get(f"{SERVER_URL}/state", timeout=5)
    if response.status_code == 200:
        state = response.json()
        player_pos = state.get("player", {}).get("position", {})
        px = player_pos.get("x", 0)
        py = player_pos.get("y", 0)
        location = state.get("player", {}).get("location", "Unknown")
        print(f"📍 Current position: ({px}, {py}) at {location}")
    else:
        px, py = None, None
except Exception as e:
    px, py = None, None

print(f"\n🎯 Strategy: Exit door is typically at BOTTOM of room")
print(f"   Will move DOWN to find and use the exit")

# In Pokemon houses, exit is typically at the bottom center
# Common exit positions: around (4-5, 8-9) but we just go DOWN
actions = []

if px is not None and py is not None:
    print(f"\n📐 Building exit path from ({px}, {py})")
    
    # First move to horizontal center (around X=4-5)
    target_center_x = 4
    if px < target_center_x:
        moves = target_center_x - px
        actions.extend(['RIGHT'] * moves)
        print(f"   → Move RIGHT {moves} to center")
    elif px > target_center_x:
        moves = px - target_center_x  
        actions.extend(['LEFT'] * moves)
        print(f"   → Move LEFT {moves} to center")
    
    # Now just go DOWN until we hit the exit
    print(f"   → Move DOWN to find exit")
    actions.extend(['DOWN'] * 8)  # Generous DOWN movements
    
else:
    print(f"\n⚠️  Using fallback - just go DOWN a lot")
    actions = ['DOWN'] * 10

print(f"\n🚀 Sending {len(actions)} actions:")
print(f"   {' → '.join(actions[:20])}")

# Send actions
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
        time.sleep(0.15)
    except Exception as e:
        print(f"\n   Error: {e}")
        break

print(f"\n\n✅ Sent {success}/{len(actions)} actions")

if success == len(actions):
    print("\n🎉 Exit navigation complete!")
    print("   The agent should now be outside in the town!")
    print("   🏘️  Welcome to Littleroot Town!")
else:
    print("\n⚠️  Some actions failed - try running again")

print("="*80)

