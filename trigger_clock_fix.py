#!/usr/bin/env python3
"""
Manual trigger for clock interaction fix
Use this if your agent is currently stuck at the clock scene
"""

import requests
import time

SERVER_URL = "http://localhost:8000"

print("="*80)
print("🕐 MANUAL CLOCK INTERACTION TRIGGER")
print("="*80)

# Check if server is running
try:
    response = requests.get(f"{SERVER_URL}/status", timeout=2)
    if response.status_code == 200:
        print("✅ Server is running")
    else:
        print("❌ Server returned error")
        exit(1)
except Exception as e:
    print(f"❌ Cannot connect to server at {SERVER_URL}")
    print(f"   Make sure the agent is running first!")
    print(f"   Error: {e}")
    exit(1)

# Get current player position
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
        print("⚠️  Could not get current position")
        px, py = None, None
except Exception as e:
    print(f"⚠️  Error getting state: {e}")
    px, py = None, None

# Calculate path to clock
clock_x, clock_y = 5, 1  # Clock position in player's bedroom
target_x, target_y = clock_x, clock_y + 1  # Stand below clock at (5, 2)

print(f"\n🎯 Target: Clock at ({clock_x}, {clock_y})")
print(f"   Will navigate to ({target_x}, {target_y}) then interact")

# Build action sequence
actions = []

if px is not None and py is not None:
    print(f"\n📐 Building navigation path from ({px}, {py}) to ({target_x}, {target_y})")
    
    # Horizontal movement
    if px < target_x:
        moves_right = target_x - px
        actions.extend(['RIGHT'] * moves_right)
        print(f"   → Move RIGHT {moves_right} steps")
    elif px > target_x:
        moves_left = px - target_x
        actions.extend(['LEFT'] * moves_left)
        print(f"   → Move LEFT {moves_left} steps")
    
    # Vertical movement
    if py < target_y:
        moves_down = target_y - py
        actions.extend(['DOWN'] * moves_down)
        print(f"   → Move DOWN {moves_down} steps")
    elif py > target_y:
        moves_up = py - target_y
        actions.extend(['UP'] * moves_up)
        print(f"   → Move UP {moves_up} steps")
    
    # Interaction sequence
    print(f"   → Face UP toward clock")
    print(f"   → Press A to interact")
    print(f"   → Navigate clock menu")
    
    actions.extend(['UP', 'A', 'A'])  # Face up and interact
    actions.extend(['A', 'A', 'A', 'A', 'A'])  # Clock menu
else:
    print(f"\n⚠️  Could not get coordinates, using fallback path")
    # Fallback path
    actions = [
        'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT',
        'UP', 'UP', 'UP',
        'LEFT', 'UP', 'A', 'A',
        'A', 'A', 'A', 'A', 'A'
    ]

print(f"\n🚀 Sending {len(actions)} actions to server:")
print(f"   {' → '.join(actions[:15])}{'...' if len(actions) > 15 else ''}")

# Send actions to server one by one
success_count = 0
for i, action in enumerate(actions):
    try:
        response = requests.post(
            f"{SERVER_URL}/action",
            json={"buttons": [action]},
            timeout=5
        )
        if response.status_code == 200:
            success_count += 1
            print(f"   [{i+1}/{len(actions)}] {action} ✓", end='\r')
        else:
            print(f"\n   [{i+1}/{len(actions)}] {action} ✗ - Error {response.status_code}")
        
        # Small delay between actions to avoid overwhelming the queue
        time.sleep(0.1)
    except Exception as e:
        print(f"\n   [{i+1}/{len(actions)}] {action} ✗ - {e}")
        break

print(f"\n\n✅ Sent {success_count}/{len(actions)} actions successfully")

if success_count == len(actions):
    print("\n🎉 Clock interaction sequence completed!")
    print("   The agent should now:")
    print("   1. Navigate to the clock")
    print("   2. Face the clock")
    print("   3. Press A to interact")
    print("   4. Complete the clock-setting menu")
    print("\n   Watch the game to see if it works!")
else:
    print("\n⚠️  Some actions failed. You may need to:")
    print("   1. Check if the server is still running")
    print("   2. Verify the agent isn't in a frozen state")
    print("   3. Try running this script again")

print("="*80)

