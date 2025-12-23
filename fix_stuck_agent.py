#!/usr/bin/env python3
"""
Emergency intervention to fix stuck agent in May's House
"""

import requests
import time
import sys

SERVER_URL = "http://localhost:8000"

def get_current_state():
    """Get current game state"""
    response = requests.get(f"{SERVER_URL}/state")
    return response.json()

def send_action(action):
    """Send a button action to the game"""
    response = requests.post(f"{SERVER_URL}/agent", json={"action": action})
    return response.json()

def get_to_littleroot_town():
    """Navigate out of May's house to Littleroot Town"""
    print("🚨 EMERGENCY INTERVENTION: Getting agent out of May's House")
    
    state = get_current_state()
    location = state['player']['location']
    pos = state['player']['position']
    
    print(f"Current location: {location}")
    print(f"Current position: ({pos['x']}, {pos['y']})")
    
    if "MAYS HOUSE" not in location.upper():
        print("✅ Agent is not in May's House. No intervention needed.")
        return True
    
    # May's House 1F - door is at bottom center around (5, 8)
    # Navigate to door and exit
    actions_sequence = []
    
    # First, get to the center
    if pos['x'] > 6:
        actions_sequence.extend(['LEFT'] * (pos['x'] - 6))
    elif pos['x'] < 5:
        actions_sequence.extend(['RIGHT'] * (5 - pos['x']))
    
    # Then go down to door
    if pos['y'] < 7:
        actions_sequence.extend(['DOWN'] * (7 - pos['y']))
    
    # Try to exit (door should trigger automatically when stepping on it)
    actions_sequence.extend(['DOWN', 'DOWN'])
    
    print(f"📝 Executing {len(actions_sequence)} actions to reach door...")
    
    for i, action in enumerate(actions_sequence):
        print(f"  Step {i+1}/{len(actions_sequence)}: {action}")
        send_action(action)
        time.sleep(0.1)
        
        # Check if we escaped
        state = get_current_state()
        new_location = state['player']['location']
        if "MAYS HOUSE" not in new_location.upper():
            print(f"✅ SUCCESS! Now in: {new_location}")
            return True
    
    # Check final location
    state = get_current_state()
    final_location = state['player']['location']
    
    if "MAYS HOUSE" not in final_location.upper():
        print(f"✅ SUCCESS! Agent escaped to: {final_location}")
        return True
    else:
        print(f"❌ FAILED: Still in {final_location}")
        print(f"Current position: {state['player']['position']}")
        return False

def go_to_oldale_pokemon_center():
    """Navigate to Oldale Town Pokemon Center for healing"""
    print("\n🏥 Navigating to Oldale Town Pokemon Center...")
    
    state = get_current_state()
    location = state['player']['location']
    
    if "OLDALE" in location.upper():
        print("✅ Already in Oldale Town")
    elif "LITTLEROOT" in location.upper():
        print("📍 In Littleroot, going north to Route 101...")
        # Go north out of Littleroot to Route 101
        for _ in range(20):
            send_action('UP')
            time.sleep(0.1)
        
        time.sleep(0.5)
        state = get_current_state()
        location = state['player']['location']
        
        if "ROUTE 101" in location.upper():
            print("✅ Reached Route 101, continuing to Oldale...")
            # Continue north through Route 101
            for _ in range(30):
                send_action('UP')
                time.sleep(0.1)
    
    time.sleep(0.5)
    state = get_current_state()
    location = state['player']['location']
    print(f"Final location: {location}")

def heal_at_pokemon_center():
    """Heal Pokemon at the Pokemon Center"""
    print("\n💊 Attempting to heal at Pokemon Center...")
    
    # Look for Pokemon Center and enter
    for _ in range(5):
        send_action('UP')
        time.sleep(0.1)
    
    time.sleep(0.5)
    
    # Talk to Nurse Joy (press A at counter)
    for _ in range(3):
        send_action('A')
        time.sleep(0.3)
    
    # Wait for healing animation
    print("⏳ Waiting for healing...")
    time.sleep(3)
    
    # Advance dialogs
    for _ in range(5):
        send_action('A')
        time.sleep(0.3)
    
    print("✅ Healing complete!")

def main():
    print("=" * 60)
    print("EMERGENCY AGENT INTERVENTION")
    print("=" * 60)
    
    # Step 1: Get out of May's House
    if not get_to_littleroot_town():
        print("\n❌ Failed to escape May's House. Manual intervention required.")
        print("\nTry running:")
        print("  python manual.py")
        print("\nThen manually navigate:")
        print("  1. Go to center of room")
        print("  2. Go DOWN to door")
        print("  3. EXIT to Littleroot Town")
        sys.exit(1)
    
    # Step 2: Get current milestone status
    state = get_current_state()
    print(f"\n📊 Current Status:")
    print(f"  Location: {state['player']['location']}")
    print(f"  Position: {state['player']['position']}")
    print(f"  Money: ${state['game']['money']}")
    
    if state['player']['party']:
        party = state['player']['party'][0]
        print(f"  Party: {party['species_name']} Lv{party['level']} ({party['current_hp']}/{party['max_hp']} HP)")
    
    print("\n" + "=" * 60)
    print("✅ INTERVENTION COMPLETE")
    print("=" * 60)
    print("\nThe agent is now outside May's House.")
    print("Training should be able to continue normally.")
    print("\nTo restart training, run:")
    print("  ./start_enhanced_training.sh")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Intervention cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
