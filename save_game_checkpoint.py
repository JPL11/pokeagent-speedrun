#!/usr/bin/env python3
"""
Save the current game state as a checkpoint.

This saves the emulator's RAM state (not the RL model weights).
You can reload this state later to continue from this exact point in the game.

Usage:
    python save_game_checkpoint.py                           # Save to default location
    python save_game_checkpoint.py my_checkpoint.state       # Save to specific file
"""

import requests
import sys
import os
from datetime import datetime

def save_game_state(filepath=None):
    """Save the current game state via the server API"""
    
    # Generate default filename with timestamp if not provided
    if not filepath:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f".pokeagent_cache/game_checkpoint_{timestamp}.state"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    
    try:
        print(f"💾 Saving game state to: {filepath}")
        
        # Call the save_state endpoint
        response = requests.post(
            "http://localhost:8000/save_state",
            json={"filepath": filepath},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ {result['message']}")
            
            # Get current game info
            state_response = requests.get("http://localhost:8000/state", timeout=5)
            if state_response.status_code == 200:
                game_state = state_response.json()
                player_pos = game_state.get('player', {}).get('position', {})
                location = game_state.get('map', {}).get('stitched_map_info', {}).get('current_area', {}).get('location', 'Unknown')
                party = game_state.get('party', [])
                
                print(f"\n📍 Game State Info:")
                print(f"   Location: {location}")
                print(f"   Position: ({player_pos.get('x', '?')}, {player_pos.get('y', '?')})")
                print(f"   Party size: {len(party)}")
                if party:
                    for i, mon in enumerate(party, 1):
                        print(f"   Pokemon {i}: {mon.get('name', 'Unknown')} - HP: {mon.get('hp', 0)}/{mon.get('max_hp', 0)}")
            
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to game server (http://localhost:8000)")
        print("   Make sure the server is running!")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def load_game_state(filepath):
    """Load a saved game state"""
    try:
        if not os.path.exists(filepath):
            print(f"❌ Error: File not found: {filepath}")
            return False
        
        print(f"📂 Loading game state from: {filepath}")
        
        response = requests.post(
            "http://localhost:8000/load_state",
            json={"filepath": filepath},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ {result['message']}")
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--load" or sys.argv[1] == "-l":
            # Load mode
            if len(sys.argv) > 2:
                load_game_state(sys.argv[2])
            else:
                print("Usage: python save_game_checkpoint.py --load <filepath>")
        else:
            # Save to specified file
            save_game_state(sys.argv[1])
    else:
        # Save to default timestamped file
        save_game_state()
