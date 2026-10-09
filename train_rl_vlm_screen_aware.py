#!/usr/bin/env python3
"""
VLM Server-Aware RL Training
Uses VLM to analyze server endpoints (/agent, /state, /milestones, action queue)
Fixes broken coordinate tracking by using server API data + visual information
"""

import argparse
import os
import sys
import logging
import time
import re
import requests
import base64
from io import BytesIO
from pathlib import Path
from PIL import Image
from typing import Optional, Tuple, List, Dict

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pokemon_env.emulator import EmeraldEmulator
from agent.rl_trainer import train_rl_agent, RLConfig, PokemonEmeraldRLEnv
from utils.vlm import create_vlm
from agent.strategic_directive_manager import StrategicDirectiveManager
from agent.terrain_navigator import TerrainNavigator, encode_terrain_for_snn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global map ID to name mapping (Emerald correct IDs)
MAP_NAMES = {
    0x09: "LITTLEROOT_TOWN",
    0x100: "BRENDANS_HOUSE_1F",
    0x101: "BRENDANS_HOUSE_2F", 
    0x102: "MAYS_HOUSE_1F",
    0x103: "MAYS_HOUSE_2F",
    0x104: "PROFESSOR_BIRCHS_LAB",
    0x10: "ROUTE_101",
    0x11: "OLDALE_TOWN",
    0x12: "ROUTE_103",
    0x0F: "ROUTE_102",
    0x200: "OLDALE_POKECENTER",  # PokeCenter interior
}

def get_map_name(map_id):
    """Convert map ID to readable name"""
    return MAP_NAMES.get(map_id, f"UNKNOWN_MAP_0x{map_id:04X}")

def _get_story_context(completed, location_name, party_size, has_pokedex_flag=False, lead_level=5):
    """
    Get current story context and next milestone goal
    Uses both milestone flags AND game state (location, party, pokedex) for accuracy
    """
    # FALLBACK: Use actual game state to infer progression if milestones aren't tracked
    has_pokemon = party_size > 0
    at_lab = "BIRCH" in location_name or "LAB" in location_name
    at_route_101 = "ROUTE" in location_name and "101" in location_name
    at_town = "LITTLEROOT" in location_name and not at_lab
    
    # Detect player house locations (May's house = female player, Brendan's house = male player)
    at_player_house_1f = ("MAYS HOUSE 1F" in location_name or "BRENDANS HOUSE 1F" in location_name or 
                          "MAY HOUSE 1F" in location_name or "BRENDAN HOUSE 1F" in location_name)
    at_player_house_2f = ("MAYS HOUSE 2F" in location_name or "BRENDANS HOUSE 2F" in location_name or
                          "MAY HOUSE 2F" in location_name or "BRENDAN HOUSE 2F" in location_name or
                          "PLAYER BEDROOM" in location_name)
    at_rival_house = ("RIVAL" in location_name.upper() or 
                      ("MAYS" not in location_name and "BRENDANS" not in location_name and 
                       ("HOUSE" in location_name and "LITTLEROOT" in location_name)))
    
    # Check milestones OR infer from game state
    # SET_CLOCK is complete if we've left our house OR have Pokemon OR visited rival
    has_set_clock = 'SET_CLOCK' in completed or has_pokemon or 'RIVAL_BEDROOM' in completed
    # If at player house 2F or beyond, clock is set (you can't reach 2F without going through cutscene)
    if at_player_house_2f:
        has_set_clock = True
    
    # CRITICAL: If you have Pokemon (STARTER_CHOSEN), you MUST have visited rival bedroom already!
    has_rival_bedroom = 'RIVAL_BEDROOM' in completed or has_pokemon or 'STARTER_CHOSEN' in completed
    has_route_101 = 'ROUTE_101' in completed or at_route_101 or has_pokemon
    has_starter = 'STARTER_CHOSEN' in completed or has_pokemon
    has_pokedex = 'RECEIVED_POKEDEX' in completed or has_pokedex_flag  # Check actual flag!
    
    # Story progression logic - START WITH SET_CLOCK (very first step)
    # Priority 1: Player needs to set the clock in their bedroom (2F) before mom lets them leave
    if not has_set_clock and at_player_house_1f:
        return {
            'story': "You just arrived at your new home! Mom says to go upstairs to your room. Go to the stairs and up to 2F to set your clock.",
            'goal': "SET_CLOCK",
            'why': "Mom won't let you leave until you set the clock upstairs in your bedroom (2F)",
            'nav_hint': "Go UP the stairs (upper-left area of 1F) to reach 2F and set your clock"
        }
    elif not has_set_clock and at_player_house_2f:
        return {
            'story': "You're in your bedroom! Go to the clock on the wall and interact with it (press A) to set the time.",
            'goal': "SET_CLOCK",
            'why': "Setting the clock triggers the cutscene and lets you leave the house",
            'nav_hint': "Walk to the clock on the wall and press A to interact"
        }
    elif not has_rival_bedroom and not has_pokemon:
        # After setting clock, go to rival's house
        if at_player_house_1f or at_player_house_2f:
            return {
                'story': "Clock is set! Now go downstairs and leave your house. Your rival lives next door - visit their bedroom.",
                'goal': "RIVAL_BEDROOM",
                'why': "Meeting your rival is required before starting your Pokemon journey",
                'nav_hint': "Go downstairs (if on 2F), then exit through the bottom door. Rival's house is next door."
            }
        elif at_rival_house and "1F" in location_name:
            return {
                'story': "You're in your rival's house! Go upstairs to their bedroom (2F) to meet them.",
                'goal': "RIVAL_BEDROOM", 
                'why': "Your rival is waiting upstairs in their room",
                'nav_hint': "Go UP the stairs to reach 2F and meet your rival"
            }
        else:
            return {
                'story': "You just moved to Littleroot Town. Your rival lives here too. You need to visit their bedroom.",
                'goal': "RIVAL_BEDROOM",
                'why': "Meeting your rival is required before starting your Pokemon journey"
            }
    elif not has_route_101 and not has_pokemon:
        return {
            'story': "You've met your rival. Professor Birch is studying Pokemon on Route 101 north of town.",
            'goal': "ROUTE_101",
            'why': "Prof. Birch needs help - this is where you'll get your first Pokemon"
        }
    elif not has_starter and not has_pokemon:
        return {
            'story': "Prof. Birch is being chased by a wild Pokemon! You must help him.",
            'goal': "STARTER_CHOSEN",
            'why': "Save Birch to receive your starter Pokemon and begin your adventure"
        }
    elif has_pokemon and not has_pokedex:
        # Have starter, but Birch might tell you to find May/Brendan first!
        # FIX: Don't trust checkpoint FIRST_RIVAL_BATTLE - may be stale!
        # Only trust if we have Pokedex (definitive proof rival was beaten)
        rival_battled = has_pokedex_flag  # Remove checkpoint trust: 'FIRST_RIVAL_BATTLE' in completed
        
        if rival_battled:
            # Battled rival, now return to lab for Pokedex
            if at_lab:
                return {
                    'story': "You defeated your rival! Now talk to Birch to receive the Pokedex.",
                    'goal': "RECEIVED_POKEDEX",
                    'why': "Birch promised the Pokedex after you find and battle your rival"
                }
            else:
                return {
                    'story': "You've battled your rival! Return to Birch's Lab for the Pokedex.",
                    'goal': "RECEIVED_POKEDEX",
                    'why': "Birch will give you the Pokedex now that you've proven yourself"
                }
        else:
            # Go to Route 103 to find and battle rival
            # REMOVED: GRIND_ROUTE_101 logic - just proceed to rival battle!
            # Wild Pokemon battles happen naturally during travel
            
            if at_lab:
                return {
                    'story': "Birch says: 'Find May/Brendan on Route 103!' Go north through Oldale Town to Route 103.",
                    'goal': "FIND_RIVAL_ROUTE_103",
                    'why': "Birch won't give Pokedex until you find and battle your rival"
                }
            elif "OLDALE" in location_name:
                return {
                    'story': "In Oldale Town. Continue north to Route 103 to find your rival.",
                    'goal': "FIND_RIVAL_ROUTE_103",
                    'why': "Need to battle rival before getting Pokedex from Birch"
                }
            elif "103" in location_name:
                return {
                    'story': "On Route 103! Look for your rival May/Brendan to battle them.",
                    'goal': "FIRST_RIVAL_BATTLE",
                    'why': "Battle your rival, then return to Birch for Pokedex"

                }
            else:
                return {
                    'story': "Birch wants you to find your rival. Head north: Route 101 → Oldale Town → Route 103.",
                    'goal': "FIND_RIVAL_ROUTE_103",
                    'why': "Must find and battle rival before receiving Pokedex"
                }
    elif has_pokedex and at_lab:
        # Already have Pokedex, still at lab - LEAVE and head north!
        return {
            'story': "Pokedex received! Time to start your journey. Exit the lab and head north through Route 101.",
            'goal': "LEAVE_LAB_TO_ROUTE_101",
            'why': "You've completed early game tasks. Now explore Route 101 and progress to Oldale Town"
        }
    else:
        # Has Pokedex, not at lab - go WEST to Route 102 → Petalburg City!
        return {
            'story': "Pokédex received! Continue your journey WEST through Oldale Town to Route 102 → Petalburg City.",
            'goal': "ROUTE_102",
            'why': "Head WEST from Oldale Town to Route 102, then continue to Petalburg City for your next adventure"
        }

def _get_direction_to_target(player_x: int, player_y: int, target_x: int, target_y: int) -> str:
    """
    Compute the primary direction to move from player position to target.
    Returns human-readable direction string for VLM guidance.
    
    Args:
        player_x, player_y: Current player position
        target_x, target_y: Target coordinates (stairs, door, etc.)
    
    Returns:
        Direction string like 'UP', 'DOWN', 'LEFT', 'RIGHT', or combination like 'UP and LEFT'
    """
    dx = target_x - player_x
    dy = target_y - player_y
    
    if dx == 0 and dy == 0:
        return "you're at the target"
    
    directions = []
    
    # Priority: Y direction first (UP/DOWN), then X (LEFT/RIGHT)
    if dy < 0:
        directions.append("UP")
    elif dy > 0:
        directions.append("DOWN")
    
    if dx < 0:
        directions.append("LEFT")
    elif dx > 0:
        directions.append("RIGHT")
    
    if len(directions) == 1:
        return directions[0]
    elif len(directions) == 2:
        return f"{directions[0]} and {directions[1]}"
    else:
        return "explore to find target"

def _get_stairs_direction_2f(player_x: int, player_y: int, is_player_house: bool = False) -> str:
    """
    Get direction to stairs on 2F of houses. Visual map shows stairs at (1,1).
    Returns VLM-friendly direction guidance.
    """
    # Stairs at (1, 1) based on visual map ground truth (not portal data!)
    stairs_x, stairs_y = 1, 1
    direction = _get_direction_to_target(player_x, player_y, stairs_x, stairs_y)
    
    if direction == "you're at the target":
        return "at stairs - press DOWN to descend to 1F"
    else:
        return f"go {direction} to reach stairs at ({stairs_x},{stairs_y}), then DOWN to 1F"

def _get_door_direction_1f(player_x: int, player_y: int) -> str:
    """
    Get direction to exit door on 1F of houses. Door is typically at (1, 8).
    Returns VLM-friendly direction guidance.
    """
    # Door at (1, 8) based on verified coordinates
    door_x, door_y = 1, 8
    direction = _get_direction_to_target(player_x, player_y, door_x, door_y)
    
    if direction == "you're at the target":
        return "at door - press DOWN to exit house"
    else:
        return f"go {direction} to door at ({door_x},{door_y}), then DOWN to exit"

def _get_milestone_action(location_name, completed, game_state, is_stuck, party_size=0, has_pokedex=False, player_x=0, player_y=0, player_gender='Boy'):
    """
    Get specific action to progress toward next milestone
    Uses both milestone flags AND game state (party_size, pokedex, location) for accuracy
    Now position-aware: computes actual direction to stairs/doors based on player coords
    """
    if game_state == 'dialog':
        return "Press A to advance dialog (story/NPC conversation)"
    
    # Infer game state from party size and location
    has_pokemon = party_size > 0
    at_lab = "BIRCH" in location_name or "LAB" in location_name
    at_route_101 = "ROUTE" in location_name and "101" in location_name
    
    # If player has Pokemon and is at lab, check if they need Pokedex
    if has_pokemon and at_lab:
        if not has_pokedex and 'RECEIVED_POKEDEX' not in completed:
            return "Talk to Professor Birch (press A near him) to receive the Pokedex - walk around lab to find him"
        else:
            # Already have Pokedex - LEAVE THE LAB!
            return "EXIT the lab (go DOWN to door, press A) and head NORTH through town to Route 101"
    
    # If player has Pokemon but not at lab, check if rival battled first!
    # MUST battle rival on Route 103 BEFORE going to lab for Pokedex!
    # FIX: Don't trust checkpoint FIRST_RIVAL_BATTLE - may be stale!
    # Only trust Pokedex as proof (received AFTER rival battle at lab)
    rival_battled = 'RECEIVED_POKEDEX' in completed  # More reliable than FIRST_RIVAL_BATTLE
    
    if has_pokemon and not at_lab and not has_pokedex and 'RECEIVED_POKEDEX' not in completed:
        if not rival_battled:
            # PHASE 1: Go NORTH to Route 103 to battle rival!
            if location_name == "LITTLEROOT_TOWN":
                return "Go UP/NORTH to Route 101 → Oldale Town → Route 103 to find and BATTLE your rival!"
            elif location_name == "ROUTE_101":
                return "Continue UP/NORTH through Route 101 to Oldale Town, then to Route 103 to BATTLE RIVAL!"
            elif "OLDALE" in location_name.upper():
                return "Go UP/NORTH to Route 103 to find your RIVAL and BATTLE them!"
            elif "103" in location_name:
                return "Find your RIVAL (May) on this route and BATTLE her! Press A when you see her."
            else:
                return "Go NORTH! Find your rival on Route 103 and BATTLE them first!"
        else:
            # PHASE 2: Rival defeated, NOW go to lab for Pokedex
            if location_name == "LITTLEROOT_TOWN":
                return "Enter Professor Birch's Lab (building on LEFT/WEST side) to get Pokedex"
            elif location_name == "ROUTE_101":
                return "Go DOWN/SOUTH to return to Littleroot Town, then enter Birch's Lab for Pokedex"
            else:
                return "Return to Professor Birch's Lab in Littleroot Town to receive Pokedex"
    
    # If player has Pokemon AND Pokedex, guide to Route 102 → Petalburg (next area)
    if has_pokemon and (has_pokedex or 'RECEIVED_POKEDEX' in completed):
        if location_name == "LITTLEROOT_TOWN":
            return "Go UP/NORTH to Route 101 → Oldale Town → Then WEST to Route 102 → Petalburg City"
        elif at_lab:
            return "Exit lab (DOWN), go NORTH to Oldale Town, then WEST to Route 102 → Petalburg City"
        elif at_route_101:
            return "Continue NORTH to Oldale Town, then go WEST to Route 102 → Petalburg City"
        elif "OLDALE" in location_name.upper():
            return "Go WEST/LEFT to Route 102 → Continue west to Petalburg City"
        elif "ROUTE 102" in location_name.upper():
            return "Continue WEST through Route 102 to reach Petalburg City"
        else:
            return "Head WEST to Route 102 → Petalburg City (the next milestone)"
    
    # Original early-game logic (no Pokemon yet)
    if not has_pokemon:
        if 'RIVAL_BEDROOM' not in completed:
            if location_name == "LITTLEROOT_TOWN":
                # Note: For Brendan, rival (May) is RIGHT. For May, rival (Brendan) is LEFT.
                # Just enter the OTHER house (not yours) to trigger story event.
                return "Enter your RIVAL's house (the OTHER house in town, not yours) and go upstairs to bedroom"
            elif "BRENDAN" in location_name.upper():
                # GENDER-AWARE LOGIC:
                # - Girl player: BRENDANS_HOUSE = RIVAL's house → go upstairs!
                # - Boy player: BRENDANS_HOUSE = PLAYER's house → EXIT!
                if player_gender == 'Girl':
                    # Girl: Brendan's = RIVAL's house - this is correct!
                    if "2F" in location_name.upper():
                        return "✅ You're in RIVAL's (Brendan's) bedroom! Walk around to trigger milestone completion"
                    else:
                        return "Go UPSTAIRS to find rival's (Brendan's) bedroom (up the stairs)"
                else:
                    # Boy: Brendan's = PLAYER's house - wrong, exit!
                    if "2F" in location_name.upper():
                        stairs_dir = _get_stairs_direction_2f(player_x, player_y)
                        return f"⚠️ This is YOUR bedroom! EXIT: {stairs_dir}, exit house, then go to MAY's house (rival's house) in town"
                    else:
                        door_dir = _get_door_direction_1f(player_x, player_y)
                        return f"⚠️ This is YOUR house! EXIT: {door_dir}, then go to MAY's house (rival's house) in town"
            elif "MAY" in location_name.upper():
                # GENDER-AWARE LOGIC:
                # - Girl player: MAYS_HOUSE = PLAYER's house → EXIT!
                # - Boy player: MAYS_HOUSE = RIVAL's house → go upstairs!
                if player_gender == 'Girl':
                    # Girl: May's = PLAYER's house - wrong, exit!
                    if "2F" in location_name.upper():
                        stairs_dir = _get_stairs_direction_2f(player_x, player_y)
                        return f"⚠️ This is YOUR bedroom! EXIT: {stairs_dir}, exit house, then go to BRENDAN's house (rival's house) in town"
                    else:
                        door_dir = _get_door_direction_1f(player_x, player_y)
                        return f"⚠️ This is YOUR house! EXIT: {door_dir}, then go to BRENDAN's house (rival's house) in town"
                else:
                    # Boy: May's = RIVAL's house - this is correct!
                    if "2F" in location_name.upper():
                        return "✅ You're in RIVAL's (May's) bedroom! Walk around to trigger milestone completion"
                    else:
                        return "Go UPSTAIRS to find rival's (May's) bedroom (up the stairs)"
            elif at_lab:
                return "Exit lab (DOWN), then go to your rival's house (the OTHER house in town)"
            else:
                return "Navigate to rival's house to complete RIVAL_BEDROOM milestone"
        

        elif 'ROUTE_101' not in completed:
            if location_name == "LITTLEROOT_TOWN":
                return "Go UP/NORTH to exit town and reach Route 101 (where Birch is)"
            elif location_name in ("MAYS_HOUSE_1F", "BRENDANS_HOUSE_1F", "MAYS_HOUSE_2F", "BRENDANS_HOUSE_2F"):
                return "Exit house (go DOWN/stairs), then go NORTH from town to Route 101"
            elif at_lab:
                return "Exit lab (DOWN), then go NORTH through town to Route 101"
            else:
                return "Head north to Route 101 to find Professor Birch"
        
        elif at_route_101:
            return "Explore tall grass to trigger Prof. Birch rescue cutscene and choose starter"
    
    if is_stuck:
        return "⚠️ STUCK - try a DIFFERENT direction to escape loop"
    
    return f"Explore {location_name} to progress story"


class ServerAPIOracle:
    """
    Enhanced VLM that uses SERVER API ENDPOINTS for real data
    Queries /agent, /state, /milestones, /recent_actions, system.log
    Provides comprehensive game state analysis without broken memory
    
    NEW: Grid search every 5-10 VLM calls to provide spatial awareness
    """
    
    def __init__(self, vlm, server_url="http://localhost:8000", log_file="server.log", use_visual=True):
        self.vlm = vlm
        self.use_visual = use_visual  # Can disable VLM visual to save VRAM
        self.server_url = server_url
        self.log_file = log_file
        self.last_log_position = 0
        self.detected_milestones = set()
        self.last_analysis = None
        
        # GRID SEARCH tracking for spatial awareness
        self.vlm_call_count = 0
        self.grid_search_interval = 7  # Do grid search every 7 VLM calls
        self.explored_tiles = {}  # {map_id: {(x,y): tile_type}}
        self.last_grid_search_map = None
        
    def get_server_state(self):
        """Get game state from /state endpoint"""
        try:
            response = requests.get(f"{self.server_url}/state", timeout=15)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.debug(f"Server state error: {e}")
        return {}
    
    def get_agent_thinking(self):
        """Get agent thinking log from /agent endpoint"""
        try:
            response = requests.get(f"{self.server_url}/agent", timeout=2)
            if response.status_code == 200:
                data = response.json()
                # Extract recent thoughts
                if 'recent_thoughts' in data:
                    return data['recent_thoughts'][-5:]  # Last 5 thoughts
        except Exception as e:
            logger.debug(f"Agent thinking error: {e}")
        return []
    
    def get_milestones(self):
        """Get milestone status from /milestones endpoint"""
        try:
            response = requests.get(f"{self.server_url}/milestones", timeout=15)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.debug(f"Milestones error: {e}")
        return {}
    
    def get_recent_actions(self):
        """Get recent button presses from /recent_actions endpoint"""
        try:
            response = requests.get(f"{self.server_url}/recent_actions", timeout=2)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.debug(f"Recent actions error: {e}")
        return {}
    
    def get_action_queue_status(self):
        """Get action queue status from /queue_status endpoint"""
        try:
            response = requests.get(f"{self.server_url}/queue_status", timeout=2)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.debug(f"Queue status error: {e}")
        return {}
    
    def read_recent_log_events(self):
        """Read new entries from system.log for additional context"""
        try:
            if not os.path.exists(self.log_file):
                return []
            
            with open(self.log_file, 'r') as f:
                f.seek(self.last_log_position)
                new_lines = f.readlines()
                self.last_log_position = f.tell()
            
            # Extract important events
            events = []
            for line in new_lines[-50:]:  # Last 50 lines
                if any(keyword in line for keyword in ['Milestone', '🎯', 'Position', '📍', 'Map', '🗺️', 'Party']):
                    events.append(line.strip())
            
            return events[-10:]  # Return last 10 events
            
        except Exception as e:
            logger.debug(f"Log reading error: {e}")
            return []
    
    def do_grid_search(self, map_id, x, y):
        """
        Query server for nearby tiles to build spatial map
        This provides the VLM with better spatial awareness for navigation
        """
        try:
            # Query 5x5 grid around current position
            grid_data = []
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    check_x, check_y = x + dx, y + dy
                    # Store in explored tiles (can be enhanced to query server if available)
                    tile_key = (check_x, check_y)
                    if map_id not in self.explored_tiles:
                        self.explored_tiles[map_id] = {}
                    if tile_key not in self.explored_tiles[map_id]:
                        self.explored_tiles[map_id][tile_key] = "unknown"
                        grid_data.append(f"({check_x},{check_y})")
            
            # Build spatial summary
            explored_count = len(self.explored_tiles.get(map_id, {}))
            map_name = get_map_name(map_id)
            
            summary = f"📊 SPATIAL AWARENESS - {map_name}:\n"
            summary += f"- Current position: ({x},{y})\n"
            summary += f"- Explored tiles on this map: {explored_count}\n"
            summary += f"- Recent scan: {len(grid_data)} tiles in 5x5 grid\n"
            
            # Add directional hints for key locations
            if map_id == 0x09:  # Littleroot Town
                if y > 10:  # Southern part
                    summary += "- You're in SOUTH town - Route 101 is NORTH (low Y coords like 3-6)\n"
                elif y > 6:
                    summary += "- You're in MIDDLE town - Continue NORTH for Route 101\n"
                else:
                    summary += "- You're in NORTH town - Route 101 entrance nearby!\n"
            
            logger.info(f"🗺️ Grid search complete: {explored_count} tiles known on {map_name}")
            return summary
            
        except Exception as e:
            logger.debug(f"Grid search error: {e}")
            return ""
    
    def _get_next_milestones_guide(self, completed):
        """Provide guidance on next milestones to achieve - comprehensive 27-milestone structure"""
        milestones_sequence = [
            # Phase 1: Game Init
            ("GAME_RUNNING", "Game loaded and running"),
            ("PLAYER_NAME_SET", "Set player name"),
            ("INTRO_CUTSCENE_COMPLETE", "Watch truck arrival cutscene"),
            
            # Phase 2: Tutorial Town
            ("LITTLEROOT_TOWN", "Arrive in Littleroot Town"),
            ("PLAYER_HOUSE_ENTERED", "Enter player's house"),
            ("PLAYER_BEDROOM", "Go upstairs to bedroom"),
            ("CLOCK_SET", "Set the clock in bedroom"),
            ("RIVAL_HOUSE", "Enter rival's house (May/Brendan)"),
            ("RIVAL_BEDROOM", "Go upstairs to rival's bedroom"),
            
            # Phase 3: Starter
            ("ROUTE_101", "Exit town north to Route 101 - save Birch"),
            ("STARTER_CHOSEN", "Save Prof Birch, receive starter Pokemon"),
            ("BIRCH_LAB_VISITED", "Return to Birch's Lab"),
            
            # Phase 4: Rival & Pokedex
            ("OLDALE_TOWN", "Travel north to Oldale Town"),
            ("ROUTE_103", "Go north to Route 103, battle rival"),
            ("RECEIVED_POKEDEX", "Return to Birch, receive Pokedex"),
            
            # Phase 5: Petalburg
            ("ROUTE_102", "Cross Route 102 westward"),
            ("PETALBURG_CITY", "Arrive in Petalburg City"),
            ("DAD_FIRST_MEETING", "Meet Dad at Petalburg Gym"),
            ("GYM_EXPLANATION", "Learn about gyms, receive Running Shoes"),
            
            # Phase 6: Road to Rustboro
            ("ROUTE_104_SOUTH", "Leave Petalburg north to Route 104"),
            ("PETALBURG_WOODS", "Enter Petalburg Woods"),
            ("TEAM_AQUA_GRUNT_DEFEATED", "Defeat Team Aqua grunt, save Devon worker"),
            ("ROUTE_104_NORTH", "Exit woods, continue north"),
            ("RUSTBORO_CITY", "Arrive in Rustboro City"),
            
            # Phase 7: First Gym
            ("RUSTBORO_GYM_ENTERED", "Enter Rustboro Gym"),
            ("ROXANNE_DEFEATED", "Defeat Gym Leader Roxanne"),
            ("FIRST_GYM_COMPLETE", "Receive Stone Badge and TM39"),
        ]
        
        next_milestones = []
        for milestone_id, description in milestones_sequence:
            if milestone_id not in completed:
                next_milestones.append(f"➡️ {milestone_id}: {description}")
                if len(next_milestones) >= 3:  # Show next 3 milestones
                    break
        
        return '\n'.join(next_milestones) if next_milestones else "✅ Early game milestones complete!"
    
    def get_comprehensive_guidance(self, game_frame, step_count, current_memory=None, is_stuck=False, recent_positions=None, recent_actions_list=None):
        """
        Provide guidance using SERVER API ENDPOINTS + game frame + STUCK DETECTION
        1. /state - game state with screenshot
        2. /agent - agent thinking log
        3. /milestones - milestone status
        4. /recent_actions - button history
        5. /queue_status - action queue
        6. system.log - additional events
        7. STUCK DETECTION - position/action loops (PokemonRedExperiments-style)
        """
        
        # Get all server data
        agent_thinking = self.get_agent_thinking()
        milestones = self.get_milestones()
        recent_actions = self.get_recent_actions()
        
        # Use current memory if provided (fresher data)
        if current_memory is not None:
            memory = current_memory
            visual = {}
        else:
            server_state = self.get_server_state()
            memory = server_state.get('memory', {})
            visual = server_state.get('visual', {})
        queue_status = self.get_action_queue_status()
        log_events = self.read_recent_log_events()
        
        # Get completed milestones
        completed = []
        if isinstance(milestones, dict):
            # API returns {"milestones": [{name, completed}, ...], "completed": count, ...}
            if 'milestones' in milestones and isinstance(milestones['milestones'], list):
                # Parse the array format
                completed = [m['name'] for m in milestones['milestones'] if m.get('completed', False)]
            else:
                # Fallback to old dict format {name: bool}
                completed = [k for k, v in milestones.items() if v]
        
        map_id = memory.get('map_id', 0)
        location_name = get_map_name(map_id)
        x, y = memory.get('x', 0), memory.get('y', 0)
        game_state = memory.get('game_state', 'overworld')
        
        # GRID SEARCH every N VLM calls for spatial awareness
        self.vlm_call_count += 1
        spatial_info = ""
        if self.vlm_call_count % self.grid_search_interval == 0:
            logger.info(f"🗺️ Performing grid search (VLM call #{self.vlm_call_count})...")
            spatial_info = self.do_grid_search(map_id, x, y)
            self.last_grid_search_map = map_id
            
            # Update spatial map tracker for CNN integration
            if hasattr(self, 'spatial_map_tracker'):
                explored_count = len(self.explored_tiles.get(map_id, {}))
                self.spatial_map_tracker.update(map_id, x, y, explored_count, spatial_info)
        
        # STORY-DRIVEN MILESTONE PROMPT - Ties actions to narrative progression
        party_size = memory.get('party_size', 0)
        has_pokedex = memory.get('has_pokedex', False)
        
        # SAFETY CHECKS: Detect if Pokedex already received
        # 1. If has_pokedex flag is true
        if has_pokedex and 'RECEIVED_POKEDEX' not in completed:
            completed.append('RECEIVED_POKEDEX')
            logger.info("✅ Detected has_pokedex=True, marking RECEIVED_POKEDEX as completed")
        
        # 2. If at lab with Pokemon and been here many steps (assume talked to Birch already)
        at_lab = "BIRCH" in location_name or "LAB" in location_name
        if party_size > 0 and at_lab and 'RECEIVED_POKEDEX' not in completed:
            # Check if we've been stuck at lab - likely already got Pokedex
            if not hasattr(self, 'lab_visit_count'):
                self.lab_visit_count = 0
            self.lab_visit_count += 1
            if self.lab_visit_count > 3:  # Been at lab for 3+ VLM calls with Pokemon
                completed.append('RECEIVED_POKEDEX')
                logger.info("✅ Assumed RECEIVED_POKEDEX (at lab with Pokemon for multiple VLM calls)")
                has_pokedex = True
        else:
            self.lab_visit_count = 0
        
        # Get lead Pokemon level for grind mode check
        lead_level = 5  # Default
        progress_ctx = memory.get('progress_context', {})
        party_data = progress_ctx.get('party', [])
        if party_data and len(party_data) > 0 and party_data[0]:
            lead_level = party_data[0].get('level', 5)
        
        story_ctx = _get_story_context(completed, location_name, party_size, has_pokedex, lead_level)
        # DEBUG: Log lead_level and goal to diagnose GRIND issue
        if self.step_count % 50 == 0:
            logger.info(f"📊 STORY_CTX DEBUG: lead_level={lead_level}, goal={story_ctx.get('goal', 'UNKNOWN')}, party_size={party_size}")
        # Get player gender for position-aware guidance
        player_gender = memory.get('player', {}).get('gender', 'Boy')
        action_guide = _get_milestone_action(location_name, completed, game_state, is_stuck, party_size, has_pokedex, x, y, player_gender)
        
        # Extract quest memory context (replaces hard-coded reminders!)
        npc_context = ""
        quest_context = ""
        
        # Get current quest from DialogueProcessor (hybrid: milestone + memory)
        try:
            import __main__
            if hasattr(__main__, 'dialogue_processor'):
                quest_mem = __main__.dialogue_processor.story_memory
                spatial_hints = __main__.dialogue_processor.get_spatial_hints_for_cnn()
                current_milestone = __main__.dialogue_processor.get_current_milestone_goal()
                
                if quest_mem['active']:
                    # Show BOTH milestone structure AND dynamic sub-quest
                    quest_context = f"""

📍 MILESTONE STRUCTURE (Fixed Goal):
Current Milestone: {current_milestone}
This is your overarching story goal.

🎯 ACTIVE SUB-QUEST (Dynamic from NPC):
Objective: {quest_mem['objective']}
Target Location: {quest_mem['target_location']}
Target NPC: {quest_mem['target_npc']}
Return to: {quest_mem['return_npc']}
Contributes to Milestone: {quest_mem.get('contributes_to_milestone', False)}

🗺️ NAVIGATION HINTS (From Dialogue):
Terrain to find: {quest_mem.get('terrain_hint', 'NONE')}
Direction to go: {quest_mem.get('direction_hint', 'NONE')}
Landmark to look for: {quest_mem.get('landmark_hint', 'NONE')}

💡 How to use this:
- MILESTONE = Your destination (where story goes)
- SUB-QUEST = Your current task (how to get there)
- HINTS = Visual cues (what to look for on screen)
- If terrain_hint = TALL_GRASS → Walk through grass!
- If direction_hint = NORTH → Prefer moving north
"""
        except:
            pass
        
        # NPC identification (less critical now that we have quest memory)
        if "BIRCH" in location_name or "LAB" in location_name:
            npc_context = "\n🧑‍🔬 NPC: Professor BIRCH (the scientist who gives you Pokemon and Pokedex)"
        elif "MAY" in location_name or "BRENDAN" in location_name or "103" in location_name:
            npc_context = "\n👤 NPC: MAY/BRENDAN (your rival - NOT Birch! A young trainer like you)"
        
        # 🆕 MULTI-MODAL PATHFINDING: SLAM + CNN + SNN + VLM integration
        # When stuck, provide real-time ground truth from server + terrain + portals
        pathfinding_context = ""
        if is_stuck and current_memory:
            # Get portal connections from server (ground truth!)
            connections = current_memory.get('connections', [])
            portal_info = ""
            if connections:
                portal_info = "\n\n🌐 AVAILABLE PORTALS (Server Ground Truth):\n"
                for conn in connections[:5]:  # Show up to 5 portals
                    to_loc = conn.get('to', 'unknown')
                    from_pos = conn.get('from_pos', [])
                    if len(from_pos) >= 2:
                        portal_info += f"  • Exit to {to_loc} at ({from_pos[0]},{from_pos[1]})\n"
            
            # Get terrain data from CNN grid search
            map_data = current_memory.get('map_data', {})
            terrain = map_data.get('terrain', [])
            terrain_info = ""
            if terrain:
                # Summarize terrain types nearby
                terrain_types = {}
                for tile in terrain[:30]:  # Check nearby tiles
                    behavior = tile.get('behavior', 'unknown')
                    terrain_types[behavior] = terrain_types.get(behavior, 0) + 1
                if terrain_types:
                    terrain_info = "\n\n🗺️ TERRAIN MAP (CNN Grid Search):\n"
                    for behavior, count in sorted(terrain_types.items(), key=lambda x: -x[1])[:5]:
                        terrain_info += f"  • {behavior}: {count} tiles\n"
            
            # SLAM position history (temporal reasoning)
            slam_info = ""
            if recent_positions and len(recent_positions) > 2:
                slam_info = f"\n\n📍 SLAM TRAJECTORY (Last {len(recent_positions)} positions):\n"
                for i, pos in enumerate(recent_positions[-5:]):
                    slam_info += f"  {i+1}. {pos}\n"
                slam_info += "  ⚠️ Agent stuck in loop! Need new navigation strategy.\n"
            
            # SNN temporal pattern (action sequence)
            snn_info = ""
            if recent_actions_list and len(recent_actions_list) > 2:
                action_counts = {}
                for act in recent_actions_list[-10:]:
                    action_counts[act] = action_counts.get(act, 0) + 1
                snn_info = f"\n\n🧠 SNN ACTION PATTERN (Last {len(recent_actions_list)} actions):\n"
                for act, count in sorted(action_counts.items(), key=lambda x: -x[1])[:5]:
                    snn_info += f"  • {act}: {count}x (repetitive pattern detected)\n"
            
            if portal_info or terrain_info or slam_info or snn_info:
                pathfinding_context = f"""

🔍 MULTI-MODAL PATHFINDING (Agent is STUCK - need fresh strategy!):
{portal_info}{terrain_info}{slam_info}{snn_info}

💡 USE THIS DATA:
- PORTALS = Real exit coordinates (walk to these exact positions!)
- TERRAIN = Obstacle map (avoid walls, use walkable tiles)
- SLAM = Your movement history (avoid repeating same loop)
- SNN = Action patterns (try different direction combo)
- VLM (you!) = High-level reasoning to combine all data

🎯 YOUR TASK: Analyze visual + data above → Find NEW path to goal!
"""
        
        prompt = f"""Pokemon Emerald Speedrun - Story Progress

**📖 STORY CONTEXT:**
{story_ctx['story']}{npc_context}{quest_context}

**🎯 CURRENT MILESTONE GOAL: {story_ctx['goal']}**
Why: {story_ctx['why']}

**📍 YOUR LOCATION:**
{location_name} at ({x},{y}) | State: {(game_state or 'unknown').upper()}
Party: {memory.get('party_size', 0)} Pokemon | Badges: {memory.get('badge_count', 0)}

**✅ PROGRESS:** {len(completed)} milestones completed
{', '.join(completed[-3:]) if completed else 'Just starting!'}

{spatial_info if spatial_info else ''}{pathfinding_context}

**➡️ TO PROGRESS STORY:**
{action_guide}

**🚨 CRITICAL RULES - MILESTONE AWARE:**
{"" if 'RIVAL_BEDROOM' not in completed else '''⚠️ RIVAL (May) IS ONLY ON ROUTE 103! NEVER IN ANY HOUSE!
⛔ You CANNOT find rival at any house!
⛔ You CANNOT battle rival at any house!
✅ RIVAL_BEDROOM done - EXIT any house!
✅ If in ANY house: ACTION = DOWN (exit immediately!)
✅ Birch = PROFESSOR at lab, NOT your rival!'''}{"" if 'RIVAL_BEDROOM' in completed else f'''🎯 RIVAL_BEDROOM NOT DONE - GO TO RIVAL'S HOUSE!
✅ For {player_gender}: {"BRENDAN's house is RIVAL's" if player_gender == 'Girl' else "MAY's house is RIVAL's"}
✅ Go to rival's house and upstairs to bedroom!
✅ If in YOUR house (wrong): EXIT with DOWN, then go to rival's!
✅ If in RIVAL's house 1F: Go UPSTAIRS to bedroom!
✅ If in RIVAL's 2F/bedroom: Walk around to trigger milestone!'''}

**RESPOND (tie your action to story/milestone):**
LOCATION: {location_name} - [visual description]
OBJECTIVE: [what milestone step this accomplishes]
ACTION: [UP/DOWN/LEFT/RIGHT/A/B]
REASON: [how this advances the story toward {story_ctx['goal']}]

**RULES:**
- Your OBJECTIVE must connect to milestone goal: {story_ctx['goal']}
- Your REASON must explain story progression
- Use directional buttons for movement, A/B only for dialogs
- If in house: ALWAYS say ACTION: DOWN (exit!)
"""
        
        try:
            # Get screenshot from server state if available
            images = []
            
            # Add game frame
            if game_frame is not None:
                images.append(Image.fromarray(game_frame))
            
            # Try to get screenshot from server
            if 'screenshot_base64' in visual:
                try:
                    img_data = base64.b64decode(visual['screenshot_base64'])
                    server_img = Image.open(BytesIO(img_data))
                    images.append(server_img)
                except:
                    pass
            
            # If VLM visual disabled, return text-only analysis
            if not self.use_visual or self.vlm is None:
                # Return rule-based guidance from server data
                response = self._get_rule_based_guidance(memory, completed, is_stuck, recent_positions, recent_actions_list)
                logger.info(f"\n📋 SERVER-AWARE ANALYSIS (Step {step_count}):\n{response}\n")
                self.last_analysis = response
                
                # Log to llm_logs for server display
                self._log_to_server(response, memory, step_count)
                
                return response
            
            if not images:
                logger.warning(f"⚠️ No images available for VLM analysis at step {step_count}")
                return None
            
            # Use VLM visual analysis
            logger.info(f"🎬 Calling VLM.get_query with {len(images)} image(s)...")
            response = self.vlm.get_query(
                img=images[0],
                text=prompt,
                module_name="ServerAwareOracle"
            )
            
            logger.info(f"🎬 VLM.get_query returned: {len(response) if response else 0} chars")
            logger.info(f"\n🧠 VLM SERVER-AWARE ANALYSIS (Step {step_count}):\n{response}\n")
            self.last_analysis = response
            
            # Log to llm_logs for server display
            self._log_to_server(response, memory, step_count)
            
            return response
            
        except Exception as e:
            import traceback
            logger.error(f"❌ VLM analysis error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _log_to_server(self, response, memory, step_count):
        """Log VLM thinking to llm_logs for server display - uses proper LLM logger format"""
        try:
            from utils.llm_logger import log_llm_interaction, get_llm_logger
            import time
            
            # Track call duration
            if not hasattr(self, 'last_call_time'):
                self.last_call_time = time.time()
            call_duration = time.time() - self.last_call_time
            self.last_call_time = time.time()
            
            # Build detailed prompt showing what VLM analyzed
            map_id = memory.get('map_id', 0)
            location_name = get_map_name(map_id)
            location_info = f"{location_name} at ({memory.get('x', 0)}, {memory.get('y', 0)})"
            
            # Get readable location name from server or memory
            server_location = memory.get('location', 'unknown')
            if server_location and server_location != 'unknown':
                location_display = f"{server_location} - {location_info}"
            else:
                location_display = location_info
            
            prompt_summary = f"[Step {step_count}] VLM analyzing: {location_display}"
            
            # DEBUG: Check if response is actually present
            logger.info(f"🔍 _log_to_server called: step={step_count}, response_len={len(response) if response else 0}")
            
            # Get logger instance and check log file
            llm_logger = get_llm_logger()
            # logger.info(f"📂 LLM Logger file: {llm_logger.log_file}")  # Removed - too verbose
            
            # REMOVED DUPLICATE: VLM already logs via utils/vlm.py as LOCAL_SERVERAWAREORACLE
            # This was causing double logging and incorrect duration (28s instead of 0.68s)
            # The actual VLM call in vlm.py already logs the interaction properly
            
            # Just log locally for training script debugging
            logger.debug(f"VLM response logged (step {step_count})")
                
        except Exception as e:
            import traceback
            logger.error(f"❌ Failed to log to server: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        # VALIDATE VLM RESPONSE - check if location matches server location
        if response:
            response_upper = (response or '').upper()
            location_name_upper = (location_name or '').upper()
            
            # Extract LOCATION line from response
            location_line = ""
            for line in response.split('\n'):
                if line.strip().startswith('LOCATION:'):
                    location_line = line.strip()
                    break
            
            # Check if VLM's location matches server's location
            if location_line and location_name_upper not in location_line.upper():
                logger.warning(f"⚠️ VLM HALLUCINATION DETECTED!")
                logger.warning(f"   Server says: {location_name}")
                logger.warning(f"   VLM said: {location_line}")
                logger.warning(f"   → Forcing correction...")
                
                # FORCE correct location in response
                if 'LOCATION:' in response:
                    lines = response.split('\n')
                    for i, line in enumerate(lines):
                        if line.strip().startswith('LOCATION:'):
                            # Replace with correct location
                            rest_of_line = line.split('-', 1)[-1] if '-' in line else ""
                            lines[i] = f"LOCATION: {location_name} - [Server-corrected]{rest_of_line}"
                            break
                    response = '\n'.join(lines)
                    logger.info(f"✅ Corrected VLM response to match server location")
    
    def _get_rule_based_guidance(self, memory, completed, is_stuck, recent_positions, recent_actions_list):
        """Rule-based guidance with dialog/naming screen handling"""
        map_names = {
            0x0000: "NAMING_SCREEN",  # Dialog/Naming typically map 0
            0x0101: "LITTLEROOT_TOWN",
            0x0201: "PLAYER_HOUSE_1F", 
            0x0309: "PLAYER_HOUSE_2F",
            0x0202: "RIVAL_HOUSE_1F",
            0x030A: "RIVAL_HOUSE_2F",
            0x020C: "BIRCH_LAB",
            0x0010: "ROUTE_101",
            0x0102: "OLDALE_TOWN",
            0x0012: "ROUTE_103",
        }
        map_id = memory.get('map_id', 0)
        location = map_names.get(map_id, f"MAP_{map_id:04X}")
        x, y = memory.get('x', 0), memory.get('y', 0)
        
        # NAMING SCREEN / DIALOG DETECTION
        if map_id == 0x0000 or (x == 0 and y == 0 and map_id < 0x0100):
            return f"🎮 DIALOG/NAMING SCREEN DETECTED\nOBJECTIVE: Skip through naming and intro dialogs\nACTION: A (confirm/advance text)\nREASON: Press A repeatedly to skip naming screen, set text speed, clock, etc."
        
        # Rule-based decision tree with event flag priority
        
        # CRITICAL: After STARTER_CHOSEN, rival (May) is on ROUTE 103, NOT in bedroom!
        starter_chosen = 'STARTER_CHOSEN' in completed
        has_pokedex = 'RECEIVED_POKEDEX' in completed
        # FIX: Don't trust checkpoint FIRST_RIVAL_BATTLE - may be stale from previous run!
        # Only trust Pokedex (received at lab AFTER rival battle)
        rival_battled = has_pokedex
        
        # =====================================================
        # CRITICAL RULE: After getting starter Pokemon:
        # RIVAL IS NEVER AT ANY HOUSE! RIVAL IS ONLY ON ROUTE 103!
        # You CANNOT find or battle rival at any house!
        # 
        # CASE 1 - MALE AVATAR (Brendan is player):
        #   Brendan's house (0x100=1F, 0x101=2F) = PLAYER's house (Mom heals here)
        #   May's house (0x0201=1F, 0x0309=2F) = RIVAL's house (but MAY is on Route 103!)
        # 
        # CASE 2 - FEMALE AVATAR (May is player):
        #   May's house (0x0201=1F, 0x0309=2F) = PLAYER's house (Mom heals here)
        #   Brendan's house (0x100=1F, 0x101=2F) = RIVAL's house (but BRENDAN is on Route 103!)
        # 
        # CURRENT: Male avatar, so Brendan's house = Player's house
        # =====================================================
        
        # CASE 1: MALE AVATAR - Brendan's house = PLAYER's house
        # Player's House 1F (Brendan's house) - EXIT or heal with Mom
        if map_id in [0x100, 0x0100]:  # Brendan's House 1F = Player's house
            return f"📍 PLAYER'S HOUSE 1F at ({x},{y})\n🎯 OBJECTIVE: EXIT and go to Route 103\n📍 ACTION: DOWN (exit door at bottom)\n📌 PATH: Exit → Route 101 → Oldale → Route 103\nNOTE: Mom can heal you. If HP low, talk to Mom then EXIT!\n⚠️ RIVAL (MAY) IS NOT HERE - SHE IS ON ROUTE 103 ONLY!"
        
        # Player's Bedroom 2F (Brendan's house) - ALWAYS EXIT
        elif map_id in [0x101, 0x0101]:  # Brendan's House 2F = Player's bedroom
            return f"⚠️ EXIT PLAYER'S BEDROOM at ({x},{y})\n🎯 OBJECTIVE: Go DOWNSTAIRS and EXIT house\n📍 ACTION: DOWN (stairs at y=1)\n📌 PATH: Bedroom → Stairs → 1F → Door → Route 101 → Route 103\n⚠️ RIVAL IS NOT IN ANY HOUSE! GO TO ROUTE 103!"
        
        # CASE 1: MALE AVATAR - May's house = RIVAL's house (but rival is NOT here!)
        # Rival's House (May's/Prof Birch's house)
        elif map_id in [0x0201, 0x0202, 0x0309, 0x030A]:
            rival_bedroom_visited = 'RIVAL_BEDROOM' in completed
            
            # If haven't visited bedroom yet, must go upstairs first!
            if not rival_bedroom_visited:
                if map_id in [0x030A]:  # Already in 2F bedroom
                    return f"✅ RIVAL BEDROOM at ({x},{y})\n🎯 OBJECTIVE: You're in rival's bedroom! Now EXIT and go to Route 103\n📍 ACTION: DOWN (stairs/exit)\n📌 PATH: Downstairs → Exit house → Route 101 → Oldale → Route 103\n⚠️ RIVAL (MAY) IS ON ROUTE 103, NOT HERE!"
                else:  # In 1F, need to go upstairs
                    return f"🏠 RIVAL HOUSE 1F at ({x},{y})\n🎯 OBJECTIVE: Go UPSTAIRS to rival's bedroom (visit it, triggers story)\n📍 ACTION: UP (find stairs, usually in center or back)\n📌 PATH: Upstairs → Bedroom → Exit → Route 101 → Route 103\nNOTE: Must visit bedroom before game allows Route 103 access!"
            else:
                # Bedroom already visited - exit house now
                return f"⚠️ EXIT THIS HOUSE at ({x},{y})\n❌ RIVAL (MAY) IS NOT HERE!\n❌ YOU CANNOT BATTLE RIVAL AT ANY HOUSE!\n❌ RIVAL IS ONLY ON ROUTE 103!\n🎯 OBJECTIVE: EXIT IMMEDIATELY\n📍 ACTION: DOWN (exit)\n📌 PATH: Exit → Route 101 → Oldale → Route 103 → FIND MAY!"
        
        # BIRCH'S LAB - Receive Pokedex after rival battle, then always EXIT
        elif map_id == 0x020C:
            if rival_battled and not has_pokedex:
                # Battled rival, need Pokedex from Birch
                return f"📍 Birch's Lab at ({x},{y})\n🎯 OBJECTIVE: Talk to Birch to RECEIVE POKEDEX\n📍 ACTION: Press A to talk to Birch, then EXIT (DOWN)\nNOTE: You beat your rival! Birch will give you the Pokedex now."
            elif has_pokedex:
                # Already have Pokedex, EXIT
                return f"📍 Birch's Lab at ({x},{y})\n🎯 OBJECTIVE: EXIT lab (you have Pokedex)\n📍 ACTION: DOWN (exit at bottom)\nNOTE: You have the Pokedex! Continue your journey."
            else:
                # Haven't battled rival yet, EXIT and go to Route 103
                return f"📍 Birch's Lab at ({x},{y})\n🎯 OBJECTIVE: EXIT lab and go to Route 103 to battle rival\n📍 ACTION: DOWN (exit at bottom)\nNOTE: Birch is PROFESSOR. Rival is on Route 103. Battle rival first, then return for Pokedex."
        
        elif map_id == 0x0101 and starter_chosen and not rival_battled:
            # In Littleroot Town with starter - GO NORTH to Route 101!
            return f"📍 LOCATION: Littleroot Town at ({x},{y})\n🎯 QUEST: Find rival on Route 103 to battle\n🗺️ PATH: Exit NORTH to Route 101 → Oldale Town → Route 103\nACTION: UP (repeatedly move north)\nREASON: Your rival (May for male, Brendan for female) is waiting on Route 103. Must navigate: Route 101 → Oldale → Route 103\nNOTE: Birch is PROFESSOR (stays at lab). May/Brendan is RIVAL (on Route 103)."
        
        elif map_id == 0x0101:
            # In town, early game
            return f"LOCATION: {location} at ({x},{y})\nOBJECTIVE: Explore town or enter houses\nACTION: UP/RIGHT\nREASON: Explore to trigger story events"
        
        elif map_id == 0x0010 and not starter_chosen:
            # In Route 101 BEFORE starter - trigger Birch rescue
            return f"LOCATION: Route 101 at ({x},{y})\nOBJECTIVE: Walk into grass to trigger Birch rescue cutscene\nACTION: UP (into tall grass)\nREASON: Walk around to find Birch being attacked"
        
        elif map_id == 0x0010 and starter_chosen and not rival_battled:
            # In Route 101 AFTER starter - continue north to Oldale
            return f"📍 LOCATION: Route 101 at ({x},{y})\n🎯 QUEST: Reach Oldale Town then Route 103\n🌿 TERRAIN: Walk through grass (it's walkable!)\nACTION: UP (move north through grass)\nREASON: Continue north to Oldale Town. Don't avoid grass - walk through it!"
        
        elif map_id == 0x11 and starter_chosen and not rival_battled:  # Fixed: Oldale Town is 0x11, not 0x0102
            # In Oldale Town - Visit PokeCenter FIRST to set respawn point!
            if not hasattr(self, 'visited_oldale_pokecenter') or not self.visited_oldale_pokecenter:
                return f"📍 LOCATION: Oldale Town at ({x},{y})\n🏥 IMPORTANT: Visit PokeCenter at (6,17) to set respawn!\n⚠️ Without this, blackouts send you to Littleroot!\nACTION: Go LEFT to PokeCenter entrance\nREASON: Visiting PokeCenter saves your respawn point here!"
            else:
                return f"📍 LOCATION: Oldale Town at ({x},{y})\n✅ Respawn set at PokeCenter!\n🎯 QUEST: May is waiting on Route 103 (north exit)\n🗺️ DIRECTION: Exit NORTH to Route 103\nACTION: UP (move to north exit)\nREASON: May (rival) is on Route 103. Battle her!"
        
        elif map_id == 0x0012 and not rival_battled:
            # On Route 103 - find May!
            return f"📍 LOCATION: Route 103 at ({x},{y})\n🎯 QUEST: Find and battle May (rival)\n👤 LOOK FOR: Trainer sprite (May/Brendan)\nACTION: Explore (UP/DOWN/LEFT/RIGHT) then A to battle\nREASON: May is somewhere on this route. Walk around to find her, press A to initiate battle"
        
        elif map_id == 0x020C and rival_battled and not has_pokedex:
            # In Birch Lab after rival battle - get Pokedex!
            return f"📍 LOCATION: Birch's Lab at ({x},{y})\n🎯 QUEST: Receive Pokedex from Professor Birch\n✅ READY: Battled rival, can now get Pokedex\nACTION: Move toward Birch, press A\nREASON: After battling rival, Birch gives you the Pokedex"
        
        elif (map_id == 0x11 or map_id == 0x0012) and rival_battled and not has_pokedex:  # Fixed: Oldale Town is 0x11
            # Battled rival, need to return to Lab
            return f"📍 LOCATION: {location} at ({x},{y})\n✅ RIVAL BATTLE COMPLETE!\n🎯 QUEST: Return to Birch's Lab for Pokedex\n🗺️ PATH: Go south to Littleroot Town\nACTION: DOWN (move south)\nREASON: Birch is waiting at his lab to give you the Pokedex"
        
        else:
            # Default exploration
            if is_stuck and recent_actions_list:
                # Try opposite of what we've been doing
                last_action = recent_actions_list[-1]
                opposite = {'UP': 'DOWN', 'DOWN': 'UP', 'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}.get(last_action, 'A')
                return f"LOCATION: {location} at ({x},{y})\nOBJECTIVE: Get unstuck\nACTION: {opposite}\nREASON: Agent stuck, trying opposite direction"
            
            return f"LOCATION: {location} at ({x},{y})\nOBJECTIVE: Explore area\nACTION: UP\nREASON: General exploration"
    
    def parse_action_from_response(self, response):
        """Extract recommended action from VLM response"""
        if not response:
            return None
        
        # Look for "ACTION: X" pattern
        match = re.search(r'ACTION:\s*([A-Z]+)', (response or '').upper())
        if match:
            action = match.group(1)
            action_map = {
                'UP': 'UP', 'DOWN': 'DOWN', 'LEFT': 'LEFT', 'RIGHT': 'RIGHT',
                'A': 'A', 'B': 'B', 'NOOP': 'NOOP', 'NONE': 'NOOP'
            }
            return action_map.get(action, None)
        return None


def main():
    parser = argparse.ArgumentParser(description="VLM Server-Aware RL Training")
    parser.add_argument('--timesteps', type=int, default=1000000,
                       help='Training timesteps (default: 1M)')
    parser.add_argument('--rom', type=str, default='emerald.gba',
                       help='Path to ROM')
    parser.add_argument('--vlm-frequency', type=int, default=30,
                       help='Call VLM every N steps')
    parser.add_argument('--vlm-backend', type=str, default='qwen2b',
                       help='VLM backend (qwen2b/local/gpt/gemini)')
    parser.add_argument('--disable-vlm-visual', action='store_true',
                       help='Disable VLM visual analysis (use server data only) to save VRAM')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint (e.g., checkpoints/server_aware_model_10000_steps.zip)')
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     🧠 VLM SERVER-AWARE RL Training                         ║
║                                                              ║
║     Uses SERVER API ENDPOINTS for guidance:                 ║
║     • /agent - Agent thinking log                           ║
║     • /state - Game state + screenshot                      ║
║     • /milestones - Milestone completion status             ║
║     • /recent_actions - Button press history                ║
║     • /queue_status - Action queue state                    ║
║     • system.log - Additional events                        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize VLM (optional - can disable to save VRAM)
    if args.disable_vlm_visual:
        logger.info("🧠 VLM visual analysis DISABLED (using server data only)")
        vlm = None
    else:
        logger.info(f"🧠 Loading VLM ({args.vlm_backend})...")
        vlm = create_vlm(args.vlm_backend)
        logger.info("✅ VLM loaded")
    
    # Create server API oracle
    oracle = ServerAPIOracle(vlm, use_visual=(not args.disable_vlm_visual))
    logger.info("✅ Server API oracle initialized" + (" (server data only)" if args.disable_vlm_visual else " (with VLM)"))
    
    # Check server connection (like train_rl_event_flags.py)
    server_url = "http://localhost:8000"
    try:
        response = requests.get(f"{server_url}/status", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Server running at http://localhost:8000")
        else:
            raise Exception(f"Server returned status {response.status_code}")
    except Exception as e:
        logger.error(f"❌ Server not accessible: {e}")
        logger.error("   Start server first: python run.py")
        return 1
    
    logger.info("📋 Configuration:")
    logger.info(f"   Timesteps: {args.timesteps:,}")
    logger.info(f"   VLM Frequency: Every {args.vlm_frequency} steps")
    logger.info(f"   Server API: ENABLED 🌐")
    logger.info(f"   Endpoints: /agent, /state, /milestones, /recent_actions, /queue_status")
    logger.info(f"   Log Monitoring: ENABLED 📋")
    logger.info(f"   SNN Features: DISABLED (using server)")
    logger.info("")
    logger.info("🚀 Starting training with server-aware VLM guidance...")
    logger.info("   VLM will analyze server API data + visual frames")
    logger.info("   to guide the agent using REAL game state!")
    logger.info("   Watch: http://localhost:8000/stream")
    logger.info("")
    
    # Create custom environment that connects to SERVER (like train_rl_event_flags.py)
    import gymnasium as gym
    from gymnasium import spaces
    import numpy as np
    
    # Import at module level for feature extractor
    if 'gym' not in dir():
        import gymnasium as gym
    
    class ServerAwareEnv(gym.Env):
        """Environment that connects to running server and uses Server API Oracle"""
        def __init__(self, server_url: str, frame_server_url: str, oracle: ServerAPIOracle, vlm_frequency: int):
            super().__init__()
            self.server_url = server_url
            self.frame_server_url = frame_server_url
            self.oracle = oracle
            self.vlm_frequency = vlm_frequency
            
            # Observation: RGB frames (84x84x3) - matches train_rl_event_flags.py
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(84, 84, 3), dtype=np.uint8
            )
            
            # Actions: 7 buttons including NOOP
            self.button_map = ["A", "B", "UP", "DOWN", "LEFT", "RIGHT", "NOOP"]
            self.action_space = spaces.Discrete(len(self.button_map))
            
            # Frame stacking (internal)
            self.frame_stack_size = 4
            self.frame_stack = []
            
            # Tracking
            self.step_count = 0
            self.seen_coords = {}
            self.last_map_id = None
            self.maps_discovered = set()
            
            # STUCK DETECTION (PokemonRedExperiments-style)
            self.position_history = []  # Last N positions
            self.action_history = []    # Last N actions
            self.snn_spike_variance_history = []  # SNN temporal reasoning
            self.position_history_size = 20
            self.last_vlm_suggestion = None
            self.vlm_suggestion_unchanged_steps = 0
            
            # STRATEGIC DIRECTIVES (Railroading)
            self.directive_manager = StrategicDirectiveManager()
            self.last_directive = None
            
            # TERRAIN NAVIGATOR (Cached A* pathfinding)
            self.terrain_nav = TerrainNavigator()
            self.current_map_id = None
        
        def get_exit_for_objective(self, connections: list, objective: str, map_id: int = None) -> tuple:
            """
            Find the best exit coordinates from dynamic server connections.
            
            Args:
                connections: List of {'to': location, 'from_pos': [x,y], 'to_pos': [x,y]}
                objective: What we're trying to reach ('route103', 'pokecenter', 'oldale', etc)
                map_id: Current map ID for location-specific overrides
            
            Returns:
                (x, y) tuple of exit coordinates, or None if not found
            """
            # --- PORTAL CONNECTION OVERRIDES ---
            # For specific maps where server connection data is incorrect
            if map_id == 0x202:  # Oldale Mart
                if objective.upper() == 'OLDALE':
                    # Server data gives entry point (14,7), but actual door is at (3,7)/(4,7)
                    return (3, 7)
            
            # Birch's Lab - exit is at (6, 12) (door at bottom center)
            if map_id in [0x020C, 0x104]:  # BIRCH_LAB map IDs
                if objective.upper() in ['LITTLEROOT', 'ROUTE101', 'EXIT']:
                    return (6, 12)
            
            # May's House 1F (0x102) - exit door at (1, 8), server wrongly says (14,9)
            if map_id == 0x102:
                if objective.upper() in ['LITTLEROOT', 'EXIT', 'TOWN']:
                    return (1, 8)  # Actual door position from user verification
            
            # May's House 2F (0x103) - stairs to 1F at (2, 2)
            if map_id == 0x103:
                if objective.upper() in ['1F', 'DOWNSTAIRS', 'EXIT', 'STAIRS']:
                    return (2, 2)  # Stairs to 1F
            
            # Brendan's House 1F (0x100) - exit door (gender-aware, same layout)
            if map_id == 0x100:
                if objective.upper() in ['LITTLEROOT', 'EXIT', 'TOWN']:
                    return (1, 8)  # Same layout as May's house
            
            # Brendan's House 2F (0x101) - stairs to 1F
            if map_id == 0x101:
                if objective.upper() in ['1F', 'DOWNSTAIRS', 'EXIT', 'STAIRS']:
                    return (2, 2)  # Stairs to 1F
            
            if not connections:
                return None
            
            objective_upper = objective.upper()
            
            # Priority mapping for objectives
            for conn in connections:
                to_location = conn.get('to', '').upper()
                from_pos = conn.get('from_pos', [])
                
                if len(from_pos) >= 2:
                    # Check if this connection matches our objective
                    if objective_upper in to_location or to_location in objective_upper:
                        return (int(from_pos[0]), int(from_pos[1]))
                    
                    # Special cases
                    if objective_upper == 'ROUTE103' and 'ROUTE 103' in to_location:
                        return (int(from_pos[0]), int(from_pos[1]))
                    if objective_upper == 'POKECENTER' and ('POKEMON CENTER' in to_location or 'POKECENTER' in to_location):
                        return (int(from_pos[0]), int(from_pos[1]))
                    if objective_upper == 'OLDALE' and 'OLDALE' in to_location:
                        return (int(from_pos[0]), int(from_pos[1]))
                    if objective_upper == 'ROUTE101' and 'ROUTE 101' in to_location:
                        return (int(from_pos[0]), int(from_pos[1]))
            
            # Fallback: return first valid connection if no match
            # EXCEPT for specific building objectives like rival_house/lab which need exact coords!
            if objective_upper not in ['RIVAL_HOUSE', 'LAB', 'PLAYER_HOUSE']:
                for conn in connections:
                    from_pos = conn.get('from_pos', [])
                    if len(from_pos) >= 2:
                        return (int(from_pos[0]), int(from_pos[1]))
            
            return None
        
        # ═══════════════════════════════════════════════════════════════════════════
        # NAVIGATION COORDINATE STRATEGY
        # ═══════════════════════════════════════════════════════════════════════════
        # 
        # When determining navigation coordinates, use different sources based on context:
        #
        # 1. INDOOR → EXIT (House, Lab, PokeCenter, Mart, etc.)
        #    ✅ USE: Visual map DD labels (real-time ground truth)
        #    ❌ AVOID: Portal connections (often have wrong coordinates)
        #    Reason: Portal data for interior exits is frequently incorrect.
        #            The visual map's DD (door) markers show exact tile positions.
        #
        # 2. OUTDOOR → ENTER BUILDING (Littleroot → Houses, Lab, etc.)
        #    ✅ USE: Visual map D labels OR Portal connections  
        #    Both are usually accurate for building entrances from outside.
        #    Portal connections work well here because they track actual warp tiles.
        #
        # 3. OUTDOOR → ANOTHER ROUTE/CITY (Route 101 → Oldale, etc.)
        #    ✅ USE: Portal connections
        #    ❌ AVOID: Visual map labels (routes are too large for label parsing)
        #    Reason: Route transitions are map-boundary warps, portals are accurate.
        #
        # Implementation:
        #   - For indoor maps: Call parse_visual_map_for_door() to find DD labels
        #   - For outdoor building entry: Use portal connections (they're accurate)
        #   - For route transitions: Use portal connections (they're accurate)
        #   - Fallback: Hardcoded overrides in get_exit_for_objective() for broken portals
        #
        # ═══════════════════════════════════════════════════════════════════════════
        
        def parse_visual_map_for_door(self, visual_map: str, door_type: str = 'DD') -> Optional[Tuple[int, int]]:
            """
            Parse the visual map string to find DD (double door) or D (door) label coordinates.
            
            The visual map from the server has format like:
                "  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15"
                "0  # # # # # # # # # # # # # # # #"
                "1  # . . . . . . D D . . . . . . #"
                ...
            
            Args:
                visual_map: The visual_map string from /state endpoint
                door_type: 'DD' for double door (exits), 'D' for single door, 'S' for stairs
            
            Returns:
                (x, y) tuple of the door/stairs position, or None if not found
                
            Usage:
                - Indoor exits: Use 'DD' to find the exit door
                - Stairs: Use 'S' to find stair tiles
                - Building entries (outdoor): Use 'D' labels
            """
            if not visual_map:
                return None
            
            try:
                lines = visual_map.split('\n')
                
                # Find the door marker
                for line_idx, line in enumerate(lines):
                    # Skip header/metadata lines
                    if 'MAP:' in line or 'Player' in line or 'Movement' in line or 'Portal' in line:
                        continue
                    
                    # Look for DD (double door) or D (single door) markers
                    if door_type == 'DD':
                        # DD is two D characters next to each other
                        dd_pos = line.find('D D')  # With spaces between grid cells
                        if dd_pos == -1:
                            dd_pos = line.find('DD')  # Without spaces
                        
                        if dd_pos != -1:
                            # Parse the line to find grid position
                            # Lines have format like: "12  # . D D . #" where 12 is y-coord
                            parts = line.strip().split()
                            if len(parts) > 0 and parts[0].isdigit():
                                y_coord = int(parts[0])
                                # Find x position by counting cells
                                cell_content = ' '.join(parts[1:])
                                cells = parts[1:]
                                for x, cell in enumerate(cells):
                                    if cell == 'D':
                                        # Check if next cell is also D (DD pattern)
                                        if x + 1 < len(cells) and cells[x + 1] == 'D':
                                            # Return the first D of DD (left door tile)
                                            return (x, y_coord)
                                        # If looking for single D, return first D found
                                        if door_type == 'D':
                                            return (x, y_coord)
                    
                    elif door_type == 'D':
                        # Single D marker
                        parts = line.strip().split()
                        if len(parts) > 0 and parts[0].isdigit():
                            y_coord = int(parts[0])
                            cells = parts[1:]
                            for x, cell in enumerate(cells):
                                if cell == 'D':
                                    return (x, y_coord)
                    
                    elif door_type == 'S':
                        # Stairs marker
                        parts = line.strip().split()
                        if len(parts) > 0 and parts[0].isdigit():
                            y_coord = int(parts[0])
                            cells = parts[1:]
                            for x, cell in enumerate(cells):
                                if cell == 'S':
                                    return (x, y_coord)
            
            except Exception as e:
                logger.warning(f"Error parsing visual map for {door_type}: {e}")
            
            return None
        
        def get_indoor_exit_from_visual_map(self, memory: dict) -> Optional[Tuple[int, int]]:
            """
            Get indoor exit coordinates from visual map DD labels.
            
            This is the preferred method for finding exits from indoor locations
            because portal connection data is often incorrect for interior exits.
            
            Args:
                memory: The memory dict from server /state endpoint
                
            Returns:
                (x, y) tuple of exit door position, or None if not found
            """
            visual_map = memory.get('map', {}).get('visual_map', '')
            if not visual_map:
                return None
            
            # Try DD first (double door = main exit)
            exit_pos = self.parse_visual_map_for_door(visual_map, 'DD')
            if exit_pos:
                return exit_pos
            
            # Fallback to single D
            return self.parse_visual_map_for_door(visual_map, 'D')
        
        def _preprocess_frame(self, frame_b64: str) -> np.ndarray:
            try:
                img_bytes = base64.b64decode(frame_b64)
                img = Image.open(BytesIO(img_bytes))
                img = img.resize((84, 84))
                img = img.convert('RGB')
                return np.array(img, dtype=np.uint8)
            except Exception as e:
                logger.error(f"Frame error: {e}")
                return np.zeros((84, 84, 3), dtype=np.uint8)
        
        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.step_count = 0
            self.frame_stack = []
            
            try:
                # Get frame from frame server (port 8001)
                frame_response = requests.get(f"{self.frame_server_url}/frame", timeout=5)
                if frame_response.status_code == 200:
                    import base64
                    from PIL import Image
                    from io import BytesIO
                    img_data = base64.b64decode(frame_response.json()['frame'])
                    pil_img = Image.open(BytesIO(img_data))
                    frame = np.array(pil_img.resize((84, 84)))
                    self.frame_stack = [frame] * self.frame_stack_size
                    return frame, {}
            except Exception as e:
                logger.warning(f"Failed to get frame from frame server: {e}")
            
            return np.zeros((84, 84, 3), dtype=np.uint8), {}
        
        def step(self, action: int):
            self.step_count += 1
            action_name = self.button_map[action]
            skip_stuck_override = False  # Flag to skip stuck detection when high-priority override active
            
            # ═══════════════════════════════════════════════════════════════════
            # 🚨 STUCK ACTION OVERRIDE - Force terrain action when stuck 50+ steps
            # Fetches position from server directly for fresh data
            # ═══════════════════════════════════════════════════════════════════
            try:
                # Get fresh position from server
                import requests
                state_resp = requests.get("http://localhost:8000/state", timeout=1).json()
                player = state_resp.get('player', {})
                position = player.get('position', {})
                current_x = position.get('x')
                current_y = position.get('y')
                
                if current_x is not None and current_y is not None:
                    current_pos = (current_x, current_y)
                    
                    if not hasattr(self, '_override_last_pos'):
                        self._override_last_pos = current_pos
                        self._override_stuck_count = 0
                    
                    if current_pos == self._override_last_pos:
                        self._override_stuck_count += 1
                    else:
                        self._override_stuck_count = 0
                        self._override_last_pos = current_pos
                    
                    # Debug: Log counter value every 10 steps
                    if self.step_count % 10 == 0:
                        logger.info(f"🔧 OVERRIDE DEBUG: count={self._override_stuck_count} pos={current_pos}")
                    
                    # Check if we're in persistent forced mode (keep forcing for 30 steps to complete waypoint path)
                    if getattr(self, '_forced_steps_remaining', 0) > 0:
                        # GET FRESH DIRECTION for CURRENT position
                        forced_action = None
                        map_id = state_resp.get('map', {}).get('id', 0)
                        metatile_info = getattr(self, 'cached_metatile', None)
                        
                        # Use terrain_nav for direction on ANY map (not just Oldale!)
                        if hasattr(self, 'terrain_nav') and metatile_info:
                            # Get goal based on context - use cached target or default cycling
                            goal = getattr(self, '_persist_goal', None)
                            if goal:
                                nav_result = self.terrain_nav.get_direction(
                                    map_id=map_id, x=current_x, y=current_y,
                                    goal=goal, metatile_info=metatile_info
                                )
                                if nav_result and nav_result.get('action') in ['LEFT', 'RIGHT', 'UP', 'DOWN']:
                                    forced_action = nav_result['action']
                        
                        if not forced_action:
                            # Fallback: CYCLE through directions (not same action!)
                            directions = ['LEFT', 'DOWN', 'RIGHT', 'UP']
                            cycle_step = 30 - self._forced_steps_remaining  # 0 to 29
                            cycle_index = (cycle_step // 5) % 4  # Change direction every 5 steps
                            forced_action = directions[cycle_index]
                        
                        action_name = forced_action
                        direction_to_action = {'UP': 2, 'DOWN': 3, 'LEFT': 4, 'RIGHT': 5}
                        action = direction_to_action.get(forced_action, action)
                        skip_stuck_override = True
                        self._forced_stuck_action = forced_action
                        self._forced_stuck_action_int = action
                        self._forced_steps_remaining -= 1
                        
                        # When persistent ends, set cooldown to prevent immediate re-trigger
                        if self._forced_steps_remaining == 0:
                            self._persist_cooldown = 30  # Don't re-trigger for 30 steps
                        
                        if self.step_count % 5 == 0:
                            logger.warning(f"🔄 PERSISTENT NAV: {forced_action} at ({current_x},{current_y}) ({self._forced_steps_remaining} steps remaining)")
                    else:
                        # Decrement cooldown if active
                        if getattr(self, '_persist_cooldown', 0) > 0:
                            self._persist_cooldown -= 1
                        
                        # Only trigger new STUCK override if: stuck >= 100 AND no cooldown active (SAFETY NET ONLY!)
                        if self._override_stuck_count >= 100 and getattr(self, '_persist_cooldown', 0) == 0:
                            # CYCLE through directions - no spamming same action!
                            directions = ['LEFT', 'DOWN', 'RIGHT', 'UP']
                            cycle_index = ((self._override_stuck_count - 100) // 5) % 4
                            forced_action = directions[cycle_index]
                            action_name = forced_action  # Override the action name!
                            # CRITICAL: Also override the action INTEGER for emulator!
                            # button_map = [A, B, UP, DOWN, LEFT, RIGHT, NOOP]
                            direction_to_action = {'UP': 2, 'DOWN': 3, 'LEFT': 4, 'RIGHT': 5}
                            action = direction_to_action.get(forced_action, action)
                            skip_stuck_override = True  # Prevent other overrides from overwriting!
                            self._forced_stuck_action = forced_action  # Save for final override!
                            self._forced_stuck_action_int = action  # Save action integer too!
                            # Start persistent forcing for 30 steps to complete waypoint path
                            self._forced_steps_remaining = 30
                            self._persistent_forced_action = forced_action
                            if self._override_stuck_count % 5 == 0:
                                logger.warning(f"🚨 STUCK OVERRIDE: {self._override_stuck_count} steps at ({current_x},{current_y}) → FORCING {forced_action} [PERSIST 30, cycling]")
            except Exception as e:
                if self.step_count % 50 == 0:
                    logger.warning(f"🚨 STUCK OVERRIDE ERROR: {e}")
            
            # ═══════════════════════════════════════════════════════════════════
            # 🏥 FORCED NURSE JOY (from previous step) - HIGHEST PRIORITY!
            # Late-stage sets force_nurse_next_action, we apply it here at step start
            # ═══════════════════════════════════════════════════════════════════
            if getattr(self, 'force_nurse_next_action', None):
                action_name = self.force_nurse_next_action
                skip_stuck_override = True  # Don't let stuck detection override this!
                coords = getattr(self, 'force_nurse_coords', ('?', '?'))
                logger.warning(f"💊 FORCED NURSE (from prev step): {action_name} at {coords}")
                
                # If this is an A press at (7,4) AND we've done enough presses (10+), healing is complete
                # Only set visited=True after 10+ A presses to complete the full healing dialog
                dialog_count = getattr(self, 'nurse_dialog_counter', 0)
                if action_name == 'A' and coords == (7, 4) and dialog_count >= 10:
                    if not getattr(self, 'visited_oldale_pokecenter', False):
                        self.visited_oldale_pokecenter = True
                        self.oldale_pokecenter_visited = True
                        logger.warning(f"✅ NURSE JOY HEALED! Respawn point set. HP restored! (after {dialog_count} A presses)")
                        # Reset dialog counter for next healing visit
                        self.nurse_dialog_counter = 0
                        # Sync milestone to server
                        try:
                            requests.post("http://localhost:8000/milestone", 
                                json={"name": "OLDALE_POKECENTER_HEALED", "completed": True}, 
                                timeout=1)
                        except Exception:
                            pass
                elif action_name == 'A' and coords == (7, 4):
                    # Still in healing dialog, log progress
                    if dialog_count % 3 == 0:
                        logger.info(f"💊 Nurse Joy dialog: {dialog_count}/10 A presses...")
                
                self.force_nurse_next_action = None  # Clear for next step
            
            # ═══════════════════════════════════════════════════════════════════
            # ⚔️ BATTLE ACTION OVERRIDE - Enhanced with PP tracking and fallbacks
            # Pokemon Emerald battle menu: FIGHT (top-left) is default selected
            # Pressing A selects FIGHT, then selects first move. Fast & efficient!
            # If stuck in battle (PP exhausted or moves fail), try switch/run fallback.
            # ═══════════════════════════════════════════════════════════════════
            in_battle_now = False
            try:
                # First check cached state (fast)
                game_state = getattr(self, 'cached_game_state', '').upper()
                if game_state == 'BATTLE':
                    in_battle_now = True
                else:
                    # Fallback: fetch LIVE from server (more accurate but slower)
                    # Only check when position seems stuck (potential battle transition)
                    if not hasattr(self, '_last_battle_check') or self.step_count - self._last_battle_check > 10:
                        self._last_battle_check = self.step_count
                        try:
                            state_resp = requests.get("http://localhost:8000/state", timeout=0.5)
                            if state_resp.ok:
                                state_data = state_resp.json()
                                live_state = state_data.get('game', {}).get('game_state', '').upper()
                                if live_state == 'BATTLE':
                                    in_battle_now = True
                                    self.cached_game_state = 'BATTLE'  # Update cache
                                    logger.info("⚔️ BATTLE DETECTED (live API check)")
                                else:
                                    # Battle ended - reset tracking
                                    if hasattr(self, '_battle_start_step'):
                                        logger.info(f"⚔️ BATTLE ENDED after {self.step_count - self._battle_start_step} steps")
                                        delattr(self, '_battle_start_step')
                                    if hasattr(self, '_battle_fallback_count'):
                                        delattr(self, '_battle_fallback_count')
                                    self.cached_game_state = live_state
                        except:
                            pass
                
                if in_battle_now:
                    # Track battle duration for stuck detection
                    if not hasattr(self, '_battle_start_step'):
                        self._battle_start_step = self.step_count
                        self._battle_fallback_count = 0
                        logger.info(f"⚔️ BATTLE STARTED at step {self.step_count}")
                    
                    battle_duration = self.step_count - self._battle_start_step
                    
                    # Check if stuck in battle for too long (100+ steps = ~10 seconds)
                    # If so, try fallback: switch Pokemon (DOWN then RIGHT to get to PKMN menu)
                    if battle_duration > 100 and battle_duration % 50 == 0:
                        self._battle_fallback_count = getattr(self, '_battle_fallback_count', 0) + 1
                        fallback_phase = self._battle_fallback_count % 4
                        
                        if fallback_phase == 1:
                            # Try to access Pokemon menu (DOWN from FIGHT)
                            action_name = 'DOWN'
                            logger.warning(f"⚔️ BATTLE STUCK ({battle_duration} steps): Trying fallback DOWN (to PKMN menu)")
                        elif fallback_phase == 2:
                            # Select Pokemon menu
                            action_name = 'A'
                            logger.warning(f"⚔️ BATTLE STUCK: Selecting Pokemon menu")
                        elif fallback_phase == 3:
                            # Try to switch to next Pokemon (DOWN then A)
                            action_name = 'DOWN'
                            logger.warning(f"⚔️ BATTLE STUCK: Trying to switch Pokemon")
                        else:
                            # Confirm switch
                            action_name = 'A'
                            logger.warning(f"⚔️ BATTLE STUCK: Confirming switch/action")
                        
                        skip_stuck_override = True
                    elif action_name in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                        # Normal battle - force A to attack
                        original_action = action_name
                        action_name = 'A'
                        skip_stuck_override = True  # Battle is handled
                        if self.step_count % 10 == 0:
                            logger.info(f"⚔️ BATTLE OVERRIDE: {original_action} → A (attack, duration: {battle_duration} steps)")
            except Exception as e:
                if self.step_count % 100 == 0:
                    logger.debug(f"Battle detection error: {e}")
            
            
            # ═══════════════════════════════════════════════════════════════════
            # 🏥 POKECENTER ENTRANCE OVERRIDE - DISABLED!
            # Now using TERRAIN A* with VISUAL MAP ground truth for navigation.
            # The visual map with . (walkable) and # (blocked) provides real-time
            # terrain data that A* can use for proper pathfinding around obstacles.
            # Portal connections provide door locations as navigation goals.
            # ═══════════════════════════════════════════════════════════════════
            # (Previously had hardcoded dx/dy navigation that caused stuck loops)
            
            # ═══════════════════════════════════════════════════════════════════
            # 🏥 POKECENTER INTERIOR: Force UP to Nurse Joy until healed!
            # When inside PokeCenter and haven't set respawn, force UP then A
            # ═══════════════════════════════════════════════════════════════════
            if hasattr(self, 'current_x') and hasattr(self, 'current_y') and hasattr(self, 'current_location'):
                location_upper = str(self.current_location).upper()
                if ('POKECENTER' in location_upper or 'POKEMON CENTER' in location_upper) and '2F' not in location_upper:
                    # Run interior navigation if EITHER:
                    # 1. Never visited (visited=False) - need to set respawn
                    # 2. Already visited BUT HP is low (need to heal again)
                    hp_needs_heal = getattr(self, 'cached_hp_ratio', 0.0) < 0.5
                    needs_nurse = not getattr(self, 'visited_oldale_pokecenter', False) or hp_needs_heal
                    
                    if needs_nurse:
                        # Inside PokeCenter and needs to talk to Nurse Joy!
                        # NURSE JOY requires agent at (7, 4) - counter is at (7, 3)
                        skip_stuck_override = True  # Don't let stuck detection mess with this!
                        
                        nurse_x, nurse_y = 7, 4  # Target position to talk to Nurse Joy (who is at 7,3)
                        npc_x, npc_y = 4, 4  # NPC blocking the corridor
                        
                        # Check if at EXACT Nurse Joy interaction position (7, 4)
                        if self.current_x == 7 and self.current_y == 4:
                            # At exact counter position - press A multiple times for dialog!
                            # Nurse Joy dialog requires: talk, confirm, wait for animation, dismiss
                            # Need ~10 A presses to fully complete healing
                            if not hasattr(self, 'nurse_dialog_counter'):
                                self.nurse_dialog_counter = 0
                            self.nurse_dialog_counter += 1
                            
                            action_name = 'A'
                            if self.step_count % 5 == 0:
                                logger.warning(f"💊 NURSE JOY: At counter, pressing A ({self.nurse_dialog_counter}/10)")
                            
                            # After 10+ A presses, healing should be complete
                            # Set flag for force_nurse to apply on NEXT step
                            if self.nurse_dialog_counter >= 10:
                                self.force_nurse_next_action = 'A'
                                self.force_nurse_coords = (self.current_x, self.current_y)
                        elif self.current_x == 6 and self.current_y == 4:
                            # One step away from counter - go RIGHT to reach (7,4)
                            action_name = 'RIGHT'
                            if self.step_count % 10 == 0:
                                logger.info(f"💊 NURSE JOY: At (6,4), going RIGHT to reach counter at (7,4)")
                        else:
                            # ═══════════════════════════════════════════════════════════════
                            # NPC AVOIDANCE: There's an NPC at (4, 4) blocking the corridor
                            # If we're at y=4 and x<4, we MUST go DOWN to y=5+ first!
                            # Path around NPC: go DOWN to y=5, RIGHT past x=4, then UP to y=4
                            # ═══════════════════════════════════════════════════════════════
                            
                            if self.current_y == npc_y and self.current_x < npc_x:
                                # On same row as NPC and to the left - go DOWN to avoid!
                                action_name = 'DOWN'
                                if self.step_count % 10 == 0:
                                    logger.warning(f"💊 NPC AVOIDANCE: At ({self.current_x},{self.current_y}), NPC blocks at ({npc_x},{npc_y}). Going DOWN!")
                                self.nurse_last_action = action_name
                            elif self.current_y == npc_y + 1 and self.current_x < npc_x:
                                # At y=5, below NPC corridor - go RIGHT to get past NPC
                                action_name = 'RIGHT'
                                if self.step_count % 10 == 0:
                                    logger.info(f"💊 NPC BYPASS: At ({self.current_x},{self.current_y}), going RIGHT to pass NPC")
                                self.nurse_last_action = action_name
                            elif self.current_y > npc_y and self.current_x >= npc_x + 1:
                                # Past NPC on x-axis, now go UP to reach y=4
                                action_name = 'UP'
                                if self.step_count % 10 == 0:
                                    logger.info(f"💊 NPC PASSED: At ({self.current_x},{self.current_y}), going UP to nurse's row")
                                self.nurse_last_action = action_name
                            else:
                                # USE TERRAIN A* to find passable path to Nurse Joy!
                                nav_action = None
                                
                                # Initialize position-keyed blocking tracker
                                # Key: (x, y) -> set of blocked directions at that position
                                if not hasattr(self, 'nurse_blocked_by_pos'):
                                    self.nurse_blocked_by_pos = {}  # {(x,y): {'RIGHT', 'UP', ...}}
                                    self.nurse_last_action = None
                                    self.nurse_last_pos = None
                                
                                current_pos = (self.current_x, self.current_y)
                                
                                # Initialize blocked set for this position if needed
                                if current_pos not in self.nurse_blocked_by_pos:
                                    self.nurse_blocked_by_pos[current_pos] = set()
                                
                                # Check if last action actually moved us
                                if self.nurse_last_pos is not None and self.nurse_last_action:
                                    if self.nurse_last_pos == current_pos:
                                        # Position didn't change - that direction is BLOCKED at last_pos
                                        self.nurse_blocked_by_pos[self.nurse_last_pos].add(self.nurse_last_action)
                                        if self.step_count % 5 == 0:
                                            logger.warning(f"💊 BLOCKED BY NPC: {self.nurse_last_action} blocked at {self.nurse_last_pos}. Memory: {dict(self.nurse_blocked_by_pos)}")
                                
                                self.nurse_last_pos = current_pos
                                
                                # Get blocked directions for CURRENT position
                                blocked_at_current = self.nurse_blocked_by_pos.get(current_pos, set())
                                
                                # Try A* first (if we haven't blocked that direction)
                                if hasattr(self, 'terrain_nav') and hasattr(self, 'cached_metatile'):
                                    map_id = getattr(self, 'current_map_id', 0x0200)
                                    metatile = getattr(self, 'cached_metatile', [])
                                    if metatile and len(metatile) >= 15:
                                        nav_result = self.terrain_nav.get_direction(
                                            map_id, self.current_x, self.current_y, 
                                            (nurse_x, nurse_y), metatile
                                        )
                                        if nav_result and isinstance(nav_result, dict):
                                            nav_action = nav_result.get('action')
                                
                                # If A* direction was already blocked, find alternative
                                if nav_action in blocked_at_current:
                                    nav_action = None
                                
                                if nav_action and nav_action in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                                    action_name = nav_action
                                    if self.step_count % 10 == 0:
                                        logger.info(f"💊 NURSE JOY A*: ({self.current_x},{self.current_y}) -> {nav_action} to ({nurse_x},{nurse_y})")
                                else:
                                    # A* failed or blocked - try any unblocked direction
                                    all_dirs = ['DOWN', 'RIGHT', 'UP', 'LEFT']
                                    available_dirs = [d for d in all_dirs if d not in blocked_at_current]
                                    
                                    if available_dirs:
                                        # Pick direction that moves toward nurse
                                        best_dir = None
                                        for d in available_dirs:
                                            dx = {'LEFT': -1, 'RIGHT': 1, 'UP': 0, 'DOWN': 0}
                                            dy = {'LEFT': 0, 'RIGHT': 0, 'UP': -1, 'DOWN': 1}
                                            new_x = self.current_x + dx[d]
                                            new_y = self.current_y + dy[d]
                                            # Prefer directions that get us closer to nurse
                                            dist_now = abs(self.current_x - nurse_x) + abs(self.current_y - nurse_y)
                                            dist_new = abs(new_x - nurse_x) + abs(new_y - nurse_y)
                                            if dist_new < dist_now:
                                                best_dir = d
                                                break
                                        
                                        action_name = best_dir if best_dir else available_dirs[0]
                                        if self.step_count % 10 == 0:
                                            logger.warning(f"💊 NURSE ALT DIR: ({self.current_x},{self.current_y}) -> {action_name} (blocked: {blocked_at_current})")
                                    else:
                                        # All 4 directions blocked at this position! Clear and try again
                                        self.nurse_blocked_by_pos[current_pos] = set()
                                        action_name = 'DOWN'  # Try DOWN to escape
                                        if self.step_count % 10 == 0:
                                            logger.error(f"💊 ALL 4 DIRS BLOCKED! Resetting at ({self.current_x},{self.current_y})")
                                
                                self.nurse_last_action = action_name
                    else:
                        # ═══════════════════════════════════════════════════════════════
                        # visited=True - BUT check HP first! If HP low, need to heal again!
                        # Only exit if HP is healthy (>= 50%)
                        # ═══════════════════════════════════════════════════════════════
                        hp_needs_heal = getattr(self, 'cached_hp_ratio', 0.0) < 0.5  # Default 0.0 = assume low HP
                        
                        if hp_needs_heal:
                            # HP is low - need to heal! Go UP to Nurse Joy just like first visit
                            skip_stuck_override = True
                            if self.current_y > 4:
                                action_name = 'UP'
                                if self.step_count % 10 == 0:
                                    logger.info(f"💉 PC HEAL NEEDED (HP<50%): At ({self.current_x},{self.current_y}), going UP to Nurse Joy")
                            else:
                                # At counter area - press A to heal
                                action_name = 'A'
                                if self.step_count % 10 == 0:
                                    logger.info(f"💉 PC HEAL NEEDED: At counter ({self.current_x},{self.current_y}), pressing A to heal")
                        else:
                            # HP is healthy - EXIT THE POKECENTER!
                            # Agent may be stuck in dialog - alternate A to dismiss, DOWN to move
                            # Exit warp is at (7, 8) - go DOWN until y=8, then position will warp
                            skip_stuck_override = True  # We're handling this ourselves
                            
                            # Track dialog dismissal attempts
                            if not hasattr(self, 'pc_exit_counter'):
                                self.pc_exit_counter = 0
                            self.pc_exit_counter += 1
                        
                            # If stuck at y<=4 (counter area), likely in dialog - press A to dismiss
                            if self.current_y <= 4:
                                if self.pc_exit_counter % 3 == 0:
                                    # Every 3rd step, press A to dismiss any dialog
                                    action_name = 'A'
                                    if self.step_count % 10 == 0:
                                        logger.info(f"🏥 PC EXIT: At ({self.current_x},{self.current_y}), pressing A to dismiss dialog")
                                else:
                                    # Otherwise try DOWN toward exit
                                    action_name = 'DOWN'
                                    if self.step_count % 10 == 0:
                                        logger.info(f"🏥 PC EXIT: At ({self.current_x},{self.current_y}), going DOWN toward exit (y=8)")
                            elif self.current_y < 8:
                                # Between counter and exit - just go DOWN
                                action_name = 'DOWN'
                                if self.step_count % 15 == 0:
                                    logger.info(f"🏥 PC EXIT: At ({self.current_x},{self.current_y}), going DOWN to exit")
                            elif self.current_y >= 8:
                                # At exit row - might need to be at x=7 for warp
                                if self.current_x < 7:
                                    action_name = 'RIGHT'
                                elif self.current_x > 7:
                                    action_name = 'LEFT'
                                else:
                                    # At (7,8) - warp should trigger, press DOWN to exit
                                    action_name = 'DOWN'
                                if self.step_count % 10 == 0:
                                    logger.info(f"🏥 PC EXIT: At exit row ({self.current_x},{self.current_y}), action={action_name}")
            
            # ═══════════════════════════════════════════════════════════════════
            # 🔄 UNIVERSAL STUCK DETECTION - Track position and force alternatives
            # When coordinates don't change for 3+ steps, cycle through directions
            # This prevents getting stuck in corners or against walls
            # MORE AGGRESSIVE: Even when PokeCenter override is active!
            # ═══════════════════════════════════════════════════════════════════
            stuck_penalty = 0.0  # Penalty for being stuck
            if hasattr(self, 'current_x') and hasattr(self, 'current_y'):
                current_pos = (self.current_x, self.current_y)
                
                # Initialize tracking if not exists
                if not hasattr(self, 'stuck_position_history'):
                    self.stuck_position_history = []
                    self.stuck_direction_cycle = 0
                    self.last_stuck_pos = None
                    self.stuck_persistence_counter = 0
                
                # Add current position to history
                self.stuck_position_history.append(current_pos)
                if len(self.stuck_position_history) > 10:
                    self.stuck_position_history.pop(0)
                
                # Check if stuck (same position for last 3 steps - MORE AGGRESSIVE)
                if len(self.stuck_position_history) >= 3:
                    recent_3 = self.stuck_position_history[-3:]
                    all_same = all(pos == recent_3[0] for pos in recent_3)
                    
                    # Count total steps stuck
                    if all_same:
                        self.stuck_persistence_counter += 1
                        
                        # PENALTY for being stuck - scales with how long stuck
                        # This provides reward signal for learning, NOT action override
                        stuck_penalty = -10.0 * min(self.stuck_persistence_counter, 10)  # Max -100
                        
                        # Log stuck status (but DO NOT override action - let policy learn!)
                        if self.stuck_persistence_counter % 20 == 0:
                            logger.info(f"🔄 STUCK PENALTY: {stuck_penalty} (position {current_pos}, {self.stuck_persistence_counter} steps)")
                        
                        # Clear terrain cache periodically to force A* recompute
                        if hasattr(self, 'terrain_nav') and self.stuck_persistence_counter % 10 == 0:
                            self.terrain_nav.clear_cache()
                        
                        # 🆕 GLOBAL ACTION DIVERSITY (30+ steps stuck at same position)
                        # Uses reward-based learning to teach PPO to try different actions
                        if self.stuck_persistence_counter >= 30:
                            # Get last action to compare
                            if not hasattr(self, 'global_last_stuck_action'):
                                self.global_last_stuck_action = None
                                self.global_action_counter = {}
                            
                            # Track action usage
                            self.global_action_counter[action_name] = self.global_action_counter.get(action_name, 0) + 1
                            most_used = max(self.global_action_counter, key=self.global_action_counter.get)
                            
                            if action_name == most_used and self.global_action_counter[most_used] > 10:
                                # HEAVY penalty for repeating same action when stuck 30+ steps
                                diversity_penalty = -150.0
                                stuck_penalty += diversity_penalty
                                if self.stuck_persistence_counter % 30 == 0:
                                    logger.warning(f"🛑 STUCK DIVERSITY: {action_name} × {self.global_action_counter[most_used]} → -150 (try different!)")
                            elif action_name != most_used:
                                # BONUS for trying different action
                                diversity_bonus = 100.0
                                stuck_penalty += diversity_bonus
                                if self.stuck_persistence_counter % 10 == 0:
                                    logger.info(f"✅ DIVERSITY BONUS: {action_name} (not {most_used}) → +100")
                            
                            self.global_last_stuck_action = action_name
                        
                        # 🆕 FORCED ACTION OVERRIDE (100+ steps stuck)
                        # PPO learning too slow - directly override policy action with terrain guidance
                        if self.stuck_persistence_counter >= 100 and hasattr(self, 'terrain_nav'):
                            try:
                                memory = getattr(self, 'cached_memory', {})
                                current_x = memory.get('x', 0)
                                current_y = memory.get('y', 0)
                                # Get terrain-recommended action
                                terrain_result = self.terrain_nav.get_next_action(current_x, current_y, memory)
                                if terrain_result and terrain_result.get('action'):
                                    forced_action = terrain_result['action']
                                    action_map = {'UP': 5, 'DOWN': 6, 'LEFT': 7, 'RIGHT': 8, 'A': 0, 'B': 1}
                                    if forced_action in action_map:
                                        # FORCE the terrain action
                                        action = action_map[forced_action]
                                        if self.stuck_persistence_counter % 50 == 0:
                                            logger.warning(f"🚨 FORCED ACTION OVERRIDE: Stuck {self.stuck_persistence_counter} steps → FORCING {forced_action}")
                            except Exception as e:
                                pass  # Fall through to normal policy
                    else:
                        # Moving again - reset
                        self.stuck_direction_cycle = 0
                        self.stuck_persistence_counter = 0
                        # Clear action counter when unstuck
                        if hasattr(self, 'global_action_counter'):
                            self.global_action_counter = {}
            
            # ═══════════════════════════════════════════════════════════════════
            # 🧭 TERRAIN NAV ACTION FORCING - Use ground truth A* instead of RL policy!
            # After healing done, force agent to follow terrain navigation to Route 103
            # This ensures agent uses real-time ground truth instead of untrained policy
            # SKIP if HP is low (< 50%) - agent needs to enter PokeCenter to heal!
            # ═══════════════════════════════════════════════════════════════════
            already_healed = getattr(self, 'visited_oldale_pokecenter', False)
            current_location = str(getattr(self, 'current_location', '')).upper()
            last_terrain = getattr(self, 'last_terrain_result', None)
            
            # Check HP - if low, don't force terrain (allow healing behavior)
            hp_low = False
            try:
                if hasattr(self, 'cached_hp_ratio') and self.cached_hp_ratio < 0.5:
                    hp_low = True
            except:
                pass
            
            # Only apply in Oldale Town EXTERIOR or ROUTE 103/101 (not inside buildings)
            in_oldale_town = 'OLDALE' in current_location and 'POKECENTER' not in current_location and 'MART' not in current_location and 'HOUSE' not in current_location
            on_route_103 = 'ROUTE' in current_location and '103' in current_location
            on_route_101 = 'ROUTE' in current_location and '101' in current_location
            
            # Also skip if near PokeCenter entrance (6,17) - let healing logic handle
            near_pokecenter = hasattr(self, 'current_x') and hasattr(self, 'current_y') and abs(self.current_x - 6) <= 1 and abs(self.current_y - 17) <= 1
            
            # Apply terrain override for all outdoor navigation areas
            should_apply_terrain = (already_healed and in_oldale_town) or on_route_103 or on_route_101
            
            # REWARD-BASED TERRAIN GUIDANCE (not action override!)
            # The compute_reward() function uses last_terrain to give:
            #   +200 for following A* direction
            #   -50 for going opposite direction
            # This allows the policy to LEARN from terrain rewards instead of being forced
            if should_apply_terrain and last_terrain and not hp_low and not near_pokecenter:
                terrain_dir = last_terrain.get('action')
                if terrain_dir and terrain_dir in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                    # Log terrain guidance (but DO NOT override action!)
                    if self.step_count % 50 == 0:
                        logger.info(f"🧭 TERRAIN REWARD GUIDANCE: Recommended {terrain_dir} (A* path, policy chooses)")
            
            # ═══════════════════════════════════════════════════════════════════
            # 🚨 FORCED ACTION OVERRIDE FOR POKECENTER 2F ESCAPE
            # When stuck on 2F, force toward escalator at (1, 6) based on position!
            # This bypasses the RL policy to physically force escalator exit.
            # ═══════════════════════════════════════════════════════════════════
            # ⚠️ DISABLED: 2F override for PPO learning
            enable_2f_override = False
            if enable_2f_override and hasattr(self, 'steps_on_2f') and hasattr(self, 'current_location') and hasattr(self, 'current_x') and hasattr(self, 'current_y'):
                if self.steps_on_2f > 30 and '2F' in str(self.current_location).upper():
                    original_action = action_name
                    escalator_x, escalator_y = 1, 6  # Escalator exit position
                    
                    # POSITION-AWARE OVERRIDE:
                    # - Force LEFT only if x > escalator_x (agent to the right of escalator)
                    # - Force RIGHT only if x < escalator_x (agent to the left of escalator, went past it)
                    # - Force UP if y > escalator_y (agent below escalator)
                    # - Force DOWN if y < escalator_y (agent above escalator)
                    
                    if self.current_x > escalator_x:
                        # Agent to the RIGHT of escalator - force LEFT
                        if action_name == 'RIGHT':
                            action_name = 'LEFT'
                            if self.step_count % 100 == 0:
                                logger.warning(f"🚨 2F OVERRIDE: Forced RIGHT → LEFT (x={self.current_x}→{escalator_x})")
                    elif self.current_x < escalator_x:
                        # Agent to the LEFT of escalator (went past it) - force RIGHT
                        if action_name == 'LEFT':
                            action_name = 'RIGHT'
                            if self.step_count % 100 == 0:
                                logger.warning(f"🚨 2F OVERRIDE: Forced LEFT → RIGHT (x={self.current_x}→{escalator_x})")
                    
                    # Only override UP/DOWN based on relative position to escalator
                    if self.current_y > escalator_y:
                        # Agent BELOW escalator (y > 6) - need to go UP
                        if action_name == 'DOWN':
                            action_name = 'UP'  # Force UP toward escalator
                            if self.step_count % 100 == 0:
                                logger.warning(f"🚨 2F OVERRIDE: Forced DOWN → UP (y={self.current_y}→{escalator_y})")
                    elif self.current_y < escalator_y:
                        # Agent ABOVE escalator (y < 6) - need to go DOWN
                        if action_name == 'UP':
                            action_name = 'DOWN'  # Force DOWN toward escalator
                            if self.step_count % 100 == 0:
                                logger.warning(f"🚨 2F OVERRIDE: Forced UP → DOWN (y={self.current_y}→{escalator_y})")
            
            # ═══════════════════════════════════════════════════════════════════
            # 🏠 EARLY GAME HOUSE A* OVERRIDE - Forces navigation to 2F stairs
            # When in player's house 1F (before SET_CLOCK) or rival's house 1F
            # (before RIVAL_BEDROOM), override terrain A* to target the stairs.
            # This ensures proper early game progression: 2F → set clock → leave
            # ═══════════════════════════════════════════════════════════════════
            current_location = str(getattr(self, 'current_location', '')).upper()
            current_x = getattr(self, 'current_x', None)
            current_y = getattr(self, 'current_y', None)
            
            # Check if we're in early game house navigation situation
            at_player_house_1f = ("MAYS HOUSE 1F" in current_location or "BRENDANS HOUSE 1F" in current_location or
                                  "MAY HOUSE 1F" in current_location or "BRENDAN HOUSE 1F" in current_location)
            at_player_house_2f = ("MAYS HOUSE 2F" in current_location or "BRENDANS HOUSE 2F" in current_location or
                                  "MAY HOUSE 2F" in current_location or "BRENDAN HOUSE 2F" in current_location or
                                  "PLAYER BEDROOM" in current_location)
            at_rival_house_1f = ("RIVAL" in current_location and "1F" in current_location)
            
            # Also detect rival house by exclusion: in a Littleroot house but not player's house
            if "LITTLEROOT" in current_location and "HOUSE" in current_location and "1F" in current_location:
                if not at_player_house_1f:
                    at_rival_house_1f = True
            
            has_pokemon = getattr(self, 'has_pokemon', False) or getattr(self, 'party_size', 0) > 0
            
            # Milestone tracking - check actual flags or infer from location
            # SET_CLOCK: True if we have Pokemon OR we've been on 2F (cutscene triggers when reaching 2F)
            has_set_clock = has_pokemon or at_player_house_2f or getattr(self, 'visited_player_2f', False)
            # Track if we've been to 2F
            if at_player_house_2f:
                self.visited_player_2f = True
                has_set_clock = True
            
            # RIVAL_BEDROOM: True if we have Pokemon OR we've visited rival's room
            has_rival_bedroom = has_pokemon or getattr(self, 'visited_rival_bedroom', False)
            # Track if at rival's 2F
            if "RIVAL" in current_location and "2F" in current_location:
                self.visited_rival_bedroom = True
                has_rival_bedroom = True
            
            # ROUTE_101: True if we have Pokemon (we've been to Route 101 to get starter)
            has_route_101 = has_pokemon
            
            # Typical stairs coordinates in Pokemon Emerald houses:
            # Player house: stairs at roughly (8, 2) or (2, 2) - upper left/right area
            # Rival house: similar layout
            # We'll use A* once we set the goal - A* will find the path
            
            early_game_override = False
            early_game_target = None
            stairs_goal = None
            
            if at_player_house_1f and not has_set_clock:
                # Force navigation to stairs in player house 1F (going UP to set clock)
                # Stairs are typically in upper-left area of 1F (around x=2, y=1)
                stairs_goal = (2, 1)  # Upper left corner where stairs are
                early_game_override = True
                early_game_target = "stairs_up"
                if self.step_count % 100 == 0:
                    logger.info(f"🏠 HOUSE OVERRIDE: Player house 1F - navigating to stairs at {stairs_goal}")
            elif at_player_house_1f and has_set_clock and not has_rival_bedroom:
                # Clock is set, need to EXIT house to go to rival's house
                # Exit door is typically at bottom center of 1F (around x=4-5, y=8-9)
                stairs_goal = (4, 8)  # Bottom door exit
                early_game_override = True
                early_game_target = "exit_door"
                if self.step_count % 100 == 0:
                    logger.info(f"🏠 HOUSE OVERRIDE: Player house 1F - clock set, navigating to exit at {stairs_goal}")
            elif at_player_house_2f and has_set_clock and not has_rival_bedroom:
                # Clock is set, on 2F, need to go DOWN stairs first
                # Stairs down are in same location as stairs up
                stairs_goal = (8, 1)  # Stairs location on 2F (usually right side)
                early_game_override = True
                early_game_target = "stairs_down"
                if self.step_count % 100 == 0:
                    logger.info(f"🏠 HOUSE OVERRIDE: Player house 2F - clock set, navigating to stairs DOWN at {stairs_goal}")
            elif at_rival_house_1f and not has_rival_bedroom:
                # Force navigation to stairs in rival house 1F (going UP to bedroom)
                # Same layout as player house
                stairs_goal = (2, 1)
                early_game_override = True
                early_game_target = "stairs_up"
                if self.step_count % 100 == 0:
                    logger.info(f"🏠 HOUSE OVERRIDE: Rival house 1F - navigating to stairs at {stairs_goal}")
            elif at_rival_house_1f and has_rival_bedroom and not has_route_101:
                # Visited rival bedroom, need to EXIT house to go to Route 101
                stairs_goal = (4, 8)  # Bottom door exit
                early_game_override = True
                early_game_target = "exit_door"
                if self.step_count % 100 == 0:
                    logger.info(f"🏠 HOUSE OVERRIDE: Rival house 1F - visited rival, navigating to exit at {stairs_goal}")
            elif "RIVAL" in current_location and "2F" in current_location and has_rival_bedroom:
                # Visited rival on 2F, need to go DOWN stairs
                stairs_goal = (8, 1)
                early_game_override = True
                early_game_target = "stairs_down"
                if self.step_count % 100 == 0:
                    logger.info(f"🏠 HOUSE OVERRIDE: Rival house 2F - visited rival, navigating to stairs DOWN at {stairs_goal}")
            
            # If early game override is active, compute A* direction to target
            if early_game_override and stairs_goal and current_x is not None and current_y is not None:
                sx, sy = stairs_goal
                dx = sx - current_x
                dy = sy - current_y
                
                # Simple directional priority: go toward target
                # Prioritize Y first (get to correct row), then X
                override_dir = None
                if abs(dy) > abs(dx):  # Prioritize vertical movement
                    if dy < 0:
                        override_dir = 'UP'
                    elif dy > 0:
                        override_dir = 'DOWN'
                else:  # Horizontal movement
                    if dx < 0:
                        override_dir = 'LEFT'
                    elif dx > 0:
                        override_dir = 'RIGHT'
                
                # If at target location, try pressing A to interact (stairs/door)
                if abs(dx) <= 1 and abs(dy) <= 1:
                    override_dir = 'A'  # Interact with stairs or door
                    target_name = early_game_target if 'early_game_target' in dir() else 'target'
                    if self.step_count % 50 == 0:
                        logger.info(f"🏠 HOUSE OVERRIDE: At {target_name} ({current_x},{current_y}) - pressing A")
                
                # Apply the override if we have a direction
                if override_dir and action_name in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                    if action_name != override_dir:
                        original_action = action_name
                        action_name = override_dir
                        skip_stuck_override = True  # Prevent terrain A* from fighting
                        if self.step_count % 50 == 0:
                            logger.warning(f"🏠 HOUSE OVERRIDE: {original_action} → {override_dir} (toward {stairs_goal})")
            
            # ═══════════════════════════════════════════════════════════════════
            # 🧭 UNIVERSAL TERRAIN A* OVERRIDE
            # Force policy to follow terrain A* when it chooses opposite direction.
            # This prevents the agent from going backwards when A* has a clear path.
            # Only overrides OPPOSITE direction - does not block A/B or perpendicular moves.
            # SKIP if house override already applied (early_game_override = True)
            # ═══════════════════════════════════════════════════════════════════
            terrain_result = getattr(self, 'last_terrain_result', None)
            if terrain_result and not early_game_override:  # Skip if house override active
                terrain_dir = terrain_result.get('action')
                if terrain_dir in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                    opposite = {'UP': 'DOWN', 'DOWN': 'UP', 'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}
                    # Force action toward terrain direction ONLY if policy chose opposite
                    if action_name == opposite.get(terrain_dir):
                        original_action = action_name
                        action_name = terrain_dir  # Follow terrain!
                        location_str = str(getattr(self, 'current_location', '')).upper()
                        if self.step_count % 50 == 0:
                            logger.warning(f"🚨 TERRAIN OVERRIDE: {original_action} → {terrain_dir} (A* path) @ {location_str}")
            
            
            # ═══════════════════════════════════════════════════════════════════
            # 🩺 UNIVERSAL HEALING INTERRUPT - FORCES ACTIONS WHEN HP CRITICAL
            # Has HIGHEST priority - overrides ALL other navigation!
            # Triggers on: 1) HP critical (<30%) OR 2) Active HEAL directive
            # Uses cached healing_target from step() SLAM pathfinding  
            # ═══════════════════════════════════════════════════════════════════
            healing_target = getattr(self, 'cached_healing_target', None)
            healing_interrupt = getattr(self, 'cached_healing_interrupt', False)
            cached_directive = str(getattr(self, 'cached_directive_target', '') or '').upper()
            current_location = str(getattr(self, 'current_location', '')).upper()
            current_x = getattr(self, 'current_x', None)
            current_y = getattr(self, 'current_y', None)
            
            # Trigger on HP critical OR active HEAL directive
            # BUT skip if already visited PokeCenter (healing done!)
            already_healed = getattr(self, 'visited_oldale_pokecenter', False)
            has_heal_directive = ('HEAL' in cached_directive or 'POKECENTER' in cached_directive) and not already_healed
            should_force_healing = ((healing_interrupt and healing_target) or has_heal_directive) and not already_healed
            
            if should_force_healing and action_name in ['UP', 'DOWN', 'LEFT', 'RIGHT', 'A']:
                heal_type = healing_target.get('type', '') if healing_target else ''
                heal_action = healing_target.get('action', '') if healing_target else ''
                heal_coords = healing_target.get('coords') if healing_target else None
                target_dir = None
                
                # Determine the direction to force based on healing target
                if heal_type == 'NURSE_JOY' or 'POKECENTER' in current_location:
                    # In PokeCenter - go to nurse and press A
                    if current_y is not None and current_y <= 4:
                        target_dir = 'A'  # At nurse counter
                    else:
                        target_dir = 'UP'  # Walk to nurse
                        
                elif heal_type == 'MOM':
                    # In player's house 1F - find and talk to Mom
                    target_dir = 'A' if action_name != 'A' else None  # Try talking
                    
                elif heal_type == 'GO_DOWNSTAIRS':
                    # In bedroom - go to stairs (need to go UP into them like house exit)
                    target_dir = 'UP'
                    
                elif heal_type in ['OLDALE_PC', 'PLAYER_HOUSE']:
                    # Heading to a destination - use coordinate pathfinding
                    if heal_coords and current_x is not None and current_y is not None:
                        tx, ty = heal_coords
                        dx = tx - current_x
                        dy = ty - current_y
                        # Prioritize larger difference
                        if abs(dx) > abs(dy):
                            target_dir = 'RIGHT' if dx > 0 else 'LEFT'
                        elif dy != 0:
                            target_dir = 'DOWN' if dy > 0 else 'UP'
                        else:
                            target_dir = 'A'  # At destination
                    elif heal_action in ['GO_NORTH', 'UP']:
                        target_dir = 'UP'
                    elif heal_action in ['GO_SOUTH', 'DOWN']:
                        target_dir = 'DOWN'
                        
                elif heal_type == 'GO_SOUTH':
                    target_dir = 'DOWN'
                    
                elif heal_type == 'FIND_HEALING':
                    # Generic - head south (toward Littleroot/Mom or Oldale)
                    target_dir = 'DOWN' if 'ROUTE 103' in current_location else 'UP'
                
                # FALLBACK: If no specific target, use location-based routing
                elif has_heal_directive and not target_dir:
                    if 'ROUTE 101' in current_location:
                        # Route 101 - go NORTH to Oldale PokeCenter
                        target_dir = 'UP'
                    elif 'ROUTE 103' in current_location:
                        # Route 103 - go SOUTH to Oldale
                        target_dir = 'DOWN'
                    elif 'OLDALE' in current_location and 'POKECENTER' not in current_location:
                        # In Oldale town - find and enter PokeCenter
                        # PokeCenter entrance at (6, 17) - SERVER GROUND TRUTH
                        if current_x is not None and current_y is not None:
                            pc_x, pc_y = 6, 17
                            dx = pc_x - current_x
                            dy = pc_y - current_y
                            if abs(dx) > abs(dy):
                                target_dir = 'RIGHT' if dx > 0 else 'LEFT'
                            elif dy != 0:
                                target_dir = 'DOWN' if dy > 0 else 'UP'
                            else:
                                target_dir = 'UP'  # At PokeCenter door - walk UP to enter!
                    elif 'LITTLEROOT' in current_location:
                        # In Littleroot - go to player's house at (5, 4)
                        if current_x is not None and current_y is not None:
                            house_x, house_y = 5, 4
                            dx = house_x - current_x
                            dy = house_y - current_y
                            if abs(dx) > abs(dy):
                                target_dir = 'RIGHT' if dx > 0 else 'LEFT'
                            elif dy != 0:
                                target_dir = 'DOWN' if dy > 0 else 'UP'
                            else:
                                target_dir = 'A'  # At house door
                
                # REWARD-BASED HEALING GUIDANCE (not action override!)
                # The terrain A* rewards will guide the agent to the healing location
                # We just log the recommended direction, don't force it
                if target_dir and target_dir in ['UP', 'DOWN', 'LEFT', 'RIGHT', 'A']:
                    # Store for reward shaping in compute_reward()
                    self._heal_target_dir = target_dir
                    if self.step_count % 30 == 0:
                        logger.info(f"🩺 HEAL REWARD GUIDANCE: Recommend {target_dir} (target={heal_type or 'DIRECTIVE'}, loc={current_location[:20]})")

            
            # ═══════════════════════════════════════════════════════════════════
            # 🚨 FORCED ACTION OVERRIDE FOR HOUSE EXITS
            # When stuck in ANY house for >30 steps, force toward exit
            # Simplified: in houses, exit is typically DOWN (toward door)
            # On 2F: when AT stairs (8,2), force A to interact with stairs!
            # ═══════════════════════════════════════════════════════════════════
            if hasattr(self, 'steps_in_house') and hasattr(self, 'current_location'):
                location_str = str(self.current_location).upper()
                is_in_house = any(kw in location_str for kw in ['HOUSE', '1F', '2F', 'BEDROOM'])
                
                if is_in_house and self.steps_in_house > 30:
                    # Use current_x/current_y directly (set after each step)
                    x = getattr(self, 'current_x', None)
                    y = getattr(self, 'current_y', None)
                    
                    # DEBUG: Log if we're in override mode
                    if self.step_count % 20 == 0:
                        logger.warning(f"🏠 HOUSE OVERRIDE CHECK: steps={self.steps_in_house}, x={x}, y={y}")
                    
                    target_dir = None
                    
                    # Only proceed if we have valid position data
                    if x is not None and y is not None:
                        # Compute exit direction using current frame's data
                        if '2F' in location_str or 'BEDROOM' in location_str:
                            exit_goal = (7, 1)  # Stairs to 1F at (7,1) - walk UP into them!
                            # Pokemon stairs: walk UP into the stair tile at (7,1)
                            # No A button needed - just walk into them
                            if x == 7 and y > 1:
                                # Need to walk UP to reach/enter stairs
                                target_dir = 'UP'
                                logger.warning(f"🪜 GO TO STAIRS (7,{y}): Forcing UP toward stairs at (7,1)!")
                            elif x != 7:
                                # Wrong column - go toward x=7
                                target_dir = 'LEFT' if x > 7 else 'RIGHT'
                                logger.warning(f"🪜 WRONG COLUMN ({x},{y}): Forcing {target_dir} toward x=7!")
                        else:
                            # 1F: Door exit at (8,8) or (9,8) - walk DOWN into it
                            exit_goal = (8, 8)  # Door position
                            if x >= 8 and x <= 9:
                                # Correct column - walk DOWN into door
                                target_dir = 'DOWN'
                                logger.warning(f"🚪 1F AT DOOR ({x},{y}): Forcing DOWN to exit!")
                            elif x > 9:
                                # Too far right - go LEFT toward door
                                target_dir = 'LEFT'
                                logger.warning(f"🚪 1F WRONG COLUMN ({x},{y}): Forcing LEFT toward x=8/9!")
                            elif x < 8:
                                # Too far left - go RIGHT toward door
                                target_dir = 'RIGHT'
                                logger.warning(f"🚪 1F WRONG COLUMN ({x},{y}): Forcing RIGHT toward x=8/9!")
                    
                    # Fallback for directional actions
                    if target_dir is None or target_dir not in ['UP', 'DOWN', 'LEFT', 'RIGHT', 'A']:
                        # On 2F: default UP toward stairs at (8,2); on 1F: default DOWN toward exit
                        if '2F' in location_str or 'BEDROOM' in location_str:
                            target_dir = 'UP'
                        else:
                            target_dir = 'DOWN'
                    
                    # ⚠️ DISABLED: Force ANY wrong movement toward exit direction
                    # Using PPO reward shaping instead of action overrides
                    enable_house_override = False
                    if enable_house_override and target_dir == 'A' and action_name != 'A':
                        original_action = action_name
                        action_name = 'A'
                        if self.step_count % 10 == 0:
                            logger.warning(f"🏠 HOUSE OVERRIDE: {original_action} → A (stuck {self.steps_in_house} steps, AT STAIRS)")
                    elif enable_house_override and action_name in ['UP', 'DOWN', 'LEFT', 'RIGHT'] and action_name != target_dir and target_dir in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                        original_action = action_name
                        action_name = target_dir
                        if self.step_count % 10 == 0:
                            logger.warning(f"🏠 HOUSE OVERRIDE: {original_action} → {target_dir} (stuck {self.steps_in_house} steps)")
            
            # ═══════════════════════════════════════════════════════════════════
            # 🚨 FINAL STUCK OVERRIDE - Apply forced action BEFORE sending to server!
            # ═══════════════════════════════════════════════════════════════════
            if getattr(self, '_forced_stuck_action', None):
                action_name = self._forced_stuck_action
                if self.step_count % 10 == 0:
                    logger.warning(f"🔥 PRE-POST STUCK OVERRIDE: Forcing {action_name} (before server send)")
            
            try:
                # Send action to server (increased timeout for stability)
                requests.post(f"{self.server_url}/action", json={"buttons": [action_name]}, timeout=15)
                state = requests.get(f"{self.server_url}/state", timeout=15).json()
                
                if state:
                    # Update frame from frame server
                    try:
                        frame_response = requests.get(f"{self.frame_server_url}/frame", timeout=5)
                        if frame_response.status_code == 200:
                            import base64
                            from PIL import Image
                            from io import BytesIO
                            img_data = base64.b64decode(frame_response.json()['frame'])
                            pil_img = Image.open(BytesIO(img_data))
                            frame = np.array(pil_img.resize((84, 84)))
                            self.frame_stack.append(frame)
                            if len(self.frame_stack) > self.frame_stack_size:
                                self.frame_stack.pop(0)
                    except Exception as e:
                        logger.debug(f"Frame update failed: {e}")
                    
                    # Get game state from player field (server format uses 'player' not 'memory')
                    player = state.get('player', {})
                    position = player.get('position', {})
                    x = position.get('x', 0)
                    y = position.get('y', 0)
                    self.current_x = x  # Store for forced action override
                    self.current_y = y  # Store for forced action override
                    game_data = state.get('game', {})
                    
                    # WORKAROUND: Server doesn't provide map_id, infer from location string
                    # FIXED MAP IDS from pokemon_env/enums.py:
                    # - Littleroot Town: 0x09
                    # - Brendan's house 1F: 0x100, 2F: 0x101
                    # - May's house 1F: 0x102, 2F: 0x103  
                    # - Birch's Lab: 0x104
                    # - Route 101: 0x10
                    location_str = player.get('location', '').upper()
                    # ═══════════════════════════════════════════════════════════════════
                    # 🗺️ GET REAL MAP_ID FROM SERVER GROUND TRUTH (NOT LOCATION STRING!)
                    # The server provides actual map_id from memory in stitched_map_info
                    # ═══════════════════════════════════════════════════════════════════
                    map_id = 0x09  # Default to Littleroot
                    
                    # Try to get map_id from server's stitched_map_info (GROUND TRUTH)
                    stitched_info = state.get('map', {}).get('stitched_map_info', {})
                    current_area = stitched_info.get('current_area', {})
                    server_map_id = current_area.get('id', '')
                    
                    if server_map_id:
                        try:
                            # Convert hex string like "0010" to int 0x10
                            map_id = int(server_map_id, 16)
                            if self.step_count % 100 == 0:
                                logger.info(f"🗺️ GROUND TRUTH MAP_ID: {map_id:#06x} from server")
                        except (ValueError, TypeError):
                            pass
                    
                    # Fallback to location string inference if server didn't provide map_id
                    if not server_map_id:
                        location_to_map = {
                            'MAYS HOUSE 2F': 0x103,  # RIVAL BEDROOM
                            'MAYS HOUSE 1F': 0x102,  # Rival house 1F
                            'BRENDANS HOUSE 2F': 0x101,  # Player bedroom
                            'BRENDANS HOUSE 1F': 0x100,  # Player house 1F
                            'PROFESSOR': 0x104, 'BIRCH': 0x104, 'LAB': 0x104,
                            'POKEMON CENTER': 0x200, 'POKECENTER': 0x200,
                            'MART': 0x202,
                            'ROUTE 103': 0x12, 'ROUTE 102': 0x0F, 'ROUTE 101': 0x10,
                            'OLDALE': 0x11, 'LITTLEROOT': 0x09,
                        }
                        for key, mid in location_to_map.items():
                            if key in location_str:
                                map_id = mid
                                break
                    
                    # Build memory dict for compatibility with oracle
                    game_state = game_data.get('game_state', 'overworld')
                    
                    # Get party size from progress_context (new server structure)
                    progress_ctx = game_data.get('progress_context', {})
                    party_size = progress_ctx.get('party_size', 0)
                    has_pokedex = progress_ctx.get('has_pokedex', False)
                    self.has_pokedex_flag = has_pokedex  # Store for forced action override
                    
                    memory = {
                        'x': x,
                        'y': y,
                        'map_id': map_id,
                        'location': player.get('location', 'unknown'),  # Add readable location name
                        'game_state': game_state,  # Track if in dialog/battle/menu
                        'party_size': party_size,  # From progress_context
                        'party': player.get('party', []),  # Full party data for HP checks!
                        'has_pokedex': has_pokedex,  # Track if player has Pokedex
                        'badge_count': len(game_data.get('badges', [])),
                        'money': game_data.get('money', 0),
                        'dialogue_text': game_data.get('dialogue_text', None),  # Actual NPC text for LLM parsing
                        'milestones': state.get('milestones', {}),  # Include milestones for dialogue processor
                        'player': player,  # Include full player object for gender detection
                        # Map terrain data for dynamic pathfinding (ground truth from server)
                        'map_data': state.get('map', {}),  # Full 15x15 tile grid with behaviors
                        # Dynamic exit/portal connections from server (ground truth!)
                        'connections': state.get('map', {}).get('stitched_map_info', {}).get('current_area', {}).get('connections', []),
                    }
                    
                    # Parse connections from location_connections (new server structure)
                    # Format: {LocName: [[ToLoc, FromPos, ToPos], ...]}
                    current_loc = player.get('location', '')
                    current_loc_upper = current_loc.upper()
                    
                    # ═══════════════════════════════════════════════════════════════════
                    # 🏥 LATE-STAGE POKECENTER OVERRIDE with FRESH coordinates!
                    # ⚠️ DISABLED: Now using pure PPO reward shaping instead of action overrides
                    # This runs AFTER coordinates are fetched from server (line 1586-1589)
                    # PokeCenter OUTDOOR entrance is at (7, 13) - NOT the exit at (6, 17)!
                    # ═══════════════════════════════════════════════════════════════════
                    enable_heuristic_overrides = False  # Disabled for PPO learning
                    if enable_heuristic_overrides and 'OLDALE' in current_loc_upper and 'POKECENTER' not in current_loc_upper and 'HOUSE' not in current_loc_upper and 'MART' not in current_loc_upper:
                        if not getattr(self, 'visited_oldale_pokecenter', False) and not skip_stuck_override:
                            pc_x, pc_y = 7, 13  # PokeCenter OUTDOOR entrance (door on south side)
                            
                            dx = pc_x - x  # -1 means need LEFT, +1 means need RIGHT
                            dy = pc_y - y  # -1 means need UP, +1 means need DOWN
                            
                            # If close to entrance, force specific direction
                            if abs(dx) <= 1 and abs(dy) <= 1:
                                # Very close - pick exact direction
                                if dx != 0:
                                    action_name = 'RIGHT' if dx > 0 else 'LEFT'
                                elif dy != 0:
                                    action_name = 'DOWN' if dy > 0 else 'UP'
                                else:
                                    action_name = 'A'  # At entrance - interact
                                skip_stuck_override = True
                                if self.step_count % 10 == 0:
                                    logger.warning(f"🏥 POKECENTER OVERRIDE: At ({x},{y}), target=({pc_x},{pc_y}), action={action_name}")
                            # If on wrong row, move vertically first
                            elif abs(dy) > 1:
                                action_name = 'DOWN' if dy > 0 else 'UP'
                                skip_stuck_override = True
                            # If on wrong column, move horizontally
                            elif abs(dx) > 1:
                                action_name = 'RIGHT' if dx > 0 else 'LEFT'
                                skip_stuck_override = True
                    
                    all_conns = state.get('location_connections', {})
                    
                    # Case-insensitive lookup
                    raw_conns = []
                    for k, v in all_conns.items():
                        if k.upper() == current_loc_upper:
                            raw_conns = v
                            break
                    
                    parsed_conns = []
                    for rc in raw_conns:
                        if isinstance(rc, list) and len(rc) >= 3:
                            parsed_conns.append({
                                'to': rc[0],
                                'from_pos': rc[1],
                                'to_pos': rc[2]
                            })
                    memory['connections'] = parsed_conns

                    
                    # Define rival_battled for local scope use (needed for navigation logic)
                    # Define rival_battled for local scope use (needed for navigation logic)
                    milestones = state.get('milestones', {})
                    # Server returns milestones as dict with milestone names as KEYS, not in 'completed' field!
                    # Check if it's the old format (has 'completed' key) or new format (milestone names are keys)
                    if isinstance(milestones, dict) and 'completed' not in milestones:
                        # New format: milestone names are the keys
                        completed = list(milestones.keys())
                    else:
                        # Old format or fallback
                        completed = milestones.get('completed', [])

                    
                    # Local override for immediate responsiveness (server lag compensation)
                    if not hasattr(self, 'completed_first_rival_battle_locally'):
                         self.completed_first_rival_battle_locally = False
                    
                    # NOTE: Don't infer from ROUTE_103 - just stepping there doesn't mean battle is done
                    # FIXED: Don't trust checkpoint milestone - it may be from a previous game save!
                    # Only use LOCAL detection from battle->overworld transition
                    rival_battled = self.completed_first_rival_battle_locally  # Don't trust checkpoint milestone!
                    
                    # CRITICAL: Check pokedex_seen from server
                    # NOTE: pokedex_seen is updated AFTER receiving Pokédex at Birch's Lab!
                    # So when FIRST_RIVAL_BATTLE milestone is complete, we should go to Lab, not re-battle!
                    try:
                        milestones_api = requests.get(f"{self.server_url}/milestones", timeout=5).json()
                        pokedex_seen = milestones_api.get('pokedex_seen', 0) if isinstance(milestones_api, dict) else 0
                    except:
                        pokedex_seen = 0  # Default to 0 if can't fetch
                    
                    # FIXED: When FIRST_RIVAL_BATTLE milestone is complete, DON'T override rival_battled!
                    # The agent should go to Birch's Lab to receive Pokédex, not keep fighting rival!
                    if rival_battled and pokedex_seen == 0:
                        # Log only - do NOT override rival_battled!
                        if self.step_count % 200 == 0:
                            logger.info(f"📦 RIVAL BATTLE WON! Now go to Birch's Lab to receive Pokédex (pokedex_seen will update there)")
                        # DO NOT set rival_battled = False - that was the bug!
                    
                    # Check for Milestone Trigger: Winning Rival Battle (Battle -> Overworld transition)
                    if not hasattr(self, 'prev_game_state'): self.prev_game_state = 'overworld'
                    if not hasattr(self, 'last_battle_was_rival'): self.last_battle_was_rival = False
                    
                    # Early HP check for win/loss detection (before milestone trigger)
                    party_data_early = progress_ctx.get('party', [])
                    early_hp_ratio = 1.0
                    if party_data_early and len(party_data_early) > 0:
                        early_total_hp = sum(p.get('hp', 0) for p in party_data_early if p)
                        early_total_max_hp = sum(p.get('max_hp', 1) for p in party_data_early if p) or 1
                        early_hp_ratio = early_total_hp / early_total_max_hp
                    
                    if self.prev_game_state == 'battle' and game_state == 'overworld':
                         if self.last_battle_was_rival and not rival_battled:
                             # Only trigger milestone if agent WON (HP > 0)
                             if early_hp_ratio > 0:
                                 logger.info("🎉 RIVAL BATTLE WON! Triggering FIRST_RIVAL_BATTLE milestone!")
                                 self.completed_first_rival_battle_locally = True
                                 rival_battled = True
                             else:
                                 # Agent LOST the battle - must return to Route 103 after healing
                                 logger.warning("💀 RIVAL BATTLE LOST! Must return to Route 103 after healing.")
                                 self.last_battle_was_rival = False  # Reset so we can try again
                    
                    # ═══════════════════════════════════════════════════════════════════
                    # 🔍 RIVAL-ABSENCE CHECK - DISABLED: Causes false positives!
                    # The FIRST_RIVAL_BATTLE milestone can come from checkpoint, not current game.
                    # Real battle detection should use game state transition (battle->overworld)
                    # at lines 2011-2021 above.
                    # ═══════════════════════════════════════════════════════════════════
                    if not hasattr(self, 'steps_on_route103_without_battle'):
                        self.steps_on_route103_without_battle = 0
                    
                    # Get location directly from memory (location_upper not defined yet!)
                    current_location = (memory.get('location', '') or '').upper()
                    self.current_location = current_location  # Store for forced action override at step start
                    on_route_103 = map_id == 0x12 or ('ROUTE' in current_location and '103' in current_location)
                    has_rival_milestone = 'FIRST_RIVAL_BATTLE' in completed
                    
                    if on_route_103 and game_state == 'overworld':
                        self.steps_on_route103_without_battle += 1
                        
                        # DISABLED - this check causes false positives when resuming from checkpoint
                        # The milestone may exist from a different game save or previous run
                        # Trust only the battle->overworld transition detection for setting rival_battled
                        # if has_rival_milestone and not self.completed_first_rival_battle_locally:
                        #     if self.steps_on_route103_without_battle >= 30:
                        #         logger.info("🔍 RIVAL-ABSENCE: On Route 103 with FIRST_RIVAL_BATTLE milestone - battle WON! Go to Birch's Lab for Pokédex!")
                        #         self.completed_first_rival_battle_locally = True
                        #         rival_battled = True
                        pass
                    else:
                        # Reset counter when not on Route 103
                        self.steps_on_route103_without_battle = 0

                    
                    self.prev_game_state = game_state

                    # ═══════════════════════════════════════════════════════════════════
                    # 🏥 FORCED POKECENTER NURSE OVERRIDE - HIGHEST PRIORITY #
                    # When inside PokeCenter and haven't talked to Nurse Joy yet,
                    # FORCE UP until at counter (y<=4), then FORCE A to talk!
                    # This runs with FRESH location data (after server response)
                    # ═══════════════════════════════════════════════════════════════════
                    force_nurse_action = None
                    # DEBUG: Log current_location to see why override isn't triggering
                    if self.step_count % 10 == 0:
                        logger.info(f"🔍 NURSE DEBUG: current_loc='{current_location}', y={y}, visited={getattr(self, 'visited_oldale_pokecenter', False)}")
                    if ('POKECENTER' in current_location or 'POKEMON CENTER' in current_location) and '2F' not in current_location:
                        if not getattr(self, 'visited_oldale_pokecenter', False):
                            # INTERNAL POKECENTER LAYOUT:
                            # - Entry at bottom: y=16-17 (internal coords)
                            # - Exit warp at x>=7, y=8 EXACTLY
                            # - Nurse Joy counter at y<=4 (top)
                            # - WARP from x=6 going UP leads to exit at (7,8)!
                            # - y DECREASES going UP (north)
                            # - Need to be at x<=5 to safely go UP!
                            # USER CLARIFICATION: Agent needs to be at (7, 4) to talk to Nurse Joy at (7, 3)
                            # TERRAIN A* will handle pathfinding - just press A when at target
                            if x == 7 and y == 4:
                                # At Nurse Joy interaction point (7,4) - press A to talk!
                                force_nurse_action = 'A'
                                logger.warning(f"💊 FORCED NURSE JOY: At counter (7,4), pressing A!")
                            # Let terrain A* handle navigation, just set flag for next step
                            # Do NOT override terrain A* pathfinding here
                    
                    # Apply forced action if set - SET FLAG FOR NEXT STEP!
                    # Late-stage runs AFTER action sent, so we save for next step
                    if force_nurse_action:
                        self.force_nurse_next_action = force_nurse_action
                        # Also set coords for logging
                        self.force_nurse_coords = (x, y)


                    
                    # ═══════════════════════════════════════════════════════════════════
                    # ⚡ HEALING INTERRUPT LAYER - CHECKED FIRST BEFORE ALL OTHER LOGIC ⚡
                    # Uses SNN priority thinking + KNN for nearest healing location
                    # ═══════════════════════════════════════════════════════════════════
                    
                    # Get HP ratio from party state (O(1) check - minimal overhead)
                    party_data = progress_ctx.get('party', [])
                    if party_data and len(party_data) > 0:
                        total_hp = sum(p.get('hp', 0) for p in party_data if p)
                        total_max_hp = sum(p.get('max_hp', 1) for p in party_data if p) or 1
                        hp_ratio = total_hp / total_max_hp
                    else:
                        hp_ratio = 1.0  # No party = assume full HP (early game before starter)
                    self.cached_hp_ratio = hp_ratio  # Cache for terrain override HP check
                    
                    # SNN PRIORITY THRESHOLDS (stored for SNN input)
                    hp_warning = hp_ratio < 0.50    # 50% - start heading to heal
                    hp_critical = hp_ratio < 0.30   # 30% - interrupt all tasks!
                    hp_emergency = hp_ratio < 0.15  # 15% - maximum priority
                    
                    # CRITICAL: Reset visited_oldale_pokecenter if HP drops below 50%!
                    # This handles the case where flag is True from checkpoint but HP is low
                    # Agent needs to re-heal, so reset flag so interior navigation runs
                    if hp_warning and getattr(self, 'visited_oldale_pokecenter', False):
                        if not getattr(self, '_hp_reset_logged', False):
                            logger.warning(f"🩺 HP LOW ({hp_ratio*100:.0f}%)! Resetting visited_oldale_pokecenter for re-healing")
                            self._hp_reset_logged = True
                        self.visited_oldale_pokecenter = False
                        self.nurse_dialog_counter = 0  # Reset dialog counter too
                    elif hp_ratio >= 0.50:
                        self._hp_reset_logged = False  # Reset the log flag when HP is healthy
                    
                    # Store HP state for SNN input encoding
                    memory['hp_ratio'] = hp_ratio
                    memory['hp_warning'] = hp_warning
                    memory['hp_critical'] = hp_critical
                    memory['hp_emergency'] = hp_emergency
                    
                    # KNN: Find nearest healing location
                    # CACHE LAST KNOWN LOCATION (for map buffer corruption recovery)
                    location = memory.get('location', '')
                    if location:  # Valid location - cache it
                        self._last_known_location = location
                        self._last_known_map_id = map_id
                    elif hasattr(self, '_last_known_location') and self._last_known_location:
                        # Use cached location when current is None
                        location = self._last_known_location
                        if self.step_count % 100 == 0:
                            logger.warning(f"⚠️ MAP CORRUPTION: Using cached location '{location}'")
                    
                    location_upper = (location or '').upper()
                    healing_interrupt_active = False
                    healing_target = None
                    
                    if hp_critical and game_state != 'battle':  # Don't interrupt during battle!
                        healing_interrupt_active = True
                        
                        # KNN-style nearest healing location
                        if 'POKECENTER' in location_upper:
                            # Already in PokeCenter! Go to Nurse Joy
                            healing_target = {'type': 'NURSE_JOY', 'action': 'UP_THEN_A', 'coords': (7, 3)}
                        elif 'MAY' in location_upper or 'BRENDAN' in location_upper:
                            # GENDER-AWARE: Check if this is actually player's house
                            # Girl: May's = player's, Brendan's = rival's
                            # Boy: Brendan's = player's, May's = rival's
                            player_gender_heal = memory.get('player', {}).get('gender', 'Boy')
                            is_player_house = False
                            if player_gender_heal == 'Girl' and 'MAY' in location_upper:
                                is_player_house = True
                            elif player_gender_heal != 'Girl' and 'BRENDAN' in location_upper:
                                is_player_house = True
                            
                            if is_player_house:
                                # In player's house - find Mom
                                if '2F' in location_upper or 'BEDROOM' in location_upper:
                                    healing_target = {'type': 'GO_DOWNSTAIRS', 'action': 'DOWN', 'coords': (1, 1)}
                                else:
                                    healing_target = {'type': 'MOM', 'action': 'TALK_TO_MOM', 'coords': (5, 8)}
                            else:
                                # In rival's house - EXIT and go to player's house!
                                healing_target = {'type': 'EXIT_RIVAL_HOUSE', 'action': 'DOWN', 'coords': None}
                        elif 'OLDALE' in location_upper:
                            # In Oldale - go to PokeCenter at (7, 13)
                            healing_target = {'type': 'OLDALE_PC', 'action': 'GO_TO_PC', 'coords': (7, 13)}
                        elif 'ROUTE 103' in location_upper:
                            # Route 103 - go south to Oldale
                            healing_target = {'type': 'GO_SOUTH', 'action': 'DOWN', 'coords': None}
                        elif 'ROUTE 101' in location_upper:
                            # Route 101 - KNN: closer to Oldale (north) or Littleroot (south)?
                            # HYSTERESIS: Once healing direction is chosen, STICK WITH IT until HP restored!
                            # This prevents oscillation at the y=15 boundary
                            cached_heal_dir = getattr(self, '_cached_healing_direction', None)
                            
                            if cached_heal_dir and hp_ratio < 0.50:
                                # Use cached direction (hysteresis) until HP restored
                                healing_target = cached_heal_dir
                                if self.step_count % 50 == 0:
                                    logger.info(f"🔄 HYSTERESIS: Sticking with cached healing dir: {cached_heal_dir['type']}")
                            else:
                                # First time choosing OR HP restored - pick based on current position
                                if y < 15:  # Closer to Oldale
                                    healing_target = {'type': 'OLDALE_PC', 'action': 'GO_NORTH', 'coords': None}
                                else:  # Closer to Littleroot/Mom
                                    healing_target = {'type': 'MOM', 'action': 'GO_SOUTH', 'coords': None}
                                # Cache this direction for hysteresis
                                self._cached_healing_direction = healing_target
                                logger.info(f"🎯 HEALING DIR CHOSEN: {healing_target['type']} (y={y})")
                        elif 'LITTLEROOT' in location_upper:
                            # In Littleroot - go to player's house for Mom
                            # GENDER-AWARE: Girl's house is May's (14,9), Boy's is Brendan's (8,8)
                            player_gender_heal = memory.get('player', {}).get('gender', 'Boy')
                            if player_gender_heal == 'Girl':
                                # May's house door
                                healing_target = {'type': 'PLAYER_HOUSE', 'action': 'ENTER_HOUSE', 'coords': (14, 9)}
                            else:
                                # Brendan's house door
                                healing_target = {'type': 'PLAYER_HOUSE', 'action': 'ENTER_HOUSE', 'coords': (8, 8)}
                        else:
                            # Default: head south to find healing
                            healing_target = {'type': 'FIND_HEALING', 'action': 'DOWN', 'coords': None}
                        
                        if self.step_count % 30 == 0:
                            logger.info(f"🚨 HEALING INTERRUPT! HP={hp_ratio:.1%} Target={healing_target['type']}")
                    
                    memory['healing_interrupt'] = healing_interrupt_active
                    memory['healing_target'] = healing_target
                    
                    # Cache for UNIVERSAL HEALING OVERRIDE in process_action()
                    self.cached_healing_interrupt = healing_interrupt_active
                    self.cached_healing_target = healing_target
                    
                    # ═══════════════════════════════════════════════════════════════════
                    # 🚪 FORCED HOUSE EXIT - DISABLED - Now handled by directive SLAM pathfinding
                    # The directive system (EXIT_HOUSE handler at line ~2790) uses:
                    # - Portal connections (real-time ground truth)
                    # - Terrain data (warp tiles)
                    # - SLAM pathfinding with get_best_direction()
                    # This old hardcoded system had wrong coordinates and is obsolete.
                    # ═══════════════════════════════════════════════════════════════════
                    
                    # Mark that we're in a house (for other systems) but DON'T provide forced exit target
                    has_starter = party_size > 0 or 'STARTER_CHOSEN' in str(memory.get('milestones', {}))
                    in_house = 'HOUSE' in location_upper or 'BEDROOM' in location_upper or '1F' in location_upper or '2F' in location_upper
                    in_lab = 'LAB' in location_upper or 'BIRCH' in location_upper
                    in_pokecenter = map_id == 0x200 or 'POKEMON CENTER' in location_upper or 'POKECENTER' in location_upper
                    in_pokecenter_2f = 'POKEMON CENTER' in location_upper and '2F' in location_upper
                    in_mart = map_id == 0x202 or (map_id == 0 and 'MART' in location_upper) or 'MART' in location_upper
                    
                    force_house_exit = has_starter and in_house and not in_lab and not in_pokecenter
                    house_exit_target = None  # DISABLED - let directive system handle it
                    
                    # Old hardcoded pathfinding removed - see directive EXIT_HOUSE handler for replacement
                    
                    memory['force_house_exit'] = force_house_exit
                    memory['house_exit_target'] = house_exit_target  # Will be None
                    
                    # ═══════════════════════════════════════════════════════════════════
                    # 🗺️ GLOBAL TERRAIN HELPER - Used by ALL maps for ground truth navigation
                    # Parses visual_map ASCII (arrows=ledges, #=obstacles) + metatile_info
                    # ═══════════════════════════════════════════════════════════════════
                    
                    # Get terrain data ONCE for all map handlers
                    map_data = memory.get('map_data', {})
                    metatile_info = map_data.get('metatile_info', [])
                    
                    # Cache position and terrain for HOUSE OVERRIDE (used at start of next step)
                    self.cached_position = (x, y, map_id)
                    self.cached_metatile = metatile_info
                    
                    def is_direction_walkable(direction):
                        """GLOBAL: Check if moving in direction is blocked by terrain
                        
                        Uses ground truth from server:
                        - visual_map ASCII: arrows (↓↘←→) = ledges, # = obstacles, . = passable
                        - metatile_info: IMPASSABLE, JUMP_* behaviors
                        
                        Works for ALL maps, not just Route 101!
                        """
                        if not metatile_info or len(metatile_info) < 15:
                            return None  # No terrain data available
                        
                        # Player is at center of 15x15 grid = position (7, 7)
                        center = 7
                        offsets = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0)}
                        dx, dy = offsets.get(direction, (0, 0))
                        
                        try:
                            tile_info = metatile_info[center + dy][center + dx]
                            behavior = tile_info.get('behavior', 'NORMAL')
                            passable = tile_info.get('passable', True)
                            
                            # === ALSO CHECK VISUAL_MAP for arrow symbols (ground truth!) ===
                            visual_map = map_data.get('visual_map', '')
                            if visual_map:
                                lines = visual_map.split('\n')
                                # Find player 'P' in visual_map to get relative position
                                for line_idx, line in enumerate(lines):
                                    if 'P' in line:
                                        p_col = line.find('P')
                                        # Check adjacent tile for arrow
                                        target_line_idx = line_idx + dy
                                        target_col = p_col + dx * 2  # ASCII uses 2 chars per tile
                                        if 0 <= target_line_idx < len(lines) and 0 <= target_col < len(lines[target_line_idx]):
                                            char = lines[target_line_idx][target_col:target_col+1]
                                            # Ledge arrows - BLOCK going UP against DOWN ledge
                                            if char in '↓vV' and direction == 'UP':
                                                return False  # Can't climb UP a DOWN ledge!
                                            if char in '↑^' and direction == 'DOWN':
                                                return False  # Can't climb DOWN an UP ledge!
                                            if char in '←<' and direction == 'RIGHT':
                                                return False  # Can't go RIGHT into LEFT ledge!
                                            if char in '→>' and direction == 'LEFT':
                                                return False  # Can't go LEFT into RIGHT ledge!
                                            if char in '↘' and direction in ['UP', 'LEFT']:
                                                return False  # Can't climb SE ledge from N or W!
                                            if char in '↙' and direction in ['UP', 'RIGHT']:
                                                return False  # Can't climb SW ledge!
                                            if char == '#':
                                                return False  # Obstacle!
                                        break
                            
                            # === IMPASSABLE (Trees, walls) - totally blocked ===
                            if 'IMPASSABLE' in behavior:
                                return False
                            
                            # === LEDGES (arrows in ground truth) - one-way only! ===
                            if 'JUMP_SOUTH' in behavior:
                                return direction == 'DOWN'
                            if 'JUMP_NORTH' in behavior:
                                return direction == 'UP'
                            if 'JUMP_EAST' in behavior:
                                return direction == 'RIGHT'
                            if 'JUMP_WEST' in behavior:
                                return direction == 'LEFT'
                            if 'JUMP_SOUTHEAST' in behavior:
                                return direction in ['DOWN', 'RIGHT']
                            if 'JUMP_SOUTHWEST' in behavior:
                                return direction in ['DOWN', 'LEFT']
                            if 'JUMP' in behavior:
                                return direction == 'DOWN'  # Assume south-facing
                            
                            # === GRASS - walkable! ===
                            if 'GRASS' in behavior:
                                return True
                            
                            return passable
                            
                        except (IndexError, KeyError, TypeError):
                            return None
                    
                    # Check all directions using GLOBAL helper (available to all map handlers!)
                    can_up = is_direction_walkable('UP')
                    can_down = is_direction_walkable('DOWN')
                    can_left = is_direction_walkable('LEFT')
                    can_right = is_direction_walkable('RIGHT')
                    
                    # ═══════════════════════════════════════════════════════════════════

                    # 🧱 ROUTE 101 WAYPOINT NAVIGATION SYSTEM
                    # Comprehensive path planning for Route 101 ledge + tree obstacles.
                    # 
                    # PROBLEM: Ledges block UP, trees block RIGHT. Simple direction
                    # logic fails because the terrain is complex.
                    #
                    # SOLUTION: Waypoint-based navigation with 3 phases:
                    #   Phase 1: Go DOWN to get south of ledge end
                    #   Phase 2: Go LEFT/RIGHT to clear tree obstacles  
                    #   Phase 3: Go UP through the gap
                    #
                    # This uses k-NN episodic memory concepts - store successful
                    # paths and recall them based on position.
                    # ═══════════════════════════════════════════════════════════════════
                    
                    route101_ledge_interrupt = False
                    ledge_target = None
                    
                    if 'ROUTE' in location_upper and '101' in location_upper:
                        # DISABLED: Let terrain A* handle Route 101 navigation using ground truth
                        # The server provides real-time metatile data (. = walkable, # = blocked, etc.)
                        # which is more accurate than hardcoded zone logic
                        route101_ledge_interrupt = False  # Use terrain A* with ground truth!
                        
                        # Uses GLOBAL is_direction_walkable and can_up/can_down/can_left/can_right
                        # defined above (before all map handlers)
                        
                        # Log terrain status every 50 steps for debugging
                        if self.step_count % 50 == 0:
                            logger.info(f"🗺️ Route 101 ({x},{y}) Terrain: UP={can_up} DOWN={can_down} LEFT={can_left} RIGHT={can_right}")

                        
                        # ═══════════════════════════════════════════════════════════════════
                        # 🗺️ ROUTE 101 WAYPOINT PATHFINDING - RE-ENABLED
                        # Route 101 has complex obstacles (ledges + immovable trees)
                        # Must go: DOWN (get below ledge) → RIGHT (around trees) → UP (through gap)
                        # ═══════════════════════════════════════════════════════════════════
                        
                        # Per Bulbapedia: Route 101 forms a "U" shape:
                        #   1. From Littleroot: Go NORTH
                        #   2. Path turns EAST near first grass patch
                        #   3. Then turns WEST (LEFT) through grass
                        #   4. Finally NORTH to Oldale Town entrance (around x=5-8)
                        #
                        # KEY INSIGHT: Upper-right area (x>=10, y<=5) means we passed
                        # the turn - must go LEFT to find Oldale entrance!
                        # ═══════════════════════════════════════════════════════════════════
                        
                        # GROUND TRUTH ZONES
                        # Route 101 has multiple zones - must handle each correctly:
                        # - Upper-right (x>=10, y<=8): Passed the turn, go LEFT!
                        # - MIDDLE LEDGE TRAP (x=7-10, y=6-9): Go DOWN to get below ledge!
                        # - Left side / Oldale path (x<=8, y<=10): Go UP to Oldale!  
                        # - Lower area (y>15): Go UP through grass
                        
                        # PRIORITY OVERRIDE: If has Pokedex, ALWAYS go UP to Oldale (then Route 102)
                        # This bypasses ledge escape logic since we need to go NORTH
                        has_pokedex = memory.get('has_pokedex', False)
                        
                        # Ledge trap is ONLY on the right side (x>=9) where you might jump down
                        # Central corridor (x=7-8) is CLEAR - UP works!
                        # Left path (x<=6) has its own navigation
                        in_middle_ledge = x >= 9 and 6 <= y <= 9  # LEDGE TRAP - narrowed bounds
                        in_upper_right = x >= 10 and y <= 8   # Passed the turn, go LEFT!
                        on_left_oldale_path = x <= 8 and y <= 10  # Left side path to Oldale - go UP!
                        at_oldale_entrance = y <= 3  # Very close to Oldale, go UP!
                        
                        # POKEDEX OVERRIDE: Already visited lab, need to go NORTH to Oldale->Route102
                        # BUT: Must respect terrain! If UP is blocked, use terrain result
                        # CRITICAL: If in upper-right, must go LEFT first (passed the turn!)
                        # CRITICAL: If in middle ledge trap, MUST go DOWN to escape first!
                        if has_pokedex:
                            if in_upper_right and can_left:
                                # PASSED THE TURN! Must go LEFT (west) to find Oldale entrance
                                ledge_target = {'action': 'LEFT', 'reason': f'Has Pokedex but at x={x}>=10! Go LEFT to find Oldale entrance!'}
                                if self.step_count % 30 == 0:
                                    logger.info(f"🚨 UPPER RIGHT ({x},{y}): Must go LEFT to reach Oldale entrance!")
                            elif in_middle_ledge and can_down:
                                # IN LEDGE TRAP! Must go DOWN to escape before going UP to Oldale
                                ledge_target = {'action': 'DOWN', 'reason': f'Has Pokedex but STUCK at y={y}! GO DOWN to escape ledge trap!'}
                                if self.step_count % 20 == 0:
                                    logger.info(f"🚨 LEDGE ESCAPE: At ({x},{y}) in middle ledge - must go DOWN first!")
                            elif can_up:
                                ledge_target = {'action': 'UP', 'reason': f'Has Pokedex! Go UP to Oldale->Route102'}
                            elif can_left:
                                ledge_target = {'action': 'LEFT', 'reason': f'Has Pokedex! UP blocked, go LEFT'}
                            elif can_right:
                                ledge_target = {'action': 'RIGHT', 'reason': f'Has Pokedex! UP blocked, go RIGHT'}
                            else:
                                # Let TerrainNavigator A* handle complex path
                                ledge_target = None  # Disable zone override, use A*

                        elif at_oldale_entrance:


                            # Very close to Oldale entrance - just go UP!
                            ledge_target = {'action': 'UP', 'reason': f'At Oldale entrance y={y} - GO UP!'}
                        
                        elif in_middle_ledge:
                            # Middle ledge zone (x>=9, y=6-9) with trees blocking
                            # DEFER TO TERRAIN A* - it has real metatile data!
                            # Only fall back to hardcoded if terrain nav fails
                            # The A* will find path around ledge/trees better than hardcoded logic
                            ledge_target = None  # Disable waypoint, let terrain nav handle it
                            if self.step_count % 100 == 0:
                                logger.info(f"🚧 LEDGE ZONE ({x},{y}): Deferring to terrain A* for navigation")
                         
                        elif on_left_oldale_path:
                            # Left side path to Oldale (x<=8, y<=10)
                            # DEFER TO TERRAIN A* - it knows about trees and ledges!
                            # The A* will find the correct path around obstacles
                            ledge_target = None  # Let terrain nav handle it
                            if self.step_count % 100 == 0:
                                logger.info(f"🚧 LEFT PATH ({x},{y}): Deferring to terrain A* for navigation")
                        
                        elif in_upper_right:
                            # Upper-right area (x>=10, y<=8)
                            # DEFER TO TERRAIN A* - it can find path around trees/obstacles
                            ledge_target = None
                            if self.step_count % 100 == 0:
                                logger.info(f"🚧 UPPER-RIGHT ({x},{y}): Deferring to terrain A* for navigation")
                        
                        # NOTE: If ledge_target is already set (from zones above), skip fallback
                        # If ledge_target is None from zone, KEEP it None to defer to terrain A*!
                        # Only use fallback if no zone matched at all
                        defer_to_terrain = (in_middle_ledge or on_left_oldale_path or in_upper_right)
                        
                        if not defer_to_terrain and ledge_target is None:
                            # FALLBACK: Only when no zone matched - use simple heuristics
                            if can_up is True:
                                ledge_target = {'action': 'UP', 'reason': 'Terrain: UP clear - GO NORTH!'}
                            elif can_up is False:
                                # UP blocked - navigate around
                                if x >= 8:
                                    if can_left is True:
                                        ledge_target = {'action': 'LEFT', 'reason': 'UP blocked, go LEFT'}
                                    elif can_right is True:
                                        ledge_target = {'action': 'RIGHT', 'reason': 'Try RIGHT'}
                                    else:
                                        ledge_target = {'action': 'DOWN', 'reason': 'Try DOWN'}
                                else:
                                    if can_right is True:
                                        ledge_target = {'action': 'RIGHT', 'reason': 'UP blocked, go RIGHT'}
                                    elif can_down is True:
                                        ledge_target = {'action': 'DOWN', 'reason': 'Try DOWN'}
                                    else:
                                        ledge_target = {'action': 'UP', 'reason': 'Force UP'}
                            elif can_up is None:
                                # No terrain data - use position fallback
                                if x >= 10:
                                    ledge_target = {'action': 'LEFT', 'reason': 'No terrain, x>=10 - go LEFT'}
                                else:
                                    ledge_target = {'action': 'UP', 'reason': 'No terrain - go UP'}
                        
                        if route101_ledge_interrupt and ledge_target and self.step_count % 30 == 0:
                            logger.info(f"🧱 TERRAIN NAV! Pos=({x},{y}) -> {ledge_target['action']} - {ledge_target['reason']}")
                    
                    memory['route101_ledge_interrupt'] = route101_ledge_interrupt
                    memory['ledge_target'] = ledge_target
                    
                    # ═══════════════════════════════════════════════════════════════════
                    # 🏥 OLDALE TOWN: HARDCODED POKECENTER NAVIGATION
                    # Fast, reliable - just go to coordinates (7, 13)
                    # ═══════════════════════════════════════════════════════════════════
                    
                    oldale_pokecenter_nav = False
                    oldale_target = None
                    
                    # Check if in outdoor Oldale (not inside any building)
                    in_oldale_outdoor = 'OLDALE' in location_upper and \
                                        'POKECENTER' not in location_upper and \
                                        'POKEMON CENTER' not in location_upper and \
                                        'CENTER' not in location_upper and \
                                        '2F' not in location_upper and \
                                        '1F' not in location_upper and \
                                        'HOUSE' not in location_upper and \
                                        'MART' not in location_upper
                    
                    if in_oldale_outdoor:

                        # Check if we already visited PokeCenter
                        if not getattr(self, 'visited_oldale_pokecenter', False):
                            oldale_pokecenter_nav = True
                            
                            # PokeCenter entrance is at (6, 17) in Oldale Town - SERVER GROUND TRUTH
                            pc_x, pc_y = 6, 17
                            
                            # USE TERRAIN_NAV A* PATHFINDING with obstacle awareness!
                            # This uses metatile_info to navigate around buildings
                            metatile_info = getattr(self, 'cached_metatile', None)
                            if hasattr(self, 'terrain_nav') and metatile_info:
                                nav_result = self.terrain_nav.get_direction(
                                    map_id=map_id,
                                    x=x, y=y,
                                    goal=(pc_x, pc_y),
                                    metatile_info=metatile_info
                                )
                                if nav_result:
                                    oldale_target = nav_result
                                    if self.step_count % 30 == 0:
                                        logger.info(f"🏥 OLDALE A* NAV! Pos=({x},{y}) -> {oldale_target['action']} - {oldale_target['reason']}")
                                else:
                                    # A* failed - fallback to simple direction
                                    if y < pc_y:
                                        oldale_target = {'action': 'DOWN', 'reason': f'A* failed, simple DOWN to ({pc_x},{pc_y})'}
                                    else:
                                        oldale_target = {'action': 'UP', 'reason': f'A* failed, simple UP to ({pc_x},{pc_y})'}
                                    logger.warning(f"⚠️ OLDALE A* FAILED at ({x},{y}), using simple fallback")
                            else:
                                # No terrain_nav - use simple direction
                                if y < pc_y:
                                    oldale_target = {'action': 'DOWN', 'reason': f'No A*, simple DOWN to ({pc_x},{pc_y})'}
                                else:
                                    oldale_target = {'action': 'UP', 'reason': f'No A*, simple UP to ({pc_x},{pc_y})'}
                            
                            # At PokeCenter door? Go UP to enter!
                            if abs(x - pc_x) <= 1 and abs(y - pc_y) <= 1:
                                oldale_target = {'action': 'UP', 'reason': 'At PokeCenter door - ENTER!'}
                    
                    memory['oldale_pokecenter_nav'] = oldale_pokecenter_nav
                    memory['oldale_target'] = oldale_target
                    # Cache for STUCK OVERRIDE access at step start
                    if oldale_target:
                        self._cached_oldale_target = oldale_target
                    
                    # ═══════════════════════════════════════════════════════════════════
                    # 💊 POKECENTER INTERIOR: TALK TO NURSE JOY
                    # When inside PokeCenter, go UP to counter and press A to heal
                    # Nurse Joy is at the counter (around x=7, y=3)
                    # Use map_id 0x200 for Oldale PokeCenter interior (not string match!)
                    # ═══════════════════════════════════════════════════════════════════
                    
                    nurse_joy_nav = False
                    nurse_target = None
                    
                    # map_id 0x200 = Oldale PokeCenter INTERIOR (ground truth)
                    # BUT: Disable on 2F - nurse is on 1F only!
                    if map_id == 0x200 and not in_pokecenter_2f:
                        nurse_joy_nav = True
                        
                        # Nurse Joy counter is at (7, 3) - AGENT NEEDS TO BE AT (7, 4) TO INTERACT!
                        # ESCALATOR to 2F is on LEFT side at x=1-2 - AVOID!
                        nurse_x, nurse_y = 7, 4  # Target (7,4) - one tile in front of counter
                        escalator_danger_zone = x <= 3  # Too far left = escalator danger!
                        
                        if escalator_danger_zone:
                            # PRIORITY: Get away from escalator area!
                            nurse_target = {'action': 'RIGHT', 'reason': f'DANGER: Near 2F escalator at x={x}! Go RIGHT to nurse area!'}
                        elif y > nurse_y + 1:
                            # Need to go UP to reach counter (but not too far left)
                            if x < 5:
                                # Go RIGHT first to avoid escalator path
                                nurse_target = {'action': 'RIGHT', 'reason': f'Go RIGHT to center before approaching nurse'}
                            else:
                                nurse_target = {'action': 'UP', 'reason': f'Go UP to Nurse Joy counter (at y={nurse_y})'}
                        elif abs(x - nurse_x) > 1:
                            # Need to move horizontally to align with counter
                            if x < nurse_x:
                                nurse_target = {'action': 'RIGHT', 'reason': f'Go RIGHT to Nurse Joy (at x={nurse_x})'}
                            else:
                                nurse_target = {'action': 'LEFT', 'reason': f'Go LEFT to Nurse Joy (at x={nurse_x})'}
                        else:
                            # At or near counter - press A to talk!
                            nurse_target = {'action': 'A', 'reason': 'At Nurse Joy counter - press A to heal!'}
                        
                        if self.step_count % 30 == 0:
                            logger.info(f"💊 NURSE JOY NAV! Pos=({x},{y}) -> {nurse_target['action']} - {nurse_target['reason']}")
                    
                    memory['nurse_joy_nav'] = nurse_joy_nav
                    memory['nurse_target'] = nurse_target

                    
                    # ═══════════════════════════════════════════════════════════════════
                    # 🗺️ ROUTE 103: TERRAIN-BASED NAVIGATION TO FIND RIVAL
                    # Uses ground truth terrain (ledges) + RL for exploration
                    # ═══════════════════════════════════════════════════════════════════
                    
                    route103_nav = False
                    route103_target = None
                    
                    if 'ROUTE' in location_upper and '103' in location_upper and not rival_battled:
                        # ONLY enable rival-finding navigation if rival NOT yet beaten!
                        # After battle won, terrain_goal at line 2051+ handles going back to Oldale
                        
                        # PRIORITY 1: Check HP - if too low, go back to Oldale PokeCenter!
                        route103_hp_low = False
                        try:
                            party_data = memory.get('party', []) or progress_ctx.get('party', [])
                            if party_data and len(party_data) > 0:
                                lead = party_data[0]
                                hp = lead.get('current_hp', lead.get('hp', 0))
                                max_hp = lead.get('max_hp', 1) or 1
                                hp_pct = (hp / max_hp) * 100
                                if hp_pct < 50:
                                    route103_hp_low = True
                                    if self.step_count % 50 == 0:
                                        logger.warning(f"🩺 ROUTE 103: HP too low ({hp_pct:.0f}%)! Redirecting to Oldale PokeCenter!")
                        except:
                            pass
                        
                        if route103_hp_low:
                            # Override: Go to Oldale to heal (entrance is SOUTH at y > 10)
                            route103_target = {'action': 'DOWN', 'reason': f'HP LOW! Go to Oldale PokeCenter to heal!'}
                            route103_nav = True
                        else:
                            route103_nav = True

                        
                        # Use terrain data for ledge detection
                        map_data = memory.get('map_data', {})
                        metatile_info = map_data.get('metatile_info', [])
                        
                        # Route 103 Layout:
                        # - Entrance from Oldale is on WEST side (x~0-5)
                        # - Rival (May/Brendan) is on EAST side (x~25-30)
                        # - LEDGES block upward movement in middle areas
                        # - Must go RIGHT (east) staying on lower path to reach rival
                        rival_x = 28  # Rival location approximately
                        
                        # Check terrain in desired direction
                        def check_route103_terrain(direction):
                            if not metatile_info or len(metatile_info) < 15:
                                return None
                            center = 7
                            offsets = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0)}
                            dx, dy = offsets.get(direction, (0, 0))
                            try:
                                tile = metatile_info[center + dy][center + dx]
                                behavior = tile.get('behavior', 'NORMAL')
                                passable = tile.get('passable', True)
                                
                                # Impassable tiles
                                if 'IMPASSABLE' in behavior:
                                    return False
                                # Ledges - one way only (can only jump DOWN from ledge)
                                if 'JUMP_SOUTH' in behavior:
                                    return direction == 'DOWN'
                                if 'JUMP' in behavior:
                                    return direction == 'DOWN'
                                if 'GRASS' in behavior:
                                    return True
                                return passable
                            except:
                                return None
                        
                        can_right = check_route103_terrain('RIGHT')
                        can_left = check_route103_terrain('LEFT')
                        can_up = check_route103_terrain('UP')
                        can_down = check_route103_terrain('DOWN')
                        
                        # ═══════════════════════════════════════════════════════════════
                        # ROUTE 103 ZONES (similar to Route 101 logic)
                        # 
                        # Zone 1: WEST ENTRANCE (x < 8) - Just entered from Oldale
                        #         Priority: Go RIGHT to reach rival
                        #
                        # Zone 2: LEDGE TRAP (8 <= x <= 20, y < 8) - Ledges block path
                        #         Priority: Go DOWN to get below ledges, then RIGHT
                        #
                        # Zone 3: EAST RIVAL AREA (x > 20) - Rival location
                        #         Priority: Search for rival, press A to battle
                        # ═══════════════════════════════════════════════════════════════
                        # ROUTE 103 ZONES - CORRECTED LAYOUT
                        # 
                        # Entrance from Oldale is on EAST (RIGHT) side (x > 18)
                        # Rival is on WEST (LEFT) side (x < 10, near top y < 8)
                        # 
                        # Zone 1: EAST ENTRANCE (x > 18) - Just entered from Oldale
                        #         Priority: Go LEFT toward rival
                        #
                        # Zone 2: MIDDLE AREA (10 <= x <= 18) - Navigate through
                        #         Priority: Go LEFT, avoid ledges by going down
                        #
                        # Zone 3: WEST RIVAL AREA (x < 10) - Rival location (top-left)
                        #         Priority: Go UP to find rival, press A to battle
                        # ═══════════════════════════════════════════════════════════════
                        
                        in_east_entrance = x > 18  # Just entered from Oldale
                        in_middle_area = 15 <= x <= 18  # Narrower middle zone
                        in_rival_area = x < 15  # Rival is slight left/center (not far left)
                        
                        if in_rival_area:
                            # In rival area (slight left/center) - search for rival
                            # Rival is toward left-center top, prioritize UP then LEFT
                            if y > 5:
                                # Need to go UP to reach rival
                                if can_up is True:
                                    route103_target = {'action': 'UP', 'reason': f'RIVAL ZONE ({x},{y}) - go UP to rival!'}
                                elif can_left is True:
                                    route103_target = {'action': 'LEFT', 'reason': f'RIVAL ZONE ({x},{y}) - go LEFT'}
                                else:
                                    route103_target = {'action': 'UP', 'reason': f'Force UP to rival'}
                            else:
                                # Near top - search for rival (prioritize UP and A for talking)
                                if can_up is True:
                                    route103_target = {'action': 'UP', 'reason': f'RIVAL ZONE ({x},{y}) - search UP'}
                                elif can_left is True:
                                    route103_target = {'action': 'LEFT', 'reason': f'RIVAL ZONE ({x},{y}) - search LEFT'}
                                else:
                                    route103_target = {'action': 'A', 'reason': f'RIVAL ZONE ({x},{y}) - talk to rival!'}

                        
                        elif in_middle_area:
                            # Middle area - go LEFT toward rival
                            # Check for ledges (can't go up if ledge is there)
                            if can_up is False and y < 8:
                                # Ledge blocking - go DOWN to get around it
                                if can_down is True:
                                    route103_target = {'action': 'DOWN', 'reason': f'LEDGE BLOCK ({x},{y}) - go DOWN!'}
                                elif can_left is True:
                                    route103_target = {'action': 'LEFT', 'reason': f'LEDGE BLOCK ({x},{y}) - try LEFT'}
                                else:
                                    route103_target = {'action': 'DOWN', 'reason': f'Force DOWN to escape ledge'}
                            else:
                                # Path clear - go LEFT toward rival
                                if can_left is True:
                                    route103_target = {'action': 'LEFT', 'reason': f'MIDDLE ({x},{y}) - go LEFT to rival!'}
                                elif can_up is True:
                                    route103_target = {'action': 'UP', 'reason': f'MIDDLE ({x},{y}) - try UP'}
                                elif can_down is True:
                                    route103_target = {'action': 'DOWN', 'reason': f'MIDDLE ({x},{y}) - go DOWN then LEFT'}
                                else:
                                    route103_target = {'action': 'LEFT', 'reason': f'Force LEFT to rival'}
                        
                        elif in_east_entrance:
                            # Just entered from Oldale (east side) - go LEFT toward rival
                            if can_left is True:
                                route103_target = {'action': 'LEFT', 'reason': f'EAST ENTRANCE ({x},{y}) - go LEFT to rival!'}
                            elif can_up is True:
                                route103_target = {'action': 'UP', 'reason': f'EAST ENTRANCE ({x},{y}) - go UP then LEFT'}
                            elif can_down is True:
                                route103_target = {'action': 'DOWN', 'reason': f'EAST ENTRANCE ({x},{y}) - go DOWN then LEFT'}
                            else:
                                route103_target = {'action': 'LEFT', 'reason': f'Force LEFT towards rival'}
                        
                        else:
                            # Fallback: Standard navigation toward rival (west side)
                            if x > 10:
                                if can_left is True:
                                    route103_target = {'action': 'LEFT', 'reason': f'Go LEFT to rival (x={x})'}
                                elif can_down is True:
                                    route103_target = {'action': 'DOWN', 'reason': f'DOWN to find path to rival'}
                                else:
                                    route103_target = {'action': 'LEFT', 'reason': f'Force LEFT'}
                            else:
                                if can_up is True:
                                    route103_target = {'action': 'UP', 'reason': f'Go UP to rival area'}
                                else:
                                    route103_target = {'action': 'LEFT', 'reason': f'Try LEFT to find rival'}
                        
                        if self.step_count % 30 == 0:
                            logger.info(f"🗺️ ROUTE 103 NAV! Pos=({x},{y}) -> {route103_target['action']} - {route103_target['reason']}")
                    
                    memory['route103_nav'] = route103_nav
                    memory['route103_target'] = route103_target


                    
                    # ═══════════════════════════════════════════════════════════════════
                    # 🧭 UNIFIED TERRAIN NAVIGATOR (Cached A* Pathfinding)
                    # Replaces manual zone checks with efficient cached pathfinding.
                    # ~0.2ms average (95% cache hits), ~2ms worst case (bounded A*)
                    # ═══════════════════════════════════════════════════════════════════
                    
                    # Clear cache on map change
                    if map_id != self.current_map_id:
                        self.terrain_nav.clear_cache()
                        self.current_map_id = map_id
                        if self.step_count % 100 == 0:
                            stats = self.terrain_nav.get_stats()
                            logger.info(f"🧭 TerrainNav stats: {stats['cache_hits']} hits, {stats['cache_misses']} misses, {stats['hit_rate']:.1%} hit rate")
                    
                    # Determine navigation goal based on current context
                    terrain_goal = None
                    terrain_nav_active = False
                    terrain_nav_result = None  # May be set directly by 2F logic
                    
                    # Get metatile info for pathfinding
                    map_data = memory.get('map_data', {})
                    metatile_info = map_data.get('metatile_info', [])
                    
                    # DEBUG: Log metatile_info status periodically
                    if self.step_count % 200 == 0:
                        has_metatile = len(metatile_info) > 0 if metatile_info else False
                        metatile_rows = len(metatile_info) if metatile_info else 0
                        visual_map = map_data.get('visual_map', '')[:100] if map_data else 'NONE'
                        logger.info(f"🗺️ MAP DATA: metatile={metatile_rows} rows, has_data={has_metatile}, visual_map={visual_map[:50]}...")

                    
                    # Goal selection based on map and state
                    # Sequences: Goal1 → Exit (for indoor maps)
                    
                    # CHECK 2F FIRST - more specific than map_id!
                    # PokeCenter 2F has escalator on LEFT side - use ground truth connections!
                    if in_pokecenter_2f:
                        connections = memory.get('connections', [])
                        
                        # PRIORITY 1: Ground truth exit from server connections
                        dynamic_exit = None
                        for conn in connections:
                            to_loc = conn.get('to', '').upper()
                            from_pos = conn.get('from_pos', [])
                            if len(from_pos) >= 2:
                                if '1F' in to_loc or ('POKECENTER' in to_loc and '2F' not in to_loc):
                                    dynamic_exit = (int(from_pos[0]), int(from_pos[1]))
                                    if self.step_count % 50 == 0:
                                        logger.info(f"🎯 DYNAMIC 2F EXIT: Using server exit {dynamic_exit} to 1F")
                                    break
                        
                        # PRIORITY 2: Fallback escalator position
                        if dynamic_exit:
                            escalator_x, escalator_y = dynamic_exit
                        else:
                            escalator_x, escalator_y = 1, 6  # Actual escalator position
                        
                        terrain_goal = (escalator_x, escalator_y)
                        
                        # === USE A* WITH METATILE DATA FOR GROUND TRUTH SLAM ===
                        # TerrainNavigator will find optimal path using real-time terrain
                        a_star_result = self.terrain_nav.get_direction(
                            map_id=0x201,  # PC 2F map ID
                            x=x, y=y,
                            goal=terrain_goal,
                            metatile_info=metatile_info
                        )
                        
                        if a_star_result:
                            # A* found a path using ground truth!
                            terrain_nav_result = a_star_result
                            if self.step_count % 30 == 0:
                                logger.info(f"🪜 PC 2F A*: ({x},{y}) -> {a_star_result['action']} | {a_star_result['reason']}")
                        else:
                            # A* failed - use heuristic fallback (DOWN first, then LEFT)
                            if y < escalator_y:
                                pc2f_action = 'DOWN'
                                pc2f_reason = f'Heuristic: DOWN to row {escalator_y} (y={y})'
                            elif x > escalator_x:
                                pc2f_action = 'LEFT'
                                pc2f_reason = f'Heuristic: LEFT to x={escalator_x} (x={x})'
                            else:
                                pc2f_action = 'LEFT'
                                pc2f_reason = 'Heuristic: Step onto escalator'
                            terrain_nav_result = {'action': pc2f_action, 'reason': pc2f_reason}
                            if self.step_count % 30 == 0:
                                logger.info(f"🪜 PC 2F: ({x},{y}) -> {pc2f_action} | {pc2f_reason}")
                        
                        terrain_nav_active = True


                    
                    elif map_id == 0x11:  # OLDALE_TOWN (outdoor)
                        connections = memory.get('connections', [])
                        has_pokedex = memory.get('has_pokedex', False)
                        
                        # Check HP for healing decision
                        lead_hp_percent = 100
                        try:
                            party_data = memory.get('party', []) or progress_ctx.get('party', [])
                            if party_data and len(party_data) > 0:
                                lead = party_data[0]
                                hp = lead.get('current_hp', lead.get('hp', 0))
                                max_hp = lead.get('max_hp', 1) or 1
                                lead_hp_percent = (hp / max_hp) * 100
                        except:
                            pass
                        
                        # Priority 1: Heal if HP < 50% and haven't won rival battle yet
                        if lead_hp_percent < 50 and not rival_battled:
                            if self.step_count % 50 == 0:
                                logger.info(f"🏥 OLDALE: HP low ({lead_hp_percent:.0f}%)! Going to PokeCenter before Route 103 battle!")
                            dynamic_exit = self.get_exit_for_objective(connections, 'pokecenter')
                            if dynamic_exit:
                                terrain_goal = dynamic_exit
                            else:
                                terrain_goal = 'pokecenter'
                            terrain_nav_active = True
                        
                        elif not getattr(self, 'visited_oldale_pokecenter', False):
                            # Step 1: Visit PokeCenter first to set spawn point
                            dynamic_exit = self.get_exit_for_objective(connections, 'pokecenter')
                            if dynamic_exit:
                                terrain_goal = dynamic_exit
                                if self.step_count % 100 == 0:
                                    logger.info(f"🎯 DYNAMIC: Using server exit {dynamic_exit} to POKECENTER")
                            else:
                                terrain_goal = 'pokecenter'  # Fallback
                            terrain_nav_active = True
                        
                        elif not rival_battled:
                            # Step 2: Head to Route 103 for rival battle (HP is OK)
                            dynamic_exit = self.get_exit_for_objective(connections, 'route103')
                            if dynamic_exit:
                                terrain_goal = dynamic_exit
                                if self.step_count % 100 == 0:
                                    logger.info(f"🎯 DYNAMIC: Using server exit {dynamic_exit} to ROUTE 103")
                            else:
                                terrain_goal = 'route103_exit'  # Fallback
                            terrain_nav_active = True
                        
                        elif not has_pokedex:
                            # Step 3: Return to Littleroot (Route 101) for Birch's Lab
                            dynamic_exit = self.get_exit_for_objective(connections, 'route101')
                            if dynamic_exit:
                                terrain_goal = dynamic_exit
                                if self.step_count % 100 == 0:
                                    logger.info(f"🎯 DYNAMIC: Using server exit {dynamic_exit} to ROUTE 101 (Lab)")
                            else:
                                terrain_goal = 'route101_exit'  # Fallback
                            terrain_nav_active = True
                        
                        else:
                            # Step 4: Has Pokedex! Head west to Route 102 -> Petalburg
                            dynamic_exit = self.get_exit_for_objective(connections, 'route102')
                            if dynamic_exit:
                                terrain_goal = dynamic_exit
                                if self.step_count % 100 == 0:
                                    logger.info(f"🎯 DYNAMIC: Using server exit {dynamic_exit} to ROUTE 102")
                            else:
                                # Fallback: Route 102 is west of Oldale
                                terrain_goal = 'route102_exit'  # Will be added to terrain nav
                            terrain_nav_active = True

                    
                    elif map_id == 0x200:  # OLDALE_POKECENTER 1F interior
                        connections = memory.get('connections', [])
                        
                        # After visiting Nurse Joy, head to exit
                        if getattr(self, 'visited_oldale_pokecenter', False):
                            # Try dynamic exit to Oldale Town from connections
                            dynamic_exit = self.get_exit_for_objective(connections, 'oldale')
                            if dynamic_exit:
                                terrain_goal = dynamic_exit
                                if self.step_count % 100 == 0:
                                    logger.info(f"🎯 DYNAMIC: Using server exit {dynamic_exit} to OLDALE TOWN")
                            else:
                                terrain_goal = 'exit'  # Fallback to hardcoded
                            terrain_nav_active = True
                        else:
                            terrain_goal = 'nurse_joy'  # Need to heal first
                            terrain_nav_active = True
                    
                    # OLDALE_MART - just exit! No reason to stay in the mart
                    elif map_id == 0x202 or (map_id == 0 and 'MART' in location_upper):
                        # Use get_exit_for_objective with map_id override
                        # The override returns (3,7) for map 0x202 -> oldale
                        connections = memory.get('connections', [])
                        current_map_id = 0x202 if map_id == 0x202 else (0x202 if 'MART' in location_upper else 0)
                        
                        dynamic_exit = self.get_exit_for_objective(connections, 'oldale', map_id=current_map_id)
                        
                        if dynamic_exit:
                            terrain_goal = dynamic_exit
                            if self.step_count % 100 == 0:
                                logger.info(f"🏪 MART NAV: Using exit {terrain_goal} to OLDALE TOWN")
                        else:
                            terrain_goal = (3, 7)  # Ultimate fallback
                            if self.step_count % 100 == 0:
                                logger.info(f"🏪 MART NAV: Using fallback exit {terrain_goal}")
                        
                        terrain_nav_active = True
                    
                    # OLDALE TOWN HOUSES (0x205, 0x206) - WRONG BUILDING! Exit immediately!
                    # These are random houses, not PokeCenter. Force exit back to Oldale Town
                    elif map_id in [0x205, 0x206] or ('OLDALE' in location_upper and 'HOUSE' in location_upper and 'POKECENTER' not in location_upper):
                        # Log that we entered wrong building
                        if self.step_count % 50 == 0:
                            logger.warning(f"⚠️ WRONG BUILDING! Entered Oldale house (map={hex(map_id) if map_id else 'None'}) instead of PokeCenter! Exiting...")
                        
                        # Use generic house exit - most houses have door at bottom
                        connections = memory.get('connections', [])
                        dynamic_exit = self.get_exit_for_objective(connections, 'oldale')
                        
                        if dynamic_exit:
                            terrain_goal = dynamic_exit
                        else:
                            terrain_goal = (4, 8)  # Generic house door location
                        
                        terrain_nav_active = True
                    
                    # BIRCH_LAB - need to exit after getting Pokedex or starter
                    elif map_id in [0x020C, 0x104] or 'LAB' in location_upper or 'BIRCH' in location_upper:
                        # Lab exit is at (6, 12) - door at bottom center
                        connections = memory.get('connections', [])
                        current_map_id = 0x020C if map_id in [0x020C, 0x104] else 0x020C
                        
                        dynamic_exit = self.get_exit_for_objective(connections, 'littleroot', map_id=current_map_id)
                        
                        if dynamic_exit:
                            terrain_goal = dynamic_exit
                            if self.step_count % 20 == 0:
                                logger.info(f"🧪 LAB NAV: pos=({x},{y}) goal={terrain_goal} map_id={hex(map_id) if map_id else 'None'}")
                        else:
                            terrain_goal = (6, 12)  # Fallback - door at bottom center
                            if self.step_count % 20 == 0:
                                logger.info(f"🧪 LAB NAV: pos=({x},{y}) goal={terrain_goal} (fallback) map_id={hex(map_id) if map_id else 'None'}")
                        
                        terrain_nav_active = True
                    
                    elif map_id == 0x10:  # ROUTE_101
                        connections = memory.get('connections', [])
                        has_pokedex = memory.get('has_pokedex', False)
                        
                        # Bidirectional navigation based on objective
                        if rival_battled and not has_pokedex:
                            # After rival, need to go SOUTH to Littleroot for Lab/Pokedex
                            dynamic_exit = self.get_exit_for_objective(connections, 'littleroot')
                            if dynamic_exit:
                                terrain_goal = dynamic_exit
                                if self.step_count % 100 == 0:
                                    logger.info(f"🎯 DYNAMIC: Using server exit {dynamic_exit} to LITTLEROOT")
                            else:
                                terrain_goal = 'littleroot_entrance'  # Fallback: Go south
                            terrain_nav_active = True
                        else:
                            # Default: Go NORTH to Oldale (for PokeCenter, Route 103, or Route 102)
                            dynamic_exit = self.get_exit_for_objective(connections, 'oldale')
                            if dynamic_exit:
                                terrain_goal = dynamic_exit
                                if self.step_count % 100 == 0:
                                    logger.info(f"🎯 DYNAMIC: Using server exit {dynamic_exit} to OLDALE")
                            else:
                                terrain_goal = 'oldale_entrance'  # Fallback: Go north
                            terrain_nav_active = True

                    
                    elif map_id == 0x12:  # ROUTE_103
                        connections = memory.get('connections', [])
                        
                        # Log rival_battled status for debugging
                        if self.step_count % 100 == 0:
                            logger.info(f"📍 ROUTE 103: rival_battled={rival_battled}, completed_locally={self.completed_first_rival_battle_locally}")
                        
                        # CRITICAL: Check HP before battling rival - if too low, go heal first!
                        lead_hp_percent = 100
                        try:
                            party_data = memory.get('party', []) or progress_ctx.get('party', [])
                            if party_data and len(party_data) > 0:
                                lead = party_data[0]
                                hp = lead.get('current_hp', lead.get('hp', 0))
                                max_hp = lead.get('max_hp', 1) or 1
                                lead_hp_percent = (hp / max_hp) * 100
                                if self.step_count % 50 == 0:
                                    logger.info(f"🩺 R103 HP CHECK: lead_hp={hp}/{max_hp} = {lead_hp_percent:.0f}%")
                            else:
                                if self.step_count % 50 == 0:
                                    logger.warning(f"🩺 R103 HP CHECK: No party data found! memory:{bool(memory.get('party'))} progress:{bool(progress_ctx.get('party'))}")
                        except Exception as e:
                            if self.step_count % 50 == 0:
                                logger.warning(f"🩺 R103 HP CHECK: Exception: {e}")
                        
                        # PRIORITY 1: If HP < 50%, go back to Oldale to heal FIRST! (regardless of battle status)
                        if lead_hp_percent < 50:
                            if self.step_count % 50 == 0:
                                logger.warning(f"⚠️ ROUTE 103: HP too low ({lead_hp_percent:.0f}%)! Go to Oldale PokeCenter to heal!")
                            dynamic_exit = self.get_exit_for_objective(connections, 'oldale')
                            if dynamic_exit:
                                terrain_goal = dynamic_exit
                            else:
                                terrain_goal = 'oldale_exit'
                            terrain_nav_active = True
                        elif not rival_battled:
                            # HP is OK, go find rival - left/middle of Route 103, y=3 is correct, x<10
                            terrain_goal = (7, 3)  # Rival's position on Route 103 (corrected: x<10, y=3)
                            terrain_nav_active = True
                        else:
                            # rival_battled=True and HP is OK - Return to Oldale after battle
                            dynamic_exit = self.get_exit_for_objective(connections, 'oldale')
                            if dynamic_exit:
                                terrain_goal = dynamic_exit
                                if self.step_count % 50 == 0:
                                    logger.info(f"🎯 R103 POST-BATTLE: Using exit {dynamic_exit} to OLDALE")
                            else:
                                terrain_goal = 'oldale_exit'
                                if self.step_count % 50 == 0:
                                    logger.info(f"🎯 R103 POST-BATTLE: Using fallback oldale_exit")
                            terrain_nav_active = True

                    
                    elif map_id == 0x0F:  # ROUTE_102 (west of Oldale, leads to Petalburg)
                        connections = memory.get('connections', [])
                        
                        # Check if we need healing before continuing to Petalburg
                        # Get lead Pokemon HP from state
                        lead_hp_percent = 100  # Default to full
                        try:
                            party = memory.get('party', [])
                            if party and len(party) > 0:
                                lead_mon = party[0]
                                if lead_mon and lead_mon.get('max_hp', 0) > 0:
                                    lead_hp_percent = (lead_mon.get('current_hp', 0) / lead_mon.get('max_hp', 1)) * 100
                        except:
                            pass
                        
                        # If HP low and not close to Petalburg (x > 5), return to Oldale PokeCenter
                        if lead_hp_percent < 50 and x > 5:
                            # Return EAST to Oldale for PokeCenter
                            dynamic_exit = self.get_exit_for_objective(connections, 'oldale')
                            if dynamic_exit:
                                terrain_goal = dynamic_exit
                                if self.step_count % 50 == 0:
                                    logger.info(f"🏥 R102: HP low ({lead_hp_percent:.0f}%), return to Oldale PokeCenter")
                            else:
                                terrain_goal = 'oldale_exit'  # Fallback: Go east
                            terrain_nav_active = True
                        else:
                            # Route 102 is the path to Petalburg after getting Pokedex
                            # Go WEST toward Petalburg City
                            dynamic_exit = self.get_exit_for_objective(connections, 'petalburg')
                            if dynamic_exit:
                                terrain_goal = dynamic_exit
                                if self.step_count % 100 == 0:
                                    logger.info(f"🎯 DYNAMIC: Using server exit {dynamic_exit} to PETALBURG")
                            else:
                                terrain_goal = 'petalburg_entrance'  # Fallback: Go west
                            terrain_nav_active = True

                    
                    elif map_id in [0x104, 0x020C]:  # BIRCH_LAB (multiple possible IDs)

                        connections = memory.get('connections', [])
                        has_pokedex = memory.get('has_pokedex', False)
                        
                        # FIXED LOGIC:
                        # 1. has_pokedex → EXIT (done with lab)
                        # 2. rival_battled && !has_pokedex → GO TO BIRCH (get Pokedex)
                        # 3. !rival_battled → EXIT (need to battle rival first)
                        
                        if has_pokedex:
                            # Already have Pokedex - head to exit
                            dynamic_exit = None
                            for conn in connections:
                                to_loc = (conn.get('to') or '').upper()
                                from_pos = conn.get('from_pos', [])
                                if len(from_pos) >= 2:
                                    if 'LITTLEROOT' in to_loc or 'TOWN' in to_loc:
                                        dynamic_exit = (int(from_pos[0]), int(from_pos[1]))
                                        if self.step_count % 100 == 0:
                                            logger.info(f"🎯 DYNAMIC LAB EXIT: Using server exit {dynamic_exit}")
                                        break
                            
                            if dynamic_exit:
                                terrain_goal = dynamic_exit
                            else:
                                terrain_goal = 'exit'  # Fallback
                            terrain_nav_active = True
                        elif rival_battled and not has_pokedex:
                            # CRITICAL: Defeated rival, need Pokedex from Birch!
                            # Birch is typically at (6, 5) in his lab
                            terrain_goal = (6, 5)  # Birch's position
                            terrain_nav_active = True
                            if self.step_count % 50 == 0:
                                logger.info(f"🎓 BIRCH LAB: Go to Birch at (6,5) to receive Pokedex!")
                        else:
                            # Haven't battled rival yet - exit and go to Route 103
                            dynamic_exit = None
                            for conn in connections:
                                to_loc = (conn.get('to') or '').upper()
                                from_pos = conn.get('from_pos', [])
                                if len(from_pos) >= 2:
                                    if 'LITTLEROOT' in to_loc or 'TOWN' in to_loc:
                                        dynamic_exit = (int(from_pos[0]), int(from_pos[1]))
                                        break
                            
                            if dynamic_exit:
                                terrain_goal = dynamic_exit
                            else:
                                terrain_goal = 'exit'
                            terrain_nav_active = True

                    
                    elif map_id == 0x09:  # LITTLEROOT_TOWN (outdoor)
                        connections = memory.get('connections', [])
                        has_pokedex = memory.get('has_pokedex', False)
                        
                        # Check milestones to determine progress
                        milestones_data = memory.get('milestones', {})
                        if isinstance(milestones_data, dict):
                            completed_milestones = list(milestones_data.keys()) if 'completed' not in milestones_data else milestones_data.get('completed', [])
                        else:
                            completed_milestones = []
                        
                        visited_route102 = 'ROUTE_102' in str(completed_milestones)
                        visited_petalburg = 'PETALBURG_CITY' in str(completed_milestones)
                        
                        # PRIORITY 1: Visit rival's bedroom FIRST before anything else!
                        # This is required for the story to progress
                        rival_bedroom_done = 'RIVAL_BEDROOM' in str(completed_milestones)
                        if not rival_bedroom_done and not (getattr(self, 'has_starter', False) or party_size > 0):
                            # MUST visit rival's bedroom before going anywhere else!
                            # Gender-aware rival house selection:
                            # Male avatar (Boy): Brendan = player, May = rival -> May's house at (17, 9)
                            # Female avatar (Girl): May = player, Brendan = rival -> Brendan's house at (4, 9)
                            player_gender = memory.get('player', {}).get('gender', 'Boy')
                            if player_gender == 'Girl':
                                rival_house_x, rival_house_y = 4, 9  # Brendan's house (rival for female)
                                rival_name = "Brendan"
                            else:
                                rival_house_x, rival_house_y = 17, 9  # May's house (rival for male)
                                rival_name = "May"
                            terrain_goal = (rival_house_x, rival_house_y)
                            terrain_nav_active = True
                            
                            # Use A* pathfinding for ground truth navigation
                            if metatile_info:
                                a_star_result = self.terrain_nav.get_direction(
                                    map_id=0x09,  # Littleroot Town
                                    x=x, y=y,
                                    goal=terrain_goal,
                                    metatile_info=metatile_info
                                )
                                if a_star_result:
                                    terrain_nav_result = a_star_result
                                    if self.step_count % 30 == 0:
                                        logger.info(f"🏠 RIVAL HOUSE A*: ({x},{y}) -> {a_star_result['action']} to {rival_name}'s house")
                            
                            if self.step_count % 50 == 0:
                                logger.info(f"🏠 LITTLEROOT: RIVAL_BEDROOM not done! Go to {rival_name}'s house at ({rival_house_x},{rival_house_y})")
                        
                        elif rival_battled and not has_pokedex:

                            # After rival battle, need to go to Birch's Lab for Pokedex
                            dynamic_exit = self.get_exit_for_objective(connections, 'lab')
                            if dynamic_exit:
                                terrain_goal = dynamic_exit
                                if self.step_count % 100 == 0:
                                    logger.info(f"🎯 DYNAMIC: Using server exit {dynamic_exit} to BIRCH'S LAB")
                            else:
                                terrain_goal = 'birch_lab'  # Fallback
                            terrain_nav_active = True
                        elif has_pokedex:
                            # Has Pokedex - head to Route 101 to continue to Petalburg
                            # This is the correct forward path: Littleroot → Route 101 → Oldale → Route 102 → Petalburg
                            dynamic_exit = self.get_exit_for_objective(connections, 'route101')
                            if dynamic_exit:
                                terrain_goal = dynamic_exit
                                if self.step_count % 100 == 0:
                                    logger.info(f"🎯 LITTLEROOT: Go Route 101 → Oldale → Route 102 → Petalburg")
                            else:
                                terrain_goal = 'route101_exit'  # Fallback
                            terrain_nav_active = True

                        elif not (getattr(self, 'has_starter', False) or party_size > 0):
                            # No starter yet - after rival bedroom, go NORTH to Route 101!
                            # Birch is in Route 101 tall grass being chased - NOT at the lab!
                            # Story order: Rival bedroom → Route 101 → Rescue Birch → Get starter
                            dynamic_exit = self.get_exit_for_objective(connections, 'route101')
                            if dynamic_exit:
                                terrain_goal = dynamic_exit
                                if self.step_count % 100 == 0:
                                    logger.info(f"🎯 LITTLEROOT: No starter - go NORTH to Route 101 to find Birch!")
                            else:
                                terrain_goal = 'route101_exit'  # Go NORTH, not to lab!
                            terrain_nav_active = True
                        else:
                            # Has starter, go to Route 101
                            dynamic_exit = self.get_exit_for_objective(connections, 'route101')
                            if dynamic_exit:
                                terrain_goal = dynamic_exit
                            else:
                                terrain_goal = 'route101_exit'  # Fallback
                            terrain_nav_active = True


                    
                    elif map_id in [0x100, 0x101, 0x102, 0x103]:  # Houses (Brendan/May)
                        # Always exit houses after getting starter

                        if getattr(self, 'has_starter', False) or party_size > 0:
                            # Check if on 2F - need to target STAIRS, not exit door!
                            # 0x101 = Brendan's house 2F, 0x103 = May's house 2F
                            # ALSO check location string since map_id can be wrong (0x0)
                            is_on_2f = (map_id in [0x101, 0x103]) or ('2F' in location_upper)
                            if is_on_2f:
                                # ON 2F: Use portal connections to find stairs coordinates
                                connections = memory.get('connections', [])
                                stairs_coords = None
                                
                                # Try to find stairs from portal connections
                                for conn in connections:
                                    to_loc = (conn.get('to') or '').upper()
                                    if '1F' in to_loc or 'FIRST FLOOR' in to_loc:
                                        # Found connection to 1F - these are the stairs!
                                        stairs_coords = tuple(conn.get('from_pos', []))
                                        if self.step_count % 100 == 0:
                                            logger.info(f"🏠 Found stairs via portal: {stairs_coords} → {to_loc}")
                                        break
                                
                                # Fallback: Check terrain for warp tiles (stairs)
                                if not stairs_coords:
                                    map_data = memory.get('map_data', {})
                                    metatile_info = map_data.get('metatile_info', [])
                                    if metatile_info and len(metatile_info) >= 15:
                                        center = 7
                                        for dy in range(-7, 8):
                                            for dx in range(-7, 8):
                                                try:
                                                    tile = metatile_info[center + dy][center + dx]
                                                    behavior = tile.get('behavior', '')
                                                    if 'WARP' in behavior or 'STAIR' in behavior or 'LADDER' in behavior:
                                                        stairs_coords = (x + dx, y + dy)
                                                        if self.step_count % 100 == 0:
                                                            logger.info(f"🏠 Found stairs via terrain: {stairs_coords}")
                                                        break
                                                except:
                                                    pass
                                            if stairs_coords:
                                                break
                                
                                # Last resort: Hardcoded fallback (but should be correct now!)
                                if not stairs_coords:
                                    # Brendan's house: stairs at (2,2) for 2F map 0x101
                                    # May's house: stairs at (2,2) for 2F map 0x103
                                    stairs_coords = (2, 2)  # CORRECT fallback for both houses
                                    if self.step_count % 100 == 0:
                                        logger.info(f"⚠️ House 2F: Using fallback stairs {stairs_coords}")
                                
                                terrain_goal = stairs_coords
                            else:
                                # ON 1F: Target exit door - use ground truth first!
                                connections = memory.get('connections', [])
                                exit_door = None
                                
                                for conn in connections:
                                    to_loc = (conn.get('to') or '').upper()
                                    from_pos = conn.get('from_pos', [])
                                    if len(from_pos) >= 2:
                                        # Look for outdoor connections (Town, Route) - EXCLUDE indoor locations!
                                        is_outdoor = ('TOWN' in to_loc or 'ROUTE' in to_loc or 'LITTLEROOT' in to_loc)
                                        is_indoor = ('HOUSE' in to_loc or '2F' in to_loc or '1F' in to_loc or 'BEDROOM' in to_loc)
                                        if is_outdoor and not is_indoor:
                                            exit_door = (int(from_pos[0]), int(from_pos[1]))
                                            if self.step_count % 100 == 0:
                                                logger.info(f"🎯 DYNAMIC HOUSE EXIT: Using server exit {exit_door} to {to_loc}")
                                            break
                                
                                if exit_door:
                                    terrain_goal = exit_door
                                else:
                                    # Debug: Log available connections to understand why no outdoor exit found
                                    if self.step_count % 50 == 0:
                                        logger.warning(f"⚠️ 1F EXIT: No outdoor exit found! Connections: {[c.get('to', '') for c in connections]}")
                                    terrain_goal = (8, 8)  # Hardcoded fallback (Brendan's house exit)
                            
                            terrain_nav_active = True
                        else:
                            # NO POKEMON - either early game or after blackout
                            # Still need to exit house to visit rival's house or find Birch!
                            # Player's house 1F: exit door at (8, 8)
                            # Player's house 2F: stairs at (2, 2)
                            if map_id in [0x101, 0x103]:  # 2F - go to stairs
                                terrain_goal = (8, 2)  # Stairs on 2F (leads to 1F)
                            else:  # 1F - go to door
                                terrain_goal = (8, 8)  # Exit door (Brendan's house)
                            terrain_nav_active = True
                            if self.step_count % 50 == 0:
                                logger.info(f"🏠 NO POKEMON: Exiting house via {terrain_goal}")

                    # === GENERIC HOUSE FALLBACK (for unknown map IDs like OLDALE TOWN HOUSE1) ===
                    elif 'HOUSE' in location_upper and not terrain_nav_active:
                        if party_size > 0 or getattr(self, 'has_starter', False):
                            # Have Pokemon - should exit the house!
                            connections = memory.get('connections', [])
                            exit_door = None
                            for conn in connections:
                                to_loc = (conn.get('to') or '').upper()
                                from_pos = conn.get('from_pos', [])
                                is_outdoor = ('TOWN' in to_loc or 'ROUTE' in to_loc)
                                is_indoor = ('HOUSE' in to_loc or '2F' in to_loc or '1F' in to_loc)
                                if len(from_pos) >= 2 and is_outdoor and not is_indoor:
                                    exit_door = (int(from_pos[0]), int(from_pos[1]))
                                    if self.step_count % 100 == 0:
                                        logger.info(f"🎯 GENERIC HOUSE EXIT: {exit_door} to {to_loc}")
                                    break
                            terrain_goal = exit_door if exit_door else (8, 8)  # Fallback exit (Brendan's house)
                            terrain_nav_active = True
                            if self.step_count % 100 == 0 and not exit_door:
                                logger.info(f"🎯 GENERIC HOUSE: Using fallback exit (4,8)")
                        else:
                            # NO POKEMON - early game, still need to exit house!
                            # Use location string to detect 2F since map_id may be wrong
                            if '2F' in location_upper or 'BEDROOM' in location_upper:
                                terrain_goal = (2, 2)  # Stairs to 1F
                            else:
                                terrain_goal = (8, 8)  # Exit door on 1F (Brendan's house)
                            terrain_nav_active = True
                            if self.step_count % 50 == 0:
                                logger.info(f"🏠 GENERIC HOUSE (no pokemon): Exit via {terrain_goal}")

                    

                    # HOUSE EXIT with SERVER GROUND TRUTH first, gender-aware hardcoded fallback
                    if 'HOUSE' in location_upper:
                        connections = memory.get('connections', [])
                        player_gender = memory.get('player', {}).get('gender', 'Boy')  # Default to Boy
                        is_player_house = 'BRENDAN' in location_upper if player_gender == 'Boy' else 'MAY' in location_upper
                        
                        # Check if CLOCK_SET milestone is done (need to set clock in player's 2F first!)
                        clock_set_done = 'CLOCK_SET' in str(memory.get('milestones', {}))
                        
                        # PRIORITY 0: If in PLAYER's house 1F and clock NOT set, go to STAIRS (2,2) not exit!
                        if '1F' in location_upper and is_player_house and not clock_set_done:
                            # Find stairs to 2F from server connections
                            stairs_from_server = None
                            for conn in connections:
                                to_loc = (conn.get('to') or '').upper()
                                from_pos = conn.get('from_pos', [])
                                if len(from_pos) >= 2 and '2F' in to_loc:
                                    stairs_from_server = (int(from_pos[0]), int(from_pos[1]))
                                    break
                            terrain_goal = stairs_from_server if stairs_from_server else (2, 2)  # Fallback to (2,2)
                            terrain_nav_active = True
                            if self.step_count % 50 == 0:
                                logger.info(f"🏠 PLAYER HOUSE 1F: Go UPSTAIRS to set clock at {terrain_goal} (gender={player_gender})")
                        
                        # PRIORITY 0.5: If in RIVAL's house 1F and RIVAL_BEDROOM NOT done, go UPSTAIRS!
                        elif '1F' in location_upper and not is_player_house:
                            # Check RIVAL_BEDROOM milestone  
                            rival_bedroom_done = 'RIVAL_BEDROOM' in str(memory.get('milestones', {}))
                            if not rival_bedroom_done:
                                # In rival's house - need to go upstairs for bedroom milestone!
                                stairs_from_server = None
                                for conn in connections:
                                    to_loc = (conn.get('to') or '').upper()
                                    from_pos = conn.get('from_pos', [])
                                    if len(from_pos) >= 2 and '2F' in to_loc:
                                        stairs_from_server = (int(from_pos[0]), int(from_pos[1]))
                                        break
                                terrain_goal = stairs_from_server if stairs_from_server else (2, 2)
                                terrain_nav_active = True
                                if self.step_count % 50 == 0:
                                    logger.info(f"🏠 RIVAL HOUSE 1F: Go UPSTAIRS to bedroom at {terrain_goal} (gender={player_gender})")
                            else:
                                # RIVAL_BEDROOM done - exit the house
                                pass  # Fall through to exit logic below
                        else:
                            # ═══════════════════════════════════════════════════
                            # OVERRIDE: Server returns WRONG coordinates for houses!
                            # May's House 1F (0x102) and Brendan's House 1F (0x100): 
                            #   Server says (14,9) but actual exit is at (1,8)
                            # Use hardcoded CORRECT values for these maps
                            # ═══════════════════════════════════════════════════
                            
                            # Get map_id from memory
                            current_map_id = memory.get('map_id')
                            if isinstance(current_map_id, str):
                                try:
                                    current_map_id = int(current_map_id, 16) if current_map_id.startswith('0x') else int(current_map_id)
                                except:
                                    current_map_id = None
                            
                            # OVERRIDE for house maps with known wrong server data
                            if current_map_id in [0x102, 0x100]:  # May's/Brendan's House 1F
                                if '1F' in location_upper:
                                    terrain_goal = (1, 8)  # CORRECT exit door (server wrongly says 14,9)
                                    terrain_nav_active = True
                                    if self.step_count % 50 == 0:
                                        logger.info(f"🏠 HOUSE 1F OVERRIDE: Exit at {terrain_goal} (map_id={hex(current_map_id) if current_map_id else 'None'})")
                            elif current_map_id in [0x103, 0x101]:  # May's/Brendan's House 2F  
                                terrain_goal = (1, 2)  # Stairs down at (1,2) per portal connection
                                terrain_nav_active = True
                                if self.step_count % 50 == 0:
                                    logger.info(f"🏠 HOUSE 2F OVERRIDE: Stairs at {terrain_goal} (map_id={hex(current_map_id) if current_map_id else 'None'})")
                            elif current_map_id == 0x104:  # Birch's Lab - portal says (7,17) but DD shows (6,12)
                                terrain_goal = (6, 12)  # CORRECT exit door from visual map DD marker
                                terrain_nav_active = True
                                if self.step_count % 50 == 0:
                                    logger.info(f"🔬 LAB OVERRIDE: Exit at {terrain_goal} (map_id={hex(current_map_id) if current_map_id else 'None'})")
                            else:
                                # No hardcoded override - use navigation strategy:
                                # For INDOOR maps: Try visual map DD labels first (more accurate)
                                # For OUTDOOR maps: Use portal connections
                                
                                is_indoor = any(x in location_upper for x in ['1F', '2F', 'HOUSE', 'LAB', 'CENTER', 'MART', 'BEDROOM'])
                                
                                exit_found = None
                                
                                if is_indoor:
                                    # PRIORITY 0: Try visual map DD label (indoor exit strategy)
                                    dd_exit = self.get_indoor_exit_from_visual_map(memory)
                                    if dd_exit:
                                        exit_found = dd_exit
                                        if self.step_count % 50 == 0:
                                            logger.info(f"📍 VISUAL MAP DD: Exit at {exit_found} (indoor={is_indoor})")
                                
                                if not exit_found:
                                    # PRIORITY 1: Server portal connections
                                    for conn in connections:
                                        to_loc = (conn.get('to') or '').upper()
                                        from_pos = conn.get('from_pos', [])
                                        if len(from_pos) >= 2:
                                            # 1F: Look for exit to TOWN or outdoor area
                                            if '1F' in location_upper and ('TOWN' in to_loc or 'ROUTE' in to_loc or 'LITTLEROOT' in to_loc):
                                                exit_found = (int(from_pos[0]), int(from_pos[1]))
                                                if self.step_count % 100 == 0:
                                                    logger.info(f"🏠 SERVER EXIT: {exit_found} to {to_loc}")
                                                break
                                            # 2F/BEDROOM: Look for stairs to 1F
                                            elif ('2F' in location_upper or 'BEDROOM' in location_upper) and '1F' in to_loc:
                                                exit_found = (int(from_pos[0]), int(from_pos[1]))
                                                if self.step_count % 100 == 0:
                                                    logger.info(f"🏠 SERVER STAIRS: {exit_found} to {to_loc}")
                                                break
                                
                                if exit_found:
                                    terrain_goal = exit_found
                                    terrain_nav_active = True
                                else:
                                    # PRIORITY 2: Hardcoded fallback for unknown maps
                                    if '1F' in location_upper:
                                        terrain_goal = (1, 8)  # Default exit door
                                        if self.step_count % 50 == 0:
                                            logger.info(f"🏠 FALLBACK 1F: Exit at {terrain_goal}")
                                    elif '2F' in location_upper or 'BEDROOM' in location_upper:
                                        terrain_goal = (2, 2)  # Default stairs
                                        if self.step_count % 50 == 0:
                                            logger.info(f"🏠 FALLBACK 2F: Stairs at {terrain_goal}")
                                    terrain_nav_active = True

                    # ═══════════════════════════════════════════════════════════════════
                    # 🚨 EMERGENCY FALLBACK: When map data is corrupted but we're indoors
                    # Uses cached location to detect if we should be in a HOUSE
                    # ═══════════════════════════════════════════════════════════════════
                    if not terrain_nav_active and not terrain_goal:
                        cached_loc = getattr(self, '_last_known_location', '')
                        cached_upper = (cached_loc or '').upper()
                        
                        # If last known location was house/indoor, use DOWN fallback
                        is_indoor_cached = any(kw in cached_upper for kw in ['HOUSE', '1F', '2F', 'BEDROOM', 'LAB'])
                        
                        if is_indoor_cached:
                            # Determine exit direction based on floor
                            if '2F' in cached_upper or 'BEDROOM' in cached_upper:
                                # 2F: Go to stairs (usually DOWN then navigate)
                                terrain_nav_result = {'action': 'DOWN', 'reason': 'EMERGENCY: Map corrupted on 2F, fallback DOWN'}
                            else:
                                # 1F: Go to exit (usually DOWN at bottom of house)
                                terrain_nav_result = {'action': 'DOWN', 'reason': 'EMERGENCY: Map corrupted on 1F, fallback DOWN'}
                            
                            terrain_goal = 'exit'
                            terrain_nav_active = True
                            
                            if self.step_count % 50 == 0:
                                logger.warning(f"⚠️ EMERGENCY NAV: Using DOWN fallback (cached: {cached_loc})")

                    # UNLESS already computed (e.g., for 2F direct calculation)
                    if terrain_nav_active and terrain_goal and terrain_nav_result is None:
                        terrain_nav_result = self.terrain_nav.get_direction(
                            map_id, x, y, terrain_goal, metatile_info
                        )
                        
                        if terrain_nav_result and self.step_count % 50 == 0:
                            logger.info(f"🧭 TERRAIN NAV: ({x},{y}) -> {terrain_nav_result['action']} | Goal={terrain_goal} | {terrain_nav_result['reason']}")
                    
                    memory['terrain_nav_active'] = terrain_nav_active
                    memory['terrain_nav_result'] = terrain_nav_result
                    self.last_terrain_result = terrain_nav_result  # Store for forced action override
                    memory['terrain_goal'] = terrain_goal
                    
                    # ═══════════════════════════════════════════════════════════════════
                    # 🚨 HOUSE STUCK TRACKING - Count steps in house after getting starter
                    # ═══════════════════════════════════════════════════════════════════
                    if not hasattr(self, 'steps_in_house'):
                        self.steps_in_house = 0
                        self.last_house_map = None
                    
                    in_house_now = 'HOUSE' in location_upper or 'BEDROOM' in location_upper
                    if in_house_now and (party_size > 0 or getattr(self, 'has_starter', False)):
                        if map_id == self.last_house_map:
                            self.steps_in_house += 1
                        else:
                            self.steps_in_house = 1
                            self.last_house_map = map_id
                    else:
                        self.steps_in_house = 0
                        self.last_house_map = None
                    
                    memory['steps_in_house'] = self.steps_in_house
                    
                    # ═══════════════════════════════════════════════════════════════════
                    
                    # STUCK DETECTION - Track position and action patterns
                    current_pos = (x, y, map_id)
                    self.position_history.append(current_pos)
                    self.action_history.append(action_name)
                    
                    # Keep only recent history
                    if len(self.position_history) > self.position_history_size:
                        self.position_history.pop(0)
                        self.action_history.pop(0)
                    
                    # ═══════════════════════════════════════════════════════════════
                    # 🔴 CONSECUTIVE A-PRESS STUCK DETECTION
                    # If pressing A too many times, suggest B to escape menus
                    # (Fixes: stuck in Pokemon summary screen during battle)
                    # ═══════════════════════════════════════════════════════════════
                    consecutive_a_count = 0
                    for act in reversed(self.action_history):
                        if act == 'A':
                            consecutive_a_count += 1
                        else:
                            break
                    
                    a_stuck = consecutive_a_count >= 5  # 5+ A presses = stuck
                    b_suggested = a_stuck  # Suggest B when stuck on A
                    
                    if a_stuck and self.step_count % 10 == 0:
                        logger.warning(f"⚠️ A-STUCK: {consecutive_a_count} consecutive A presses! Suggesting B to escape.")
                    
                    memory['consecutive_a_count'] = consecutive_a_count
                    memory['a_stuck'] = a_stuck
                    memory['b_suggested'] = b_suggested
                    
                    # Calculate reward
                    reward = 0.0 + stuck_penalty  # Apply stuck penalty (0 if not stuck, -10 to -100 if stuck)
                    info_parts = []
                    if stuck_penalty < 0:
                        info_parts.append(f"🔄 STUCK: {stuck_penalty:.0f}")
                    
                    # A-STUCK OVERRIDE: Penalize A and reward B when stuck
                    if a_stuck:
                        if action_name == 'A':
                            reward -= 50.0  # Penalize continuing to press A
                            info_parts.append(f"⚠️ A-STUCK ({consecutive_a_count}x): A -50")
                        elif action_name == 'B':
                            reward += 100.0  # Reward pressing B to escape
                            info_parts.append(f"✅ B to escape A-stuck! +100")

                    
                    # DIALOG STATE HANDLING - Reward progressing dialog, small penalty for wrong buttons
                    # Note: PokemonRedExperiments doesn't special-case dialog, treating A/B as normal actions
                    # However, we slightly encourage dialog progression since it can unlock story events
                    if game_state == 'dialog':
                        # In dialog - reward A and B presses to advance dialog
                        if action_name in ['A', 'B']:
                            reward += 2.0  # Reduced from 3.0 - dialog can be helpful but not overprioritized
                            info_parts.append(f"💬 Dialog {action_name} +2.0")
                        else:
                            # Very small penalty for movement during dialog (can't move anyway)
                            reward -= 0.1  # Reduced from 0.5 - not too harsh
                    
                    # 🚨 HOUSE STUCK PENALTY - Growing penalty for staying in house too long after getting starter
                    # Also track if stuck in Littleroot house specifically (for exit rewards)
                    in_littleroot_house = in_house_now and ('LITTLEROOT' in location_upper or map_id in [0x09, 0x0A, 0x0B, 0x0C])
                    if self.steps_in_house > 20:
                        house_penalty = -2.0 * (self.steps_in_house - 20)  # -2 per extra step
                        house_penalty = max(house_penalty, -200.0)  # Cap at -200
                        reward += house_penalty
                        if self.step_count % 20 == 0:
                            info_parts.append(f"🚨 STUCK IN HOUSE x{self.steps_in_house} {house_penalty:.0f}")
                    
                    # 🚪 EXIT-HOUSE REWARD: When in Littleroot house and seeking rival, reward actions toward exit
                    milestone_status = memory.get('milestones', {})
                    first_rival_pending = not milestone_status.get('FIRST_RIVAL_BATTLE', False)
                    if in_littleroot_house and first_rival_pending and party_size > 0:
                        # Use terrain nav result (ground truth pathfinding to stairs/exit)
                        terrain_nav_result = memory.get('terrain_nav_result')
                        
                        if terrain_nav_result:
                            # Terrain nav knows the correct direction to exit!
                            correct_dir = terrain_nav_result.get('action')
                            if action_name == correct_dir:
                                exit_reward = 2.0 + (0.1 * min(self.steps_in_house, 20))  # Up to +4.0 if stuck
                                reward += exit_reward
                                info_parts.append(f"🚪 EXIT HOUSE: {correct_dir} +{exit_reward:.1f}")
                            else:
                                # Wrong direction penalty
                                reward -= 0.5
                        else:
                            # Fallback: On 2F, stairs are at (8,2) so usually need UP; on 1F need DOWN
                            if '2F' in location_upper or 'BEDROOM' in location_upper:
                                # 2F: Stairs typically at y=2, so need UP if y > 2
                                if y > 2 and action_name == 'UP':
                                    exit_reward = 2.0 + (0.1 * min(self.steps_in_house, 20))
                                    reward += exit_reward
                                    info_parts.append(f"🚪 EXIT 2F: UP to stairs +{exit_reward:.1f}")
                                elif y <= 2 and action_name == 'A':
                                    # At stairs, press A to use them
                                    exit_reward = 3.0
                                    reward += exit_reward
                                    info_parts.append(f"🚪 USE STAIRS: A +{exit_reward:.1f}")
                            else:
                                # 1F: Exit door typically at bottom (y=8)
                                if action_name == 'DOWN':
                                    exit_reward = 0.8 + (0.1 * min(self.steps_in_house, 20))
                                    reward += exit_reward
                                    info_parts.append(f"🚪 EXIT 1F: DOWN +{exit_reward:.1f}")
                    
                    # 🚨 STUCK-IN-HOUSE DETECTOR: Progressive penalty after 50 steps
                    if in_littleroot_house and first_rival_pending and self.steps_in_house > 50:
                        stuck_penalty = -0.2 * (self.steps_in_house - 50)  # -0.2 per step after 50
                        stuck_penalty = max(stuck_penalty, -50.0)  # Cap at -50
                        reward += stuck_penalty
                        if self.step_count % 10 == 0:
                            info_parts.append(f"⚠️ STUCK IN HOUSE {self.steps_in_house} steps {stuck_penalty:.1f}")
                    
                    # 🧹 AGGRESSIVE HOUSE MEMORY PRUNING: Remove house entries from k-NN when stuck
                    if in_littleroot_house and first_rival_pending and self.steps_in_house > 100:
                        # Prune house-related navigation memories so agent doesn't recall going into houses
                        if hasattr(self, 'episodic_memory') and hasattr(self.episodic_memory, 'nav_memory'):
                            house_keywords = ['HOUSE', 'BEDROOM', '1F', '2F', 'MAYS', 'BRENDANS', 'PLAYER']
                            original_count = len(self.episodic_memory.nav_memory)
                            # Filter out memories that mention houses
                            self.episodic_memory.nav_memory = [
                                mem for mem in self.episodic_memory.nav_memory
                                if not any(kw in str(mem.get('location', '')).upper() for kw in house_keywords)
                            ]
                            pruned_count = original_count - len(self.episodic_memory.nav_memory)
                            if pruned_count > 0 and self.step_count % 50 == 0:
                                logger.warning(f"🧹 Pruned {pruned_count} house memories from k-NN (stuck {self.steps_in_house} steps)")
                                info_parts.append(f"🧹 Pruned {pruned_count} house memories")
                    
                    # 🧪 LAB STUCK FALLBACK: If in Birch's Lab but haven't WON rival battle, redirect
                    in_birch_lab = map_id == 0x104  # PROFESSOR_BIRCHS_LAB
                    if in_birch_lab and first_rival_pending and party_size > 0:
                        # Agent is trying to enter lab for Pokedex but hasn't beaten rival yet
                        if not hasattr(self, 'lab_stuck_counter'):
                            self.lab_stuck_counter = 0
                        self.lab_stuck_counter += 1
                        
                        if self.lab_stuck_counter > 30:
                            # Strong penalty for wrong objective - should be at Route 103
                            lab_penalty = -5.0 - (0.5 * (self.lab_stuck_counter - 30))  # Increasing penalty
                            lab_penalty = max(lab_penalty, -50.0)  # Cap
                            reward += lab_penalty
                            if self.step_count % 10 == 0:
                                logger.warning(f"⚠️ LAB STUCK: In lab x{self.lab_stuck_counter} steps without winning rival battle! Go to Route 103!")
                                info_parts.append(f"⚠️ LAB STUCK: Go to Route 103! {lab_penalty:.1f}")
                    else:
                        # Reset counter when not in lab
                        if hasattr(self, 'lab_stuck_counter'):
                            self.lab_stuck_counter = 0

                    
                    # ═══════════════════════════════════════════════════════════════════
                    # ⚡ HEALING INTERRUPT REWARD OVERRIDE ⚡
                    # When HP is critical, override ALL other rewards with healing pathfinding
                    # ═══════════════════════════════════════════════════════════════════
                    
                    if healing_interrupt_active and healing_target:
                        heal_type = healing_target['type']
                        heal_coords = healing_target.get('coords')
                        
                        # GROUND TRUTH PATHFINDING for each healing target
                        if heal_type == 'NURSE_JOY':
                            # Inside PokeCenter - go UP to nurse, press A
                            if y <= 4:  # At nurse counter
                                if action_name == 'A':
                                    reward += 200.0  # HUGE reward for healing!
                                    info_parts.append("💚 HEALING AT NURSE JOY! +200")
                                    # THIS IS WHERE RESPAWN IS SET - when talking to Nurse Joy!
                                    if not getattr(self, 'visited_oldale_pokecenter', False):
                                        self.visited_oldale_pokecenter = True
                                        reward += 500.0  # MASSIVE bonus for first time!
                                        info_parts.append("🏥 RESPAWN SET AT OLDALE POKECENTER! +500")
                                        logger.info("✅ RESPAWN POINT SET BY TALKING TO NURSE JOY!")
                                elif action_name == 'UP':
                                    reward += 50.0
                            else:
                                if action_name == 'UP':
                                    reward += 100.0
                                    info_parts.append("🏥 POKECENTER: Going UP to nurse")
                                elif action_name == 'DOWN':
                                    reward -= 50.0  # Don't exit!
                                    
                        elif heal_type == 'MOM':
                            # In player's house 1F - find Mom and talk
                            if action_name == 'A':
                                reward += 150.0  # Talk to Mom
                                info_parts.append("👩 TALKING TO MOM FOR HEAL")
                            elif action_name in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                                reward += 20.0  # Move to find Mom
                                
                        elif heal_type == 'GO_DOWNSTAIRS':
                            # In bedroom - go to stairs
                            if action_name == 'DOWN':
                                reward += 80.0
                                info_parts.append("🏠 Going downstairs to Mom")
                            else:
                                reward -= 20.0
                                
                        elif heal_type == 'OLDALE_PC':
                            # Heading to Oldale PokeCenter
                            if heal_coords:
                                pc_x, pc_y = heal_coords
                                # Check if at PC door
                                if abs(x - pc_x) <= 2 and abs(y - pc_y) <= 2:
                                    if action_name == 'A':
                                        reward += 100.0
                                        info_parts.append("🏥 ENTERING POKECENTER!")
                                else:
                                    # Pathfind to PC
                                    best_dir = get_best_direction(x, y, pc_x, pc_y)
                                    if best_dir and action_name == best_dir:
                                        reward += 80.0
                                        info_parts.append(f"🏥 HEAL PATH: {best_dir} to PC")
                                    elif action_name in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                                        reward -= 20.0
                                        
                        elif heal_type == 'GO_SOUTH':
                            # Route 103 → Oldale: go SOUTH
                            if action_name == 'DOWN':
                                reward += 100.0
                                info_parts.append("🚨 EMERGENCY: Going SOUTH to heal")
                            elif action_name == 'UP':
                                reward -= 80.0  # WRONG WAY when critical HP!
                            else:
                                reward -= 20.0
                                
                        elif heal_type == 'PLAYER_HOUSE':
                            # In Littleroot - go to player's house
                            if heal_coords:
                                house_x, house_y = heal_coords
                                if abs(x - house_x) <= 2 and abs(y - house_y) <= 2:
                                    if action_name == 'A':
                                        reward += 80.0
                                        info_parts.append("🏠 ENTERING HOUSE FOR MOM")
                                else:
                                    best_dir = get_best_direction(x, y, house_x, house_y)
                                    if best_dir and action_name == best_dir:
                                        reward += 60.0
                                    elif action_name in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                                        reward -= 10.0
                                        
                        elif heal_type in ['FIND_HEALING', 'OLDALE_PC'] and healing_target.get('action') in ['UP', 'DOWN']:
                            # Generic direction to healing
                            target_dir = healing_target['action']
                            if action_name == target_dir:
                                reward += 80.0
                                info_parts.append(f"🚨 HEAL INTERRUPT: {target_dir}")
                            elif action_name in ['UP', 'DOWN', 'LEFT', 'RIGHT'] and action_name != target_dir:
                                reward -= 40.0
                    
                    # ═══════════════════════════════════════════════════════════════════
                    # 🚪 FORCED HOUSE EXIT REWARD OVERRIDE - DISABLED
                    # This system is obsolete - directive-based SLAM pathfinding handles exits now.
                    # The directive EXIT_HOUSE (line ~2790) uses portal connections + terrain data.
                    # This old hardcoded reward was overriding the new portal-based system!
                    # ═══════════════════════════════════════════════════════════════════
                    
                    # if force_house_exit and house_exit_target:  # DISABLED - house_exit_target is now always None
                    #     # Old hardcoded reward system removed
                    #     pass
                    
                    # ═══════════════════════════════════════════════════════════════════
                    # 🧱 ROUTE 101 LEDGE REWARD OVERRIDE
                    # When in ledge zone, override VLM with ground truth rewards
                    # ═══════════════════════════════════════════════════════════════════
                    
                    if route101_ledge_interrupt and ledge_target:
                        target_action = ledge_target['action']
                        
                        # ★★★ CRITICAL: During ledge escape, this reward MUST override terrain nav! ★★★
                        # STRONG rewards for following waypoint navigation
                        if action_name == target_action:
                            reward += 300.0  # VERY HIGH reward to override terrain nav!
                            info_parts.append(f"🧱 WAYPOINT: {target_action} +300")
                        elif action_name == 'A':
                            reward += 5.0  # Neutral for A (might interact)
                        else:
                            # Penalties based on how wrong the action is
                            opposite_actions = {
                                'UP': 'DOWN', 'DOWN': 'UP',
                                'LEFT': 'RIGHT', 'RIGHT': 'LEFT'
                            }
                            
                            if action_name == opposite_actions.get(target_action):
                                # Going exactly opposite direction - VERY WRONG during ledge escape!
                                # Must be higher than terrain_nav's +200 to actually work!
                                reward -= 400.0
                                info_parts.append(f"❌ OPPOSITE! Need {target_action} not {action_name}!")
                            elif action_name in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                                # Perpendicular movement - less wrong but still not ideal
                                reward -= 100.0
                                info_parts.append(f"⚠️ Wrong direction - need {target_action}")
                            else:
                                reward -= 20.0  # B button etc

                    
                    # ═══════════════════════════════════════════════════════════════════
                    # 🏥 OLDALE POKECENTER NAVIGATION REWARDS
                    # Hardcoded path to PokeCenter - sets respawn point!
                    # ═══════════════════════════════════════════════════════════════════
                    
                    if oldale_pokecenter_nav and oldale_target:
                        target_action = oldale_target['action']
                        
                        # Strong rewards for going to PokeCenter (highest priority!)
                        if action_name == target_action:
                            reward += 200.0  # HIGHEST reward for heading to PokeCenter!
                            info_parts.append(f"🏥 PC NAV: {target_action} +200")
                        elif action_name == 'A':
                            reward += 10.0  # Might be entering door
                            info_parts.append("🏥 A pressed - entering?")
                        else:
                            opposite_actions = {'UP': 'DOWN', 'DOWN': 'UP', 'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}
                            if action_name == opposite_actions.get(target_action):
                                reward -= 100.0
                                info_parts.append(f"❌ Wrong way to PC! Need {target_action}")
                            elif action_name in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                                reward -= 30.0
                                info_parts.append(f"⚠️ Not toward PC - need {target_action}")
                    
                    # ═══════════════════════════════════════════════════════════════════
                    # 🗺️ ROUTE 103 RIVAL NAVIGATION REWARDS
                    # DISABLED: Now using terrain A* which correctly navigates to rival at (12,3)
                    # The hardcoded logic assumed rival was to LEFT but rival is actually to RIGHT!
                    # Terrain A* in terrain_navigator.py handles this correctly.
                    # ═══════════════════════════════════════════════════════════════════
                    
                    # DISABLED - route103_target gave +200 for LEFT but rival is at (12,3) to RIGHT!
                    # if route103_nav and route103_target:
                    #     target_action = route103_target['action']
                    #     
                    #     if action_name == target_action:
                    #         reward += 200.0  # STRONG reward for following terrain guidance (match Route 101)
                    #         info_parts.append(f"🗺️ R103 WAYPOINT: {target_action} +200")
                    #     elif action_name == 'A':
                    #         reward += 50.0  # Might be talking to rival!
                    #         info_parts.append("🗺️ A pressed - initiating battle?")
                    #     else:
                    #         opposite_actions = {'UP': 'DOWN', 'DOWN': 'UP', 'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}
                    #         if action_name == opposite_actions.get(target_action):
                    #             reward -= 150.0  # STRONG penalty for opposite (match Route 101)
                    #             info_parts.append(f"❌ OPPOSITE on R103! Need {target_action}")
                    #         elif action_name in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                    #             reward -= 50.0
                    #             info_parts.append(f"⚠️ Off-path on R103 - need {target_action}")

                    
                    # ═══════════════════════════════════════════════════════════════════
                    # 💊 NURSE JOY NAVIGATION REWARDS
                    # Inside PokeCenter - guide to Nurse Joy counter to heal
                    # ═══════════════════════════════════════════════════════════════════
                    
                    if nurse_joy_nav and nurse_target:
                        target_action = nurse_target['action']
                        
                        if action_name == target_action:
                            if target_action == 'A':
                                reward += 300.0  # HUGE reward for talking to Nurse Joy!
                                info_parts.append("💊 NURSE JOY! A +300")
                            else:
                                reward += 200.0  # Good reward for moving to counter
                                info_parts.append(f"💊 NURSE NAV: {target_action} +200")
                        else:
                            # Penalize wrong direction in PokeCenter
                            opposite_actions = {'UP': 'DOWN', 'DOWN': 'UP', 'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}
                            if action_name == opposite_actions.get(target_action):
                                reward -= 100.0
                                info_parts.append(f"❌ Wrong way in PC! Need {target_action}")
                            elif action_name in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                                reward -= 30.0
                                info_parts.append(f"⚠️ Not toward Nurse - need {target_action}")
                    
                    # ═══════════════════════════════════════════════════════════════════
                    # 🚫 POKECENTER 2F PENALTY - 2F is USELESS for speedrun (trading only)
                    # Penalize being on 2F and reward leaving ASAP
                    # ═══════════════════════════════════════════════════════════════════
                    
                    if in_pokecenter_2f:
                        # Track steps spent on 2F for increasing penalty
                        if not hasattr(self, 'steps_on_2f'):
                            self.steps_on_2f = 0
                        self.steps_on_2f += 1
                        
                        # HEAVY penalty for wasting time on 2F - increases with time!
                        base_penalty = -300.0
                        time_penalty = -10.0 * min(self.steps_on_2f, 50)  # Up to -500 extra
                        total_2f_penalty = base_penalty + time_penalty
                        reward += total_2f_penalty
                        info_parts.append(f"🚫 2F PENALTY: {total_2f_penalty:.0f} (step {self.steps_on_2f})")
                        
                        # Reward for going LEFT toward escalator (exit)
                        if action_name == 'LEFT':
                            reward += 500.0  # Increased from 200
                            info_parts.append("🪜 Going LEFT to exit 2F! +500")
                        elif action_name == 'DOWN':
                            # DOWN is also needed to reach escalator at y=6
                            reward += 300.0
                            info_parts.append("🪜 Going DOWN toward escalator! +300")
                        elif action_name == 'A':
                            # HEAVY PENALTY for pressing A on 2F - trading/PC is useless!
                            # Nothing to trade, no Pokemon in PC, just wasting time
                            reward -= 500.0
                            info_parts.append("🚫 A on 2F = WASTE! -500 (no trading/PC needed)")
                        
                        # FORCED ACTION OVERRIDE when stuck on 2F for too long!
                        # If steps_on_2f > 50, massively boost LEFT/DOWN and penalize UP
                        if self.steps_on_2f > 50:
                            if action_name == 'LEFT':
                                reward += 1000.0  # MASSIVE override to force LEFT toward escalator
                                info_parts.append("🚨 2F FORCE LEFT +1000!")
                            elif action_name == 'DOWN':
                                reward += 1000.0  # Also good for escalator
                                info_parts.append("🚨 2F FORCE DOWN +1000!")
                            elif action_name == 'UP':
                                reward -= 500.0  # VERY strong penalty for UP on 2F
                                info_parts.append("🚨 2F UP BLOCK -500!")
                            elif action_name == 'RIGHT':
                                reward -= 300.0  # Penalty for going further right (away from escalator)
                                
                        # Track position to detect complete stuckness  
                        if not hasattr(self, 'last_2f_pos'):
                            self.last_2f_pos = (x, y)
                        if (x, y) != self.last_2f_pos:
                            # Position changed - reward movement!
                            reward += 100.0
                            info_parts.append("✓ 2F movement +100")
                        self.last_2f_pos = (x, y)
                            
                    else:
                        # Reset 2F counter when not on 2F
                        if hasattr(self, 'steps_on_2f'):
                            self.steps_on_2f = 0
                        if hasattr(self, 'last_2f_pos'):
                            self.last_2f_pos = None


                    
                    # ═══════════════════════════════════════════════════════════════════
                    # 🧭 UNIFIED TERRAIN NAVIGATOR REWARDS
                    # Applies rewards based on TerrainNavigator's cached A* pathfinding
                    # ═══════════════════════════════════════════════════════════════════
                    
                    terrain_nav_result = memory.get('terrain_nav_result')
                    terrain_nav_active = memory.get('terrain_nav_active', False)
                    terrain_goal = memory.get('terrain_goal')
                    
                    if terrain_nav_active and terrain_nav_result:
                        target_action = terrain_nav_result['action']
                        
                        # MASSIVE rewards for house exits (to overcome random early policy)
                        is_house_exit = 'HOUSE' in location_upper or 'BEDROOM' in location_upper
                        house_exit_multiplier = 2.0 if is_house_exit else 1.0
                        
                        if action_name == target_action:
                            if target_action == 'A':
                                # Interaction (talking to NPC, entering door)
                                base_reward = 300.0 * house_exit_multiplier
                                reward += base_reward
                                info_parts.append(f"🧭 TERRAIN: A (interact) +{base_reward:.0f}")
                            else:
                                # Movement in right direction - UP TO +400 FOR HOUSE EXITS!
                                base_reward = 200.0 * house_exit_multiplier
                                reward += base_reward
                                info_parts.append(f"🧭 TERRAIN: {target_action} +{base_reward:.0f} ({terrain_goal})")
                        else:
                            # Wrong direction - penalties also scaled for houses
                            opposite_actions = {'UP': 'DOWN', 'DOWN': 'UP', 'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}
                            if action_name == opposite_actions.get(target_action):
                                penalty = 150.0 * house_exit_multiplier  # Up to -300 for house
                                reward -= penalty
                                info_parts.append(f"❌ TERRAIN: opposite! Need {target_action}")
                            elif action_name in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                                penalty = 50.0 * house_exit_multiplier  # Up to -100 for house
                                reward -= penalty
                                info_parts.append(f"⚠️ TERRAIN: wrong dir, need {target_action}")
                    
                    
                    # MILESTONE REWARDS (HIGHEST PRIORITY - like train_rl_event_flags.py)
                    try:
                        milestones_resp = requests.get(f"{self.server_url}/milestones", timeout=15).json()
                        if milestones_resp:
                            # Update milestone tracker for SNN temporal reasoning
                            if hasattr(self, 'milestone_tracker'):
                                self.milestone_tracker.update(milestones_resp)
                            
                            completed_now = sum(1 for v in milestones_resp.values() if v)
                            if not hasattr(self, 'last_milestone_count'):
                                self.last_milestone_count = completed_now
                                self.completed_rival_bedroom = milestones_resp.get('RIVAL_BEDROOM', False)
                            
                            if completed_now > self.last_milestone_count:
                                new_milestones = completed_now - self.last_milestone_count
                                milestone_reward = new_milestones * 100.0  # HUGE reward!
                                reward += milestone_reward
                                info_parts.append(f"🎯 MILESTONE +{milestone_reward}")
                                self.last_milestone_count = completed_now
                                
                                # EXTRA bonus for RIVAL_BEDROOM (critical gatekeeper)
                                if not self.completed_rival_bedroom and milestones_resp.get('RIVAL_BEDROOM', False):
                                    reward += 50.0  # Extra bonus!
                                    info_parts.append("👥 RIVAL_BEDROOM unlocked! +50")
                                    self.completed_rival_bedroom = True
                    except:
                        pass
                    
                    # Coordinate exploration (PokemonRedExperiments-style)
                    coord_key = (x, y, map_id)
                    if coord_key not in self.seen_coords:
                        self.seen_coords[coord_key] = 1
                        # First visit bonus - INCREASED to encourage exploration
                        explore_reward = 3.0  # Was 2.0
                        info_parts.append(f"🆕 tile +{explore_reward:.1f}")
                        
                        # DIRECTIONAL BONUS: Exploring northern coordinates ONLY in outdoor Littleroot Town
                        # Route 101 entrance is at top/north of OUTDOOR town (low Y values like Y=0-3)
                        # NOT in Lab (0x104), Player House (0x100/0x101), or Rival House (0x102/0x103)
                        # This prevents rewarding northern coords inside buildings where it's just walls
                        if map_id == 0x09:  # ONLY outdoor Littleroot Town map
                            if y <= 3:  # Very north of outdoor town (Route 101 entrance area)
                                north_bonus = 10.0  # Increased from 5.0 - this is the goal!
                                reward += north_bonus
                                info_parts.append(f"🧭 FAR NORTH TOWN +{north_bonus:.1f}")
                            elif y <= 6:  # Northern half of outdoor town
                                north_bonus = 3.0  # Increased from 2.0
                                reward += north_bonus
                                info_parts.append(f"🧭 NORTH TOWN +{north_bonus:.1f}")
                    else:
                        self.seen_coords[coord_key] += 1
                        explore_reward = 0.0
                        visit_count = self.seen_coords[coord_key]
                        # Only penalize EXTREME looping (>400 visits, reduced from 600)
                        if visit_count > 400:
                            stuck_penalty = -0.1  # Increased penalty
                            reward += stuck_penalty
                            if visit_count % 100 == 0:
                                info_parts.append(f"🔁 STUCK x{visit_count} {stuck_penalty}")
                    
                    # CUMULATIVE exploration reward (PokemonRedExperiments key insight!)
                    # This gives continuous reward for maintaining high exploration count
                    cumulative_explore = len(self.seen_coords) * 0.1
                    reward += explore_reward + cumulative_explore
                    
                    # New map discovery - BIG REWARDS for entering buildings!
                    if map_id != self.last_map_id and self.last_map_id is not None:
                        # 🆕 MAP TRANSITION RESET: Fix SLAM/A* desync
                        # Reset stuck counters when map changes
                        self.stuck_persistence_counter = 0
                        self.stuck_direction_cycle = 0
                        if hasattr(self, 'global_action_counter'):
                            self.global_action_counter = {}
                        if hasattr(self, 'stuck_position_history'):
                            self.stuck_position_history.clear()
                        # Clear terrain cache to force A* recompute on new map
                        if hasattr(self, 'terrain_nav'):
                            self.terrain_nav.clear_cache()
                        logger.info(f"🗺️ MAP TRANSITION: Reset stuck counters & terrain cache")
                        
                        # Track map transition pattern to detect lab/town loops
                        if not hasattr(self, 'map_transition_history'):
                            self.map_transition_history = []
                        self.map_transition_history.append((self.last_map_id, map_id))
                        if len(self.map_transition_history) > 10:
                            self.map_transition_history.pop(0)
                        
                        # Check for repetitive lab<->town loops (0x104 <-> 0x09)
                        if len(self.map_transition_history) >= 4:
                            recent_transitions = self.map_transition_history[-4:]
                            lab_town_transitions = sum(1 for (a,b) in recent_transitions 
                                                      if (a == 0x104 and b == 0x09) or (a == 0x09 and b == 0x104))
                            if lab_town_transitions >= 3:
                                # Penalize excessive lab<->town loops without exploring
                                loop_penalty = -5.0
                                reward += loop_penalty
                                info_parts.append(f"🔄 Lab/Town LOOP {loop_penalty}")
                        
                        map_transition_bonus = 20.0  # Increased from 10!
                        reward += map_transition_bonus
                        prev_name = get_map_name(self.last_map_id) if self.last_map_id is not None else "START"
                        curr_name = get_map_name(map_id)
                        info_parts.append(f"🚪 {prev_name}→{curr_name} +{map_transition_bonus}")
                        
                        if map_id not in self.maps_discovered:
                            self.maps_discovered.add(map_id)
                            new_map_bonus = 30.0  # Extra for first time!
                            reward += new_map_bonus
                            info_parts.append(f"🗺️ NEW map +{new_map_bonus}")
                        
                        # EXTRA BONUS: Entering Rival House from town
                        if self.last_map_id == 0x09 and map_id == 0x102:
                            reward += 40.0
                            info_parts.append("🏠 Entered Rival House! +40")
                        
                        # MASSIVE BONUS: Reaching Rival Bedroom (2F)
                        if map_id == 0x103:
                            reward += 80.0
                            info_parts.append("🛏️ RIVAL BEDROOM! +80")
                        
                        # HUGE BONUS: Reaching Route 101 FIRST TIME (means story progressed!)
                        # Only give this once to avoid pulling agent back from Oldale
                        if map_id == 0x10:
                            if not hasattr(self, 'reached_route_101'):
                                reward += 100.0
                                info_parts.append("🌳 ROUTE 101! Story progress! +100")
                                self.reached_route_101 = True
                        
                        # 🏥 CRITICAL: OLDALE POKECENTER - Sets respawn point!
                        # Map ID 0x200 = Oldale PokeCenter 1F interior
                        # Also detect via location string (1F or 2F)
                        # Visiting saves respawn here instead of Littleroot on blackout!
                        in_oldale_pokecenter = (map_id == 0x200) or (in_pokecenter and 'OLDALE' in location_upper)
                        
                        # Debug logging to trace why flag isn't being set
                        if self.step_count % 50 == 0 and ('POKECENTER' in location_upper or 'POKEMON CENTER' in location_upper or map_id == 0x200):
                            logger.info(f"🏥 PC DEBUG: map_id={hex(map_id)}, in_pokecenter={in_pokecenter}, location={location_upper[:40]}, in_oldale_pc={in_oldale_pokecenter}")
                        
                        if in_oldale_pokecenter:
                            if not getattr(self, 'visited_oldale_pokecenter', False):
                                # Not yet talked to Nurse Joy - give small bonus for entering
                                reward += 50.0
                                info_parts.append("🏥 ENTERED PC - go UP to Nurse Joy! +50")
                                # SET FLAG FOR NEXT STEP: Force UP to Nurse Joy!
                                self.force_nurse_next = True
                                self.force_nurse_y = y  # Store current y for UP vs A decision
                            else:
                                # Already visited AND healed - but check HP first!
                                # If HP is low, agent SHOULD re-enter to heal
                                hp_needs_heal = getattr(self, 'cached_hp_ratio', 0.0) < 0.5  # Default 0.0 = assume low HP
                                if hp_needs_heal:
                                    reward += 50.0  # Allow healing!
                                    info_parts.append("💉 PC RE-ENTRY for HEALING (HP low)! +50")
                                else:
                                    reward -= 200.0
                                    info_parts.append("🚫 PC RE-ENTRY! Already healed! -200")
                    
                    # DIRECTIONAL EXPLORATION BONUS - encourage moving UP/NORTH
                    # In LITTLEROOT_TOWN (0x0101), Route 101 is at the TOP
                    if map_id == 0x0101:  # LITTLEROOT_TOWN
                        if not hasattr(self, 'last_y'):
                            self.last_y = y
                        if y < self.last_y:  # Moving UP (y decreases)
                            north_bonus = 0.5
                            reward += north_bonus
                            if self.step_count % 50 == 0:
                                info_parts.append(f"⬆️ north +{north_bonus}")
                        self.last_y = y
                    
                    # SPECIAL BONUS: Going upstairs in Rival House (before bedroom milestone)
                    if map_id == 0x0302 and not getattr(self, 'completed_rival_bedroom', False):  # RIVAL_BEDROOM_2F
                        if not hasattr(self, 'reached_rival_bedroom'):
                            reward += 20.0  # Big bonus for finding the bedroom!
                            info_parts.append("🏠 Found Rival's bedroom! +20")
                            self.reached_rival_bedroom = True
                    
                    self.last_map_id = map_id
                    
                    # Log rewards: ALWAYS log map transitions, otherwise every 50 steps
                    has_important_event = any(marker in ' '.join(info_parts) for marker in ['🚪', '🗺️', '🏠', '🛏️', '🏆', '🧱', '❌'])
                    if info_parts and (has_important_event or self.step_count % 50 == 0):
                        map_name = get_map_name(map_id)
                        logger.info(f"Step {self.step_count}: ({x},{y}) {map_name} | {' | '.join(info_parts)} | R={reward:.1f}")
                    
                    # STUCK DETECTION (PokemonRedExperiments-style)
                    is_stuck = False
                    if len(self.position_history) >= self.position_history_size:
                        # Check if positions are looping (same 3-4 positions repeated)
                        unique_positions = len(set(self.position_history[-12:]))
                        if unique_positions <= 3:
                            is_stuck = True
                            info_parts.append("🔒 STUCK detected!")
                    
                    # === STRATEGIC DIRECTIVES (RAILROADING) ===
                    # Strict guidance for early game "Rival Arc" without VLM overhead
                    try:
                        # Get party data from progress_context (correct path!)
                        progress_ctx = state.get('game', {}).get('progress_context', {})
                        party_size = progress_ctx.get('party_size', 0)
                        party_species = progress_ctx.get('party_species', [])
                        party_levels = progress_ctx.get('party_levels', [])
                        
                        # Build party_state for directive manager with ACTUAL HP!
                        # Get actual party data from player.party (NOT progress_ctx.party which is None!)
                        actual_party = state.get('player', {}).get('party', []) or []
                        party_state = []
                        for i in range(party_size):
                            # Use actual HP from party data if available
                            actual_mon = actual_party[i] if i < len(actual_party) and actual_party[i] else {}
                            pokemon = {
                                'species': party_species[i] if i < len(party_species) else 'UNKNOWN',
                                'level': party_levels[i] if i < len(party_levels) else 1,
                                'hp': actual_mon.get('current_hp', actual_mon.get('hp', 1)),
                                'max_hp': actual_mon.get('max_hp', 1) or 1
                            }
                            party_state.append(pokemon)
                        
                        # Extract completed milestones safely from server response
                        current_milestones = milestones_resp if 'milestones_resp' in locals() else {}
                        completed_names = []
                        if isinstance(current_milestones, dict):
                            m_data = current_milestones.get('milestones', [])
                            if isinstance(m_data, list):
                                # Format: {"milestones": [{"name": "X", "completed": true}, ...]}
                                completed_names = [m.get('name') for m in m_data if isinstance(m, dict) and m.get('completed')]
                            elif isinstance(m_data, dict):
                                # Format: {\"milestones\": {\"X\": {\"completed\": true}, ...}}
                                completed_names = [k for k, v in m_data.items() if isinstance(v, dict) and v.get('completed')]
                        
                        # SYNC: Add OLDALE_POKECENTER to completed_names when internal flag is set
                        # This syncs train script's visited_oldale_pokecenter flag with directive manager's milestone check
                        if getattr(self, 'visited_oldale_pokecenter', False) and 'OLDALE_POKECENTER' not in completed_names:
                            completed_names.append('OLDALE_POKECENTER')
                        
                        location_name = memory.get('location', 'unknown')
                        location_name_upper = (location_name or 'unknown').upper()
                        has_pokedex = memory.get('has_pokedex', False)
                        
                        # Call update_directive with list of completed milestone names AND pokedex flag
                        # CRITICAL: Pass rival_battle_won - use the OVERRIDDEN rival_battled value
                        # which has already been checked against pokedex_seen == 0
                        rival_battle_won = rival_battled  # Uses the already-overridden value
                        directive_info = self.directive_manager.update_directive(
                            party_state,
                            completed_names,
                            location_name,
                            self.step_count,
                            has_pokedex=has_pokedex,
                            rival_battle_won=rival_battle_won
                        )


                        
                        if directive_info:
                            self.last_directive = directive_info
                            # The 'reason' field contains the specific navigational instruction
                            target = directive_info.get('reason', '')
                            priority = 10  # Implicit high priority for these directives
                            
                            # Cache directive target for UNIVERSAL HEALING OVERRIDE in process_action()
                            self.cached_directive_target = target
                            
                            # Log directive periodically
                            if self.step_count % 50 == 0:
                                directive_type = directive_info.get('directive', 'UNKNOWN')
                                logger.info(f"🧭 DIRECTIVE: {directive_type} - {target}")
                            
                            # GROUND TRUTH DOOR COORDINATES (Littleroot/Oldale/Routes)
                            # Complete journey pathfinding: 2F→1F→town→routes→rival→back
                            DOOR_COORDS = {
                                # HOUSE INTERIORS (for 2F→1F→exit)
                                'player_house_2f': {
                                    'stairs': (8, 2),       # Brendan's house 2F - stairs
                                },
                                'player_house_1f': {
                                    'door': (8, 8),         # Brendan's house 1F - exit door
                                    'mom': (5, 5),          # Mom for healing
                                },
                                'rival_house_2f': {
                                    'stairs': (8, 1),       # Go down to 1F
                                },
                                'rival_house_1f': {
                                    'door': (3, 8),         # Exit door (LEFT-BOTTOM side, visual position)
                                },
                                # LITTLEROOT TOWN (PORTAL GROUND TRUTH!)
                                # For Girl: May's(14,9)=player, Brendan's(8,8)=rival
                                # For Boy: Brendan's(8,8)=player, May's(14,9)=rival
                                'littleroot': {
                                    # Use server portal connections for gender-aware detection!
                                    'brendan_house': (8, 8),   # Portal ground truth
                                    'may_house': (14, 9),      # Portal ground truth
                                    'player_house': (8, 8),    # Fallback (Boy default)
                                    'rival_house': (8, 8),     # WRONG - need gender check! Use portal!
                                    'lab': (7, 17),            # Portal ground truth
                                    'route_101_north': (11, 0) # GROUND TRUTH from server!
                                },
                                # OLDALE TOWN
                                'oldale': {
                                    'pokecenter': (7, 13),
                                    'mart': (14, 13),
                                    'route_101_south': (0, 9),   # Exit south to Route 101
                                    'route_102_west': (0, 7),    # Exit west to Route 102
                                    'route_103_north': (10, 0)   # Exit north to Route 103
                                },
                                # ROUTE 101 WAYPOINTS (use server portal data when available!)
                                'route_101': {
                                    'south_entry': (10, 19),     # GROUND TRUTH! From Littleroot (server portal)
                                    'north_exit': (8, 0),        # To Oldale
                                    'safe_corridor_x': 8,        # Safe x to avoid ledges
                                },
                                # ROUTE 103 (to find rival)
                                'route_103': {
                                    'rival_area': (25, 5),       # May/Brendan location
                                    'south_exit': (5, 15),       # Back to Oldale
                                }
                            }
                            
                            # HEALING LOCATIONS (Mom's house + PokeCenters)
                            # Mom heals for free in Littleroot!
                            HEALING_COORDS = {
                                'littleroot': ('mom', 8, 8),    # Player's house - Mom heals!
                                'oldale': ('pokecenter', 7, 13),
                                'petalburg': ('pokecenter', 20, 17),
                                'rustboro': ('pokecenter', 16, 20)
                            }
                            
                            # Helper: Get best direction toward target coordinates
                            def get_best_direction(current_x, current_y, target_x, target_y):
                                dx = target_x - current_x
                                dy = target_y - current_y
                                # Prioritize larger distance
                                if abs(dx) > abs(dy):
                                    return 'RIGHT' if dx > 0 else 'LEFT'
                                elif dy != 0:
                                    return 'DOWN' if dy > 0 else 'UP'
                                elif dx != 0:
                                    return 'RIGHT' if dx > 0 else 'LEFT'
                                return None  # Already at target
                            
                            # EARLY GAME AREAS (hardcoded coords available)
                            EARLY_GAME_AREAS = ['LITTLEROOT', 'OLDALE', 'ROUTE 101', 'ROUTE 102', 'ROUTE 103', 'PETALBURG']
                            
                            def is_early_game(location):
                                loc_upper = (location or '').upper()
                                return any(area in loc_upper for area in EARLY_GAME_AREAS)
                            
                            # Reward based on directive instruction string
                            directive_reward = 0.0
                            
                            # PRIORITY: EXIT_LAB must be checked BEFORE GO_NORTH (since EXIT_LAB_GO_NORTH contains both!)
                            if 'EXIT_LAB' in target:
                                # Get current position in lab
                                current_x = memory.get('x', 0)
                                current_y = memory.get('y', 0)
                                location_upper = (memory.get('location', '') or '').upper()
                                
                                # Lab door is typically at bottom-center (similar to houses)
                                # Use portal connection if available!
                                if 'LAB' in location_upper:
                                    # Try to get door coordinates from server connections
                                    connections = memory.get('connections', [])
                                    # Look for exits from lab (to Littleroot)
                                    lab_exit = None
                                    for conn in connections:
                                        if 'LITTLEROOT' in conn.get('to', '').upper():
                                            from_pos = conn.get('from_pos', [])
                                            if len(from_pos) >= 2:
                                                lab_exit = (int(from_pos[0]), int(from_pos[1]))
                                                break
                                    
                                    # Fallback: Lab doors are typically at bottom-center
                                    if not lab_exit:
                                        lab_exit = (6, 14)  # Typical lab door position
                                    
                                    # Use SLAM pathfinding with terrain awareness
                                    best_dir = get_best_direction(current_x, current_y, lab_exit[0], lab_exit[1])
                                    
                                    # Check terrain to avoid walls
                                    map_data = memory.get('map_data', {})
                                    terrain = map_data.get('terrain', [])
                                    
                                    if best_dir and action_name == best_dir:
                                        directive_reward = 100.0  # Correct direction toward door
                                        if self.step_count % 50 == 0:
                                            logger.info(f"🚪 EXIT LAB: ({current_x},{current_y}) → door({lab_exit[0]},{lab_exit[1]}) = {best_dir} (+100)")
                                    elif action_name == 'DOWN':
                                        # DOWN is generally good (going toward exit)
                                        directive_reward = 60.0
                                        logger.info(f"🚪 EXIT LAB: DOWN (+60 general)")
                                    elif action_name in ['UP', 'LEFT', 'RIGHT']:
                                        # Check if terrain allows this direction (might be valid detour)
                                        directive_reward = -30.0  # Moderate penalty (not -50, might be avoiding obstacle)
                                        if self.step_count % 50 == 0:
                                            logger.info(f"🚫 EXIT LAB: {action_name} (-30, away from door)")
                                    elif action_name == 'A':
                                        # Check if at door (within 2 tiles)
                                        at_door = abs(current_x - lab_exit[0]) <= 2 and abs(current_y - lab_exit[1]) <= 2
                                        if at_door:
                                            directive_reward = 100.0  # Press A at door!
                                            logger.info(f"🚪 EXIT LAB: A at door (+100)")
                                        else:
                                            directive_reward = 10.0  # Might be interacting with something
                                else:
                                    # Not in lab anymore but directive still active
                                    # Just give small rewards to get out of Littleroot
                                    if action_name == 'UP':
                                        directive_reward = 20.0  # Going north toward Route 101
                            
                            # If directive is movement-based (e.g., GO_NORTH_TO_ROUTE_101)
                            elif 'GO_NORTH' in target:
                                current_x = memory.get('x', 0)
                                current_y = memory.get('y', 0)
                                location_upper = (memory.get('location', '') or '').upper()
                                
                                # GROUND TRUTH for Littleroot Town outdoor
                                if 'LITTLEROOT' in location_upper and 'HOUSE' not in location_upper and 'LAB' not in location_upper:
                                    # Use REAL-TIME portal connection from server (ground truth!)
                                    connections = memory.get('connections', [])
                                    dynamic_exit = self.get_exit_for_objective(connections, 'route101')
                                    
                                    if dynamic_exit:
                                        target_x, target_y = dynamic_exit
                                        if self.step_count % 100 == 0:
                                            logger.info(f"🌐 PORTAL: LITTLEROOT → ROUTE_101 at ({target_x},{target_y}) [SERVER]")
                                    else:
                                        # Fallback to hardcoded if server data unavailable
                                        target_x, target_y = 11, 0  # Updated from your ground truth!
                                    
                                    # Track stuck state
                                    if not hasattr(self, 'littleroot_stuck_count'):
                                        self.littleroot_last_pos = (current_x, current_y)
                                        self.littleroot_stuck_count = 0
                                    
                                    if (current_x, current_y) == self.littleroot_last_pos:
                                        self.littleroot_stuck_count += 1
                                    else:
                                        self.littleroot_stuck_count = 0
                                        self.littleroot_last_pos = (current_x, current_y)
                                    
                                    is_littleroot_stuck = self.littleroot_stuck_count > 5
                                    
                                    # Use ground truth pathfinding with CNN grid search awareness
                                    best_dir = get_best_direction(current_x, current_y, target_x, target_y)
                                    
                                    # Get terrain data from CNN grid search for obstacle awareness
                                    map_data = memory.get('map_data', {})
                                    terrain_info = map_data.get('terrain', [])
                                    
                                    if is_littleroot_stuck:
                                        # SLAM+CNN+SNN: Multi-modal pathfinding when stuck
                                        # - SLAM: Position tracking & loop closure
                                        # - CNN: Grid search terrain identification
                                        # - SNN: Temporal sequence reasoning (stuck pattern)
                                        # - VLM: High-level objective (Route 101)
                                        if best_dir and action_name == best_dir:
                                            directive_reward = 100.0  # Very strong for correct direction!
                                            if self.step_count % 50 == 0:
                                                logger.info(f"🗺️🧠 SLAM+CNN: ({current_x},{current_y}) → portal({target_x},{target_y}) = {best_dir} | terrain={len(terrain_info)} tiles")
                                        elif action_name in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                                            directive_reward = -40.0  # Strong penalty for wrong dir
                                        elif action_name == 'NOOP':
                                            directive_reward = -60.0
                                    else:
                                        # Normal guidance - but still strong to override exploration!
                                        if best_dir and action_name == best_dir:
                                            directive_reward = 80.0  # Strong reward for correct direction
                                            if self.step_count % 100 == 0:
                                                logger.info(f"🗺️ LITTLEROOT: ({current_x},{current_y}) → portal({target_x},{target_y}) = {best_dir} (+80)")
                                        elif action_name == 'UP':
                                            directive_reward = 30.0  # UP is good (going north!)
                                        elif action_name in ['LEFT', 'RIGHT']:
                                            # Only reward lateral movement if it's closer to target
                                            if action_name == 'RIGHT' and current_x < target_x:
                                                directive_reward = 40.0  # Need to go right
                                            elif action_name == 'LEFT' and current_x > target_x:
                                                directive_reward = 40.0  # Need to go left
                                            else:
                                                directive_reward = -20.0  # Wrong lateral direction
                                        elif action_name == 'DOWN':
                                            directive_reward = -50.0  # WRONG direction - strong penalty!
                                # Route 101 → Oldale pathfinding (also use server portals!)
                                elif 'ROUTE' in location_upper and '101' in location_upper:
                                    # Use real-time portal to Oldale from server
                                    connections = memory.get('connections', [])
                                    dynamic_exit = self.get_exit_for_objective(connections, 'oldale')
                                    
                                    if dynamic_exit:
                                        target_x, target_y = dynamic_exit
                                        best_dir = get_best_direction(current_x, current_y, target_x, target_y)
                                        # Check if terrain A* conflicts with simple best_dir
                                        terrain_result = memory.get('terrain_nav_result')
                                        terrain_conflicts = terrain_result and terrain_result.get('action') != best_dir
                                        
                                        if best_dir and action_name == best_dir:
                                            if terrain_conflicts:
                                                # Terrain knows about obstacles/ledges - reduced reward
                                                directive_reward = 20.0
                                            else:
                                                directive_reward = 80.0  # Strong guidance on Route 101
                                            if self.step_count % 100 == 0:
                                                logger.info(f"🌐 PORTAL: ROUTE_101 → OLDALE at ({target_x},{target_y}) = {best_dir}")
                                        elif action_name == 'UP':
                                            # DEFER TO TERRAIN: If terrain says something different, respect it!
                                            if terrain_conflicts:
                                                # Terrain knows about obstacles/ledges, don't reward UP
                                                directive_reward = 0.0  # Let terrain handle it
                                            else:
                                                directive_reward = 60.0  # General north is good
                                    elif action_name == 'UP':
                                        # DEFER TO TERRAIN: If terrain says something different, respect it!
                                        terrain_result = memory.get('terrain_nav_result')
                                        if terrain_result and terrain_result.get('action') != 'UP':
                                            directive_reward = 0.0  # Let terrain handle it
                                        else:
                                            directive_reward = 40.0  # Fallback: go north
                                
                                # Oldale → Route 103 pathfinding
                                elif 'OLDALE' in location_upper:
                                    # Use real-time portal to Route 103 from server
                                    connections = memory.get('connections', [])
                                    dynamic_exit = self.get_exit_for_objective(connections, 'route103')
                                    
                                    if dynamic_exit:
                                        target_x, target_y = dynamic_exit
                                        best_dir = get_best_direction(current_x, current_y, target_x, target_y)
                                        if best_dir and action_name == best_dir:
                                            directive_reward = 80.0  # Strong guidance in Oldale
                                            if self.step_count % 100 == 0:
                                                logger.info(f"🌐 PORTAL: OLDALE → ROUTE_103 at ({target_x},{target_y}) = {best_dir}")
                                        elif action_name == 'UP':
                                            directive_reward = 50.0
                                    else:
                                        # Fallback to hardcoded (only if portal unavailable)
                                        connections = memory.get('connections', [])
                                        dynamic_exit = self.get_exit_for_objective(connections, 'route103')
                                        target_x, target_y = dynamic_exit if dynamic_exit else DOOR_COORDS['oldale']['route_103_north']
                                        best_dir = get_best_direction(current_x, current_y, target_x, target_y)
                                        if best_dir and action_name == best_dir:
                                            directive_reward = 60.0
                                        elif action_name == 'UP':
                                            directive_reward = 40.0
                                else:
                                    # Generic GO_NORTH fallback - BUT NOT INSIDE BUILDINGS!
                                    # Indoor navigation is handled by terrain_nav with ground truth A*
                                    if 'HOUSE' not in location_upper and 'LAB' not in location_upper and 'POKECENTER' not in location_upper:
                                        if action_name == 'UP':
                                            directive_reward = 20.0
                            
                            # GO_SOUTH: Return journey after beating rival
                            elif 'GO_SOUTH' in target:
                                current_x = memory.get('x', 0)
                                current_y = memory.get('y', 0)
                                location_upper = (memory.get('location', '') or '').upper()
                                
                                # Route 103 → Oldale
                                if 'ROUTE' in location_upper and '103' in location_upper:
                                    # Go south to Oldale (y increases = south)
                                    if action_name == 'DOWN':
                                        directive_reward = 30.0
                                    elif action_name in ['LEFT', 'RIGHT']:
                                        directive_reward = 5.0
                                    elif action_name == 'UP':
                                        directive_reward = -10.0  # Wrong way
                                        
                                # Oldale → Route 101
                                elif 'OLDALE' in location_upper:
                                    # Use portal connection
                                    connections = memory.get('connections', [])
                                    dynamic_exit = self.get_exit_for_objective(connections, 'route101')
                                    target_x, target_y = dynamic_exit if dynamic_exit else DOOR_COORDS['oldale']['route_101_south']
                                    
                                    best_dir = get_best_direction(current_x, current_y, target_x, target_y)
                                    if best_dir and action_name == best_dir:
                                        directive_reward = 30.0
                                        if self.step_count % 50 == 0:
                                            portal_src = "portal" if dynamic_exit else "fallback"
                                            logger.info(f"🗺️ OLDALE→101: ({current_x},{current_y}) → ({target_x},{target_y}) = {best_dir} [{portal_src}]")
                                    elif action_name == 'DOWN':
                                        directive_reward = 10.0
                                    elif action_name in ['LEFT', 'RIGHT']:
                                        directive_reward = 5.0
                                        
                                # Route 101 → Littleroot
                                elif 'ROUTE' in location_upper and '101' in location_upper:
                                    # Going SOUTH to Littleroot
                                    if action_name == 'DOWN':
                                        directive_reward = 30.0
                                    elif action_name in ['LEFT', 'RIGHT']:
                                        directive_reward = 5.0  # May need to go around ledges
                                    elif action_name == 'UP':
                                        directive_reward = -15.0  # Wrong way
                                        
                                # Littleroot → Lab (use portal connections!)
                                elif 'LITTLEROOT' in location_upper and 'HOUSE' not in location_upper:
                                    # PRIORITY: Portal connection for lab entrance
                                    connections = memory.get('connections', [])
                                    dynamic_exit = self.get_exit_for_objective(connections, 'lab')
                                    target_x, target_y = dynamic_exit if dynamic_exit else DOOR_COORDS['littleroot']['lab']
                                    
                                    best_dir = get_best_direction(current_x, current_y, target_x, target_y)
                                    if best_dir and action_name == best_dir:
                                        directive_reward = 30.0
                                        if self.step_count % 50 == 0:
                                            portal_src = "portal" if dynamic_exit else "fallback"
                                            logger.info(f"🗺️ LITTLEROOT→LAB: ({current_x},{current_y}) → ({target_x},{target_y}) = {best_dir} [{portal_src}]")
                                    elif action_name in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                                        directive_reward = 5.0
                                else:
                                    # Fallback: simple DOWN
                                    if action_name == 'DOWN':
                                        directive_reward = 10.0
                            
                            # ROUTE 103: Find rival (exploration with ledge avoidance)
                            elif 'FIND_RIVAL' in target or 'BATTLE' in target:
                                current_x = memory.get('x', 0)
                                current_y = memory.get('y', 0)
                                location_upper = (memory.get('location', '') or '').upper()
                                
                                if 'ROUTE' in location_upper and '103' in location_upper:
                                    # Route 103 layout: Rival is typically in the upper RIGHT area
                                    # Ledges block certain paths, rival at approximately (25-30, 3-5)
                                    
                                    # Track stuck state
                                    if not hasattr(self, 'route103_stuck_count'):
                                        self.route103_last_pos = (current_x, current_y)
                                        self.route103_stuck_count = 0
                                    
                                    if (current_x, current_y) == self.route103_last_pos:
                                        self.route103_stuck_count += 1
                                    else:
                                        self.route103_stuck_count = 0
                                        self.route103_last_pos = (current_x, current_y)
                                    
                                    is_stuck = self.route103_stuck_count > 5
                                    
                                    # Rival location: approximately x=25-30, y=3-5 (upper right)
                                    # Strategy: Go RIGHT, then UP when possible
                                    if is_stuck:
                                        # SLAM scan: if stuck, try different direction
                                        if current_x < 20:
                                            # Need to go more RIGHT
                                            if action_name == 'RIGHT':
                                                directive_reward = 50.0
                                                logger.info(f"🗺️ ROUTE103 STUCK: x={current_x}<20, going RIGHT")
                                            elif action_name == 'UP':
                                                directive_reward = 20.0
                                            elif action_name == 'DOWN':
                                                directive_reward = 10.0  # Try to get around ledge
                                            else:
                                                directive_reward = -10.0
                                        else:
                                            # x >= 20, try going UP to find rival
                                            if action_name == 'UP':
                                                directive_reward = 50.0
                                                logger.info(f"🗺️ ROUTE103 STUCK: x={current_x}>=20, going UP")
                                            elif action_name == 'RIGHT':
                                                directive_reward = 20.0
                                            else:
                                                directive_reward = -10.0
                                    else:
                                        # Normal exploration: encourage RIGHT + UP
                                        if action_name == 'RIGHT':
                                            directive_reward = 20.0
                                        elif action_name == 'UP':
                                            directive_reward = 15.0
                                        elif action_name == 'A':
                                            directive_reward = 10.0  # Press A to talk to rival
                                        elif action_name in ['DOWN', 'LEFT']:
                                            directive_reward = 3.0  # Some exploration OK for snake path around ledges
                                else:
                                    # Not on Route 103 yet
                                    if action_name in ['UP', 'RIGHT']:
                                        directive_reward = 10.0
                            
                            # EXIT_HOUSE/RIVAL_BEDROOM pathfinding (EXIT_LAB handled above)
                            elif 'EXIT_HOUSE' in target or 'EXIT_RIVAL_BEDROOM' in target:
                                # PRIORITY: Portal connections (real-time ground truth from server)
                                # Use SLAM + portal for dynamic house interior navigation
                                current_x = memory.get('x', 0)
                                current_y = memory.get('y', 0)
                                location_upper = (memory.get('location', '') or '').upper()
                                
                                # Get real-time connections from server
                                connections = memory.get('connections', [])
                                map_data = memory.get('map_data', {})
                                terrain = map_data.get('terrain', [])
                                
                                if '2F' in location_upper or 'BEDROOM' in location_upper:
                                    # ===== 2F: Find stairs using multi-level fallback =====
                                    stairs_exit = None
                                    source = "unknown"
                                    
                                    # LEVEL 1: Portal connections (HIGHEST PRIORITY - real-time server ground truth)
                                    if connections:
                                        for conn in connections:
                                            to_loc = conn.get('to', '').upper()
                                            # Look for connection to 1F or outside
                                            if '1F' in to_loc or 'LITTLEROOT' in to_loc or 'TOWN' in to_loc or 'HOUSE' in to_loc:
                                                from_pos = conn.get('from_pos', [])
                                                if len(from_pos) >= 2:
                                                    stairs_exit = (int(from_pos[0]), int(from_pos[1]))
                                                    source = "portal"
                                                    break
                                    
                                    # LEVEL 2: Terrain data (server map - warp tiles)
                                    if not stairs_exit and terrain:
                                        for tile in terrain:
                                            behavior = (tile.get('behavior', '') or '').upper()
                                            # Warp tiles = stairs/doors
                                            if 'WARP' in behavior or 'STAIRS' in behavior or 'DOOR' in behavior:
                                                tx, ty = tile.get('x'), tile.get('y')
                                                if tx is not None and ty is not None:
                                                    stairs_exit = (int(tx), int(ty))
                                                    source = "terrain"
                                                    break
                                    
                                    # LEVEL 3: Hardcoded fallback (only if no server data)
                                    if not stairs_exit:
                                        stairs_exit = (2, 2)  # Known correct position for houses
                                        source = "fallback"
                                        if self.step_count % 100 == 0:
                                            logger.info(f"⚠️ House 2F: Using fallback stairs (2,2) - no portal/terrain data available")
                                    
                                    # SLAM pathfinding to stairs
                                    target_x, target_y = stairs_exit
                                    best_dir = get_best_direction(current_x, current_y, target_x, target_y)
                                    
                                    if best_dir and action_name == best_dir:
                                        directive_reward = 100.0
                                        if self.step_count % 50 == 0:
                                            logger.info(f"🏠 2F EXIT: ({current_x},{current_y}) → stairs({target_x},{target_y}) = {best_dir} (+100) [{source}]")
                                    elif action_name in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                                        directive_reward = -50.0
                                        
                                elif '1F' in location_upper:
                                    # ===== 1F: PRIORITY 1 - Real-time portal connections (ground truth!) =====
                                    door_exit = None
                                    source = "unknown"
                                    
                                    # Try portal connections first (server ground truth)
                                    if connections:
                                        for conn in connections:
                                            to_loc = conn.get('to', '').upper()
                                            # Look for outdoor connections (TOWN, LITTLEROOT, ROUTE)
                                            # CRITICAL: Exclude INDOOR destinations (2F, 1F, HOUSE, BEDROOM)
                                            is_outdoor = 'TOWN' in to_loc or 'LITTLEROOT' in to_loc or 'ROUTE' in to_loc
                                            is_indoor = '2F' in to_loc or '1F' in to_loc or 'HOUSE' in to_loc or 'BEDROOM' in to_loc
                                            if is_outdoor and not is_indoor:
                                                from_pos = conn.get('from_pos', [])
                                                if len(from_pos) >= 2:
                                                    door_exit = (int(from_pos[0]), int(from_pos[1]))
                                                    source = "portal"
                                                    if self.step_count % 100 == 0:
                                                        logger.info(f"🌐 1F EXIT: Using portal ground truth → {to_loc} at {door_exit}")
                                                    break
                                    
                                    # FALLBACK: If no portal, use DOWN direction (exits at bottom)
                                    if door_exit:
                                        target_x, target_y = door_exit
                                        best_dir = get_best_direction(current_x, current_y, target_x, target_y)
                                        
                                        if best_dir and action_name == best_dir:
                                            directive_reward = 100.0
                                            if self.step_count % 50 == 0:
                                                logger.info(f"🏠 1F EXIT: ({current_x},{current_y}) → door({target_x},{target_y}) = {best_dir} (+100) [{source}]")
                                        elif action_name in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                                            directive_reward = -30.0
                                    else:
                                        # No portal data - use DOWN as generic direction (exits at bottom)
                                        if action_name == 'DOWN':
                                            directive_reward = 80.0  # Strong reward for DOWN
                                            if self.step_count % 50 == 0:
                                                logger.info(f"🏠 1F EXIT: No portal - using DOWN to find exit")
                                        elif action_name in ['LEFT', 'RIGHT']:
                                            directive_reward = 10.0  # Allow lateral exploration
                                        elif action_name == 'UP':
                                            directive_reward = -50.0  # Penalty for going away from exit
                                else:
                                    # Generic - encourage DOWN
                                    if action_name == 'DOWN':
                                        directive_reward = 50.0
                            elif 'GO_EAST' in target and action_name == 'RIGHT':
                                directive_reward = 10.0
                            elif 'GO_WEST' in target and action_name == 'LEFT':
                                directive_reward = 10.0
                            
                            # POST-HEAL: Exit house immediately after healing
                            elif 'EXIT_HOUSE_AFTER_HEAL' in target or 'AFTER_HEAL' in target:
                                # Use portal connections (ground truth!) or DOWN fallback
                                current_x = memory.get('x', 0)
                                current_y = memory.get('y', 0)
                                location_upper = (memory.get('location', '') or '').upper()
                                connections = memory.get('connections', [])
                                
                                if '2F' in location_upper or 'BEDROOM' in location_upper:
                                    # 2F: Find stairs via portal or use DOWN
                                    stairs_exit = None
                                    for conn in connections:
                                        to_loc = conn.get('to', '').upper()
                                        if '1F' in to_loc:
                                            from_pos = conn.get('from_pos', [])
                                            if len(from_pos) >= 2:
                                                stairs_exit = (int(from_pos[0]), int(from_pos[1]))
                                                break
                                    if stairs_exit:
                                        target_x, target_y = stairs_exit
                                    else:
                                        target_x, target_y = current_x, current_y + 1  # DOWN
                                else:
                                    # 1F: Find exit via portal or use DOWN
                                    door_exit = None
                                    for conn in connections:
                                        to_loc = conn.get('to', '').upper()
                                        if 'TOWN' in to_loc or 'LITTLEROOT' in to_loc or 'ROUTE' in to_loc:
                                            from_pos = conn.get('from_pos', [])
                                            if len(from_pos) >= 2:
                                                door_exit = (int(from_pos[0]), int(from_pos[1]))
                                                break
                                    if door_exit:
                                        target_x, target_y = door_exit
                                    else:
                                        # No portal - go DOWN (exits at bottom)
                                        if action_name == 'DOWN':
                                            directive_reward = 80.0
                                            if self.step_count % 50 == 0:
                                                logger.info(f"🏠 POST-HEAL: No portal - using DOWN")
                                        elif action_name == 'UP':
                                            directive_reward = -50.0
                                        else:
                                            directive_reward = 5.0  # Allow lateral
                                        target_x, target_y = None, None  # Skip pathfinding
                                
                                if target_x is not None:
                                    best_dir = get_best_direction(current_x, current_y, target_x, target_y)
                                    if best_dir and action_name == best_dir:
                                        directive_reward = 100.0  # HIGH priority to exit!
                                        if self.step_count % 50 == 0:
                                            logger.info(f"🏠 POST-HEAL EXIT: ({current_x},{current_y}) → ({target_x},{target_y}) = {best_dir}")
                                    elif action_name in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                                        directive_reward = -20.0  # Strong penalty for wrong dir
                                    elif action_name == 'NOOP':
                                        directive_reward = -30.0  # Very strong penalty
                            
                            # ROUTE 101 OBSTACLE PATHFINDING: Ledges + Trees
                            # NOTE: This is DISABLED when route101_ledge_interrupt is active
                            # The new waypoint system (lines 1242-1322) takes priority
                            elif 'GO_NORTH' in target and ('ROUTE' in target or 'OLDALE' in target):
                                # Check if new waypoint system is handling this
                                if memory.get('route101_ledge_interrupt', False) and memory.get('ledge_target'):
                                    # Defer to new waypoint system - it already assigned rewards
                                    if self.step_count % 100 == 0:
                                        logger.info(f"🗺️ Deferring to waypoint system for Route 101 navigation")
                                    pass  # Skip old logic
                                else:
                                    # Fallback: Old SLAM navigation (for areas not covered by waypoint system)
                                    current_x = memory.get('x', 0)
                                    current_y = memory.get('y', 0)
                                    location_upper = (memory.get('location', '') or '').upper()
                                
                                    # ROUTE 101 WAYPOINTS (ground truth path around ledge+trees)
                                    # From Littleroot, the safe path is:
                                    # - Stay on the RIGHT side (x ~= 7-9) to avoid west trees
                                    # - Ledges are around y=15-18, go DOWN then RIGHT to clear
                                    ROUTE_101_PATH = [
                                        # South entrance (from Littleroot) → safe corridor
                                        {'zone': (0, 30), 'target_x': 8, 'dir': 'UP'},  # Start: stay center
                                        # Ledge zone - need to go RIGHT to find gap
                                        {'zone': (12, 22), 'target_x': 10, 'dir': 'RIGHT_THEN_UP'},  # Go right around ledge
                                        # North section after ledge
                                        {'zone': (0, 12), 'target_x': 8, 'dir': 'UP'},  # Clear, go north
                                    ]
                                
                                # Track if stuck at obstacle
                                if not hasattr(self, 'route101_last_pos'):
                                    self.route101_last_pos = (current_x, current_y)
                                    self.route101_stuck_count = 0
                                
                                if (current_x, current_y) == self.route101_last_pos:
                                    self.route101_stuck_count += 1
                                else:
                                    self.route101_stuck_count = 0
                                    self.route101_last_pos = (current_x, current_y)
                                
                                is_stuck = self.route101_stuck_count > 5  # Stuck for 5+ steps
                                # LEDGE ZONES: Main ledge is around y=12-20
                                # Upper zone (y<12) is above ledge - clear path
                                # South zone (y>20) is below ledge - need to navigate up
                                near_ledge = 12 <= current_y <= 20  # Main ledge zone
                                upper_zone = current_y < 12  # Above ledge, clear path north
                                
                                if 'ROUTE' in location_upper and '101' in location_upper:
                                    # GUARD: Skip old SLAM if new waypoint system is active
                                    if memory.get('route101_ledge_interrupt', False):
                                        pass  # New waypoint system handles this
                                    elif is_stuck and near_ledge:
                                        # SLAM GRID SCAN: Find walkable path around ledge+trees
                                        # Uses metatile data from server map to detect obstacles
                                        # Behavior 101 = SOUTH_ARROW_WARP (ledge)
                                        # Trees show as '#' (blocked) in map
                                        
                                        # Simple pathfinding: check which directions are clear
                                        # Based on common Route 101 layout: gap is to the RIGHT
                                        # Trees often block LEFT, ledge blocks UP
                                        
                                        # Prioritize based on position:
                                        # If x < 5: Need to go RIGHT to find gap
                                        # If x >= 5: Can try UP or RIGHT
                                        if current_x < 5:
                                            # On left side - trees block LEFT, go RIGHT
                                            if action_name == 'RIGHT':
                                                directive_reward = 60.0
                                                logger.info(f"🗺️ SLAM SCAN: x={current_x}<5, going RIGHT to find gap")
                                            elif action_name == 'DOWN':
                                                directive_reward = 20.0  # Down also OK
                                            elif action_name == 'UP':
                                                directive_reward = -30.0  # Ledge blocks
                                            elif action_name == 'LEFT':
                                                directive_reward = -30.0  # Trees block
                                        elif current_x >= 8:
                                            # On right side - try UP, gap may be here
                                            if action_name == 'UP':
                                                directive_reward = 40.0
                                                logger.info(f"🗺️ SLAM SCAN: x={current_x}>=8, trying UP through gap")
                                            elif action_name == 'RIGHT':
                                                directive_reward = 20.0
                                            elif action_name == 'DOWN':
                                                directive_reward = 5.0
                                            elif action_name == 'LEFT':
                                                directive_reward = -10.0
                                        else:
                                            # Middle zone (5-7) - keep going RIGHT
                                            if action_name == 'RIGHT':
                                                directive_reward = 50.0
                                                logger.info(f"🗺️ SLAM SCAN: x={current_x} in middle, going RIGHT")
                                            elif action_name == 'DOWN':
                                                directive_reward = 15.0
                                            elif action_name == 'UP':
                                                directive_reward = -20.0
                                            elif action_name == 'LEFT':
                                                directive_reward = -10.0
                                    elif near_ledge:
                                        # Ledge zone (y=12-20) - careful here!
                                        # Must go RIGHT to x>=12 to find gap, then UP
                                        if current_x >= 12:
                                            # Far right - gap should be here, go UP!
                                            if action_name == 'UP':
                                                directive_reward = 60.0
                                                if self.step_count % 30 == 0:
                                                    logger.info(f"🗺️ LEDGE GAP: x={current_x}>=12, going UP through gap!")
                                            elif action_name == 'RIGHT':
                                                directive_reward = 20.0
                                            else:
                                                directive_reward = -15.0
                                        else:
                                            # Not at gap yet - must go RIGHT!
                                            if action_name == 'RIGHT':
                                                directive_reward = 80.0  # Strong reward - need to reach gap!
                                                if self.step_count % 30 == 0:
                                                    logger.info(f"🗺️ LEDGE ZONE: x={current_x}<12, going RIGHT to gap")
                                            elif action_name == 'DOWN':
                                                directive_reward = 5.0   # Backtrack if needed
                                            elif action_name == 'UP':
                                                directive_reward = -80.0  # STRONG penalty - ledge blocks!
                                            elif action_name == 'LEFT':
                                                directive_reward = -40.0  # Wrong direction
                                    elif upper_zone:
                                        # UPPER ZONE (y<12) - above ledge, clear path north!
                                        # Just go UP to reach Oldale
                                        near_oldale_exit = current_y < 5  # Very close to Oldale
                                        
                                        if near_oldale_exit:
                                            # Almost at Oldale! Make sure not blocked by trees
                                            if current_x > 10:
                                                # Too far right - tree might block
                                                if action_name == 'LEFT':
                                                    directive_reward = 40.0
                                                    if self.step_count % 30 == 0:
                                                        logger.info(f"🗺️ NEAR OLDALE: x={current_x}>10, adjusting LEFT")
                                                elif action_name == 'UP':
                                                    directive_reward = 30.0
                                            elif current_x < 5:
                                                # Too far left
                                                if action_name == 'RIGHT':
                                                    directive_reward = 40.0
                                                elif action_name == 'UP':
                                                    directive_reward = 30.0
                                            else:
                                                # Good position - go UP!
                                                if action_name == 'UP':
                                                    directive_reward = 80.0
                                                    if self.step_count % 30 == 0:
                                                        logger.info(f"🗺️ ENTERING OLDALE! y={current_y}, going UP")
                                                else:
                                                    directive_reward = -10.0
                                        else:
                                            # Upper zone but not at exit yet - keep going UP!
                                            if action_name == 'UP':
                                                directive_reward = 60.0
                                                if self.step_count % 50 == 0:
                                                    logger.info(f"🗺️ UPPER ZONE: y={current_y}<12, going UP to Oldale")
                                            elif action_name == 'RIGHT':
                                                directive_reward = 10.0  # Minor adjustment OK
                                            elif action_name == 'LEFT':
                                                directive_reward = 5.0
                                            elif action_name == 'DOWN':
                                                directive_reward = -30.0  # Don't go back!
                                    else:
                                        # SOUTH ZONE (y > 20) - below main ledge, just go UP
                                        if action_name == 'UP':
                                            directive_reward = 40.0
                                        elif action_name in ['LEFT', 'RIGHT']:
                                            directive_reward = 5.0
                                        elif action_name == 'DOWN':
                                            directive_reward = -15.0
                            elif 'ENTER_LAB' in target or 'GET_POKEDEX' in target:
                                # Use dynamic portal connection for lab entrance
                                current_x = memory.get('x', 0)
                                current_y = memory.get('y', 0)
                                
                                # Try portal first, fallback to hardcoded
                                connections = memory.get('connections', [])
                                dynamic_exit = self.get_exit_for_objective(connections, 'lab')
                                lab_x, lab_y = dynamic_exit if dynamic_exit else DOOR_COORDS['littleroot']['lab']
                                
                                best_dir = get_best_direction(current_x, current_y, lab_x, lab_y)
                                if best_dir and action_name == best_dir:
                                    directive_reward = 150.0  # BOOSTED: Lab is critical for Pokedex!
                                    if self.step_count % 100 == 0:
                                        portal_src = "portal" if dynamic_exit else "fallback"
                                        logger.info(f"🧪 LAB PATHFINDING: ({current_x},{current_y}) → lab({lab_x},{lab_y}) = {best_dir} (+150) [{portal_src}]")
                                elif action_name in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                                    directive_reward = -20.0  # Stronger penalty for wrong direction
                                    if self.step_count % 50 == 0:
                                        logger.info(f"⚠️ Wrong direction to lab! Need {best_dir} to reach ({lab_x},{lab_y})")
                            elif 'RIVAL_HOUSE' in target and 'GO_UPSTAIRS' not in target:
                                # CRITICAL: Rival house visit required for NPC trigger
                                # CONDITIONAL GROUND TRUTH PATHFINDING - only when stuck!
                                # Uses hierarchical approach: Ground truth → SLAM → CNN → Grid search
                                # NOTE: This handles ENTERING the house from outside only!
                                #       GO_UPSTAIRS_RIVAL_HOUSE is handled separately below
                                
                                # FULL SNN INTEGRATION: Detect stuck using temporal reasoning
                                current_x = memory.get('x', 0)
                                current_y = memory.get('y', 0)
                                current_pos = (current_x, current_y)
                                
                                # Track position, action, and SNN spike variance
                                self.position_history.append(current_pos)
                                self.action_history.append(action_name)
                                
                                # Get SNN spike variance from policy model (if available)
                                snn_variance = 0.0
                                try:
                                    if hasattr(self, 'model') and hasattr(self.model.policy, 'snn_spike_history'):
                                        # Calculate variance of recent SNN spikes (temporal reasoning)
                                        spike_history = self.model.policy.snn_spike_history
                                        if spike_history is not None and spike_history.numel() > 0:
                                            # Low variance = repetitive neural patterns = stuck
                                            snn_variance = spike_history.var().item()
                                    self.snn_spike_variance_history.append(snn_variance)
                                except:
                                    self.snn_spike_variance_history.append(0.0)
                                
                                # Maintain rolling windows (20 steps)
                                for hist in [self.position_history, self.action_history, self.snn_spike_variance_history]:
                                    if len(hist) > 20:
                                        hist.pop(0)
                                
                                # MULTI-MODAL STUCK DETECTION:
                                # 1. Position repetition (SLAM)
                                position_counts = {}
                                for pos in self.position_history:
                                    position_counts[pos] = position_counts.get(pos, 0) + 1
                                max_position_rep = max(position_counts.values()) if position_counts else 0
                                position_stuck = max_position_rep > 10
                                
                                # 2. Action repetition (behavioral loops)
                                action_counts = {}
                                for act in self.action_history[-10:]:  # Last 10 actions
                                    action_counts[act] = action_counts.get(act, 0) + 1
                                max_action_rep = max(action_counts.values()) if action_counts else 0
                                action_stuck = max_action_rep > 7  # Same action 7/10 times
                                
                                # 🛑 DIRECTION-SPAM PENALTY: If same direction used 7+ times
                                if action_stuck:
                                    most_spammed = max(action_counts, key=action_counts.get) if action_counts else None
                                    if most_spammed in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                                        direction_spam_penalty = -50.0
                                        reward += direction_spam_penalty
                                        if self.step_count % 50 == 0:
                                            logger.warning(f"🛑 DIRECTION SPAM: {most_spammed} × {max_action_rep}/10 → -50")
                                            info_parts.append(f"🛑 {most_spammed}-SPAM -50")
                                
                                # 3. SNN temporal pattern (low variance = stuck)
                                if len(self.snn_spike_variance_history) >= 10:
                                    recent_snn_var = self.snn_spike_variance_history[-10:]
                                    avg_snn_variance = sum(recent_snn_var) / len(recent_snn_var)
                                    snn_stuck = avg_snn_variance < 0.01  # Very low neural variance
                                else:
                                    snn_stuck = False
                                
                                # COMBINED STUCK DETECTION (any 2 of 3 indicators)
                                stuck_indicators = sum([position_stuck, action_stuck, snn_stuck])
                                is_stuck = stuck_indicators >= 2
                                
                                if is_stuck and self.step_count % 100 == 0:
                                    logger.info(f"🔴 STUCK DETECTED: Pos={position_stuck}, Act={action_stuck}, SNN={snn_stuck} (var={avg_snn_variance if 'avg_snn_variance' in locals() else 0:.4f})")
                                
                                # 🆕 AGGRESSIVE ACTION DIVERSITY (30+ steps stuck at same position)
                                # If stuck too long, heavily penalize the most-used action to force diversity
                                if max_position_rep >= 30:
                                    most_used_action = max(action_counts, key=action_counts.get) if action_counts else None
                                    if most_used_action and action_name == most_used_action:
                                        diversity_penalty = -200.0
                                        reward += diversity_penalty
                                        logger.warning(f"🔄 FORCED DIVERSITY: Stuck {max_position_rep} steps! {most_used_action} -200 to try different action")
                                        info_parts.append(f"🔄 STUCK {max_position_rep}: {most_used_action} -200")
                                    # Also boost the least-used action
                                    elif action_counts:
                                        least_used = min(action_counts, key=action_counts.get)
                                        if action_name == least_used:
                                            diversity_bonus = 100.0
                                            reward += diversity_bonus
                                            logger.info(f"✅ DIVERSITY BONUS: Trying underused action {least_used} +100")
                                
                                # CONDITIONAL: Use ground truth pathfinding ONLY if stuck + high priority
                                directive_priority = directive_info.get('target_level', 0) if 'directive_info' in locals() else 10
                                use_pathfinding = is_stuck and directive_priority >= 8
                                
                                # RIVAL HOUSE ENTRANCE: ONLY when OUTSIDE in Littleroot Town!
                                # This is for approaching the door from outside
                                # If agent is INSIDE the house, this logic should NOT run!
                                location_str = memory.get('location', '').upper()
                                map_id = memory.get('map_id', 0)
                                in_rival_house = (map_id in [0x0202, 0x030A]) or ('HOUSE' in location_str and ('RIVAL' in location_str or '2F' in location_str or '1F' in location_str))
                                in_littleroot = 'LITTLEROOT' in location_str and not in_rival_house
                                
                                if self.step_count % 100 == 0:
                                    logger.info(f"🏠 LOCATION CHECK: '{location_str}' | map_id={hex(map_id)} | in_rival_house={in_rival_house} | in_littleroot={in_littleroot}")
                                
                                if 'RIVAL_HOUSE' in target and in_littleroot:
                                    # ONLY when OUTSIDE - navigate to rival house door
                                    # Rival house door: approach door position, then go UP to enter
                                    # House is solid - need to navigate AROUND it to the door
                                    
                                    # PRIORITY: Portal connection for rival house entrance
                                    connections = memory.get('connections', [])
                                    dynamic_exit = self.get_exit_for_objective(connections, 'rival_house')
                                    if dynamic_exit:
                                        rival_x, rival_y = dynamic_exit
                                        if self.step_count % 100 == 0:
                                            logger.info(f"🏠 Using PORTAL rival_house coords: ({rival_x},{rival_y})")
                                    else:
                                        # Gender-aware fallback using PORTAL GROUND TRUTH coords
                                        player_gender = memory.get('player', {}).get('gender', 'Boy')
                                        if player_gender == 'Girl':
                                            # Girl's rival is Brendan's house (8, 8)
                                            rival_x, rival_y = DOOR_COORDS['littleroot']['brendan_house']
                                        else:
                                            # Boy's rival is May's house (14, 9)
                                            rival_x, rival_y = DOOR_COORDS['littleroot']['may_house']
                                        if self.step_count % 100 == 0:
                                            logger.info(f"🏠 Using GENDER-AWARE rival_house coords: ({rival_x},{rival_y}) [{player_gender}]")
                                    
                                    # Check if at the door position (17, 9)
                                    at_door = (current_x == rival_x and current_y == rival_y)
                                    
                                    if at_door and action_name == 'UP':
                                        # At (17, 9), go UP to (17, 8) to enter!
                                        directive_reward = 200.0
                                        logger.info(f"🚪 ENTERING RIVAL HOUSE: UP from ({current_x},{current_y}) (+200)")
                                    elif not at_door:
                                        # Not at door yet - use waypoint navigation around the house
                                        # House blocks direct path, so we need to:
                                        # 1. Get to x=17 (same column as door)
                                        # 2. Then move UP to y=9
                                        
                                        if current_x != rival_x:
                                            # Need to move horizontally toward rival house
                                            if current_x < rival_x and action_name == 'RIGHT':
                                                directive_reward = 200.0
                                                logger.info(f"🏠 RIVAL HOUSE: x{current_x}→{rival_x}, going RIGHT (+200)")
                                            elif current_x > rival_x and action_name == 'LEFT':
                                                directive_reward = 200.0
                                                logger.info(f"🏠 RIVAL HOUSE: x{current_x}→{rival_x}, going LEFT (+200)")
                                        
                                        if current_y != rival_y:
                                            # ALSO need to move vertically! (not elif - check BOTH!)
                                            if current_y < rival_y and action_name == 'DOWN':
                                                # Agent NORTH of door (y < 9), go DOWN!
                                                directive_reward = 200.0
                                                logger.info(f"🏠 RIVAL HOUSE: y{current_y}→{rival_y}, going DOWN (+200)")
                                            elif current_y > rival_y and action_name == 'UP':
                                                # Agent SOUTH of door (y > 9), go UP
                                                directive_reward = 200.0
                                                logger.info(f"🏠 RIVAL HOUSE: y{current_y}→{rival_y}, going UP (+200)")
                                        
                                        # Penalize wrong direction only if we didn't already reward
                                        if directive_reward == 0.0:
                                            # Check if action is moving AWAY from goal
                                            dx = rival_x - current_x
                                            dy = rival_y - current_y
                                            if (action_name == 'LEFT' and dx > 0) or \
                                               (action_name == 'RIGHT' and dx < 0) or \
                                               (action_name == 'UP' and dy > 0) or \
                                               (action_name == 'DOWN' and dy < 0):
                                                directive_reward = -50.0

                                    else:
                                        # At door but wrong action
                                        if action_name in ['DOWN', 'LEFT', 'RIGHT']:
                                            directive_reward = -50.0
                            
                            # INSIDE HOUSE: Go upstairs - gender-aware for player vs rival
                            elif 'GO_UPSTAIRS' in target:
                                # Determine if in PLAYER's house or RIVAL's house
                                current_loc = memory.get('location', '').upper() if memory else ''
                                player_gender = memory.get('player', {}).get('gender', 'Boy') if memory else 'Boy'
                                
                                # Gender-aware house detection:
                                # Male (Boy): Brendan's = player's, May's = rival's
                                # Female (Girl): May's = player's, Brendan's = rival's
                                in_player_house = ('BRENDAN' in current_loc and player_gender == 'Boy') or \
                                                  ('MAY' in current_loc and player_gender == 'Girl')
                                in_rival_house = ('MAY' in current_loc and player_gender == 'Boy') or \
                                                 ('BRENDAN' in current_loc and player_gender == 'Girl')
                                
                                # Check milestones to determine if we should go upstairs or exit
                                clock_set_done = 'CLOCK_SET' in str(memory.get('milestones', {}))
                                rival_bedroom_done = 'RIVAL_BEDROOM' in str(memory.get('milestones', {}))
                                
                                if in_player_house:
                                    if not clock_set_done:
                                        # PLAYER'S HOUSE: Go upstairs to SET CLOCK
                                        if action_name == 'UP':
                                            directive_reward = 50.0
                                            logger.info(f"⬆️ GOING UPSTAIRS TO SET CLOCK (+50) [{player_gender}]")
                                        elif action_name == 'LEFT':
                                            # LEFT towards stairs at (2,2)
                                            directive_reward = 20.0
                                        elif action_name in ['DOWN', 'RIGHT']:
                                            directive_reward = -20.0  # Wrong direction
                                    else:
                                        # CLOCK done - should EXIT house, not go upstairs
                                        if action_name == 'DOWN':
                                            directive_reward = 30.0  # Reward exiting
                                            logger.info(f"🚪 CLOCK DONE, EXIT HOUSE (+30) [{player_gender}]")
                                        elif action_name == 'UP':
                                            directive_reward = -30.0  # Penalize going upstairs again
                                elif in_rival_house:
                                    if not rival_bedroom_done:
                                        # RIVAL'S HOUSE: Go upstairs to RIVAL BEDROOM
                                        if action_name == 'UP':
                                            directive_reward = 100.0  # Strong reward to go upstairs
                                            logger.info(f"⬆️ GOING UPSTAIRS TO RIVAL BEDROOM (+100) [{player_gender}]")
                                        elif action_name in ['DOWN', 'LEFT', 'RIGHT']:
                                            directive_reward = -30.0  # Stronger penalty for wrong direction
                                    else:
                                        # RIVAL_BEDROOM done - should EXIT house
                                        if action_name == 'DOWN':
                                            directive_reward = 30.0  # Reward exiting
                                            logger.info(f"🚪 RIVAL_BEDROOM DONE, EXIT HOUSE (+30) [{player_gender}]")
                                        elif action_name == 'UP':
                                            directive_reward = -30.0
                                else:
                                    # Generic house or unknown
                                    if action_name == 'UP':
                                        directive_reward = 30.0
                                    elif action_name in ['DOWN']:
                                        directive_reward = -10.0
                            
                            # EXIT_RIVAL_BEDROOM is now handled by EXIT_HOUSE logic above (lines 2790+)
                            # which uses SLAM pathfinding to stairs - NO LEGACY HANDLER NEEDED
                            
                            elif 'SET_CLOCK' in target:
                                # Setting clock - reward A button press
                                if action_name == 'A':
                                    directive_reward = 15.0
                                elif action_name in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                                    directive_reward = 2.0  # Moving to clock
                            elif 'GO_UPSTAIRS' in target and action_name == 'UP':
                                # Going upstairs in house
                                directive_reward = 10.0
                            elif 'FIND_BIRCH' in target:
                                # Finding Birch on Route 101 - reward any movement to trigger encounter
                                if action_name == 'UP':
                                    directive_reward = 10.0  # North into grass
                                elif action_name in ['DOWN', 'LEFT', 'RIGHT']:
                                    directive_reward = 3.0  # Exploring
                            elif 'WALK_GRASS' in target:
                                # Walking in grass to trigger Birch - reward movement
                                if action_name in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                                    directive_reward = 5.0
                            
                            # HEALING PATHFINDING: Go to Oldale PokeCenter
                            elif 'HEAL' in target or 'POKECENTER' in target or 'CRITICAL_HP' in target or 'VISIT_OLDALE' in target:
                                current_x = memory.get('x', 0)
                                current_y = memory.get('y', 0)
                                location_upper = (memory.get('location', '') or '').upper()
                                
                                # INSIDE POKECENTER: Go UP to Nurse Joy and press A!
                                # EXCEPTION: On 2F, go DOWN to escalator to exit!
                                if 'POKECENTER' in location_upper:
                                    on_2f = '2F' in location_upper
                                    
                                    if on_2f:
                                        # ON 2F: Go DOWN to escalator to return to 1F!
                                        if action_name == 'DOWN':
                                            directive_reward = 100.0  # Correct direction to escalator!
                                            if self.step_count % 30 == 0:
                                                logger.info(f"🪜 POKECENTER 2F: Going DOWN to escalator to exit!")
                                        elif action_name in ['LEFT', 'RIGHT']:
                                            directive_reward = 20.0  # Might need to navigate around
                                        elif action_name == 'UP':
                                            directive_reward = -50.0  # Wrong direction on 2F!
                                        else:
                                            directive_reward = 0.0
                                    else:
                                        # ON 1F: Normal nurse navigation
                                        # Nurse Joy is at counter - typically at (7, 3) or similar
                                        nurse_x, nurse_y = 7, 3  # Nurse Joy counter position
                                        
                                        # Check if at Nurse Joy counter (y <= 4)
                                        at_nurse = current_y <= 4
                                        
                                        if at_nurse:
                                            # At Nurse Joy - press A to heal!
                                            if action_name == 'A':
                                                directive_reward = 200.0  # Big reward for healing!
                                                logger.info(f"💚 HEALING AT NURSE JOY! Position: ({current_x},{current_y})")
                                            elif action_name == 'UP':
                                                directive_reward = 50.0  # Still go up to get closer
                                            else:
                                                directive_reward = -10.0
                                        else:
                                            # Walk UP to Nurse Joy counter
                                            if action_name == 'UP':
                                                directive_reward = 100.0
                                                if self.step_count % 30 == 0:
                                                    logger.info(f"🏥 POKECENTER INTERIOR: Going UP to Nurse Joy")
                                            elif action_name in ['LEFT', 'RIGHT']:
                                                directive_reward = 10.0  # Slight reward for adjusting position
                                            elif action_name == 'DOWN':
                                                directive_reward = -50.0  # Don't exit!
                                
                                # If in Oldale TOWN (not PokeCenter), pathfind to PokeCenter door
                                elif 'OLDALE' in location_upper:
                                    # Try dynamic portal, fallback to hardcoded
                                    connections = memory.get('connections', [])
                                    dynamic_exit = self.get_exit_for_objective(connections, 'pokecenter')
                                    pc_x, pc_y = dynamic_exit if dynamic_exit else DOOR_COORDS['oldale']['pokecenter']
                                    
                                    # Check if at PokeCenter door (within 2 tiles)
                                    at_pokecenter = abs(current_x - pc_x) <= 2 and abs(current_y - pc_y) <= 2
                                    
                                    # NOTE: Do NOT set visited flag here - this is just approaching the door!
                                    # visited_oldale_pokecenter should ONLY be set when agent talks to Nurse Joy
                                    # at (7,4) INSIDE the PokeCenter - handled in late-stage override
                                    if at_pokecenter:
                                        # Reward pressing A to enter PokeCenter
                                        if action_name == 'A':
                                            directive_reward = 300.0  # MASSIVE reward - critical for respawn!
                                            logger.info(f"🏥 ENTERING POKECENTER at ({current_x},{current_y})")
                                    else:
                                        # Still pathfinding to PokeCenter - USE A* oldale_target!
                                        oldale_target = memory.get('oldale_target')
                                        if oldale_target:
                                            best_dir = oldale_target.get('action')
                                        else:
                                            best_dir = get_best_direction(current_x, current_y, pc_x, pc_y)
                                        if best_dir and action_name == best_dir:
                                            directive_reward = 200.0  # Huge reward - need respawn point!
                                            if self.step_count % 50 == 0:
                                                logger.info(f"🏥 HEAL A* PATHFINDING: → Oldale PokeCenter ({pc_x},{pc_y}) via {best_dir}")
                                        elif action_name in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                                            directive_reward = -10.0
                                # If in Littleroot, go to Mom's house (she heals for free!)
                                elif 'LITTLEROOT' in location_upper:
                                    _, mom_x, mom_y = HEALING_COORDS['littleroot']
                                    best_dir = get_best_direction(current_x, current_y, mom_x, mom_y)
                                    if best_dir and action_name == best_dir:
                                        directive_reward = 100.0  # Mom heals!
                                        if self.step_count % 50 == 0:
                                            logger.info(f"🏠 HEAL PATHFINDING: → Mom's House ({mom_x},{mom_y})")
                                    elif action_name in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                                        directive_reward = -10.0
                                # EMERGENCY: If on Route 103 with low HP, go SOUTH to Oldale!
                                elif 'ROUTE' in location_upper and '103' in location_upper:
                                    if action_name == 'DOWN':  # Go south to Oldale
                                        directive_reward = 100.0  # Emergency!
                                        if self.step_count % 50 == 0:
                                            logger.info(f"🚨 EMERGENCY HEAL: Route 103 → Oldale PokeCenter")
                                    elif action_name in ['UP', 'LEFT', 'RIGHT']:
                                        directive_reward = -30.0  # Strong penalty - need to heal!
                                # If on Route 101, go NORTH to Oldale (not south to Littleroot)
                                elif 'ROUTE' in location_upper and '101' in location_upper:
                                    if action_name == 'UP':  # Go to Oldale (PokeCenter is there!)
                                        directive_reward = 50.0
                                    elif action_name in ['DOWN', 'LEFT', 'RIGHT']:
                                        directive_reward = -5.0
                            
                            # POST-BLACKOUT RECOVERY: Use KNN-style distance to return to progress
                            elif 'RECOVERY' in target or 'POST_BLACKOUT' in target:
                                current_x = memory.get('x', 0)
                                current_y = memory.get('y', 0)
                                location_upper = (memory.get('location', '') or '').upper()
                                
                                # KNN-style progress tracking: Remember furthest location
                                # Priority order: Route 103 > Oldale > Route 101 > Littleroot
                                PROGRESS_LOCATIONS = {
                                    'ROUTE 103': {'priority': 4, 'target': (12, 5)},     # Rival area
                                    'OLDALE': {'priority': 3, 'target': (7, 13)},        # PokeCenter
                                    'ROUTE 101': {'priority': 2, 'target': (8, 5)},      # Mid-route
                                    'LITTLEROOT': {'priority': 1, 'target': (8, 0)},     # North exit
                                }
                                
                                # Track furthest progress point
                                if not hasattr(self, 'furthest_progress'):
                                    self.furthest_progress = {'priority': 0, 'location': 'LITTLEROOT', 'x': 8, 'y': 8}
                                
                                # Update furthest progress if current location is further
                                for loc_name, info in PROGRESS_LOCATIONS.items():
                                    if loc_name in location_upper:
                                        if info['priority'] > self.furthest_progress['priority']:
                                            self.furthest_progress = {
                                                'priority': info['priority'],
                                                'location': loc_name,
                                                'x': current_x, 'y': current_y
                                            }
                                            logger.info(f"📍 Progress updated: {loc_name} ({current_x},{current_y})")
                                
                                # After blackout: pathfind back to furthest progress location
                                target_loc = self.furthest_progress['location']
                                
                                # KNN-style: Find direction to get back to progress area
                                if 'LITTLEROOT' in location_upper:
                                    # In Littleroot after blackout - heal at Mom then go north
                                    if 'HOUSE' in location_upper:
                                        # Exit house first
                                        if action_name == 'DOWN':
                                            directive_reward = 50.0
                                            logger.info(f"🔄 BLACKOUT RECOVERY: Exiting house to return to {target_loc}")
                                    else:
                                        # Go north toward last progress
                                        if action_name == 'UP':
                                            directive_reward = 50.0
                                        elif action_name == 'RIGHT':
                                            directive_reward = 20.0
                                        elif action_name == 'LEFT':
                                            directive_reward = 5.0
                                elif 'OLDALE' in location_upper:
                                    # At Oldale - continue north if progress was Route 103
                                    if self.furthest_progress['priority'] >= 4:  # Was at Route 103
                                        if action_name == 'UP':
                                            directive_reward = 40.0
                                        logger.info(f"🔄 BLACKOUT RECOVERY: Returning to Route 103")
                                    else:
                                        # Just got to Oldale first time - heal then continue
                                        if action_name in ['UP', 'RIGHT']:
                                            directive_reward = 20.0
                            # POST-POKEDEX: Exit Littleroot via Route 101 north
                            elif 'ROUTE_101' in target and 'NORTH' in target:
                                current_x = memory.get('x', 0)
                                current_y = memory.get('y', 0)
                                location_upper = (memory.get('location', '') or '').upper()
                                
                                if 'LITTLEROOT' in location_upper:
                                    # Use dynamic portal connection (ground truth!)
                                    connections = memory.get('connections', [])
                                    dynamic_exit = self.get_exit_for_objective(connections, 'route101')
                                    
                                    if dynamic_exit:
                                        exit_x, exit_y = dynamic_exit
                                    else:
                                        # Fallback only if server unavailable
                                        exit_x, exit_y = DOOR_COORDS['littleroot']['route_101_north']
                                    
                                    best_dir = get_best_direction(current_x, current_y, exit_x, exit_y)
                                    if best_dir and action_name == best_dir:
                                        directive_reward = 50.0
                                        if self.step_count % 100 == 0:
                                            logger.info(f"🚪 EXIT PATHFINDING: → Route 101 North ({exit_x},{exit_y}) [portal]")
                                    elif action_name in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                                        directive_reward = -10.0
                            
                            # LATE GAME FALLBACK: SNN-based exploration for unknown areas
                            # If we're NOT in early game areas with hardcoded coords, use SNN temporal reasoning
                            if directive_reward == 0:
                                location_name = memory.get('location', '')
                                if not is_early_game(location_name):
                                    # Use SNN spike variance to encourage exploration
                                    try:
                                        if hasattr(self, 'model') and hasattr(self.model.policy, 'snn_spike_history'):
                                            spike_history = self.model.policy.snn_spike_history
                                            if spike_history is not None and spike_history.numel() > 0:
                                                snn_var = spike_history.var().item()
                                                # High variance = novel situation → encourage movement
                                                if snn_var > 0.1:
                                                    if action_name in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                                                        directive_reward = 5.0  # Encourage exploration
                                                # Low variance = repetitive → penalize to break loop
                                                elif snn_var < 0.01:
                                                    if action_name == 'NOOP':
                                                        directive_reward = -10.0  # Strong penalty for inaction
                                                    logger.debug(f"🧠 SNN: Low variance ({snn_var:.4f}), encouraging change")
                                    except:
                                        pass  # SNN not available, fall through
                                
                            # ═══════════════════════════════════════════════════════════════
                            # 🚨 FINAL STUCK OVERRIDE - HIGHEST PRIORITY! Bypass all other logic
                            # ═══════════════════════════════════════════════════════════════
                            if getattr(self, '_forced_stuck_action', None):
                                action_name = self._forced_stuck_action
                                self._forced_stuck_action = None  # Clear for next step
                                if self.step_count % 10 == 0:
                                    logger.warning(f"🔥 FINAL STUCK OVERRIDE: Forcing {action_name} (bypassed all other logic)")
                                
                            if directive_reward != 0:
                                reward += directive_reward
                                if self.step_count % 10 == 0:
                                     logger.info(f"✅ Followed directive {target}: {action_name} ({directive_reward:+.1f})")
                    except Exception as e:
                        logger.error(f"Directive error: {e}")
                    
                    # VLM SERVER-AWARE GUIDANCE (NEW!)
                    # SKIP VLM in early game areas - use ground truth only!
                    # VLM only called when stuck (for exploration fallback)
                    has_starter = memory.get('party_size', 0) > 0 or 'STARTER_CHOSEN' in str(memory.get('milestones', {}))
                    in_player_house = map_id in [0x100, 0x101, 0x0100, 0x0101]  # Brendan's house = player's house
                    in_any_house = map_id in [0x100, 0x101, 0x0100, 0x0101, 0x102, 0x103, 0x0102, 0x0103]
                    in_littleroot = map_id == 0x09  # Littleroot Town outdoor
                    in_route_101 = map_id == 0x10   # Route 101
                    in_oldale = map_id == 0x11      # Oldale Town
                    in_route_103 = map_id == 0x12   # Route 103
                    in_lab = map_id == 0x104        # Birch's Lab
                    
                    # EARLY GAME AREAS: Use ground truth only, skip VLM!
                    # VLM hallucinates about what to do, ground truth is more reliable
                    in_early_game_area = in_any_house or in_littleroot or in_route_101 or in_oldale or in_route_103 or in_lab
                    
                    # Check if in battle (suppress SLAM to save CPU/memory!)
                    game_state = memory.get('game_state', '').upper()
                    in_battle = game_state == 'BATTLE'
                    self.cached_game_state = game_state  # Cache for battle override in next step()
                    
                    # Stuck detection: no position change for 30+ steps
                    # SUPPRESSED during battles (normal to be "stuck" at same position)
                    if not hasattr(self, 'vlm_stuck_counter'):
                        self.vlm_stuck_counter = 0
                        self.vlm_last_pos = (x, y, map_id)
                    
                    if (x, y, map_id) == self.vlm_last_pos:
                        self.vlm_stuck_counter += 1
                    else:
                        self.vlm_stuck_counter = 0
                        self.vlm_last_pos = (x, y, map_id)
                    
                    is_vlm_stuck = self.vlm_stuck_counter > 50  # Stuck for 50+ steps
                    
                    # Skip VLM in early game UNLESS stuck
                    skip_vlm_for_pathfinding = has_starter and in_early_game_area and not is_vlm_stuck
                    
                    should_call_vlm = (
                        not skip_vlm_for_pathfinding and  # Skip VLM in early game!
                        (self.step_count % self.vlm_frequency == 0 or
                        (is_stuck and self.step_count % 30 == 0))
                    )
                    
                    if skip_vlm_for_pathfinding and self.step_count % 200 == 0:
                        logger.info(f"🚫 VLM SKIPPED: Early game area (map {map_id:04X}) - using ground truth only!")
                    elif is_vlm_stuck and self.step_count % 50 == 0:
                        logger.info(f"🔄 VLM ENABLED: Stuck for {self.vlm_stuck_counter} steps - calling VLM for exploration")
                    
                    if should_call_vlm:
                        logger.info(f"🔍 Step {self.step_count}: Calling VLM oracle at ({x},{y}) map {map_id:04X}...")
                        
                        # Process dialog with structured quest memory system
                        if hasattr(self, 'dialogue_processor') and game_state == 'dialog':
                            map_name = get_map_name(map_id)
                            has_pokemon = memory.get('party_size', 0) > 0
                            has_pokedex = memory.get('has_pokedex', False)
                            
                            # Extract completed milestones from memory
                            milestones = memory.get('milestones', {})
                            completed = [k for k, v in milestones.items() if v] if isinstance(milestones, dict) else []
                            
                            if self.dialogue_processor.is_important_dialog(map_name, game_state):
                                # Extract actual dialogue text if available from memory
                                dialogue_text = memory.get('dialogue_text', None)
                                
                                # Parse dialog into structured quest objective
                                # Uses LLM if dialogue_text provided, otherwise rule-based
                                parsed = self.dialogue_processor.parse_and_update(
                                    map_name, 
                                    completed,
                                    has_pokemon,
                                    has_pokedex,
                                    dialogue_text=dialogue_text
                                )
                                if parsed:
                                    obj = self.dialogue_processor.story_memory
                                    logger.info(f"📋 Quest Updated: {obj['objective']} → {obj['target_location']} (NPC: {obj['target_npc']})")
                                    logger.info(f"   ↳ Event spike sent to SNN (dialogue_event={self.dialogue_processor.dialogue_event_spike})")
                        
                        current_frame = self.frame_stack[-1] if len(self.frame_stack) > 0 else None
                        # Pass CURRENT memory for fresh data + stuck status
                        guidance = self.oracle.get_comprehensive_guidance(
                            game_frame=current_frame, 
                            step_count=self.step_count, 
                            current_memory=memory,
                            is_stuck=is_stuck,
                            recent_positions=self.position_history[-5:] if is_stuck else None,
                            recent_actions_list=self.action_history[-5:] if is_stuck else None
                        )
                        if guidance:
                            logger.info(f"✅ VLM returned guidance: {guidance[:100]}...")
                        
                        if guidance:
                            suggested_action = self.oracle.parse_action_from_response(guidance)
                            
                            # Update VLM guidance tracker for SNN integration
                            if hasattr(self, 'vlm_guidance_tracker'):
                                confidence = 1.0 if is_stuck else 0.8  # Higher confidence when stuck
                                self.vlm_guidance_tracker.update(suggested_action, confidence)
                            
                            # Track if VLM suggestion is unchanged
                            if suggested_action == self.last_vlm_suggestion:
                                self.vlm_suggestion_unchanged_steps += 1
                            else:
                                self.vlm_suggestion_unchanged_steps = 0
                                self.last_vlm_suggestion = suggested_action
                            
                            # Skip VLM reward if high-priority directive is active (prevents conflict)
                            # High-priority directives (Rival Arc railroading) use priority >= 8
                            suppress_vlm = False
                            if hasattr(self, 'last_directive') and self.last_directive:
                                directive_target = str(self.last_directive.get('target', '')).upper()
                                directive_priority = self.last_directive.get('target_level', 0)
                                
                                # Suppress VLM for EXIT_HOUSE (VLM might say "go upstairs" wrongly)
                                if 'EXIT_HOUSE' in directive_target or 'EXIT_LAB' in directive_target:
                                    suppress_vlm = True
                                # Suppress during post-blackout recovery
                                if 'RECOVERY' in directive_target or 'POST_BLACKOUT' in directive_target:
                                    suppress_vlm = True
                                # During Rival Arc (priority 10), suppress VLM to avoid UP/DOWN conflicts
                                if directive_priority >= 8:
                                    suppress_vlm = True
                                
                                if suppress_vlm and self.step_count % 50 == 0:
                                    logger.info(f"⚠️ VLM reward suppressed: {directive_target}")
                            
                            # Bigger bonus if stuck and following advice (only if not suppressed)
                            if suggested_action == action_name and not suppress_vlm:
                                bonus = 5.0 if is_stuck else 2.0
                                reward += bonus
                                logger.info(f"✅ Agent followed VLM advice: {action_name} (+{bonus}) {'[STUCK]' if is_stuck else ''}")
                    
                    # ===== ENHANCED REWARDS FROM POKEMONREDEXPERIMENTS =====
                    
                    # 1. BADGE REWARDS (DOUBLED - badges are critical milestones)
                    try:
                        badge_count = len(game_data.get('badges', []))
                        if not hasattr(self, 'last_badge_count'):
                            self.last_badge_count = badge_count
                        if badge_count > self.last_badge_count:
                            badge_reward = 200.0 * (badge_count - self.last_badge_count)  # HUGE!
                            reward += badge_reward
                            info_parts.append(f"🏆 BADGE +{badge_reward}")
                            self.last_badge_count = badge_count
                    except:
                        pass
                    
                    # 2. HP/HEALING REWARDS (PokemonRed-style survival incentive)
                    try:
                        party = state.get('game', {}).get('party', [])
                        if party:
                            total_hp = sum(p.get('current_hp', 0) for p in party)
                            max_hp = sum(p.get('max_hp', 1) for p in party)
                            hp_fraction = total_hp / max(max_hp, 1)
                            
                            if not hasattr(self, 'last_hp_fraction'):
                                self.last_hp_fraction = hp_fraction
                                self.last_party_size = len(party)
                            
                            # Reward healing (HP increased without party size change)
                            if hp_fraction > self.last_hp_fraction and len(party) == self.last_party_size:
                                heal_amount = hp_fraction - self.last_hp_fraction
                                healing_reward = 5.0 * (heal_amount ** 2)  # Squared like PokemonRed
                                reward += healing_reward
                                if healing_reward > 0.5:
                                    info_parts.append(f"❤️ Heal +{healing_reward:.1f}")
                            
                            # Penalize blackout (all Pokemon fainted)
                            elif hp_fraction == 0 and self.last_hp_fraction > 0:
                                blackout_penalty = -50.0
                                reward += blackout_penalty
                                info_parts.append(f"☠️ Blackout {blackout_penalty}")
                            
                            self.last_hp_fraction = hp_fraction
                            self.last_party_size = len(party)
                            
                            # ===== RIVAL BATTLE REWARD INCENTIVES =====
                            # Problem: Agent blacks out on Route 103 before reaching rival
                            # Solution: Strong rewards for reaching rival with HP intact
                            
                            # 4a. RIVAL PROXIMITY BONUS (Route 103 only)
                            if map_id == 0x12:  # Route 103
                                rival_pos = (12, 3)  # Rival location
                                current_dist = abs(x - rival_pos[0]) + abs(y - rival_pos[1])
                                if not hasattr(self, 'last_rival_dist'):
                                    self.last_rival_dist = current_dist
                                if current_dist < self.last_rival_dist:
                                    # Reward for getting closer to rival
                                    approach_bonus = 50.0 * (self.last_rival_dist - current_dist)
                                    reward += approach_bonus
                                    info_parts.append(f"🎯 Rival approach +{approach_bonus:.0f}")
                                self.last_rival_dist = current_dist
                                
                                # 4b. HP CONSERVATION BONUS (Route 103)
                                # Encourage reaching rival with HP intact
                                if hp_fraction > 0.5 and not hasattr(self, '_hp_conserve_logged'):
                                    hp_conserve = 20.0 * hp_fraction
                                    reward += hp_conserve
                                    if self.step_count % 50 == 0:
                                        info_parts.append(f"💪 HP conserve +{hp_conserve:.0f}")
                            
                            # 4c. LEVEL-UP BONUS (encourages grinding before rival)
                            if party:
                                lead_level = party[0].get('level', 5)
                                if not hasattr(self, 'last_lead_level'):
                                    self.last_lead_level = lead_level
                                if lead_level > self.last_lead_level:
                                    level_bonus = 500.0 * (lead_level - self.last_lead_level)
                                    reward += level_bonus
                                    info_parts.append(f"⬆️ LEVEL UP +{level_bonus:.0f}")
                                    self.last_lead_level = lead_level
                    except:
                        pass
                    
                    # 4d. FIRST_RIVAL_BATTLE WIN MILESTONE BONUS
                    if hasattr(self, 'completed_first_rival_battle_locally') and self.completed_first_rival_battle_locally:
                        if not hasattr(self, '_rival_win_rewarded') or not self._rival_win_rewarded:
                            rival_win_bonus = 1000.0
                            reward += rival_win_bonus
                            info_parts.append(f"🎉 RIVAL BATTLE WON +{rival_win_bonus:.0f}")
                            self._rival_win_rewarded = True
                    
                    # 5. KNN EPISODIC MEMORY - Reward-based action bias (NeurIPS/CVPR 2024)
                    # Query memory only when SNN is uncertain (event-driven sparse retrieval)
                    try:
                        if hasattr(self, 'episodic_memory'):
                            # Build state key from current context
                            state_key = np.zeros(32, dtype=np.float32)
                            state_key[0] = x / 30.0  # Normalized position
                            state_key[1] = y / 30.0
                            state_key[2] = map_id / 256.0 if map_id else 0
                            state_key[3] = hp_fraction if 'hp_fraction' in dir() else 0.5
                            # Fill rest with milestone progress
                            milestones_completed = set(game_data.get('milestones_completed', []))
                            for i, m in enumerate(['STARTER_CHOSEN', 'OLDALE_TOWN', 'ROUTE_103', 'FIRST_RIVAL_BATTLE']):
                                if i + 4 < 32:
                                    state_key[i + 4] = 1.0 if m in milestones_completed else 0.0
                            
                            # SNN-gated query (only when uncertain)
                            snn_confidence = getattr(self, 'snn_confidence', 0.5)
                            knn_result = self.episodic_memory.query(state_key, map_id, snn_confidence)
                            
                            if not knn_result.get('skipped', True) and knn_result['confidence'] > 0.3:
                                # Reward bias for matching remembered successful actions
                                action_weights = knn_result['action_weights']
                                if hasattr(self, 'last_action_idx'):
                                    action_match = action_weights[self.last_action_idx]
                                    if action_match > 0.3 and knn_result['avg_reward'] > 0:
                                        knn_bonus = 10.0 * action_match * knn_result['confidence']
                                        reward += knn_bonus
                                        if self.step_count % 100 == 0:
                                            info_parts.append(f"🧠 KNN +{knn_bonus:.1f}")
                            
                            # Store current experience and add exploration bonus
                            if hasattr(self, 'last_action_idx'):
                                is_novel = self.episodic_memory.store(state_key, self.last_action_idx, reward, map_id or 0)
                                
                                # Exploration bonus for discovering new states (PokemonRedExperiments style)
                                if is_novel:
                                    explore_bonus = self.episodic_memory.explore_weight * 10.0  # +0.1 per unique state
                                    reward += explore_bonus
                                    if self.step_count % 100 == 0:
                                        info_parts.append(f"🔍 NOVEL +{explore_bonus:.1f} ({self.episodic_memory.unique_states})")
                    except Exception as e:
                        pass  # Don't crash if KNN fails
                    
                    # 3. OPPONENT LEVEL TRACKING (Reward fighting stronger trainers)
                    try:
                        battle_info = state.get('game', {}).get('battle_info', {})
                        if battle_info and battle_info.get('in_battle'):
                            opp_pokemon = battle_info.get('opponent_pokemon', {})
                            opp_level = opp_pokemon.get('level', 0)
                            # Trust enhanced memory_reader heuristics (L5=Rival, L<5=Wild)
                            is_wild = battle_info.get('is_wild_battle', True)
                            is_trainer = battle_info.get('is_trainer_battle', not is_wild)

                            # Track if fighting Rival (Route 103 Trainer Battle)
                            if not hasattr(self, 'last_battle_was_rival'):
                                self.last_battle_was_rival = False
                                
                            if map_id == 0x12 and is_trainer:
                                self.last_battle_was_rival = True
                            elif map_id != 0x12:
                                self.last_battle_was_rival = False
                            
                            # BATTLE STRATEGY: RUN from wild battles when:
                            # 1. HP is low (< 30%)
                            # 2. HEAL directive is active (traveling to PokeCenter/Mom)
                            # 3. Early game speedrun (no need to grind wild encounters)
                            should_run = False
                            run_reason = ""
                            
                            # Check HP
                            party = state.get('game', {}).get('party', [])
                            if party:
                                total_hp = sum(p.get('hp', 0) for p in party)
                                max_hp = sum(p.get('max_hp', 1) for p in party) or 1
                                hp_percent = total_hp / max_hp
                                
                                if hp_percent < 0.30 and is_wild:
                                    should_run = True
                                    run_reason = f"Low HP ({hp_percent:.0%})"
                            
                            # Check if healing directive active
                            if hasattr(self, 'last_directive') and self.last_directive:
                                directive_target = str(self.last_directive.get('target', '')).upper()
                                if ('HEAL' in directive_target or 'POKECENTER' in directive_target or 
                                    'MOM' in directive_target or 'CRITICAL_HP' in directive_target):
                                    if is_wild:
                                        should_run = True
                                        run_reason = "Traveling to heal"
                            
                            # Early game speedrun - prioritize story over wild grinding
                            has_pokedex = state.get('game', {}).get('progress_context', {}).get('has_pokedex', False)
                            if not has_pokedex and is_wild:
                                should_run = True
                                run_reason = "Early game speedrun"
                            
                            # ═══════════════════════════════════════════════════════════════════
                            # 🏃 ROUTE 103 STRATEGY: RUN from wild battles, FIGHT rival trainer!
                            # Wild battles waste time - RUN to reach rival faster
                            # Rival trainer battle = MUST FIGHT to complete FIRST_RIVAL_BATTLE milestone
                            # Note: You can't run from trainer battles anyway, so this is just for wild
                            # ═══════════════════════════════════════════════════════════════════
                            if map_id == 0x12:  # Route 103
                                if is_wild:
                                    should_run = True  # RUN from wild battles to reach rival faster!
                                    run_reason = "Route 103 - rushing to rival!"
                                    if self.step_count % 50 == 0:
                                        logger.info("🏃 ROUTE 103: Running from wild battle to reach rival faster!")
                                else:
                                    should_run = False  # FIGHT trainer (rival) battle!
                                    if self.step_count % 50 == 0:
                                        logger.info("⚔️ ROUTE 103 RIVAL BATTLE! MUST FIGHT to get Pokédex!")
                            
                            # Apply RUN reward/penalty - STRONG incentive to run!
                            if should_run:
                                # Reward running (DOWN+RIGHT to reach RUN, then A to select)
                                if action_name in ['RIGHT', 'DOWN', 'B']:
                                    reward += 50.0  # Strong run incentive
                                    if self.step_count % 20 == 0:
                                        logger.info(f"🏃 RUN encouraged: {run_reason}")
                                elif action_name == 'A':
                                    reward -= 20.0  # Strong penalty for attacking when should run
                            else:
                                # ═══════════════════════════════════════════════════════════════════
                                # ⚔️ FIGHT REWARDS: Strongly reward attacking in trainer battles!
                                # Especially important for Route 103 rival battle to advance story!
                                # ═══════════════════════════════════════════════════════════════════
                                if action_name == 'A':
                                    reward += 100.0  # Strong reward for attacking!
                                    if map_id == 0x12:  # Extra bonus for Route 103 rival
                                        reward += 200.0  # HUGE bonus - this is the critical story battle!
                                        if self.step_count % 30 == 0:
                                            logger.info("⚔️ FIGHTING RIVAL! A button +300 reward!")
                                elif action_name in ['RIGHT', 'DOWN']:
                                    # Penalty for trying to run when should fight
                                    if map_id == 0x12:
                                        reward -= 100.0  # Strong penalty for running from rival!
                                        if self.step_count % 30 == 0:
                                            logger.info("🚫 NO RUNNING from rival battle! -100")
                            
                            # Level tracking (existing logic)
                            if not hasattr(self, 'max_opponent_level'):
                                self.max_opponent_level = 0
                            
                            if opp_level > self.max_opponent_level:
                                level_increase = opp_level - self.max_opponent_level
                                opponent_reward = 2.0 * level_increase
                                reward += opponent_reward
                                info_parts.append(f"⚔️ Stronger foe +{opponent_reward:.1f}")
                                self.max_opponent_level = opp_level
                    except:
                        pass
                    
                    # 4. PARTY LEVEL SUM (Reward training Pokemon)
                    try:
                        party = state.get('game', {}).get('party', [])
                        if party:
                            level_sum = sum(p.get('level', 0) for p in party)
                            
                            if not hasattr(self, 'max_level_sum'):
                                self.max_level_sum = level_sum
                            
                            if level_sum > self.max_level_sum:
                                level_gain = level_sum - self.max_level_sum
                                level_reward = 1.0 * level_gain
                                reward += level_reward
                                if level_reward >= 1.0:
                                    info_parts.append(f"📈 Level up +{level_reward:.1f}")
                                self.max_level_sum = level_sum
                    except:
                        pass
                    
                    # 5. TERRAIN-AWARE REWARDS (Solve Route 101 stuck issue)
                    try:
                        if hasattr(self, 'terrain_feature_extractor'):
                            # Update terrain tracker with current state
                            old_pos = (memory.get('x', 0), memory.get('y', 0))
                            new_pos = (x, y)
                            
                            # Get current tile behavior
                            map_tiles = state.get('map', {}).get('tiles', [])
                            current_tile_behavior = None
                            if map_tiles and len(map_tiles) > 0:
                                center = len(map_tiles) // 2
                                if 0 <= center < len(map_tiles) and 0 <= center < len(map_tiles[center]):
                                    tile = map_tiles[center][center]
                                    if len(tile) >= 2:
                                        current_tile_behavior = tile[1]
                            
                            self.terrain_feature_extractor.update_from_action(
                                old_pos, new_pos, action_name, current_tile_behavior
                            )
                            
                            # REWARD walking through grass (don't avoid it!)
                            if current_tile_behavior:
                                bval = current_tile_behavior.value if hasattr(current_tile_behavior, 'value') else current_tile_behavior
                                if isinstance(bval, int) and bval in [2, 3]:  # TALL_GRASS, LONG_GRASS
                                    grass_reward = 1.0
                                    reward += grass_reward
                                    if self.step_count % 25 == 0:
                                        info_parts.append(f"🌿 Grass OK +{grass_reward:.1f}")
                            
                            # PENALIZE trying to climb ledges (they're one-way!)
                            if old_pos == new_pos and action_name in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                                # Didn't move when trying - might be hitting ledge
                                if self.terrain_feature_extractor.ledge_collision_count > 5:
                                    ledge_penalty = -0.5
                                    reward += ledge_penalty
                                    if self.step_count % 20 == 0:
                                        info_parts.append(f"⛰️ Ledge wrong way {ledge_penalty}")
                    except:
                        pass
                    
                    obs = self.frame_stack[-1] if len(self.frame_stack) > 0 else np.zeros((84, 84, 3), dtype=np.uint8)
                    
                    info = {
                        'coordinates': (x, y),
                        'map_id': map_id,
                        'tiles_explored': len(self.seen_coords),
                        'maps_discovered': len(self.maps_discovered)
                    }
                    
                    return obs, reward, False, False, info
                
            except Exception as e:
                logger.error(f"Step error: {e}")
            
            obs = self.frame_stack[-1] if len(self.frame_stack) > 0 else np.zeros((84, 84, 3), dtype=np.uint8)
            return obs, 0.0, False, False, {}
        
        def render(self, mode='rgb_array'):
            if len(self.frame_stack) > 0:
                return self.frame_stack[-1]
            return np.zeros((84, 84, 3), dtype=np.uint8)
    
    # Create environment that connects to SERVER
    frame_server_url = "http://localhost:8001"
    env = ServerAwareEnv(server_url, frame_server_url, oracle, args.vlm_frequency)
    
    # RGB-based terrain analysis for CNN guidance
    class RGBTerrainAnalyzer:
        """
        Analyzes RGB pixels to differentiate trees from grass.
        Trees: Darker green (RGB ~30-60), vertical patterns, denser pixels
        Grass: Lighter green (RGB ~80-150), horizontal patterns, sparser pixels
        
        This provides pixel-level understanding that complements map-based TerrainFeatureExtractor.
        """
        def __init__(self):
            self.last_analysis = None
            self.frame_buffer = []  # Store recent frames for temporal consistency
            
        def analyze_frame(self, rgb_frame):
            """
            Analyze 84x84x3 RGB frame to identify trees vs grass.
            Returns dict with walkability masks and attention guidance.
            """
            if rgb_frame is None or rgb_frame.size == 0:
                return self._empty_analysis()
            
            try:
                import numpy as np
                
                # Extract color channels
                if len(rgb_frame.shape) == 3:
                    r = rgb_frame[:, :, 0].astype(float)
                    g = rgb_frame[:, :, 1].astype(float)
                    b = rgb_frame[:, :, 2].astype(float)
                else:
                    return self._empty_analysis()
                
                # Green dominance mask (green channel > red and blue)
                green_dominant = (g > r) & (g > b)
                
                # TREE DETECTION: Dark green + vertical density
                # Trees have RGB ~(30-60, 40-80, 30-60)
                dark_green = (g > 40) & (g < 80) & (r < 60) & (b < 60) & green_dominant
                
                # Check vertical density (trees are vertically structured)
                vertical_density = np.sum(dark_green, axis=0) / max(dark_green.shape[0], 1)  # Column-wise
                tree_columns = vertical_density > 0.3  # High vertical density = tree
                tree_mask = dark_green & tree_columns
                
                # GRASS DETECTION: Light green + horizontal spread
                # Grass has RGB ~(80-150, 100-180, 60-120)
                light_green = (g > 100) & (g < 180) & (r > 80) & (r < 150) & green_dominant
                
                # Check horizontal spread (grass covers ground horizontally)
                horizontal_density = np.sum(light_green, axis=1) / max(light_green.shape[1], 1)  # Row-wise
                grass_rows = horizontal_density > 0.2  # Moderate horizontal density = grass
                grass_mask = light_green & grass_rows[:, np.newaxis]
                
                # WALKABILITY: Grass + non-dark areas (roads, floors)
                # Exclude trees and very dark pixels (walls, deep water)
                very_dark = (r < 30) & (g < 30) & (b < 30)
                walkable_mask = (grass_mask | ~(tree_mask | very_dark)).astype(float)
                
                # Generate attention focus (center more on walkable areas)
                # Use Gaussian blur to create smooth attention field
                from scipy.ndimage import gaussian_filter
                attention_field = gaussian_filter(walkable_mask.astype(float), sigma=5)
                attention_field = (attention_field - attention_field.min()) / (attention_field.max() - attention_field.min() + 1e-8)
                
                analysis = {
                    'tree_mask': tree_mask.astype(float),           # 84x84: 1.0 = tree obstacle
                    'grass_mask': grass_mask.astype(float),         # 84x84: 1.0 = walkable grass
                    'walkable_mask': walkable_mask,                 # 84x84: 1.0 = can walk here
                    'attention_field': attention_field,             # 84x84: smooth attention weights
                    'tree_fraction': float(np.mean(tree_mask)),     # % of screen that's trees
                    'grass_fraction': float(np.mean(grass_mask)),   # % of screen that's grass
                    'walkable_fraction': float(np.mean(walkable_mask))  # % walkable
                }
                
                self.last_analysis = analysis
                return analysis
                
            except Exception as e:
                logger.debug(f"RGB analysis error: {e}")
                return self._empty_analysis()
        
        def _empty_analysis(self):
            """Return empty analysis when frame unavailable"""
            return {
                'tree_mask': np.zeros((84, 84)),
                'grass_mask': np.zeros((84, 84)),
                'walkable_mask': np.ones((84, 84)),  # Assume all walkable if unknown
                'attention_field': np.ones((84, 84)),
                'tree_fraction': 0.0,
                'grass_fraction': 0.0,
                'walkable_fraction': 1.0
            }
        
        def get_summary_features(self):
            """
            Get compact feature vector (8-dim) summarizing RGB analysis.
            Used as input to SNN for high-level reasoning.
            """
            if self.last_analysis is None:
                return [0.0] * 8
            
            return [
                self.last_analysis['tree_fraction'],
                self.last_analysis['grass_fraction'],
                self.last_analysis['walkable_fraction'],
                1.0 if self.last_analysis['tree_fraction'] > 0.3 else 0.0,  # Heavy tree obstruction
                1.0 if self.last_analysis['grass_fraction'] > 0.2 else 0.0,  # Grass route available
                0.0,  # Reserved
                0.0,  # Reserved
                0.0   # Reserved
            ]
    
    # Global state trackers for SNN integration
    class VLMGuidanceTracker:
        """Tracks VLM action suggestions for SNN temporal reasoning"""
        def __init__(self):
            # Action encoding: A=0, B=1, UP=2, DOWN=3, LEFT=4, RIGHT=5, NOOP=6
            self.action_map = {"A": 0, "B": 1, "UP": 2, "DOWN": 3, "LEFT": 4, "RIGHT": 5, "NOOP": 6}
            self.last_vlm_action = 6  # NOOP initially
            self.vlm_confidence = 0.0  # How confident/relevant the suggestion is
            self.steps_since_vlm = 0  # Decay suggestion over time
            
        def update(self, action_string, confidence=1.0):
            """Update from VLM response"""
            action = (action_string or '').upper().strip()
            if action in self.action_map:
                self.last_vlm_action = self.action_map[action]
                self.vlm_confidence = confidence
                self.steps_since_vlm = 0
            
        def get_encoding(self):
            """Get one-hot encoding of VLM suggestion for SNN"""
            # 7 actions + confidence + time decay
            encoding = [0.0] * 7
            encoding[self.last_vlm_action] = 1.0
            
            # Add confidence and temporal decay
            decay_factor = max(0.0, 1.0 - (self.steps_since_vlm / 50.0))  # Decay over 50 steps
            encoding.append(self.vlm_confidence * decay_factor)
            encoding.append(float(self.steps_since_vlm) / 100.0)  # Normalize time
            
            self.steps_since_vlm += 1
            return encoding  # 9-dim vector
    
    # Global spatial map tracker for CNN integration
    class SpatialMapTracker:
        """Tracks grid search spatial data for CNN spatial awareness"""
        def __init__(self):
            self.current_map_id = 0
            self.current_pos = (0, 0)
            self.explored_tile_count = 0
            self.directional_hint = [0.0, 0.0, 0.0, 0.0]  # [north, south, east, west] goal direction
            self.in_north_town = 0.0  # Binary: 0=south/middle, 1=north area
            self.in_middle_town = 0.0
            self.in_south_town = 0.0
            
        def update(self, map_id, x, y, explored_count, spatial_summary=""):
            """Update from grid search results"""
            self.current_map_id = map_id
            self.current_pos = (x, y)
            self.explored_tile_count = min(explored_count / 100.0, 1.0)  # Normalize to [0,1]
            
            # Parse directional hints from spatial summary
            self.directional_hint = [0.0, 0.0, 0.0, 0.0]
            if "NORTH" in spatial_summary and "Route 101" in spatial_summary:
                self.directional_hint[0] = 1.0  # Goal is NORTH
            elif "SOUTH" in spatial_summary:
                self.directional_hint[1] = 1.0
            
            # Detect town region for Littleroot (map 0x09)
            if map_id == 0x09:
                if y <= 6:
                    self.in_north_town = 1.0
                    self.in_middle_town = 0.0
                    self.in_south_town = 0.0
                elif y <= 10:
                    self.in_north_town = 0.0
                    self.in_middle_town = 1.0
                    self.in_south_town = 0.0
                else:
                    self.in_north_town = 0.0
                    self.in_middle_town = 0.0
                    self.in_south_town = 1.0
            else:
                self.in_north_town = 0.0
                self.in_middle_town = 0.0
                self.in_south_town = 0.0
        
        def get_encoding(self):
            """Get 11-dim spatial encoding for CNN"""
            return [
                float(self.current_pos[0]) / 100.0,  # Normalized X
                float(self.current_pos[1]) / 100.0,  # Normalized Y
                self.explored_tile_count,
                self.directional_hint[0],  # North
                self.directional_hint[1],  # South
                self.directional_hint[2],  # East
                self.directional_hint[3],  # West
                self.in_north_town,
                self.in_middle_town,
                self.in_south_town,
                float(self.current_map_id) / 256.0  # Normalized map ID
            ]
    
    # Terrain Feature Extraction for Route 101 Navigation
    class TerrainFeatureExtractor:
        """
        Extracts terrain reasoning features to help agent understand:
        - Ledges are ONE-WAY (can't climb up)
        - Grass is WALKABLE (don't avoid it!)
        - Water needs Surf HM
        """
        def __init__(self):
            from pokemon_env.enums import MetatileBehavior
            self.MetatileBehavior = MetatileBehavior
            
            # Track agent behavior patterns
            self.ledge_collision_count = 0
            self.grass_entries = 0
            self.last_tile_behavior = None
            self.position_history = []
            
        def extract_features(self, game_state):
            """
            Extract 64-dim terrain features from game state
            Returns: list of 64 floats encoding terrain information
            """
            features = []
            
            # Get map tiles if available
            map_data = game_state.get('map', {})
            tiles = map_data.get('tiles', [])
            player_pos = game_state.get('player', {}).get('position', {})
            px, py = player_pos.get('x', 0), player_pos.get('y', 0)
            
            # Extract adjacent tile features (4 directions × 4 features = 16)
            directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # N, S, W, E
            for dx, dy in directions:
                tile_info = self._get_tile_at(tiles, px + dx, py + dy)
                features.extend([
                    1.0 if tile_info['is_blocked'] else 0.0,
                    1.0 if tile_info['is_grass'] else 0.0,
                    1.0 if tile_info['is_ledge'] else 0.0,
                    tile_info['ledge_direction_value']
                ])
            
            # Current tile state (8)
            current = self._get_tile_at(tiles, px, py)
            features.extend([
                1.0 if current['is_grass'] else 0.0,
                1.0 if current['is_ledge'] else 0.0,
                1.0 if current['is_water'] else 0.0,
                1.0 if current['is_door'] else 0.0,
                1.0 if current['ledge_allows_north'] else 0.0,
                1.0 if current['ledge_allows_south'] else 0.0,
                1.0 if current['ledge_allows_east'] else 0.0,
                1.0 if current['ledge_allows_west'] else 0.0,
            ])
            
            # Movement pattern signals (8)
            features.extend([
                min(self.ledge_collision_count / 10.0, 1.0),  # Stuck on ledges
                min(self.grass_entries / 20.0, 1.0),          # Walking through grass (good!)
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Reserved
            ])
            
            # Padding to 64 dims (32 more)
            features.extend([0.0] * 32)
            
            return features[:64]
        
        def _get_tile_at(self, tiles, x, y):
            """Extract tile information at position"""
            info = {
                'is_blocked': False,
                'is_grass': False,
                'is_ledge': False,
                'is_water': False,
                'is_door': False,
                'ledge_direction_value': 0.0,
                'ledge_allows_north': False,
                'ledge_allows_south': False,
                'ledge_allows_east': False,
                'ledge_allows_west': False,
            }
            
            # Find tile at (x,y) in tiles array
            # tiles is usually a 2D list centered on player
            if not tiles or not isinstance(tiles, list):
                return info
            
            try:
                # Assuming tiles is list of rows
                if len(tiles) > 0 and isinstance(tiles[0], list):
                    # Rough center detection (tiles are usually 11x11 or 15x15)
                    center = len(tiles) // 2
                    row_idx = center + (y - 0)  # Relative to something
                    col_idx = center + (x - 0)
                    
                    if 0 <= row_idx < len(tiles) and 0 <= col_idx < len(tiles[row_idx]):
                        tile = tiles[row_idx][col_idx]
                        
                        if len(tile) >= 2:
                            behavior = tile[1]
                            
                            # Check behavior type
                            if hasattr(behavior, 'value'):
                                bval = behavior.value
                            elif isinstance(behavior, int):
                                bval = behavior
                            else:
                                return info
                            
                            # Grass (WALKABLE!)
                            if bval in [2, 3, 7]:  # TALL_GRASS, LONG_GRASS, SHORT_GRASS
                                info['is_grass'] = True
                            
                            # Ledges (ONE-WAY)
                            elif bval >= 56 and bval <= 63:  # JUMP_EAST through JUMP_SOUTHWEST
                                info['is_ledge'] = True
                                info['ledge_direction_value'] = (bval - 56) / 8.0  # Normalize
                                
                                # Ledge directions
                                if bval == 56:  # JUMP_EAST - can only jump TO this from WEST
                                    info['ledge_allows_east'] = True
                                elif bval == 57:  # JUMP_WEST
                                    info['ledge_allows_west'] = True
                                elif bval == 58:  # JUMP_NORTH
                                    info['ledge_allows_north'] = True
                                elif bval == 59:  # JUMP_SOUTH
                                    info['ledge_allows_south'] = True
                            
                            # Water (BLOCKED until Surf)
                            elif bval in [16, 17, 18, 21, 23]:  # Various water types
                                info['is_water'] = True
                                info['is_blocked'] = True
                            
                            # Doors/Warps
                            elif bval in [97, 98, 99, 100, 101, 105]:  # LADDER, arrow warps, doors
                                info['is_door'] = True
                            
                            # Solid obstacles
                            if len(tile) >= 3:
                                collision = tile[2]
                                if collision > 0:
                                    info['is_blocked'] = True
            except Exception as e:
                pass  # Return default info
            
            return info
        
        def update_from_action(self, old_pos, new_pos, action, tile_behavior):
            """Update tracking based on agent's action"""
            self.position_history.append(new_pos)
            if len(self.position_history) > 20:
                self.position_history.pop(0)
            
            # Track grass entries
            if tile_behavior and hasattr(tile_behavior, 'value'):
                bval = tile_behavior.value
                if bval in [2, 3]:  # TALL_GRASS, LONG_GRASS
                    self.grass_entries += 1
            
            # Track ledge collisions (if position didn't change when it should)
            if old_pos == new_pos and action in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                self.ledge_collision_count += 1
        
        def get_encoding(self):
            """Get current terrain encoding (for global access)"""
            # Return default if no recent state
            return [0.0] * 64
    
    # HYBRID Quest System: Hard-coded milestones + Memory-driven navigation
    class DialogueProcessor:
        """
        Hybrid quest processor: Combines milestone structure with dynamic memory.
        
        ARCHITECTURE:
        ┌─────────────────────────────────────────┐
        │  HARD-CODED MILESTONES (Structure)     │  ← Overarching story
        │  START → POKEDEX → GYM1 → ... → E4    │
        └──────────────┬──────────────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────────────┐
        │  MEMORY-DRIVEN NAVIGATION (How to get) │  ← Dynamic from NPCs
        │  Dialogue → Sub-quests → Actions       │
        └──────────────┬──────────────────────────┘
                       │
                ┌──────┴──────┐
                │             │
                ▼             ▼
        ┌──────────┐   ┌──────────┐
        │   SNN    │   │   CNN    │
        │ Temporal │   │ Spatial  │
        │  Memory  │   │  Hints   │
        └──────────┘   └──────────┘
        
        EXAMPLE:
        Milestone: RECEIVED_POKEDEX (hard-coded goal)
        Memory: Birch says "Find May in tall grass on Route 103" (dynamic nav)
        → SNN: Stores quest chain toward RECEIVED_POKEDEX milestone
        → CNN: Uses "tall grass" hint for spatial navigation
        → Agent: Navigates dynamically toward fixed milestone
        """
        def __init__(self, llm=None):
            self.llm = llm  # Optional LLM for parsing actual dialog text
            
            # Important NPCs whose dialog affects main quest
            self.important_npcs = {'BIRCH', 'MAY', 'BRENDAN', 'MOM', 'DAD', 'RIVAL',
                                   'ROXANNE', 'BRAWLY', 'WATTSON', 'FLANNERY', 'NORMAN',
                                   'WALLY', 'STEVEN', 'LEADER', 'PROFESSOR'}
            
            # Keywords that indicate quest-relevant dialog
            self.quest_keywords = {
                'locations': {'ROUTE', 'TOWN', 'CITY', 'GYM', 'LAB', 'CAVE', 'FOREST', 'NORTH', 'SOUTH', 'EAST', 'WEST'},
                'actions': {'FIND', 'GO', 'MEET', 'BATTLE', 'GET', 'BRING', 'RETURN', 'COME', 'DEFEAT', 'EXPLORE'},
                'items': {'POKEDEX', 'BADGE', 'HM', 'TM', 'POKEMON'},
                'terrain': {'GRASS', 'TALL', 'WATER', 'CAVE', 'TREE', 'ROCK', 'LEDGE', 'DOOR', 'STAIRS'}
            }
            
            # OVERARCHING MILESTONES (hard-coded structure)
            # Complete milestone tracking for progression verification (submission.log compatible)
            self.milestones = [
                # Phase 1: Game Initialization
                'GAME_RUNNING', 'PLAYER_NAME_SET', 'INTRO_CUTSCENE_COMPLETE',
                
                # Phase 2: Tutorial & Starting Town
                'LITTLEROOT_TOWN', 'PLAYER_HOUSE_ENTERED', 'PLAYER_BEDROOM',
                'CLOCK_SET', 'RIVAL_HOUSE', 'RIVAL_BEDROOM',
                
                # Phase 3: Professor Birch & Starter
                'ROUTE_101', 'STARTER_CHOSEN', 'BIRCH_LAB_VISITED',
                
                # Phase 4: Rival & Pokedex
                'OLDALE_TOWN', 'ROUTE_103', 'RECEIVED_POKEDEX',
                
                # Phase 5: Route 102 & Petalburg
                'ROUTE_102', 'PETALBURG_CITY', 'DAD_FIRST_MEETING', 'GYM_EXPLANATION',
                
                # Phase 6: Road to Rustboro City
                'ROUTE_104_SOUTH', 'PETALBURG_WOODS', 'TEAM_AQUA_GRUNT_DEFEATED',
                'ROUTE_104_NORTH', 'RUSTBORO_CITY',
                
                # Phase 7: First Gym Challenge
                'RUSTBORO_GYM_ENTERED', 'ROXANNE_DEFEATED', 'FIRST_GYM_COMPLETE',
                
                # Phase 8+: Additional gyms (for future expansion)
                'SECOND_GYM_BADGE', 'THIRD_GYM_BADGE', 'FOURTH_GYM_BADGE',
                'FIFTH_GYM_BADGE', 'SIXTH_GYM_BADGE', 'SEVENTH_GYM_BADGE',
                'EIGHTH_GYM_BADGE', 'ELITE_FOUR_COMPLETE'
            ]
            self.current_milestone_index = 0  # Which milestone we're working toward
            
            # DYNAMIC SUB-QUESTS (memory-driven navigation)
            # These are the steps to reach current milestone, learned from NPCs
            self.quest_chain = []  # List of sub-quest steps
            self.current_quest_index = 0  # Which sub-quest we're on
            
            # Current active sub-quest (what we're doing RIGHT NOW)
            self.story_memory = {
                'active': False,
                'objective': 'NONE',  # FIND_MAY, BATTLE_RIVAL, RETURN_TO_BIRCH, etc.
                'target_location': 'NONE',  # ROUTE_103, BIRCH_LAB, OLDALE_TOWN, etc.
                'target_npc': 'NONE',  # MAY, ROXANNE, WALLY, etc.
                'return_npc': 'NONE',  # Who to return to after completing objective
                'linked_milestone': 'NONE',  # Which OVERARCHING milestone this connects to
                'contributes_to_milestone': True,  # Does this help reach milestone?
                'steps_since_update': 0,
                # Spatial hints from dialogue for CNN guidance
                'terrain_hint': 'NONE',  # TALL_GRASS, WATER, CAVE, etc.
                'direction_hint': 'NONE',  # NORTH, SOUTH, EAST, WEST
                'landmark_hint': 'NONE'  # TREE, ROCK, DOOR, STAIRS, etc.
            }
            
            # Event spike: 1 on step when new dialog parsed, 0 otherwise
            self.dialogue_event_spike = 0.0
            
            # SPATIAL MEMORY: Terrain/landmark hints that guide CNN attention
            self.spatial_hints = {
                'look_for_grass': False,
                'look_for_water': False,
                'look_for_cave': False,
                'look_for_door': False,
                'look_for_stairs': False,
                'prefer_direction': None,  # 'NORTH', 'SOUTH', 'EAST', 'WEST'
                'avoid_terrain': None  # What terrain to avoid
            }
            
        def is_important_dialog(self, location_name, game_state):
            """Check if we should process dialog (simple heuristics)"""
            if game_state != 'dialog':
                return False
            
            # Check if at important NPC location
            for npc in self.important_npcs:
                if npc in (location_name or '').upper():
                    return True
            
            return False
        
        def parse_with_llm(self, npc_name, dialogue_text, milestones_completed, map_id):
            """
            Use LLM to parse actual dialogue text (when available)
            Extracts: objectives, locations, NPCs, AND spatial/terrain hints
            Returns structured quest data or None if not important
            """
            if not self.llm or not dialogue_text:
                return None
            
            prompt = f"""Parse this Pokemon NPC dialogue for quest information:

NPC: {npc_name}
Map ID: {map_id:04X}
Completed Milestones: {', '.join(milestones_completed) if milestones_completed else 'None'}
Dialogue: "{dialogue_text}"

Extract:
1. Is this important for main story quest? (yes/no)
2. What objective does it set? (FIND_MAY, BATTLE_RIVAL, RETURN_TO_BIRCH, GET_BADGE, etc.)
3. What location is mentioned? (ROUTE_103, BIRCH_LAB, OLDALE_TOWN, PETALBURG_CITY, etc.)
4. Which future milestone is this tied to? (RECEIVED_POKEDEX, FIRST_GYM_BADGE, etc.)
5. Should player return to this NPC after? (yes/no)
6. SPATIAL HINTS - What terrain to look for? (TALL_GRASS, WATER, CAVE, DOOR, STAIRS, NONE)
7. DIRECTION HINT - Which direction to go? (NORTH, SOUTH, EAST, WEST, NONE)
8. QUEST CHAIN - Are there multiple steps? If yes, list them.

Format your response as:
important: yes/no
objective: [OBJECTIVE]
target_location: [LOCATION]
linked_milestone: [MILESTONE]
return_to_npc: yes/no
terrain_hint: [TERRAIN or NONE]
direction_hint: [DIRECTION or NONE]
quest_steps: [STEP1 → STEP2 → STEP3 or SINGLE]
"""
            
            try:
                response = self.llm.query(prompt, max_tokens=200)
                
                # Parse LLM response
                important = 'yes' in response.lower().split('important:')[1].split('\n')[0]
                if not important:
                    return None
                
                lines = {line.split(':')[0].strip(): line.split(':')[1].strip() 
                        for line in response.split('\n') if ':' in line}
                
                # Extract spatial hints for CNN guidance
                terrain_hint = lines.get('terrain_hint', 'NONE').upper()
                direction_hint = lines.get('direction_hint', 'NONE').upper()
                quest_steps = lines.get('quest_steps', 'SINGLE')
                
                # Update spatial memory based on terrain hints
                if 'GRASS' in terrain_hint:
                    self.spatial_hints['look_for_grass'] = True
                if 'WATER' in terrain_hint:
                    self.spatial_hints['look_for_water'] = True
                if 'CAVE' in terrain_hint:
                    self.spatial_hints['look_for_cave'] = True
                if 'DOOR' in terrain_hint:
                    self.spatial_hints['look_for_door'] = True
                if 'STAIRS' in terrain_hint:
                    self.spatial_hints['look_for_stairs'] = True
                if direction_hint in ['NORTH', 'SOUTH', 'EAST', 'WEST']:
                    self.spatial_hints['prefer_direction'] = direction_hint
                
                return {
                    'important': True,
                    'objective': lines.get('objective', 'NONE').upper(),
                    'target_location': lines.get('target_location', 'NONE').upper(),
                    'target_npc': (npc_name or 'UNKNOWN').upper(),
                    'return_npc': npc_name if 'yes' in lines.get('return_to_npc', '') else 'NONE',
                    'linked_milestone': lines.get('linked_milestone', 'NONE').upper(),
                    'terrain_hint': terrain_hint,
                    'direction_hint': direction_hint,
                    'quest_chain': quest_steps
                }
            except Exception as e:
                logger.debug(f"LLM dialogue parse error: {e}")
                return None
        
        def parse_and_update(self, location_name, milestones_completed, has_pokemon, has_pokedex, dialogue_text=None):
            """
            Parse dialog based on context (location + progress)
            Uses LLM if dialogue_text provided, otherwise rule-based
            """
            self.dialogue_event_spike = 0.0  # Reset spike
            
            # Try LLM parsing first if text available
            if dialogue_text and self.llm:
                npc_name = next((npc for npc in self.important_npcs if npc in (location_name or '').upper()), 'UNKNOWN')
                parsed = self.parse_with_llm(npc_name, dialogue_text, milestones_completed, 0)
                if parsed:
                    self.story_memory.update({
                        'active': True,
                        'objective': parsed['objective'],
                        'target_location': parsed['target_location'],
                        'target_npc': parsed['target_npc'],
                        'return_npc': parsed['return_npc'],
                        'linked_milestone': parsed['linked_milestone'],
                        'terrain_hint': parsed.get('terrain_hint', 'NONE'),
                        'direction_hint': parsed.get('direction_hint', 'NONE'),
                        'landmark_hint': 'NONE',
                        'steps_since_update': 0
                    })
                    self.dialogue_event_spike = 1.0
                    logger.info(f"📝 Quest Updated: {parsed['objective']} at {parsed['target_location']} | Terrain: {parsed.get('terrain_hint', 'NONE')} | Direction: {parsed.get('direction_hint', 'NONE')}")
                    return True
            
            # BIRCH'S LAB logic (with spatial hints + milestone awareness!)
            if 'BIRCH' in location_name or 'LAB' in location_name:
                if has_pokemon and not has_pokedex and 'FIRST_RIVAL_BATTLE' not in milestones_completed:
                    # Birch says: "Go find May/Brendan in the tall grass on Route 103"
                    # This is a SUB-QUEST that contributes to RECEIVED_POKEDEX milestone
                    
                    current_milestone = self.get_current_milestone_goal()
                    objective = 'FIND_RIVAL'
                    
                    # Check if this quest helps reach current milestone
                    contributes = self.is_quest_relevant_to_milestone(objective, current_milestone)
                    
                    self.story_memory.update({
                        'active': True,
                        'objective': objective,
                        'target_location': 'ROUTE_103',
                        'target_npc': 'MAY',
                        'return_npc': 'BIRCH',
                        'linked_milestone': 'RECEIVED_POKEDEX',  # HARD-CODED: Overarching goal
                        'contributes_to_milestone': contributes,  # Does this help reach milestone?
                        'terrain_hint': 'TALL_GRASS',  # MEMORY: Dynamic spatial hint
                        'direction_hint': 'NORTH',  # MEMORY: Dynamic direction hint
                        'landmark_hint': 'NONE',
                        'steps_since_update': 0
                    })
                    # Update spatial memory: look for grass, prefer going north
                    self.spatial_hints['look_for_grass'] = True
                    self.spatial_hints['prefer_direction'] = 'NORTH'
                    self.dialogue_event_spike = 1.0
                    logger.info(f"📝 Sub-Quest: {objective} | Milestone: {current_milestone} | Terrain: TALL_GRASS, Direction: NORTH")
                    return True
                
                elif has_pokemon and 'FIRST_RIVAL_BATTLE' in milestones_completed and not has_pokedex:
                    # After rival battle, Birch gives Pokedex
                    self.story_memory.update({
                        'active': True,
                        'objective': 'RECEIVE_POKEDEX',
                        'target_location': 'BIRCH_LAB',
                        'target_npc': 'BIRCH',
                        'return_npc': 'NONE',
                        'linked_milestone': 'RECEIVED_POKEDEX',
                        'steps_since_update': 0
                    })
                    self.dialogue_event_spike = 1.0
                    return True
            
            # ROUTE 103 logic (rival battle)
            if '103' in location_name and has_pokemon and 'FIRST_RIVAL_BATTLE' not in milestones_completed:
                self.story_memory.update({
                    'active': True,
                    'objective': 'BATTLE_RIVAL',
                    'target_location': 'ROUTE_103',
                    'target_npc': 'MAY',
                    'return_npc': 'BIRCH',
                    'linked_milestone': 'FIRST_RIVAL_BATTLE',
                    'steps_since_update': 0
                })
                self.dialogue_event_spike = 1.0
                return True
            
            return False
        
        def clear_if_milestone_achieved(self, milestone):
            """Clear objective when linked milestone is completed"""
            if self.story_memory['linked_milestone'] == milestone:
                self.story_memory['active'] = False
                self.story_memory['objective'] = 'NONE'
                self.dialogue_event_spike = 1.0  # Signal completion
                # Keep spatial hints until new quest is given (persistent memory)
                logger.info(f"✅ Quest completed: {milestone} | Spatial hints retained for navigation")
        
        def get_current_milestone_goal(self):
            """
            Get the current overarching milestone we're working toward.
            This is the hard-coded structure that guides overall progression.
            """
            if self.current_milestone_index < len(self.milestones):
                return self.milestones[self.current_milestone_index]
            return 'GAME_COMPLETE'
        
        def is_quest_relevant_to_milestone(self, quest_objective, milestone):
            """
            Check if a sub-quest contributes to reaching the current milestone.
            This connects dynamic memory to fixed milestone structure.
            
            Example:
            Milestone: RECEIVED_POKEDEX
            Quest: FIND_MAY → Yes, needed for Pokedex
            Quest: CATCH_ZIGZAGOON → No, optional side quest
            """
            milestone_quest_map = {
                # Phase 1: Game Init
                'GAME_RUNNING': ['START_GAME', 'LOAD_ROM'],
                'PLAYER_NAME_SET': ['SET_NAME', 'CONFIRM_NAME'],
                'INTRO_CUTSCENE_COMPLETE': ['WATCH_CUTSCENE', 'TRUCK_ARRIVAL'],
                
                # Phase 2: Tutorial Town
                'LITTLEROOT_TOWN': ['EXIT_TRUCK', 'EXPLORE_TOWN'],
                'PLAYER_HOUSE_ENTERED': ['FIND_PLAYER_HOUSE', 'ENTER_HOUSE'],
                'PLAYER_BEDROOM': ['GO_UPSTAIRS', 'ENTER_BEDROOM'],
                'CLOCK_SET': ['SET_CLOCK', 'CONFIRM_TIME'],
                'RIVAL_HOUSE': ['EXIT_HOUSE', 'FIND_RIVAL_HOUSE', 'ENTER_RIVAL_HOUSE'],
                'RIVAL_BEDROOM': ['GO_UPSTAIRS_RIVAL_HOUSE', 'MEET_RIVAL'],
                
                # Phase 3: Starter
                'ROUTE_101': ['LEAVE_LITTLEROOT_NORTH', 'FIND_BIRCH'],
                'STARTER_CHOSEN': ['HELP_BIRCH', 'CHOOSE_STARTER', 'DEFEAT_ZIGZAGOON'],
                'BIRCH_LAB_VISITED': ['RETURN_TO_LITTLEROOT', 'GO_TO_LAB', 'TALK_TO_BIRCH'],
                
                # Phase 4: Rival & Pokedex
                'OLDALE_TOWN': ['LEAVE_ROUTE_101_NORTH', 'ENTER_OLDALE'],
                'ROUTE_103': ['FIND_RIVAL_ON_103', 'GO_TO_ROUTE_103'],
                'RECEIVED_POKEDEX': ['BATTLE_RIVAL', 'RETURN_TO_BIRCH', 'GET_POKEDEX'],
                
                # Phase 5: Petalburg
                'ROUTE_102': ['LEAVE_OLDALE_WEST', 'CROSS_ROUTE_102'],
                'PETALBURG_CITY': ['ARRIVE_PETALBURG', 'EXPLORE_CITY'],
                'DAD_FIRST_MEETING': ['FIND_GYM', 'MEET_DAD', 'TALK_TO_DAD'],
                'GYM_EXPLANATION': ['LEARN_ABOUT_GYMS', 'GET_RUNNING_SHOES'],
                
                # Phase 6: Road to Rustboro
                'ROUTE_104_SOUTH': ['LEAVE_PETALBURG_NORTH', 'ENTER_ROUTE_104'],
                'PETALBURG_WOODS': ['CROSS_ROUTE_104', 'ENTER_WOODS'],
                'TEAM_AQUA_GRUNT_DEFEATED': ['FIND_GRUNT', 'BATTLE_GRUNT', 'SAVE_DEVON_WORKER'],
                'ROUTE_104_NORTH': ['EXIT_WOODS', 'CONTINUE_NORTH'],
                'RUSTBORO_CITY': ['ARRIVE_RUSTBORO', 'ENTER_CITY'],
                
                # Phase 7: First Gym
                'RUSTBORO_GYM_ENTERED': ['FIND_GYM', 'ENTER_GYM', 'NAVIGATE_TRAINERS'],
                'ROXANNE_DEFEATED': ['BATTLE_ROXANNE', 'DEFEAT_ROXANNE'],
                'FIRST_GYM_COMPLETE': ['RECEIVE_BADGE', 'GET_TM39', 'EXIT_GYM'],
                
                # Additional Gyms (future expansion)
                'SECOND_GYM_BADGE': ['GO_TO_DEWFORD', 'FIND_GYM', 'BATTLE_BRAWLY'],
            }
            
            relevant_quests = milestone_quest_map.get(milestone, [])
            return quest_objective in relevant_quests
        
        def get_spatial_hints_for_cnn(self):
            """
            Get spatial hints that CNN can use to modulate attention.
            Returns dict with terrain/direction/landmark guidance.
            
            Example: If Birch says "Find May in tall grass",
            returns {'look_for_grass': True, 'prefer_direction': 'NORTH'}
            → CNN boosts attention on grass tiles in north areas
            """
            return {
                'look_for_grass': self.spatial_hints['look_for_grass'],
                'look_for_water': self.spatial_hints['look_for_water'],
                'look_for_cave': self.spatial_hints['look_for_cave'],
                'look_for_door': self.spatial_hints['look_for_door'],
                'look_for_stairs': self.spatial_hints['look_for_stairs'],
                'prefer_direction': self.spatial_hints['prefer_direction'],
                'terrain_hint': self.story_memory.get('terrain_hint', 'NONE'),
                'direction_hint': self.story_memory.get('direction_hint', 'NONE'),
                'current_milestone': self.get_current_milestone_goal(),  # NEW: Guide toward milestone
                'quest_supports_milestone': self.story_memory.get('contributes_to_milestone', False)
            }
        
        def get_encoding(self):
            """
            Get 16-dim structured encoding for SNN:
            - objective one-hot (5 dims): NONE, FIND_RIVAL, BATTLE_RIVAL, RETURN_TO_BIRCH, RECEIVE_POKEDEX
            - target_location one-hot (4 dims): NONE, ROUTE_103, BIRCH_LAB, OLDALE_TOWN
            - target_npc one-hot (3 dims): NONE, MAY, BIRCH
            - return_npc one-hot (2 dims): NONE, BIRCH
            - active (1 dim): 1.0 if objective active
            - dialogue_event_spike (1 dim): 1.0 on new dialog, else 0.0
            """
            objectives = ['NONE', 'FIND_RIVAL', 'BATTLE_RIVAL', 'RETURN_TO_BIRCH', 'RECEIVE_POKEDEX']
            locations = ['NONE', 'ROUTE_103', 'BIRCH_LAB', 'OLDALE_TOWN']
            npcs = ['NONE', 'MAY', 'BIRCH']
            return_npcs = ['NONE', 'BIRCH']
            
            obj_idx = objectives.index(self.story_memory['objective']) if self.story_memory['objective'] in objectives else 0
            loc_idx = locations.index(self.story_memory['target_location']) if self.story_memory['target_location'] in locations else 0
            npc_idx = npcs.index(self.story_memory['target_npc']) if self.story_memory['target_npc'] in npcs else 0
            ret_idx = return_npcs.index(self.story_memory['return_npc']) if self.story_memory['return_npc'] in return_npcs else 0
            
            # One-hot encodings
            obj_vec = [1.0 if i == obj_idx else 0.0 for i in range(5)]
            loc_vec = [1.0 if i == loc_idx else 0.0 for i in range(4)]
            npc_vec = [1.0 if i == npc_idx else 0.0 for i in range(3)]
            ret_vec = [1.0 if i == ret_idx else 0.0 for i in range(2)]
            
            return obj_vec + loc_vec + npc_vec + ret_vec + [
                1.0 if self.story_memory['active'] else 0.0,
                self.dialogue_event_spike
            ]
    
    # Global milestone state tracker for SNN integration
    # This will be updated by env and read by the SNN extractor
    class MilestoneStateTracker:
        """Tracks milestone state for SNN temporal reasoning"""
        def __init__(self):
            self.milestones = {
                'RIVAL_BEDROOM': 0, 'ROUTE_101': 0, 'STARTER_CHOSEN': 0,
                'BIRCH_LAB_VISITED': 0, 'OLDALE_TOWN': 0, 'ROUTE_103': 0,
                'RECEIVED_POKEDEX': 0, 'FIRST_RIVAL_BATTLE': 0
            }
            self.story_phase = 0  # 0=start, 1=rival, 2=route101, 3=starter, etc.
            
        def update(self, milestone_dict):
            """Update from server milestone response"""
            if milestone_dict:
                for key in self.milestones:
                    if key in milestone_dict:
                        self.milestones[key] = 1 if milestone_dict[key] else 0
                
                # Update story phase based on progression
                if self.milestones['RECEIVED_POKEDEX']:
                    self.story_phase = 5
                elif self.milestones['STARTER_CHOSEN']:
                    self.story_phase = 3
                elif self.milestones['ROUTE_101']:
                    self.story_phase = 2
                elif self.milestones['RIVAL_BEDROOM']:
                    self.story_phase = 1
                    
        def get_vector(self):
            """Get milestone state as vector for SNN"""
            return list(self.milestones.values()) + [self.story_phase / 5.0]  # Normalize phase
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SPARSE EPISODIC MEMORY - Lightweight KNN for Edge SNN (NeurIPS/CVPR 2024)
    # Memory: ~100KB (fits in L2 cache)
    # Features: FP16 storage, SNN-gated queries, reward-based action bias
    # ═══════════════════════════════════════════════════════════════════════════
    class SparseEpisodicMemory:
        """
        Memory-efficient episodic KNN for edge SNN.
        Based on NeurIPS 2024 FEEL-SNN and CVPR 2024 hybrid SNN research.
        
        Memory budget: ~100KB
        - 1000 entries × 32-dim keys (FP16) = 64KB
        - 1000 × 8 action weights (FP16) = 16KB  
        - 1000 rewards + metadata = ~20KB
        """
        def __init__(self, max_entries: int = 1000, key_dim: int = 32, k: int = 5):
            self.max_entries = max_entries
            self.key_dim = key_dim
            self.k = k
            self.count = 0
            
            # FP16 storage for memory efficiency (from CVPR 2024 quantization research)
            self.keys = np.zeros((max_entries, key_dim), dtype=np.float16)
            self.actions = np.zeros((max_entries, 8), dtype=np.float16)  # 8 game actions
            self.rewards = np.zeros(max_entries, dtype=np.float16)
            self.map_ids = np.zeros(max_entries, dtype=np.int16)
            
            # SNN-gated query threshold (from NeurIPS FEEL-SNN)
            self.query_threshold = 0.3  # Only query when SNN confidence < this
            self.last_confidence = 0.0
            
            # Exploration bonus (inspired by PokemonRedExperiments)
            self.unique_states = 0  # Count of unique states discovered
            self.similar_state_dist = 0.3  # Similarity threshold (like sim_frame_dist)
            self.explore_weight = 0.01  # Reward per unique state
            
            logger.info(f"🧠 SparseEpisodicMemory initialized: {max_entries} entries × {key_dim}D (FP16)")
            logger.info(f"   Memory usage: ~{(max_entries * key_dim * 2 + max_entries * 8 * 2 + max_entries * 4) / 1024:.1f}KB")
            logger.info(f"   Exploration: +{self.explore_weight:.3f} per unique state (threshold={self.similar_state_dist})")
        
        def store(self, state_key: np.ndarray, action_idx: int, reward: float, map_id: int) -> bool:
            """
            Store experience with FP16 compression.
            Returns True if this was a NOVEL state (for exploration reward).
            Inspired by PokemonRedExperiments update_frame_knn_index().
            """
            # Normalize key
            key_norm = state_key[:self.key_dim].astype(np.float16)
            key_norm = key_norm / (np.linalg.norm(key_norm) + 1e-8)
            
            is_novel = False
            
            # Check if state is novel (PokemonRedExperiments style)
            if self.count == 0:
                is_novel = True
            else:
                # Query nearest neighbor
                valid_count = min(self.count, self.max_entries)
                distances = np.linalg.norm(self.keys[:valid_count] - key_norm, axis=1)
                min_dist = np.min(distances)
                
                # Only add if sufficiently different from all stored states
                if min_dist > self.similar_state_dist:
                    is_novel = True
            
            # Only store if novel (to maximize diversity like PokemonRedExperiments)
            if is_novel:
                # Circular buffer for fixed memory
                idx = self.count % self.max_entries
                self.keys[idx] = key_norm
                
                # One-hot action
                action_vec = np.zeros(8, dtype=np.float16)
                action_vec[min(action_idx, 7)] = 1.0
                self.actions[idx] = action_vec
                
                # Store reward and map
                self.rewards[idx] = np.float16(reward)
                self.map_ids[idx] = np.int16(map_id)
                
                self.count += 1
                self.unique_states += 1
            
            return is_novel
        
        def get_exploration_reward(self) -> float:
            """Get exploration bonus based on unique states discovered.
            Inspired by PokemonRedExperiments get_knn_reward().
            """
            return self.explore_weight * self.unique_states
        
        def query(self, state_key: np.ndarray, map_id: int = None, snn_confidence: float = 1.0) -> dict:
            """
            SNN-gated KNN query - only runs when SNN is uncertain.
            Returns action bias weights for reward-based learning.
            """
            self.last_confidence = snn_confidence
            
            # Skip query if SNN is confident (event-driven sparse retrieval)
            if snn_confidence > self.query_threshold:
                return {'action_weights': np.ones(8, dtype=np.float32) / 8, 
                        'avg_reward': 0.0, 'confidence': 0.0, 'skipped': True}
            
            if self.count == 0:
                return {'action_weights': np.ones(8, dtype=np.float32) / 8,
                        'avg_reward': 0.0, 'confidence': 0.0, 'skipped': False}
            
            # Normalize query
            query_key = state_key[:self.key_dim].astype(np.float16)
            query_key = query_key / (np.linalg.norm(query_key) + 1e-8)
            
            # Compute distances (vectorized for speed)
            valid_count = min(self.count, self.max_entries)
            distances = np.linalg.norm(self.keys[:valid_count] - query_key, axis=1)
            
            # Filter by map if provided
            if map_id is not None:
                map_mask = self.map_ids[:valid_count] == map_id
                if map_mask.sum() >= self.k:
                    distances[~map_mask] = np.inf
            
            # Get k nearest
            k_indices = np.argpartition(distances, min(self.k, valid_count - 1))[:self.k]
            k_distances = distances[k_indices]
            
            # Compute weights (inverse distance)
            weights = 1.0 / (k_distances + 0.1)
            weights = weights / weights.sum()
            
            # Weighted action distribution (reward-based, not override)
            action_weights = np.zeros(8, dtype=np.float32)
            for i, idx in enumerate(k_indices):
                action_weights += weights[i] * self.actions[idx].astype(np.float32)
            
            # Average reward for value estimation
            avg_reward = np.sum(weights * self.rewards[k_indices].astype(np.float32))
            
            # Confidence based on distance (closer = more confident)
            confidence = min(1.0, np.mean(weights) * 10)
            
            return {
                'action_weights': action_weights,
                'avg_reward': float(avg_reward),
                'confidence': float(confidence),
                'skipped': False
            }
        
        def get_stats(self) -> dict:
            """Get memory statistics."""
            return {
                'entries': min(self.count, self.max_entries),
                'max_entries': self.max_entries,
                'memory_kb': (self.max_entries * self.key_dim * 2 + 
                              self.max_entries * 8 * 2 + 
                              self.max_entries * 4) / 1024,
                'last_confidence': self.last_confidence
            }
    
    # Initialize episodic memory
    episodic_memory = SparseEpisodicMemory(max_entries=1000, key_dim=32, k=5)
    env.episodic_memory = episodic_memory
    logger.info("✅ SparseEpisodicMemory attached to env")
    
    milestone_tracker = MilestoneStateTracker()
    vlm_guidance_tracker = VLMGuidanceTracker()
    spatial_map_tracker = SpatialMapTracker()
    terrain_feature_extractor = TerrainFeatureExtractor()  # Map-based: Ledge/grass reasoning
    rgb_terrain_analyzer = RGBTerrainAnalyzer()  # NEW: Pixel-based tree/grass detection
    # Pass VLM for optional LLM-based dialogue parsing (user's architecture suggestion)
    dialogue_processor = DialogueProcessor(llm=vlm)  # Structured quest memory (not raw dialog dump)
    
    # Attach trackers to env so they update
    env.milestone_tracker = milestone_tracker
    env.vlm_guidance_tracker = vlm_guidance_tracker
    env.spatial_map_tracker = spatial_map_tracker
    env.terrain_feature_extractor = terrain_feature_extractor
    env.rgb_terrain_analyzer = rgb_terrain_analyzer
    env.dialogue_processor = dialogue_processor
    
    # Make globally accessible for CNN/SNN extractors
    import __main__
    __main__.milestone_tracker = milestone_tracker
    __main__.vlm_guidance_tracker = vlm_guidance_tracker
    __main__.spatial_map_tracker = spatial_map_tracker
    __main__.terrain_feature_extractor = terrain_feature_extractor
    __main__.rgb_terrain_analyzer = rgb_terrain_analyzer
    __main__.dialogue_processor = dialogue_processor
    
    logger.info("🎬 Training starting now!")
    logger.info("   Watch the mGBA window to see progress")
    logger.info("   📊 Architecture:")
    logger.info("      CNN ← Visual pixels + Grid search spatial map (WHERE to go)")
    logger.info("      SNN ← VLM high-level reasoning + Story/milestone progression (WHAT to do)")
    logger.info("      Fusion → Policy learns from both spatial awareness & temporal reasoning")
    logger.info("")
    
    # Use standard trainer but with our custom env
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    import torch
    import torch.nn as nn
    
    # ===== ADAPTIVE SURROGATE GRADIENT (NeurIPS 2024 inspired) =====
    # Makes spike generation differentiable for gradient-based learning
    class AdaptiveSurrogateSpike(torch.autograd.Function):
        """Spike function with adaptive surrogate gradient.
        
        Forward: Hard threshold (Heaviside step)
        Backward: Differentiable surrogate with learnable slope
        
        Reference: "Adaptive Surrogate Gradients for Sequential RL in SNNs" NeurIPS 2024
        """
        @staticmethod
        def forward(ctx, membrane, threshold, slope):
            """Forward pass: spike if membrane > threshold"""
            spikes = (membrane > threshold).float()
            ctx.save_for_backward(membrane, threshold, slope)
            return spikes
        
        @staticmethod
        def backward(ctx, grad_output):
            """Backward pass: smooth surrogate gradient"""
            membrane, threshold, slope = ctx.saved_tensors
            # Adaptive surrogate gradient: sigmoid derivative with learnable slope
            # slope controls sharpness: higher = sharper approximation to Heaviside
            diff = membrane - threshold
            surrogate_grad = slope * torch.sigmoid(slope * diff) * (1 - torch.sigmoid(slope * diff))
            return grad_output * surrogate_grad, None, None
    
    def spike_fn(membrane, threshold, slope):
        """Wrapper for AdaptiveSurrogateSpike"""
        return AdaptiveSurrogateSpike.apply(membrane, threshold, slope)
    
    # Hierarchical Architecture: SNN as Central Temporal Core
    class StoryDrivenCNNSNNExtractor(BaseFeaturesExtractor):
        """
        Hierarchical Architecture with SNN as Central Temporal Reasoning Core:
        
        DESIGN PHILOSOPHY (Track 2 Speedrunning):
        ┌──────────────────┐
        │   VLM (Qwen2B)   │  ← High-level semantic + story reasoning
        │  "Go to Gym 1"   │     Milestone understanding, quest logic
        └─────────┬────────┘
                  │ goal embeddings
                  ▼
        ┌─────────────────────────┐
        │    SNN CORE (256 LIF)   │  ← Central temporal reasoning
        │  Event memory, timing   │     "What just happened?"
        │  Story progression      │     "Where in storyline?"
        └──┬────────────┬─────────┘
           │            │
    CNN features    Grid/Map position
    (local vision)  (global location)
           │            │
           └──────┬─────┘
                  ▼
           Low-Level Policy
                  ↓
              Actions
        
        This mirrors human speedrunning:
        • VLM = Think about quest progression ("I need Pokedex before Route 103")
        • SNN = Remember what just happened + time-based events
        • CNN = See where you are locally (doors, NPCs, terrain)
        • Grid = Know where you are globally (town, route, gym)
        
        Key advantages:
        1. Modular specialization - each component does what it's best at
        2. Robust generalization - story reasoning not tied to pixels
        3. Long-horizon capability - hierarchical temporal memory
        """
        def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
            super().__init__(observation_space, features_dim)
            
            n_input_channels = observation_space.shape[0]
            
            # ===== CNN PATH: SPATIAL AWARENESS =====
            self.cnn = nn.Sequential(
                # First conv block - large receptive field
                nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                
                # Second conv block - medium receptive field
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                
                # Third conv block - fine details
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                
                # Fourth conv block - high-level features
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                
                nn.Flatten(),
            )
            
            # Compute CNN output size
            with torch.no_grad():
                sample = torch.zeros(1, n_input_channels, 84, 84)
                cnn_output_size = self.cnn(sample).shape[1]
            
            # Feedback-modulated spatial attention for CNN features
            # Receives SNN navigation hints (64-dim) + CNN features
            self.spatial_attention = nn.Sequential(
                nn.Linear(cnn_output_size + 64, cnn_output_size // 2),  # CNN + SNN feedback
                nn.ReLU(),
                nn.Linear(cnn_output_size // 2, cnn_output_size),
                nn.Sigmoid()
            )
            
            # ===== SNN CORE: CENTRAL TEMPORAL REASONING WITH FEEDBACK =====
            # The SNN is the heart of temporal reasoning, receiving inputs from:
            # 1. VLM (high-level goals: "go to gym", "get starter")
            # 2. Milestones (story flags: starter_chosen, pokedex_received)
            # 3. Dialog Memory (NPC hints: "Find May on Route 103")
            # 4. CNN (visual features: what's on screen now)
            # 5. Grid Search (global position: which town/route)
            # 6. Terrain Features (map-based: ledges, grass walkability)
            # 7. RGB Analysis (pixel-based: tree vs grass detection)
            #
            # FEEDBACK LOOP: SNN also outputs navigation hints back to CNN
            # This creates hierarchical perception: Map→SNN→CNN guidance
            
            snn_hidden_dim = 448  # LIF neurons for temporal patterns (7 inputs × 64 each)
            snn_input_per_source = 64  # Each source contributes 64 neurons
            
            # Input encoders to SNN core
            # 1. VLM high-level goal encoding (9-dim: action suggestion + confidence + decay)
            self.vlm_goal_encoder = nn.Sequential(
                nn.Linear(9, 32),
                nn.ReLU(),
                nn.Linear(32, snn_input_per_source),  # Exactly 64 neurons
                nn.Tanh()
            )
            
            # 2. Milestone/Story flags (9-dim: 8 milestones + story phase)
            self.milestone_flag_encoder = nn.Sequential(
                nn.Linear(9, 32),
                nn.ReLU(),
                nn.Linear(32, snn_input_per_source),  # Exactly 64 neurons
                nn.Tanh()
            )
            
            # 3. Structured Quest Memory (16-dim: objective, location, NPC, event spike)
            # This is NOT raw dialog dump - it's parsed quest facts
            self.quest_memory_encoder = nn.Sequential(
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, snn_input_per_source),  # Exactly 64 neurons
                nn.Tanh()
            )
            
            # 4. CNN visual features (local spatial perception)
            self.cnn_to_snn_encoder = nn.Sequential(
                nn.Linear(cnn_output_size, 128),
                nn.ReLU(),
                nn.Linear(128, snn_input_per_source),  # Exactly 64 neurons
                nn.Tanh()
            )
            
            # 5. Grid/Map position (global spatial grounding)
            self.map_to_snn_encoder = nn.Sequential(
                nn.Linear(11, 32),
                nn.ReLU(),
                nn.Linear(32, snn_input_per_source),  # Exactly 64 neurons
                nn.Tanh()
            )
            
            # 6. Terrain Features (map-based: ledges, grass, one-way tiles)
            self.terrain_feature_encoder = nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, snn_input_per_source),  # Exactly 64 neurons
                nn.Tanh()
            )
            
            # 7. RGB Analysis (pixel-based: tree vs grass detection)
            self.rgb_analysis_encoder = nn.Sequential(
                nn.Linear(8, 32),  # 8-dim summary features from RGB analyzer
                nn.ReLU(),
                nn.Linear(32, snn_input_per_source),  # Exactly 64 neurons
                nn.Tanh()
            )
            
            # SNN LIF neuron parameters (learnable)
            self.snn_threshold = nn.Parameter(torch.ones(snn_hidden_dim) * 0.5)
            self.snn_decay = nn.Parameter(torch.ones(snn_hidden_dim) * 0.8)
            
            # ADAPTIVE SURROGATE: Learnable slope for spike gradient (NeurIPS 2024)
            # Higher slope = sharper spike approximation, lower = smoother gradients
            self.snn_surrogate_slope = nn.Parameter(torch.ones(1) * 5.0)
            
            # SYNAPTIC DELAYS: Buffer for temporal offset learning
            # Different inputs arrive with different delays (3-step max)
            # This helps capture "X happens BEFORE Y" relationships
            self.synaptic_delay_steps = 3
            self.register_buffer('synaptic_delay_buffer', torch.zeros(1, snn_hidden_dim, self.synaptic_delay_steps))
            
            # Learnable delay weights per input source (7 sources)
            self.delay_weights = nn.Parameter(torch.ones(7, self.synaptic_delay_steps) / self.synaptic_delay_steps)
            
            # SNN state (temporal memory across timesteps)
            self.register_buffer('snn_membrane', torch.zeros(1, snn_hidden_dim))
            self.register_buffer('snn_spike_history', torch.zeros(1, snn_hidden_dim, 10))  # Last 10 timesteps
            
            # ===== SNN FEEDBACK: Top-Down Guidance to CNN =====
            # SNN outputs navigation hints that modulate CNN attention
            # This creates feedback: Map/Terrain → SNN reasoning → CNN focus
            self.snn_feedback_decoder = nn.Sequential(
                nn.Linear(snn_hidden_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),  # 64-dim navigation hints
                nn.Tanh()
            )
            
            # ===== LOW-LEVEL POLICY: FUSE SNN REASONING + CNN LOCAL VISION =====
            # SNN provides temporal reasoning (what to do based on story/goals)
            # CNN provides local spatial awareness (where obstacles/NPCs are)
            # This separation allows robust generalization
            fusion_input_size = snn_hidden_dim + cnn_output_size  # SNN core + CNN local vision
            
            self.fusion_network = nn.Sequential(
                nn.Linear(fusion_input_size, 768),
                nn.LayerNorm(768),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(768, features_dim),
                nn.ReLU(),
            )
            
            logger.info(f"🧠 Hierarchical Architecture with FEEDBACK LOOP:")
            logger.info(f"   INPUT STREAMS → SNN CORE:")
            logger.info(f"   🎯 VLM (Qwen2B): High-level goals → SNN (9-dim → 64 neurons)")
            logger.info(f"   📖 Milestones: Story flags → SNN (9-dim → 64 neurons)")
            logger.info(f"   💬 Quest Memory: Structured objectives → SNN (16-dim → 64 neurons)")
            logger.info(f"      ↳ NOT raw dialog - parsed quest facts (WHO/WHAT/WHERE/WHEN)")
            logger.info(f"   👁️  CNN: Local vision → SNN ({cnn_output_size} → 64 neurons)")
            logger.info(f"   🗺️  Grid Search: Global map → SNN (11-dim → 64 neurons)")
            logger.info(f"   🌳 Terrain (Map): Ledges/grass/one-way → SNN (64-dim → 64 neurons)")
            logger.info(f"   🎨 RGB Analysis: Tree vs grass pixels → SNN (8-dim → 64 neurons)")
            logger.info(f"      ↳ Trees: Dark green + vertical | Grass: Light green + horizontal")
            logger.info(f"   ")
            logger.info(f"   🧠 SNN CORE: {snn_hidden_dim} LIF neurons (7 inputs × 64)")
            logger.info(f"      • Integrates: VLM + Story + Quest + Vision + Map + Terrain + RGB")
            logger.info(f"      • 10-timestep spike history (event memory)")
            logger.info(f"      • Learnable threshold & decay parameters")
            logger.info(f"   ")
            logger.info(f"   🔄 FEEDBACK LOOP: SNN → CNN Attention")
            logger.info(f"      • SNN outputs 64-dim navigation hints")
            logger.info(f"      • Modulates CNN spatial attention (focus on walkable areas)")
            logger.info(f"      • Architecture: Map/Terrain → SNN reasoning → CNN guidance")
            logger.info(f"   ")
            logger.info(f"   🎮 Low-Level Policy: SNN reasoning + CNN local vision")
            logger.info(f"      • Fusion: {fusion_input_size} → 768 → {features_dim}")
            logger.info(f"   ")
            logger.info(f"   📊 Total params: ~{sum(p.numel() for p in self.parameters()):,}")
        
        def forward(self, observations: torch.Tensor) -> torch.Tensor:
            batch_size = observations.shape[0]
            
            # ===== STEP 1: Extract initial CNN features (before SNN feedback) =====
            cnn_features = self.cnn(observations)
            
            # ===== HIERARCHICAL FLOW WITH FEEDBACK LOOP =====
            # Flow: Map/Terrain → SNN reasoning → Feedback to CNN → Final policy
            # This mirrors human cognition: Use map knowledge to guide visual attention
            
            # Expand or resize membrane to match batch size
            if self.snn_membrane.shape[0] != batch_size:
                self.snn_membrane = torch.zeros(batch_size, self.snn_membrane.shape[1], 
                                               device=self.snn_membrane.device, dtype=self.snn_membrane.dtype)
                self.snn_spike_history = torch.zeros(batch_size, self.snn_spike_history.shape[1], 
                                                     self.snn_spike_history.shape[2],
                                                     device=self.snn_spike_history.device, 
                                                     dtype=self.snn_spike_history.dtype)
            
            # ===== INPUT 1: VLM HIGH-LEVEL GOALS → SNN =====
            # "Go to Route 101", "Get Pokedex", "Battle Rival"
            vlm_input = torch.zeros(batch_size, 64, device=observations.device)
            try:
                import __main__
                if hasattr(__main__, 'vlm_guidance_tracker'):
                    vlm_encoding = __main__.vlm_guidance_tracker.get_encoding()
                    vlm_tensor = torch.tensor(
                        [vlm_encoding] * batch_size,
                        dtype=torch.float32,
                        device=observations.device
                    )
                    vlm_input = self.vlm_goal_encoder(vlm_tensor)
            except:
                pass
            
            # ===== INPUT 2: MILESTONE/STORY FLAGS → SNN =====
            # starter_chosen, pokedex_received, gym1_beaten, etc.
            milestone_input = torch.zeros(batch_size, 64, device=observations.device)
            try:
                import __main__
                if hasattr(__main__, 'milestone_tracker'):
                    milestone_vector = __main__.milestone_tracker.get_vector()
                    milestone_tensor = torch.tensor(
                        [milestone_vector] * batch_size,
                        dtype=torch.float32,
                        device=observations.device
                    )
                    milestone_input = self.milestone_flag_encoder(milestone_tensor)
            except:
                pass
            
            # ===== INPUT 3: STRUCTURED QUEST MEMORY → SNN =====
            # Parsed quest objectives: FIND_RIVAL on ROUTE_103, return to BIRCH
            # NOT raw dialog - structured facts extracted from important NPC conversations
            quest_input = torch.zeros(batch_size, 64, device=observations.device)
            try:
                import __main__
                if hasattr(__main__, 'dialogue_processor'):
                    quest_encoding = __main__.dialogue_processor.get_encoding()
                    quest_tensor = torch.tensor(
                        [quest_encoding] * batch_size,
                        dtype=torch.float32,
                        device=observations.device
                    )
                    quest_input = self.quest_memory_encoder(quest_tensor)
            except:
                pass
            
            # ===== INPUT 4: CNN LOCAL VISION → SNN =====
            # What's visible on screen right now (doors, NPCs, terrain)
            cnn_input = self.cnn_to_snn_encoder(cnn_features)  # 64 neurons
            
            # ===== INPUT 5: GRID/MAP GLOBAL POSITION → SNN =====
            # Which town/route am I in? Where is my goal?
            map_input = torch.zeros(batch_size, 64, device=observations.device)
            try:
                import __main__
                if hasattr(__main__, 'spatial_map_tracker'):
                    spatial_encoding = __main__.spatial_map_tracker.get_encoding()
                    spatial_tensor = torch.tensor(
                        [spatial_encoding] * batch_size,
                        dtype=torch.float32,
                        device=observations.device
                    )
                    map_input = self.map_to_snn_encoder(spatial_tensor)
            except:
                pass
            
            # ===== INPUT 6: TERRAIN FEATURES (MAP-BASED) → SNN =====
            # Ledges (one-way), grass (walkable), water (blocked)
            terrain_input = torch.zeros(batch_size, 64, device=observations.device)
            try:
                import __main__
                if hasattr(__main__, 'terrain_feature_extractor'):
                    terrain_encoding = __main__.terrain_feature_extractor.get_encoding()
                    terrain_tensor = torch.tensor(
                        [terrain_encoding] * batch_size,
                        dtype=torch.float32,
                        device=observations.device
                    )
                    terrain_input = self.terrain_feature_encoder(terrain_tensor)
            except:
                pass
            
            # ===== INPUT 7: RGB ANALYSIS (PIXEL-BASED) → SNN =====
            # Tree vs grass detection from actual pixels
            rgb_input = torch.zeros(batch_size, 64, device=observations.device)
            try:
                import __main__
                if hasattr(__main__, 'rgb_terrain_analyzer'):
                    # Analyze current frame to differentiate trees from grass
                    # observations shape: (batch, channels, height, width)
                    # Convert first sample to numpy for analysis
                    frame_np = observations[0].cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
                    # Denormalize if needed (assuming 0-1 range)
                    frame_np = (frame_np * 255).astype('uint8')
                    
                    # Run RGB analysis
                    __main__.rgb_terrain_analyzer.analyze_frame(frame_np)
                    rgb_features = __main__.rgb_terrain_analyzer.get_summary_features()
                    
                    rgb_tensor = torch.tensor(
                        [rgb_features] * batch_size,
                        dtype=torch.float32,
                        device=observations.device
                    )
                    rgb_input = self.rgb_analysis_encoder(rgb_tensor)
            except:
                pass
            
            # ===== SNN CORE: INTEGRATE ALL 7 INPUTS =====
            # Concatenate all input streams into SNN input current
            snn_input_current = torch.cat([
                vlm_input,        # 64 neurons: High-level goals
                milestone_input,  # 64 neurons: Story flags
                quest_input,      # 64 neurons: Quest objectives (parsed dialog)
                cnn_input,        # 64 neurons: Local vision
                map_input,        # 64 neurons: Global position
                terrain_input,    # 64 neurons: Map-based terrain (ledges/grass)
                rgb_input         # 64 neurons: Pixel-based tree/grass detection
            ], dim=1)  # Total: 448 neurons (7 × 64)
            
            # ===== SYNAPTIC DELAYS: Process temporal offsets =====
            # Store current input in delay buffer
            if self.synaptic_delay_buffer.shape[0] != batch_size:
                self.synaptic_delay_buffer = torch.zeros(
                    batch_size, self.synaptic_delay_buffer.shape[1], self.synaptic_delay_steps,
                    device=observations.device, dtype=observations.dtype
                )
            
            # Roll buffer and add new input
            self.synaptic_delay_buffer = torch.roll(self.synaptic_delay_buffer, shifts=1, dims=2)
            self.synaptic_delay_buffer[:, :, 0] = snn_input_current
            
            # Apply learnable delay weights (soft attention over delay steps)
            # Each of 7 sources has different temporal importance
            delay_weights_normalized = torch.softmax(self.delay_weights, dim=1)  # (7, 3)
            
            # Compute weighted sum over delays for each source (64 neurons each)
            delayed_input = torch.zeros_like(snn_input_current)
            source_size = 64
            for i in range(7):
                start_idx = i * source_size
                end_idx = (i + 1) * source_size
                # Weight past inputs for this source
                weighted_delay = torch.zeros(batch_size, source_size, device=observations.device)
                for delay_step in range(self.synaptic_delay_steps):
                    weighted_delay += delay_weights_normalized[i, delay_step] * \
                                     self.synaptic_delay_buffer[:, start_idx:end_idx, delay_step]
                delayed_input[:, start_idx:end_idx] = weighted_delay
            
            # LIF dynamics: V(t+1) = decay * V(t) + I(t) with delays
            # SNN integrates delayed information over time (temporal pattern learning)
            self.snn_membrane = self.snn_decay * self.snn_membrane + delayed_input
            
            # ===== ADAPTIVE SURROGATE SPIKE (NeurIPS 2024) =====
            # Use differentiable spike function for gradient-based learning
            # Forward: hard threshold, Backward: smooth surrogate gradient
            spikes = spike_fn(self.snn_membrane, self.snn_threshold, self.snn_surrogate_slope)
            
            # Reset membrane where spikes occurred (soft reset proportional to spike)
            self.snn_membrane = self.snn_membrane * (1.0 - spikes)
            
            # Update spike history (rolling window of last 10 timesteps)
            # This is the temporal memory: "what happened in last N seconds?"
            self.snn_spike_history = torch.roll(self.snn_spike_history.detach(), shifts=-1, dims=2)
            self.snn_spike_history[:, :, -1] = spikes.detach()  # Detach to prevent backward graph reuse
            
            # Aggregate spike history for temporal reasoning features
            snn_temporal_features = self.snn_spike_history.mean(dim=2)
            
            # ===== FEEDBACK LOOP: SNN → CNN ATTENTION =====
            # SNN generates navigation hints based on map/terrain/RGB knowledge
            # These hints modulate CNN attention to focus on relevant areas
            snn_feedback = self.snn_feedback_decoder(snn_temporal_features)  # 64-dim navigation hints
            
            # Apply feedback-modulated spatial attention to CNN features
            # Concatenate CNN features with SNN feedback for attention computation
            attention_input = torch.cat([cnn_features, snn_feedback], dim=1)
            spatial_weights = self.spatial_attention(attention_input)
            attended_cnn = cnn_features * spatial_weights
            
            # ===== LOW-LEVEL POLICY: SNN REASONING + FEEDBACK-GUIDED CNN =====
            # Architecture flow:
            # 1. Map/Terrain/RGB → SNN (temporal reasoning)
            # 2. SNN → Navigation hints (feedback signal)
            # 3. Feedback + CNN features → Modulated attention
            # 4. SNN reasoning + Attended CNN → Final policy
            fused_features = torch.cat([
                snn_temporal_features,  # 448: Temporal reasoning from SNN core (7 inputs)
                attended_cnn,           # 12,800: Feedback-guided CNN vision
            ], dim=1)
            
            # Process through fusion network
            final_features = self.fusion_network(fused_features)
            
            return final_features
    
    # Wrap env - CRITICAL: VecTransposeImage fixes CnnPolicy obs space issue
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecTransposeImage(vec_env)  # Fixes channel ordering for CnnPolicy
    
    logger.info("✅ Environment wrapped with VecTransposeImage")
    logger.info(f"   Observation space: {vec_env.observation_space}")
    
    # Load existing model or create new one
    if args.resume and os.path.exists(args.resume):
        logger.info(f"📂 Resuming from checkpoint: {args.resume}")
        model = PPO.load(args.resume, env=vec_env)
        logger.info("✅ Model loaded! Continuing training...")
    else:
        logger.info("🤖 Creating PPO with Story-Driven CNN+SNN Hybrid policy...")
        
        # Use story-driven CNN+SNN hybrid extractor
        policy_kwargs = dict(
            features_extractor_class=StoryDrivenCNNSNNExtractor,
            features_extractor_kwargs=dict(features_dim=512),
            net_arch=[dict(pi=[256, 128], vf=[256, 128])]  # Deeper policy/value networks
        )
        
        model = PPO(
            "CnnPolicy",
            vec_env,
            policy_kwargs=policy_kwargs,  # Custom extractor!
            verbose=1,
            n_steps=64,
            batch_size=64,
            n_epochs=10,  # Match train_rl_event_flags.py
            learning_rate=0.0003,
            gamma=0.98,  # Match train_rl_event_flags.py
            gae_lambda=0.95,  # Match train_rl_event_flags.py
            clip_range=0.2,
            ent_coef=0.15,  # Match train_rl_event_flags.py
            tensorboard_log=None,  # Disable to prevent hanging
            device='cuda'
        )
        logger.info("✅ PPO model with Story-Driven CNN+SNN Hybrid policy created!")
    logger.info("   Architecture:")
    logger.info("     • CNN PATH: Visual pixels + Grid search spatial map → WHERE to go")
    logger.info("     • SNN PATH: VLM high-level reasoning + Story/milestone → WHAT to do (temporal)")
    logger.info("     • Grid Search → CNN: 11-dim spatial encoding (position, direction, region)")
    logger.info("     • VLM + Story → SNN: Action guidance + narrative progression modulate LIF neurons")
    logger.info("     • Fusion Network: Combines CNN spatial awareness + SNN temporal reasoning")
    logger.info("   Grid Search Frequency: Every 7 VLM calls")
    logger.info("   Rewards: +100 per milestone, +2.0 VLM guidance bonus, +3.0 exploration")
    logger.info("   Server API: /agent, /state, /milestones, /recent_actions, /queue_status")
    
    # Checkpoint callback with timestamp for unique identification
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"./checkpoints_{timestamp}/",
        name_prefix="rl_model"
    )
    
    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    logger.info("✅ Training complete!")
    final_model_name = f"final_model_{timestamp}"
    model.save(final_model_name)
    logger.info(f"💾 Final model saved: {final_model_name}.zip")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("\n🛑 Training interrupted by user")
        sys.exit(0)
