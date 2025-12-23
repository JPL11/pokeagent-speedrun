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

def _get_story_context(completed, location_name, party_size, has_pokedex_flag=False):
    """
    Get current story context and next milestone goal
    Uses both milestone flags AND game state (location, party, pokedex) for accuracy
    """
    # FALLBACK: Use actual game state to infer progression if milestones aren't tracked
    has_pokemon = party_size > 0
    at_lab = "BIRCH" in location_name or "LAB" in location_name
    at_route_101 = "ROUTE" in location_name and "101" in location_name
    at_town = "LITTLEROOT" in location_name and not at_lab
    
    # Check milestones OR infer from game state
    # CRITICAL: If you have Pokemon (STARTER_CHOSEN), you MUST have visited rival bedroom already!
    has_rival_bedroom = 'RIVAL_BEDROOM' in completed or has_pokemon or 'STARTER_CHOSEN' in completed
    has_route_101 = 'ROUTE_101' in completed or at_route_101 or has_pokemon
    has_starter = 'STARTER_CHOSEN' in completed or has_pokemon
    has_pokedex = 'RECEIVED_POKEDEX' in completed or has_pokedex_flag  # Check actual flag!
    
    # Story progression logic
    if not has_rival_bedroom and not has_pokemon:
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
        # Check if FIRST_RIVAL_BATTLE is completed
        # NOTE: We should NOT infer from ROUTE_103 - just stepping in Route 103 doesn't mean battle is done!
        # Check if actually received Pokedex (which means rival battle is complete)
        rival_battled = 'FIRST_RIVAL_BATTLE' in completed or has_pokedex_flag
        
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
            # Birch wants you to find May/Brendan on Route 103 first
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
        # Has Pokedex, not at lab - continue adventure
        return {
            'story': "Early game complete! Time to explore Hoenn region and head to Oldale Town.",
            'goal': "OLDALE_TOWN",
            'why': "Continue north through Route 101 to reach Oldale Town and progress your adventure"
        }

def _get_milestone_action(location_name, completed, game_state, is_stuck, party_size=0, has_pokedex=False):
    """
    Get specific action to progress toward next milestone
    Uses both milestone flags AND game state (party_size, pokedex, location) for accuracy
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
    # NOTE: Don't infer from ROUTE_103 - just stepping there doesn't mean battle is done
    rival_battled = 'FIRST_RIVAL_BATTLE' in completed
    
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
    
    # If player has Pokemon AND Pokedex, guide to next area
    if has_pokemon and (has_pokedex or 'RECEIVED_POKEDEX' in completed):
        if location_name == "LITTLEROOT_TOWN":
            return "Go UP/NORTH to Route 101, then continue north to Oldale Town"
        elif at_lab:
            return "Exit lab (DOWN), then go NORTH through town to Route 101 → Oldale Town"
        elif at_route_101:
            return "Continue UP/NORTH through Route 101 to reach Oldale Town"
        else:
            return "Head north to Oldale Town (Route 101 → Oldale)"
    
    # Original early-game logic (no Pokemon yet)
    if not has_pokemon:
        if 'RIVAL_BEDROOM' not in completed:
            if location_name == "LITTLEROOT_TOWN":
                return "Enter the house on the RIGHT/EAST (rival's house) to reach bedroom upstairs"
            elif location_name in ("MAYS_HOUSE_1F", "BRENDANS_HOUSE_1F"):
                return "Go UP-RIGHT to find stairs, then press UP to go upstairs to bedroom"
            elif location_name in ("MAYS_HOUSE_2F", "BRENDANS_HOUSE_2F"):
                return "Walk around bedroom to trigger milestone completion"
            elif at_lab:
                return "Exit lab (DOWN), then go to rival's house (right building in town)"
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
        
        story_ctx = _get_story_context(completed, location_name, party_size, has_pokedex)
        action_guide = _get_milestone_action(location_name, completed, game_state, is_stuck, party_size, has_pokedex)
        
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

**🚨 CRITICAL RULES - YOU HAVE A STARTER POKEMON:**
⚠️ RIVAL (May) IS ONLY ON ROUTE 103! NEVER IN ANY HOUSE!
⛔ You CANNOT find rival at any house!
⛔ You CANNOT battle rival at any house!
⛔ Do NOT go to rival's house!
⛔ Do NOT go upstairs in any house!
✅ Brendan's house = YOUR house (Mom heals here)
✅ If in ANY house: ACTION = DOWN (exit immediately!)
✅ If in 2F/bedroom: ACTION = DOWN (go to stairs!)
✅ Birch = PROFESSOR at lab, NOT your rival!

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
        # NOTE: Don't infer from ROUTE_103 - just stepping there doesn't mean battle is done
        rival_battled = 'FIRST_RIVAL_BATTLE' in completed
        
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
        
        elif map_id == 0x0102 and starter_chosen and not rival_battled:
            # In Oldale Town - Visit PokeCenter FIRST to set respawn point!
            if not hasattr(self, 'visited_oldale_pokecenter') or not self.visited_oldale_pokecenter:
                return f"📍 LOCATION: Oldale Town at ({x},{y})\n🏥 IMPORTANT: Visit PokeCenter at (7,13) to set respawn!\n⚠️ Without this, blackouts send you to Littleroot!\nACTION: Go to PokeCenter (center of town)\nREASON: Visiting PokeCenter saves your respawn point here!"
            else:
                return f"📍 LOCATION: Oldale Town at ({x},{y})\n✅ Respawn set at PokeCenter!\n🎯 QUEST: May is waiting on Route 103 (north exit)\n🗺️ DIRECTION: Exit NORTH to Route 103\nACTION: UP (move to north exit)\nREASON: May (rival) is on Route 103. Battle her!"
        
        elif map_id == 0x0012 and not rival_battled:
            # On Route 103 - find May!
            return f"📍 LOCATION: Route 103 at ({x},{y})\n🎯 QUEST: Find and battle May (rival)\n👤 LOOK FOR: Trainer sprite (May/Brendan)\nACTION: Explore (UP/DOWN/LEFT/RIGHT) then A to battle\nREASON: May is somewhere on this route. Walk around to find her, press A to initiate battle"
        
        elif map_id == 0x020C and rival_battled and not has_pokedex:
            # In Birch Lab after rival battle - get Pokedex!
            return f"📍 LOCATION: Birch's Lab at ({x},{y})\n🎯 QUEST: Receive Pokedex from Professor Birch\n✅ READY: Battled rival, can now get Pokedex\nACTION: Move toward Birch, press A\nREASON: After battling rival, Birch gives you the Pokedex"
        
        elif (map_id == 0x0102 or map_id == 0x0012) and rival_battled and not has_pokedex:
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
        
        def get_exit_for_objective(self, connections: list, objective: str) -> tuple:
            """
            Find the best exit coordinates from dynamic server connections.
            
            Args:
                connections: List of {'to': location, 'from_pos': [x,y], 'to_pos': [x,y]}
                objective: What we're trying to reach ('route103', 'pokecenter', 'oldale', etc)
            
            Returns:
                (x, y) tuple of exit coordinates, or None if not found
            """
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
                    game_data = state.get('game', {})
                    
                    # WORKAROUND: Server doesn't provide map_id, infer from location string
                    # FIXED MAP IDS from pokemon_env/enums.py:
                    # - Littleroot Town: 0x09
                    # - Brendan's house 1F: 0x100, 2F: 0x101
                    # - May's house 1F: 0x102, 2F: 0x103  
                    # - Birch's Lab: 0x104
                    # - Route 101: 0x10
                    location_str = player.get('location', '').upper()
                    location_to_map = {
                        'MAYS HOUSE 2F': 0x103,  # RIVAL BEDROOM - check FIRST before general MAY!
                        'MAYS HOUSE 1F': 0x102,  # Rival house 1F
                        'BRENDANS HOUSE 2F': 0x101,  # Player bedroom
                        'BRENDANS HOUSE 1F': 0x100,  # Player house 1F
                        'PROFESSOR': 0x104,  # Professor Birch's Lab
                        'BIRCH': 0x104,
                        'LAB': 0x104,
                        'POKECENTER': 0x200,  # PokeCenter interior - check before OLDALE!
                        'ROUTE 103': 0x12,  # Where rival is!
                        'ROUTE 102': 0x0F,  # West of Oldale
                        'ROUTE 101': 0x10,  # Between Littleroot and Oldale
                        'OLDALE': 0x11,     # Oldale Town
                        'LITTLEROOT': 0x09,  # Check last as it might be in other location names
                    }
                    map_id = 0x09  # Default to town
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
                    
                    memory = {
                        'x': x,
                        'y': y,
                        'map_id': map_id,
                        'location': player.get('location', 'unknown'),  # Add readable location name
                        'game_state': game_state,  # Track if in dialog/battle/menu
                        'party_size': party_size,  # From progress_context
                        'has_pokedex': has_pokedex,  # Track if player has Pokedex
                        'badge_count': len(game_data.get('badges', [])),
                        'money': game_data.get('money', 0),
                        'dialogue_text': game_data.get('dialogue_text', None),  # Actual NPC text for LLM parsing
                        'milestones': state.get('milestones', {}),  # Include milestones for dialogue processor
                        # Map terrain data for dynamic pathfinding (ground truth from server)
                        'map_data': state.get('map', {}),  # Full 15x15 tile grid with behaviors
                        # Dynamic exit/portal connections from server (ground truth!)
                        'connections': state.get('map', {}).get('stitched_map_info', {}).get('current_area', {}).get('connections', []),
                    }
                    
                    # Define rival_battled for local scope use (needed for navigation logic)
                    # Define rival_battled for local scope use (needed for navigation logic)
                    milestones = state.get('milestones', {})
                    completed = milestones.get('completed', [])
                    
                    # Local override for immediate responsiveness (server lag compensation)
                    if not hasattr(self, 'completed_first_rival_battle_locally'):
                         self.completed_first_rival_battle_locally = False
                    
                    # NOTE: Don't infer from ROUTE_103 - just stepping there doesn't mean battle is done
                    rival_battled = 'FIRST_RIVAL_BATTLE' in completed or self.completed_first_rival_battle_locally
                    
                    # Check for Milestone Trigger: Winning Rival Battle (Battle -> Overworld transition)
                    if not hasattr(self, 'prev_game_state'): self.prev_game_state = 'overworld'
                    if not hasattr(self, 'last_battle_was_rival'): self.last_battle_was_rival = False
                    
                    if self.prev_game_state == 'battle' and game_state == 'overworld':
                         if self.last_battle_was_rival and not rival_battled:
                             logger.info("🎉 RIVAL BATTLE WON! Triggering FIRST_RIVAL_BATTLE milestone!")
                             self.completed_first_rival_battle_locally = True
                             rival_battled = True
                    
                    self.prev_game_state = game_state
                    
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
                    
                    # SNN PRIORITY THRESHOLDS (stored for SNN input)
                    hp_warning = hp_ratio < 0.50    # 50% - start heading to heal
                    hp_critical = hp_ratio < 0.30   # 30% - interrupt all tasks!
                    hp_emergency = hp_ratio < 0.15  # 15% - maximum priority
                    
                    # Store HP state for SNN input encoding
                    memory['hp_ratio'] = hp_ratio
                    memory['hp_warning'] = hp_warning
                    memory['hp_critical'] = hp_critical
                    memory['hp_emergency'] = hp_emergency
                    
                    # KNN: Find nearest healing location
                    location_upper = (memory.get('location', '') or '').upper()
                    healing_interrupt_active = False
                    healing_target = None
                    
                    if hp_critical and game_state != 'battle':  # Don't interrupt during battle!
                        healing_interrupt_active = True
                        
                        # KNN-style nearest healing location
                        if 'POKECENTER' in location_upper:
                            # Already in PokeCenter! Go to Nurse Joy
                            healing_target = {'type': 'NURSE_JOY', 'action': 'UP_THEN_A', 'coords': (7, 3)}
                        elif 'PLAYER' in location_upper or 'BRENDAN' in location_upper:
                            # In player's house - find Mom
                            if '2F' in location_upper or 'BEDROOM' in location_upper:
                                healing_target = {'type': 'GO_DOWNSTAIRS', 'action': 'DOWN', 'coords': (1, 1)}
                            else:
                                healing_target = {'type': 'MOM', 'action': 'TALK_TO_MOM', 'coords': (5, 8)}
                        elif 'OLDALE' in location_upper:
                            # In Oldale - go to PokeCenter at (7, 13)
                            healing_target = {'type': 'OLDALE_PC', 'action': 'GO_TO_PC', 'coords': (7, 13)}
                        elif 'ROUTE 103' in location_upper:
                            # Route 103 - go south to Oldale
                            healing_target = {'type': 'GO_SOUTH', 'action': 'DOWN', 'coords': None}
                        elif 'ROUTE 101' in location_upper:
                            # Route 101 - KNN: closer to Oldale (north) or Littleroot (south)?
                            if y < 15:  # Closer to Oldale
                                healing_target = {'type': 'OLDALE_PC', 'action': 'UP', 'coords': None}
                            else:  # Closer to Littleroot/Mom
                                healing_target = {'type': 'MOM', 'action': 'DOWN', 'coords': None}
                        elif 'LITTLEROOT' in location_upper:
                            # In Littleroot - go to player's house for Mom
                            healing_target = {'type': 'PLAYER_HOUSE', 'action': 'ENTER_HOUSE', 'coords': (5, 4)}
                        else:
                            # Default: head south to find healing
                            healing_target = {'type': 'FIND_HEALING', 'action': 'DOWN', 'coords': None}
                        
                        if self.step_count % 30 == 0:
                            logger.info(f"🚨 HEALING INTERRUPT! HP={hp_ratio:.1%} Target={healing_target['type']}")
                    
                    memory['healing_interrupt'] = healing_interrupt_active
                    memory['healing_target'] = healing_target
                    
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
                    
                    force_house_exit = has_starter and in_house and not in_lab and not in_pokecenter
                    house_exit_target = None  # DISABLED - let directive system handle it
                    
                    # Old hardcoded pathfinding removed - see directive EXIT_HOUSE handler for replacement
                    
                    memory['force_house_exit'] = force_house_exit
                    memory['house_exit_target'] = house_exit_target  # Will be None
                    
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
                        route101_ledge_interrupt = True  # Always active on Route 101
                        
                        # === USE GROUND TRUTH TERRAIN DATA ===
                        # Server provides 15x15 metatile grid centered on player (position 7,7)
                        # Check which directions are walkable vs blocked
                        map_data = memory.get('map_data', {})
                        metatile_info = map_data.get('metatile_info', [])
                        
                        # Helper to check if a direction is walkable (uses server terrain)
                        def is_direction_walkable(direction):
                            """Check if moving in direction is blocked by terrain
                            
                            Ground truth symbols (from user):
                            - Arrows (↓) = Ledges = JUMP_SOUTH, JUMP_EAST, etc. - one-way!
                            - Dashes (-) = Grass = TALL_GRASS, LONG_GRASS - walkable!
                            - Blocked = Trees = IMPASSABLE_* - cannot pass!
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
                                
                                # === IMPASSABLE (Trees, walls) - totally blocked ===
                                if 'IMPASSABLE' in behavior:
                                    return False
                                
                                # === LEDGES (arrows in ground truth) - one-way only! ===
                                # JUMP_SOUTH means you can only GO south (down), not climb up
                                if 'JUMP_SOUTH' in behavior:
                                    return direction == 'DOWN'  # Can only go DOWN
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
                                if 'JUMP' in behavior:  # Generic ledge
                                    return direction == 'DOWN'  # Assume south-facing
                                
                                # === GRASS (dashes in ground truth) - walkable! ===
                                if 'GRASS' in behavior:
                                    return True  # Grass is always walkable
                                
                                # === NORMAL tiles - check passable flag ===
                                return passable
                                
                            except (IndexError, KeyError, TypeError):
                                return None  # Error reading terrain
                        
                        # Check all directions
                        can_up = is_direction_walkable('UP')
                        can_down = is_direction_walkable('DOWN')
                        can_left = is_direction_walkable('LEFT')
                        can_right = is_direction_walkable('RIGHT')
                        
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
                        
                        in_middle_ledge = 7 <= x <= 10 and 6 <= y <= 9  # LEDGE TRAP - go DOWN!
                        in_upper_right = x >= 10 and y <= 8   # Passed the turn, go LEFT!
                        on_left_oldale_path = x <= 8 and y <= 10  # Left side path to Oldale - go UP!
                        at_oldale_entrance = y <= 3  # Very close to Oldale, go UP!
                        
                        if at_oldale_entrance:
                            # Very close to Oldale entrance - just go UP!
                            ledge_target = {'action': 'UP', 'reason': f'At Oldale entrance y={y} - GO UP!'}
                        
                        elif in_middle_ledge:
                            # CRITICAL FIX: Middle ledge trap at y=6-9!
                            # Agent oscillates LEFT/RIGHT at y=7 because ledge blocks UP
                            # Solution: Go DOWN to get below ledge (y>=10), then navigate around
                            if y < 9 and can_down is True:
                                ledge_target = {'action': 'DOWN', 'reason': f'LEDGE TRAP at ({x},{y}) - GO DOWN to escape!'}
                            elif y >= 9 and can_left is True:
                                # Below ledge now - go LEFT to get to left path
                                ledge_target = {'action': 'LEFT', 'reason': f'Below ledge at ({x},{y}) - go LEFT to path!'}
                            elif can_down is True:
                                ledge_target = {'action': 'DOWN', 'reason': 'Continue DOWN to escape ledge zone'}
                            else:
                                ledge_target = {'action': 'LEFT', 'reason': 'Force LEFT to find path'}
                        
                        elif on_left_oldale_path:
                            # On the left side path to Oldale - go UP!
                            # This is where the agent was getting stuck at (2-4, 8)
                            if can_up is True:
                                ledge_target = {'action': 'UP', 'reason': f'Left path x={x},y={y} - go UP to Oldale!'}
                            elif can_right is True:
                                ledge_target = {'action': 'RIGHT', 'reason': 'UP blocked on left path, try RIGHT'}
                            else:
                                ledge_target = {'action': 'UP', 'reason': 'Force UP to Oldale!'}
                        
                        elif in_upper_right:
                            # CRITICAL: In upper-right area. Per U-shape, must go LEFT!
                            # Oldale entrance is at x~5-8, we're at x>=10
                            if can_left is True:
                                ledge_target = {'action': 'LEFT', 'reason': f'Upper-right x={x}, go LEFT to Oldale entrance!'}
                            elif can_up is True:
                                ledge_target = {'action': 'UP', 'reason': 'LEFT blocked, try UP'}
                            elif can_down is True:
                                ledge_target = {'action': 'DOWN', 'reason': 'Try DOWN to find path'}
                            else:
                                ledge_target = {'action': 'LEFT', 'reason': 'Force LEFT to Oldale!'}
                        
                        elif can_up is True:
                            # Default: If UP is clear, go UP
                            ledge_target = {'action': 'UP', 'reason': 'Terrain: UP clear - GO NORTH!'}
                        
                        elif can_up is False:
                            # UP blocked (ledge) - navigate around based on position
                            if x >= 8:
                                # Right side - try LEFT first
                                if can_left is True:
                                    ledge_target = {'action': 'LEFT', 'reason': 'UP blocked right side, go LEFT'}
                                elif can_right is True:
                                    ledge_target = {'action': 'RIGHT', 'reason': 'Try RIGHT'}
                                else:
                                    ledge_target = {'action': 'DOWN', 'reason': 'Try DOWN'}
                            else:
                                # Left side - try RIGHT first
                                if can_right is True:
                                    ledge_target = {'action': 'RIGHT', 'reason': 'UP blocked left side, go RIGHT'}
                                elif can_down is True:
                                    ledge_target = {'action': 'DOWN', 'reason': 'Try DOWN'}
                                else:
                                    ledge_target = {'action': 'UP', 'reason': 'Force UP'}
                        
                        else:
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
                    
                    if 'OLDALE' in location_upper and 'POKECENTER' not in location_upper:
                        # Check if we already visited PokeCenter
                        if not getattr(self, 'visited_oldale_pokecenter', False):
                            oldale_pokecenter_nav = True
                            
                            # PokeCenter is at (7, 13) in Oldale Town
                            pc_x, pc_y = 7, 13
                            
                            # Calculate direction to PokeCenter
                            if abs(x - pc_x) > abs(y - pc_y):
                                # Horizontal distance greater - move horizontally first
                                if x < pc_x:
                                    oldale_target = {'action': 'RIGHT', 'reason': f'Go RIGHT to PokeCenter ({pc_x},{pc_y})'}
                                elif x > pc_x:
                                    oldale_target = {'action': 'LEFT', 'reason': f'Go LEFT to PokeCenter ({pc_x},{pc_y})'}
                                else:
                                    # Same x - move vertically
                                    if y < pc_y:
                                        oldale_target = {'action': 'DOWN', 'reason': f'Go DOWN to PokeCenter ({pc_x},{pc_y})'}
                                    else:
                                        oldale_target = {'action': 'UP', 'reason': f'Go UP to PokeCenter ({pc_x},{pc_y})'}
                            else:
                                # Vertical distance greater - move vertically first
                                if y < pc_y:
                                    oldale_target = {'action': 'DOWN', 'reason': f'Go DOWN to PokeCenter ({pc_x},{pc_y})'}
                                elif y > pc_y:
                                    oldale_target = {'action': 'UP', 'reason': f'Go UP to PokeCenter ({pc_x},{pc_y})'}
                                else:
                                    # Same y - move horizontally
                                    if x < pc_x:
                                        oldale_target = {'action': 'RIGHT', 'reason': f'Go RIGHT to PokeCenter ({pc_x},{pc_y})'}
                                    else:
                                        oldale_target = {'action': 'LEFT', 'reason': f'Go LEFT to PokeCenter ({pc_x},{pc_y})'}
                            
                            # At PokeCenter door? Go UP to enter!
                            if abs(x - pc_x) <= 1 and abs(y - pc_y) <= 1:
                                oldale_target = {'action': 'UP', 'reason': 'At PokeCenter door - ENTER!'}
                            
                            if self.step_count % 30 == 0:
                                logger.info(f"🏥 OLDALE NAV! Pos=({x},{y}) -> {oldale_target['action']} - {oldale_target['reason']}")
                    
                    memory['oldale_pokecenter_nav'] = oldale_pokecenter_nav
                    memory['oldale_target'] = oldale_target
                    
                    # ═══════════════════════════════════════════════════════════════════
                    # 💊 POKECENTER INTERIOR: TALK TO NURSE JOY
                    # When inside PokeCenter, go UP to counter and press A to heal
                    # Nurse Joy is at the counter (around x=7, y=3)
                    # Use map_id 0x200 for Oldale PokeCenter interior (not string match!)
                    # ═══════════════════════════════════════════════════════════════════
                    
                    nurse_joy_nav = False
                    nurse_target = None
                    
                    # map_id 0x200 = Oldale PokeCenter INTERIOR (ground truth)
                    if map_id == 0x200:
                        nurse_joy_nav = True
                        
                        # Nurse Joy counter is at approximately (7, 3) in PokeCenter
                        nurse_x, nurse_y = 7, 3
                        
                        if y > nurse_y + 1:
                            # Need to go UP to reach counter
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
                    
                    if 'ROUTE' in location_upper and '103' in location_upper:
                        # DISABLED: Use Unified TerrainNavigator instead!
                        route103_nav = False
                        # route103_nav = True
                        
                        # Use terrain data for ledge detection
                        map_data = memory.get('map_data', {})
                        metatile_info = map_data.get('metatile_info', [])
                        
                        # Rival is typically at east side of Route 103 (around x=25)
                        rival_x = 25
                        
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
                                # Ledges - one way only
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
                        
                        # Priority: Go RIGHT (east) to find rival, avoid ledges
                        if x < rival_x - 5:
                            # Need to go east to reach rival
                            if can_right is True:
                                route103_target = {'action': 'RIGHT', 'reason': 'Go EAST to find rival'}
                            elif can_up is True:
                                route103_target = {'action': 'UP', 'reason': 'RIGHT blocked, try UP'}
                            elif can_down is True:
                                route103_target = {'action': 'DOWN', 'reason': 'Try DOWN to find path'}
                            else:
                                route103_target = {'action': 'RIGHT', 'reason': 'Force RIGHT towards rival'}
                        elif x > rival_x + 5:
                            # Overshot - go back west
                            if can_left is True:
                                route103_target = {'action': 'LEFT', 'reason': 'Go WEST, overshot rival area'}
                            else:
                                route103_target = {'action': 'UP', 'reason': 'Try UP to find path'}
                        else:
                            # In rival area - explore to find them
                            if can_up is True:
                                route103_target = {'action': 'UP', 'reason': 'In rival zone - search UP'}
                            elif can_right is True:
                                route103_target = {'action': 'RIGHT', 'reason': 'In rival zone - search RIGHT'}
                            elif can_down is True:
                                route103_target = {'action': 'DOWN', 'reason': 'In rival zone - search DOWN'}
                            else:
                                route103_target = {'action': 'A', 'reason': 'In rival zone - try talking to NPC'}
                        
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
                    
                    # Goal selection based on map and state
                    # Sequences: Goal1 → Exit (for indoor maps)
                    
                    # CHECK 2F FIRST - more specific than map_id!
                    # PokeCenter 2F has escalator on LEFT side at (1,6), access from (2,6)
                    # Use DIRECT direction calculation for 2F (bypass TerrainNavigator)
                    if in_pokecenter_2f:
                        # On 2F - need to take escalator down
                        # Escalator is at (1,6), you stand at (2,6) and go LEFT to use it
                        escalator_access_x, escalator_access_y = 2, 6
                        
                        # Calculate direction directly - 2F is open, no obstacles
                        if x == escalator_access_x and y == escalator_access_y:
                            # At access point - go LEFT to step onto escalator
                            pc2f_action = 'LEFT'
                            pc2f_reason = 'At (2,5) - stepping LEFT onto escalator!'
                        else:
                            # Move toward access point
                            dx = escalator_access_x - x  # 2 - current_x
                            dy = escalator_access_y - y  # 5 - current_y
                            
                            if abs(dx) > abs(dy):
                                pc2f_action = 'LEFT' if dx < 0 else 'RIGHT'
                            elif abs(dy) > 0:
                                pc2f_action = 'UP' if dy < 0 else 'DOWN'
                            else:
                                pc2f_action = 'LEFT'  # Default toward escalator
                            pc2f_reason = f'Heading to escalator access (2,5), dx={dx}, dy={dy}'
                        
                        # Set terrain nav result DIRECTLY (bypass TerrainNavigator for 2F)
                        terrain_nav_result = {'action': pc2f_action, 'reason': pc2f_reason}
                        terrain_nav_active = True
                        terrain_goal = (escalator_access_x, escalator_access_y)
                        
                        if self.step_count % 30 == 0:
                            logger.info(f"🪜 PC 2F: ({x},{y}) -> {pc2f_action} | {pc2f_reason}")
                    
                    elif map_id == 0x11:  # OLDALE_TOWN (outdoor)
                        connections = memory.get('connections', [])
                        
                        if not getattr(self, 'visited_oldale_pokecenter', False):
                            # Try dynamic exit to PokeCenter from connections
                            dynamic_exit = self.get_exit_for_objective(connections, 'pokecenter')
                            if dynamic_exit:
                                terrain_goal = dynamic_exit
                                if self.step_count % 100 == 0:
                                    logger.info(f"🎯 DYNAMIC: Using server exit {dynamic_exit} to POKECENTER")
                            else:
                                terrain_goal = 'pokecenter'  # Fallback to hardcoded
                            terrain_nav_active = True
                        else:
                            # Try dynamic exit to Route 103 from connections
                            dynamic_exit = self.get_exit_for_objective(connections, 'route103')
                            if dynamic_exit:
                                terrain_goal = dynamic_exit
                                if self.step_count % 100 == 0:
                                    logger.info(f"🎯 DYNAMIC: Using server exit {dynamic_exit} to ROUTE 103")
                            else:
                                terrain_goal = 'route103_exit'  # Fallback to hardcoded
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
                    
                    elif map_id == 0x10:  # ROUTE_101
                        terrain_goal = 'oldale_entrance'
                        terrain_nav_active = True
                    
                    elif map_id == 0x12:  # ROUTE_103
                        connections = memory.get('connections', [])
                        
                        if not rival_battled:
                            terrain_goal = 'rival_area'
                            terrain_nav_active = True
                        else:
                            # Return to Oldale after battle
                            dynamic_exit = self.get_exit_for_objective(connections, 'oldale')
                            if dynamic_exit:
                                terrain_goal = dynamic_exit
                                if self.step_count % 100 == 0:
                                    logger.info(f"🎯 DYNAMIC: Using server exit {dynamic_exit} to OLDALE TOWN")
                            else:
                                terrain_goal = 'oldale_exit'
                            terrain_nav_active = True
                    
                    elif map_id == 0x020C:  # BIRCH_LAB
                        # After getting starter, head to exit
                        if getattr(self, 'has_starter', False) or party_size > 0:
                            terrain_goal = 'exit'
                            terrain_nav_active = True
                        else:
                            terrain_goal = 'birch'  # Need to talk to Birch
                            terrain_nav_active = True
                    
                    elif map_id in [0x100, 0x101, 0x102, 0x103]:  # Houses (Brendan/May)
                        # Always exit houses after getting starter
                        if getattr(self, 'has_starter', False) or party_size > 0:
                            # Check if on 2F - need to target STAIRS, not exit door!
                            # 0x101 = Brendan's house 2F, 0x103 = May's house 2F
                            if map_id in [0x101, 0x103]:
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
                                # ON 1F: Target exit door
                                terrain_goal = (4, 8)  # House exit door (LEFT-BOTTOM side, visual position)
                            
                            terrain_nav_active = True
                    
                    # Get optimal direction from TerrainNavigator 
                    # UNLESS already computed (e.g., for 2F direct calculation)
                    if terrain_nav_active and terrain_goal and terrain_nav_result is None:
                        terrain_nav_result = self.terrain_nav.get_direction(
                            map_id, x, y, terrain_goal, metatile_info
                        )
                        
                        if terrain_nav_result and self.step_count % 50 == 0:
                            logger.info(f"🧭 TERRAIN NAV: ({x},{y}) -> {terrain_nav_result['action']} | Goal={terrain_goal} | {terrain_nav_result['reason']}")
                    
                    memory['terrain_nav_active'] = terrain_nav_active
                    memory['terrain_nav_result'] = terrain_nav_result
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
                    
                    # Calculate reward
                    reward = 0.0
                    info_parts = []
                    
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
                    if self.steps_in_house > 20:
                        house_penalty = -2.0 * (self.steps_in_house - 20)  # -2 per extra step
                        house_penalty = max(house_penalty, -200.0)  # Cap at -200
                        reward += house_penalty
                        if self.step_count % 20 == 0:
                            info_parts.append(f"🚨 STUCK IN HOUSE x{self.steps_in_house} {house_penalty:.0f}")
                    
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
                        
                        # STRONG rewards for following waypoint navigation
                        if action_name == target_action:
                            reward += 200.0  # Massive reward for correct direction!
                            info_parts.append(f"🧱 WAYPOINT: {target_action} +200")
                        elif action_name == 'A':
                            reward += 5.0  # Neutral for A (might interact)
                        else:
                            # Penalties based on how wrong the action is
                            opposite_actions = {
                                'UP': 'DOWN', 'DOWN': 'UP',
                                'LEFT': 'RIGHT', 'RIGHT': 'LEFT'
                            }
                            
                            if action_name == opposite_actions.get(target_action):
                                # Going exactly opposite direction - very wrong!
                                reward -= 150.0
                                info_parts.append(f"❌ OPPOSITE! Need {target_action} not {action_name}!")
                            elif action_name in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                                # Perpendicular movement - less wrong but still not ideal
                                reward -= 50.0
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
                    # Terrain-based path to find rival for battle
                    # ═══════════════════════════════════════════════════════════════════
                    
                    if route103_nav and route103_target:
                        target_action = route103_target['action']
                        
                        if action_name == target_action:
                            reward += 100.0  # Good reward for following terrain guidance
                            info_parts.append(f"🗺️ R103 NAV: {target_action} +100")
                        elif action_name == 'A':
                            reward += 20.0  # Might be talking to rival!
                            info_parts.append("🗺️ A pressed - initiating battle?")
                        else:
                            opposite_actions = {'UP': 'DOWN', 'DOWN': 'UP', 'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}
                            if action_name == opposite_actions.get(target_action):
                                reward -= 80.0
                                info_parts.append(f"❌ Wrong direction on R103! Need {target_action}")
                            elif action_name in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                                reward -= 20.0
                                info_parts.append(f"⚠️ Off-path on R103 - need {target_action}")
                    
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
                        # Heavy penalty for wasting time on 2F
                        reward -= 100.0
                        info_parts.append("🚫 2F PENALTY: -100 (trading floor, useless!)")
                        
                        # Reward for going LEFT toward escalator (exit)
                        if action_name == 'LEFT':
                            reward += 200.0
                            info_parts.append("🪜 Going LEFT to exit 2F! +200")
                    
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
                        
                        if in_oldale_pokecenter:
                            if not hasattr(self, 'visited_oldale_pokecenter'):
                                reward += 500.0  # MASSIVE bonus - this saves so much time on resets!
                                info_parts.append("🏥 OLDALE POKECENTER! RESPAWN SET! +500")
                                self.visited_oldale_pokecenter = True
                                logger.info("✅ RESPAWN POINT SET AT OLDALE POKECENTER!")
                            else:
                                # RE-ENTRY PENALTY - already visited, stop coming back!
                                reward -= 200.0
                                info_parts.append("🚫 PC RE-ENTRY! Already visited! -200")
                    
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
                        
                        # Build party_state for directive manager
                        # Format: [{'species': 'TORCHIC', 'level': 11, 'hp': 1, 'max_hp': 1}, ...]
                        party_state = []
                        for i in range(party_size):
                            pokemon = {
                                'species': party_species[i] if i < len(party_species) else 'UNKNOWN',
                                'level': party_levels[i] if i < len(party_levels) else 1,
                                'hp': 1,  # Assume alive unless we get HP data
                                'max_hp': 1
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
                                # Format: {"milestones": {"X": {"completed": true}, ...}}
                                completed_names = [k for k, v in m_data.items() if isinstance(v, dict) and v.get('completed')]
                        
                        location_name = memory.get('location', 'unknown')
                        location_name_upper = (location_name or 'unknown').upper()
                        has_pokedex = memory.get('has_pokedex', False)
                        
                        # Call update_directive with list of completed milestone names AND pokedex flag
                        directive_info = self.directive_manager.update_directive(
                            party_state,
                            completed_names,
                            location_name,
                            self.step_count,
                            has_pokedex=has_pokedex
                        )
                        
                        if directive_info:
                            self.last_directive = directive_info
                            # The 'reason' field contains the specific navigational instruction
                            target = directive_info.get('reason', '')
                            priority = 10  # Implicit high priority for these directives
                            
                            # Log directive periodically
                            if self.step_count % 50 == 0:
                                directive_type = directive_info.get('directive', 'UNKNOWN')
                                logger.info(f"🧭 DIRECTIVE: {directive_type} - {target}")
                            
                            # GROUND TRUTH DOOR COORDINATES (Littleroot/Oldale/Routes)
                            # Complete journey pathfinding: 2F→1F→town→routes→rival→back
                            DOOR_COORDS = {
                                # HOUSE INTERIORS (for 2F→1F→exit)
                                'player_house_2f': {
                                    'stairs': (8, 1),       # Go down at y=1 to reach stairs
                                },
                                'player_house_1f': {
                                    'door': (3, 8),         # Exit door (LEFT-BOTTOM side, visual position)
                                    'mom': (5, 5),          # Mom for healing
                                },
                                'rival_house_2f': {
                                    'stairs': (8, 1),       # Go down to 1F
                                },
                                'rival_house_1f': {
                                    'door': (3, 8),         # Exit door (LEFT-BOTTOM side, visual position)
                                },
                                # LITTLEROOT TOWN (use server ground truth when available!)
                                'littleroot': {
                                    'player_house': (8, 8),
                                    'rival_house': (17, 9),  # Enter from below (bottom)
                                    'lab': (12, 14),
                                    'route_101_north': (11, 0)  # GROUND TRUTH from server! Portal to Route 101
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
                                        if best_dir and action_name == best_dir:
                                            directive_reward = 80.0  # Strong guidance on Route 101
                                            if self.step_count % 100 == 0:
                                                logger.info(f"🌐 PORTAL: ROUTE_101 → OLDALE at ({target_x},{target_y}) = {best_dir}")
                                        elif action_name == 'UP':
                                            directive_reward = 60.0  # General north is good
                                    elif action_name == 'UP':
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
                                    # Generic GO_NORTH fallback
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
                                            directive_reward = 3.0  # Some exploration OK
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
                                    # ===== 1F: Find door using hardcoded coordinates =====
                                    # NOTE: Portal data from server shows wrong coordinates [14,9] for house exits
                                    # Visual inspection shows door is at LEFT-BOTTOM (3,8), not right side
                                    # Using ONLY hardcoded coordinates until server portal data is fixed
                                    door_exit = (3, 8)  # House door (LEFT-BOTTOM side, visual position)
                                    source = "hardcoded"
                                    
                                    # DISABLED: Portal connections (server provides incorrect [14,9])
                                    # if connections:
                                    #     for conn in connections:
                                    #         to_loc = conn.get('to', '').upper()
                                    #         if 'TOWN' in to_loc or 'LITTLEROOT' in to_loc:
                                    #             from_pos = conn.get('from_pos', [])
                                    #             if len(from_pos) >= 2:
                                    #                 door_exit = (int(from_pos[0]), int(from_pos[1]))
                                    #                 source = "portal"
                                    #                 break
                                    
                                    # DISABLED: Terrain data (no warp tiles in server response)
                                    # if not door_exit and terrain:
                                    #     for tile in terrain:
                                    #         behavior = (tile.get('behavior', '') or '').upper()
                                    #         if 'WARP' in behavior or 'DOOR' in behavior:
                                    #             tx, ty = tile.get('x'), tile.get('y')
                                    #             if tx is not None and ty is not None:
                                    #                 door_exit = (int(tx), int(ty))
                                    #                 source = "terrain"
                                    #                 break
                                    
                                    # SLAM pathfinding to door
                                    target_x, target_y = door_exit
                                    best_dir = get_best_direction(current_x, current_y, target_x, target_y)
                                    
                                    if best_dir and action_name == best_dir:
                                        directive_reward = 100.0
                                        if self.step_count % 50 == 0:
                                            logger.info(f"🏠 1F EXIT: ({current_x},{current_y}) → door({target_x},{target_y}) = {best_dir} (+100) [{source}]")
                                    elif action_name in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                                        directive_reward = -30.0
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
                                # Same as EXIT_HOUSE but with logging
                                current_x = memory.get('x', 0)
                                current_y = memory.get('y', 0)
                                location_upper = (memory.get('location', '') or '').upper()
                                
                                if '2F' in location_upper or 'BEDROOM' in location_upper:
                                    target_x, target_y = 8, 1  # Stairs
                                else:
                                    target_x, target_y = 3, 8  # Door (LEFT-BOTTOM side, visual position)
                                
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
                                        rival_x, rival_y = DOOR_COORDS['littleroot']['rival_house']  # Fallback: (17, 9)
                                        if self.step_count % 100 == 0:
                                            logger.info(f"🏠 Using FALLBACK rival_house coords: ({rival_x},{rival_y})")
                                    
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
                                            # First priority: get to x=17 (door's column)
                                            if current_x < rival_x and action_name == 'RIGHT':
                                                directive_reward = 200.0
                                                logger.info(f"🏠 RIVAL HOUSE: x{current_x}→{rival_x}, going RIGHT (+200)")
                                            elif current_x > rival_x and action_name == 'LEFT':
                                                directive_reward = 200.0
                                                logger.info(f"🏠 RIVAL HOUSE: x{current_x}→{rival_x}, going LEFT (+200)")
                                            elif action_name in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                                                directive_reward = -50.0  # Wrong direction
                                        elif current_y > rival_y:
                                            # At correct x=17, but south of door (y > 9)
                                            if action_name == 'UP':
                                                directive_reward = 200.0
                                                logger.info(f"🏠 RIVAL HOUSE: y{current_y}→{rival_y}, going UP (+200)")
                                            elif action_name in ['DOWN', 'LEFT', 'RIGHT']:
                                                directive_reward = -50.0
                                        elif current_y < rival_y:
                                            # At correct x=17, but north of door (shouldn't happen, but handle it)
                                            if action_name == 'DOWN':
                                                directive_reward = 200.0
                                                logger.info(f"🏠 RIVAL HOUSE: y{current_y}→{rival_y}, going DOWN (+200)")
                                            elif action_name in ['UP', 'LEFT', 'RIGHT']:
                                                directive_reward = -50.0
                                    else:
                                        # At door but wrong action
                                        if action_name in ['DOWN', 'LEFT', 'RIGHT']:
                                            directive_reward = -50.0
                            
                            # INSIDE RIVAL HOUSE: Go upstairs to bedroom
                            elif 'GO_UPSTAIRS' in target and 'RIVAL' in target:
                                # In rival's house 1F - go UP to stairs, then UP again
                                if action_name == 'UP':
                                    directive_reward = 100.0  # Strong reward to go upstairs
                                    logger.info(f"⬆️ GOING UPSTAIRS TO RIVAL BEDROOM (+100)")
                                elif action_name in ['DOWN', 'LEFT', 'RIGHT']:
                                    directive_reward = -30.0  # Stronger penalty for wrong direction
                            
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
                                if 'POKECENTER' in location_upper:
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
                                    
                                    # Mark as visited when near PokeCenter door
                                    if at_pokecenter:
                                        if not hasattr(self, 'oldale_pokecenter_visited'):
                                            self.oldale_pokecenter_visited = False
                                        if not self.oldale_pokecenter_visited:
                                            self.oldale_pokecenter_visited = True
                                            logger.info(f"✅ OLDALE POKECENTER VISITED! Respawn point updated.")
                                            # Sync milestone to server!
                                            try:
                                                requests.post("http://localhost:8000/milestone", 
                                                    json={"name": "OLDALE_POKECENTER", "completed": True}, 
                                                    timeout=1)
                                            except Exception:
                                                pass
                                        
                                        # Reward pressing A to enter
                                        if action_name == 'A':
                                            directive_reward = 300.0  # MASSIVE reward - critical for respawn!
                                            logger.info(f"🏥 ENTERING POKECENTER at ({current_x},{current_y})")
                                    else:
                                        # Still pathfinding to PokeCenter
                                        best_dir = get_best_direction(current_x, current_y, pc_x, pc_y)
                                        if best_dir and action_name == best_dir:
                                            directive_reward = 200.0  # Huge reward - need respawn point!
                                            if self.step_count % 50 == 0:
                                                logger.info(f"🏥 HEAL PATHFINDING: → Oldale PokeCenter ({pc_x},{pc_y})")
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
                    except:
                        pass
                    
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
                            
                            # Apply RUN reward/penalty - STRONG incentive to run!
                            if should_run:
                                # Reward running (DOWN+RIGHT to reach RUN, then A to select)
                                if action_name in ['RIGHT', 'DOWN', 'B']:
                                    reward += 50.0  # Strong run incentive
                                    if self.step_count % 20 == 0:
                                        logger.info(f"🏃 RUN encouraged: {run_reason}")
                                elif action_name == 'A':
                                    reward -= 20.0  # Strong penalty for attacking when should run
                            
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
            # This is the central temporal reasoning that integrates:
            # - What should I do? (VLM goals)
            # - Where am I in the story? (Milestones)
            # - What quest am I on? (Structured objectives from dialog)
            # - What do I see? (CNN vision)
            # - Where am I? (Grid/Map position)
            # - What's walkable? (Map-based terrain)
            # - Trees vs grass? (Pixel-based RGB analysis)
            snn_input_current = torch.cat([
                vlm_input,        # 64 neurons: High-level goals
                milestone_input,  # 64 neurons: Story flags
                quest_input,      # 64 neurons: Quest objectives (parsed dialog)
                cnn_input,        # 64 neurons: Local vision
                map_input,        # 64 neurons: Global position
                terrain_input,    # 64 neurons: Map-based terrain (ledges/grass)
                rgb_input         # 64 neurons: Pixel-based tree/grass detection
            ], dim=1)  # Total: 448 neurons (7 × 64)
            
            # LIF dynamics: V(t+1) = decay * V(t) + I(t)
            # SNN integrates information over time (event memory, temporal patterns)
            self.snn_membrane = self.snn_decay * self.snn_membrane + snn_input_current
            
            # Spike generation: spike if V > threshold
            spikes = (self.snn_membrane > self.snn_threshold).float()
            
            # Reset membrane where spikes occurred
            self.snn_membrane = self.snn_membrane * (1.0 - spikes)
            
            # Update spike history (rolling window of last 10 timesteps)
            # This is the temporal memory: "what happened in last N seconds?"
            self.snn_spike_history = torch.roll(self.snn_spike_history, shifts=-1, dims=2)
            self.snn_spike_history[:, :, -1] = spikes
            
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
