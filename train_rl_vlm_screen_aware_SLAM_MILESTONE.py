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
    has_rival_bedroom = 'RIVAL_BEDROOM' in completed
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
        if 'FIRST_RIVAL_BATTLE' in completed:
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
    
    # If player has Pokemon but not at lab, guide them back (unless has Pokedex)
    if has_pokemon and not at_lab and not has_pokedex and 'RECEIVED_POKEDEX' not in completed:
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
        
        prompt = f"""Pokemon Emerald Speedrun - Story Progress

**📖 STORY CONTEXT:**
{story_ctx['story']}{npc_context}{quest_context}

**🎯 CURRENT MILESTONE GOAL: {story_ctx['goal']}**
Why: {story_ctx['why']}

**📍 YOUR LOCATION:**
{location_name} at ({x},{y}) | State: {game_state.upper()}
Party: {memory.get('party_size', 0)} Pokemon | Badges: {memory.get('badge_count', 0)}

**✅ PROGRESS:** {len(completed)} milestones completed
{', '.join(completed[-3:]) if completed else 'Just starting!'}

{spatial_info if spatial_info else ''}

**➡️ TO PROGRESS STORY:**
{action_guide}

**RESPOND (tie your action to story/milestone):**
LOCATION: {location_name} - [visual description]
OBJECTIVE: [what milestone step this accomplishes]
ACTION: [UP/DOWN/LEFT/RIGHT/A/B]
REASON: [how this advances the story toward {story_ctx['goal']}]

**RULES:**
- Your OBJECTIVE must connect to milestone goal: {story_ctx['goal']}
- Your REASON must explain story progression
- Use directional buttons for movement, A/B only for dialogs
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
            logger.info(f"📂 LLM Logger file: {llm_logger.log_file}")
            
            # Log using proper LLM logger that server reads
            log_llm_interaction(
                interaction_type="ServerAwareOracle",
                prompt=prompt_summary,
                response=response if response else "VLM processing...",
                metadata={
                    "step": step_count,
                    "location": location_info,
                    "server_location": server_location,
                    "x": memory.get('x', 0),
                    "y": memory.get('y', 0),
                    "map_id": map_id,
                    "party_size": memory.get('party_size', 0),
                    "badge_count": memory.get('badge_count', 0)
                },
                duration=call_duration
            )
            
            logger.info(f"✅ Logged VLM thinking to server (step {step_count}, duration: {call_duration:.2f}s)")
                
        except Exception as e:
            import traceback
            logger.error(f"❌ Failed to log to server: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        # VALIDATE VLM RESPONSE - check if location matches server location
        if response:
            response_upper = response.upper()
            location_name_upper = location_name.upper()
            
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
        rival_battled = 'FIRST_RIVAL_BATTLE' in completed
        
        if map_id == 0x0202 and not starter_chosen:
            # In rival house BEFORE starter - bedroom visit may be needed
            return f"LOCATION: {location} at ({x},{y})\nOBJECTIVE: Explore rival house (bedroom optional)\nACTION: UP then RIGHT (stairs)\nREASON: May explore bedroom before getting starter"
        
        elif map_id == 0x0202 and starter_chosen and not rival_battled:
            # In rival house AFTER starter - WRONG PLACE! May is on Route 103!
            return f"⚠️ WRONG LOCATION: {location} at ({x},{y})\n❌ May (rival) is NOT here - she's on ROUTE 103!\nOBJECTIVE: EXIT this house and go to Route 103\nACTION: DOWN (exit house)\nREASON: After getting starter, rival is waiting on Route 103 (north of Oldale Town). Path: Exit house → Route 101 (north) → Oldale Town → Route 103"
        
        elif map_id == 0x030A:
            # In rival bedroom
            return f"LOCATION: {location} at ({x},{y})\nOBJECTIVE: Explore bedroom then exit\nACTION: Move around then DOWN\nREASON: Exit to continue quest"
        
        elif map_id == 0x020C:
            # In Birch lab
            return f"LOCATION: {location} at ({x},{y})\nOBJECTIVE: Exit lab to town\nACTION: DOWN\nREASON: Exit is at bottom of lab"
        
        elif map_id == 0x0101 and starter_chosen and not rival_battled:
            # In Littleroot Town with starter - GO NORTH to Route 101!
            return f"📍 LOCATION: Littleroot Town at ({x},{y})\n🎯 QUEST: Find May on Route 103 to battle\n🗺️ PATH: Exit NORTH to Route 101 → Oldale Town → Route 103\nACTION: UP (repeatedly move north)\nREASON: May (rival) is waiting on Route 103. Must navigate: Route 101 → Oldale → Route 103"
        
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
            # In Oldale Town - go to Route 103!
            return f"📍 LOCATION: Oldale Town at ({x},{y})\n🎯 QUEST: May is waiting on Route 103 (north exit)\n🗺️ DIRECTION: Exit NORTH to Route 103\nACTION: UP (move to north exit)\nREASON: May (rival) is on Route 103. Battle her, then return to Birch's Lab for Pokedex"
        
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
        match = re.search(r'ACTION:\s*([A-Z]+)', response.upper())
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
    parser.add_argument('--timesteps', type=int, default=50000,
                       help='Training timesteps')
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
            self.position_history_size = 20
            self.last_vlm_suggestion = None
            self.vlm_suggestion_unchanged_steps = 0
        
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
                        'ROUTE 101': 0x10,
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
                        'milestones': state.get('milestones', {})  # Include milestones for dialogue processor
                    }
                    
                    # STUCK DETECTION - Track position and action patterns
                    current_pos = (x, y, map_id)
                    self.position_history.append(current_pos)
                    self.action_history.append(action_name)
                    
                    # Keep only recent history
                    if len(self.position_history) > self.position_history_size:
                        self.position_history.pop(0)
                        self.action_history.pop(0)
                    
                    # ===== UPDATE SLAM MAPPER (Spatial Memory & Path Planning) =====
                    if hasattr(self, 'slam_mapper'):
                        # Update occupancy grid with current tile info
                        # Get tile behavior from terrain extractor if available
                        tile_info = {}
                        if hasattr(self, 'terrain_feature_extractor'):
                            # terrain_feature_extractor tracks tile info
                            tile_info = {
                                'is_blocked': False,  # TODO: Get from terrain extractor
                                'is_grass': False,
                                'is_door': False
                            }
                        self.slam_mapper.update_position(map_id, x, y, tile_info)
                        
                        # Plan path to next milestone target if available
                        if hasattr(self, 'milestone_feedback') and self.milestone_feedback.next_milestone_target:
                            target_map, target_x, target_y, _ = self.milestone_feedback.next_milestone_target
                            path = self.slam_mapper.plan_path(map_id, x, y, target_map, target_x, target_y)
                            if path:
                                self.slam_mapper.current_path = path
                    
                    # ===== UPDATE MILESTONE FEEDBACK (Server Status) =====
                    if hasattr(self, 'milestone_feedback'):
                        # Query server every 100 steps to avoid overhead
                        if self.step_count % 100 == 0:
                            self.milestone_feedback.update_from_server()
                            # Get next milestone target from dialogue processor's milestone sequence
                            if hasattr(self, 'dialogue_processor'):
                                self.milestone_feedback.get_next_milestone_target(
                                    self.dialogue_processor.milestones
                                )
                    
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
                    
                    # Coordinate exploration (PokemonRedExperiments-style + SLAM bonus)
                    coord_key = (x, y, map_id)
                    if coord_key not in self.seen_coords:
                        self.seen_coords[coord_key] = 1
                        # First visit bonus - INCREASED to encourage exploration
                        explore_reward = 3.0  # Was 2.0
                        
                        # SLAM exploration bonus (novelty reward)
                        if hasattr(self, 'slam_mapper'):
                            slam_bonus = self.slam_mapper.get_exploration_bonus(map_id, x, y)
                            explore_reward += slam_bonus
                            if slam_bonus > 0.5:
                                info_parts.append(f"🗺️ SLAM +{slam_bonus:.1f}")
                        
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
                        
                        # HUGE BONUS: Reaching Route 101 (means story progressed!)
                        if map_id == 0x10:
                            reward += 100.0
                            info_parts.append("🌳 ROUTE 101! Story progress! +100")
                    
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
                    has_important_event = any(marker in ' '.join(info_parts) for marker in ['🚪', '🗺️', '🏠', '🛏️', '🏆'])
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
                    
                    # VLM SERVER-AWARE GUIDANCE (NEW!)
                    # Call more frequently if stuck, or at regular interval
                    should_call_vlm = (
                        self.step_count % self.vlm_frequency == 0 or
                        (is_stuck and self.step_count % 10 == 0)  # Call every 10 steps if stuck
                    )
                    
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
                            
                            # Bigger bonus if stuck and following advice
                            if suggested_action == action_name:
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
            action = action_string.upper().strip()
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
                if npc in location_name.upper():
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
                    'target_npc': npc_name.upper(),
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
                npc_name = next((npc for npc in self.important_npcs if npc in location_name.upper()), 'UNKNOWN')
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
    rgb_terrain_analyzer = RGBTerrainAnalyzer()  # Pixel-based tree/grass detection
    # Pass VLM for optional LLM-based dialogue parsing (user's architecture suggestion)
    dialogue_processor = DialogueProcessor(llm=vlm)  # Structured quest memory (not raw dialog dump)
    # NEW: SLAM-like spatial mapper for path planning
    slam_mapper = SLAMSpatialMapper(map_size=256)
    # NEW: Milestone feedback system (server integration)
    milestone_feedback = MilestoneFeedbackSystem(server_url="http://localhost:8000")
    
    # Attach trackers to env so they update
    env.milestone_tracker = milestone_tracker
    env.vlm_guidance_tracker = vlm_guidance_tracker
    env.spatial_map_tracker = spatial_map_tracker
    env.terrain_feature_extractor = terrain_feature_extractor
    env.rgb_terrain_analyzer = rgb_terrain_analyzer
    env.dialogue_processor = dialogue_processor
    env.slam_mapper = slam_mapper
    env.milestone_feedback = milestone_feedback
    
    # Make globally accessible for CNN/SNN extractors
    import __main__
    __main__.milestone_tracker = milestone_tracker
    __main__.vlm_guidance_tracker = vlm_guidance_tracker
    __main__.spatial_map_tracker = spatial_map_tracker
    __main__.terrain_feature_extractor = terrain_feature_extractor
    __main__.rgb_terrain_analyzer = rgb_terrain_analyzer
    __main__.dialogue_processor = dialogue_processor
    __main__.slam_mapper = slam_mapper
    __main__.milestone_feedback = milestone_feedback
    
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
            # 8. SLAM Path Planning (spatial memory: path waypoints, exploration)
            # 9. Milestone Feedback (server status: what's done, what's next)
            #
            # FEEDBACK LOOP: SNN also outputs navigation hints back to CNN
            # This creates hierarchical perception: Map→SNN→CNN guidance
            
            snn_hidden_dim = 576  # LIF neurons for temporal patterns (9 inputs × 64 each)
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
            
            # 8. SLAM Spatial Planning (path waypoints, exploration, landmarks)
            self.slam_spatial_encoder = nn.Sequential(
                nn.Linear(64, 64),  # Already 64-dim from SLAM mapper
                nn.Tanh()
            )
            
            # 9. Milestone Feedback (server status: progress, next target)
            self.milestone_feedback_encoder = nn.Sequential(
                nn.Linear(32, 48),
                nn.ReLU(),
                nn.Linear(48, snn_input_per_source),  # Exactly 64 neurons
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
            logger.info(f"   🗺️  SLAM Mapper: Path planning → SNN (64-dim → 64 neurons)")
            logger.info(f"      ↳ A* waypoints, exploration bonus, landmark proximity")
            logger.info(f"   📊 Milestone Feedback: Server status → SNN (32-dim → 64 neurons)")
            logger.info(f"      ↳ Progress tracking, next target, what's done/needed")
            logger.info(f"   ")
            logger.info(f"   🧠 SNN CORE: {snn_hidden_dim} LIF neurons (9 inputs × 64)")
            logger.info(f"      • Integrates: VLM + Story + Quest + Vision + Map + Terrain + RGB + SLAM + Milestones")
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
            
            # ===== INPUT 8: SLAM SPATIAL PLANNING → SNN =====
            # Path planning, exploration bonus, landmark proximity
            slam_input = torch.zeros(batch_size, 64, device=observations.device)
            try:
                import __main__
                if hasattr(__main__, 'slam_mapper'):
                    # Get current position from server
                    try:
                        state_response = requests.get('http://localhost:8000/state', timeout=0.5)
                        if state_response.status_code == 200:
                            state = state_response.json()
                            player = state.get('player', {})
                            pos = player.get('position', {})
                            map_id = player.get('map_id', 0)
                            x, y = pos.get('x', 0), pos.get('y', 0)
                            
                            slam_hints = __main__.slam_mapper.get_navigation_hints(map_id, x, y)
                            slam_tensor = torch.tensor(
                                [slam_hints] * batch_size,
                                dtype=torch.float32,
                                device=observations.device
                            )
                            slam_input = self.slam_spatial_encoder(slam_tensor)
                    except:
                        pass
            except:
                pass
            
            # ===== INPUT 9: MILESTONE FEEDBACK (SERVER) → SNN =====
            # What's done, what's needed, where to go next
            milestone_feedback_input = torch.zeros(batch_size, 64, device=observations.device)
            try:
                import __main__
                if hasattr(__main__, 'milestone_feedback'):
                    progress_encoding = __main__.milestone_feedback.get_progress_encoding()
                    feedback_tensor = torch.tensor(
                        [progress_encoding] * batch_size,
                        dtype=torch.float32,
                        device=observations.device
                    )
                    milestone_feedback_input = self.milestone_feedback_encoder(feedback_tensor)
            except:
                pass
            
            # ===== SNN CORE: INTEGRATE ALL 9 INPUTS =====
            # Concatenate all input streams into SNN input current
            # This is the central temporal reasoning that integrates:
            # - What should I do? (VLM goals)
            # - Where am I in the story? (Milestones)
            # - What quest am I on? (Structured objectives from dialog)
            # - What do I see? (CNN vision)
            # - Where am I? (Grid/Map position)
            # - What's walkable? (Map-based terrain)
            # - Trees vs grass? (Pixel-based RGB analysis)
            # - How do I get there? (SLAM path planning)
            # - What's my progress? (Server milestone feedback)
            snn_input_current = torch.cat([
                vlm_input,                    # 64 neurons: High-level goals
                milestone_input,              # 64 neurons: Story flags
                quest_input,                  # 64 neurons: Quest objectives (parsed dialog)
                cnn_input,                    # 64 neurons: Local vision
                map_input,                    # 64 neurons: Global position
                terrain_input,                # 64 neurons: Map-based terrain (ledges/grass)
                rgb_input,                    # 64 neurons: Pixel-based tree/grass detection
                slam_input,                   # 64 neurons: Path planning waypoints
                milestone_feedback_input      # 64 neurons: Server progress feedback
            ], dim=1)  # Total: 576 neurons (9 × 64)
            
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
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./checkpoints_server_aware/",
        name_prefix="server_aware_model"
    )
    
    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    logger.info("✅ Training complete!")
    model.save("server_aware_final_model")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("\n🛑 Training interrupted by user")
        sys.exit(0)
