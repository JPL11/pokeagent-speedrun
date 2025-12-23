"""
Strategic Directive Manager - High-level goal coordination with feedback loops
Manages VLM strategic directives and updates based on progress
"""

import logging
from collections import deque

logger = logging.getLogger(__name__)


# Milestone level requirements (when agent needs to be strong enough)
MILESTONE_LEVEL_REQUIREMENTS = {
    'STARTER_CHOSEN': 5,
    'FIRST_RIVAL_BATTLE': 5,
    'RECEIVED_POKEDEX': 7,
    'RUSTBORO_GYM': 13,
    'ROXANNE_DEFEATED': 13,
    'BRAWLY_GYM': 18,
    'WATTSON_GYM': 23,
    'FLANNERY_GYM': 28,
    'NORMAN_GYM': 33,
    'WINONA_GYM': 35,
    'TATE_LIZA_GYM': 43,
    'WALLACE_GYM': 45,
    'ELITE_FOUR': 50,
    'DEFAULT': 10
}


class StrategicDirectiveManager:
    """
    Manages high-level strategic directives with feedback loops.
    
    Directives:
    - PROCEED: Continue to next milestone (strong enough)
    - GRIND: Train Pokemon to reach level target
    - OPTIONAL_GRIND: Slightly under-leveled, can grind or proceed
    - HEAL: HP too low, must heal before continuing
    - PREPARE_GYM: Near gym, final preparation needed
    - POST_BLACKOUT: Recovery mode after blackout
    - GYM_BATTLE: Currently in gym (special battle mode)
    """
    
    def __init__(self):
        self.current_directive = 'PROCEED'
        self.directive_history = deque(maxlen=50)
        
        # Feedback tracking for grinding sessions
        self.grinding_session_start_level = 0
        self.grinding_session_start_step = 0
        self.grinding_session_battles = 0
        self.levels_gained_this_session = 0
        
        # State tracking
        self.step_count = 0
        self.just_blackouted = False
        self.last_update_step = 0
        
        # Performance metrics
        self.grinding_effectiveness = {}  # location -> levels_per_1000_steps
    
    def _get_completed_milestones_list(self, milestones):
        """Extract list of completed milestones safely"""
        if isinstance(milestones, list):
            return milestones
        if isinstance(milestones, dict):
            return [k for k, v in milestones.items() if v.get('completed', False)]
        return []

    def update_directive(self, party_state, milestones_completed, location, step_count, gender="Boy", has_pokedex=False):
        """
        Update strategic directive based on current game state.
        Called every VLM invocation (~25 steps).
        
        Returns:
            dict: {
                'directive': str,
                'reason': str,
                'target_level': int,
                'should_transition': bool
            }
        """
        self.step_count = step_count
        
        # CLIENT-SIDE MILESTONE INFERENCE
        # If the server fails to mark CLOCK_SET but we are outside, assume it's done.
        # (Mom blocks exit if clock isn't set)
        implied_milestones = set(self._get_completed_milestones_list(milestones_completed))
        
        # Infer Pokedex from game state flag
        if has_pokedex:
            implied_milestones.add('RECEIVED_POKEDEX')
        
        # CRITICAL: If we have Pokemon in party, we MUST have chosen starter!
        if party_state and len(party_state) > 0:
            implied_milestones.add('STARTER_CHOSEN')
            implied_milestones.add('CLOCK_SET')  # Clock must be set to leave bedroom
            implied_milestones.add('RIVAL_BEDROOM')  # Must have visited to trigger story
        location_upper = (location or '').upper()
        
        # Infer OLDALE_POKECENTER visited if we're past Oldale (on Route 103 or beyond)
        # This prevents re-entering PC after we've already moved on
        if 'ROUTE' in location_upper and '103' in location_upper:
            implied_milestones.add('OLDALE_POKECENTER')
        
        if 'LITTLEROOT' in location_upper and 'HOUSE' not in location_upper:
            if 'CLOCK_SET' not in implied_milestones:
                # Force inference
                implied_milestones.add('CLOCK_SET')
                # If we are outside, we also must have entered our house and left our bedroom
                implied_milestones.add('PLAYER_HOUSE_ENTERED')
                implied_milestones.add('PLAYER_BEDROOM')
        
        if 'LITTLEROOT' in location_upper and 'HOUSE' not in location_upper:
            if 'CLOCK_SET' not in implied_milestones:
                # Force inference
                implied_milestones.add('CLOCK_SET')
                # If we are outside, we also must have entered our house and left our bedroom
                implied_milestones.add('PLAYER_HOUSE_ENTERED')
                implied_milestones.add('PLAYER_BEDROOM')
        
        # PRIORITY 0: EARLY GAME STORY - Rival bedroom visit required!
        # In Pokemon Emerald, you MUST visit rival's bedroom to trigger NPC to stop blocking Route 101
        # Sequence: Clock set → Visit rival house + bedroom → NPC moves → Route 101 accessible
        
        # Phase 1: No Pokemon yet AND no starter chosen - need to visit rival and then get starter
        # CRITICAL: Check milestones too because party_state may be empty after faint/blackout!
        has_starter = (party_state and len(party_state) > 0) or 'STARTER_CHOSEN' in implied_milestones
        if not has_starter:
             # CRITICAL: Must visit rival bedroom FIRST before anything else!
             # If haven't visited rival bedroom yet, go there first (triggers NPC)
             if 'RIVAL_BEDROOM' not in implied_milestones:
                 # If in lab, exit to go find rival house
                 if 'LAB' in location_upper or 'PROFESSOR' in location_upper:
                     return self._get_directive_info('PROCEED', 'EXIT_LAB_GO_TO_RIVAL_HOUSE', 10)
                 
                 # In Littleroot Town (outside) - navigate to rival house
                 if 'LITTLEROOT' in location_upper and 'HOUSE' not in location_upper and 'LAB' not in location_upper:
                     if gender == "Girl":
                         return self._get_directive_info('PROCEED', 'GO_LEFT_TO_RIVAL_HOUSE', 10)
                     else:
                         return self._get_directive_info('PROCEED', 'GO_RIGHT_TO_RIVAL_HOUSE', 10)
                 
                 # Inside rival's house (various location string formats)
                 # MALE AVATAR: May's house = rival's house (Brendan's = player's)
                 # FEMALE AVATAR: Brendan's house = rival's house (May's = player's)
                 is_in_rival_house = (
                     (gender != "Girl" and 'MAY' in location_upper) or  # Male: May's house is rival's
                     (gender == "Girl" and 'BRENDAN' in location_upper)  # Female: Brendan's house is rival's
                 )
                 if is_in_rival_house and 'BEDROOM' not in location_upper:
                      return self._get_directive_info('PROCEED', 'GO_UPSTAIRS_RIVAL_BEDROOM', 10)
                 if is_in_rival_house and 'BEDROOM' in location_upper:
                     # In rival's bedroom - talk to them, then exit
                     return self._get_directive_info('PROCEED', 'EXIT_RIVAL_BEDROOM', 10)
             
             # After rival bedroom visit, go to Route 101 to find Birch
             else:
                 if 'LITTLEROOT' in location_upper and 'HOUSE' not in location_upper:
                     return self._get_directive_info('PROCEED', 'GO_NORTH_TO_ROUTE_101_FIND_BIRCH', 10)
                 if 'ROUTE' in location_upper and '101' in location_upper:
                     return self._get_directive_info('PROCEED', 'FIND_BIRCH_WALK_GRASS', 10)
        
        # FALLBACK: If in house/bedroom and no party data, handle appropriately
        # POST-BLACKOUT FIX: If implied_milestones show we have Pokemon, skip clock and exit!
        if not party_state or len(party_state) == 0:
            # Check if we've already made progress (has Pokemon = clock already done)
            has_made_progress = (
                'STARTER_CHOSEN' in implied_milestones or 
                'SAVED_BIRCH' in implied_milestones or
                'FIRST_RIVAL_BATTLE' in implied_milestones or
                'RECEIVED_POKEDEX' in implied_milestones
            )
            
            if has_made_progress:
                # POST-BLACKOUT: Already have Pokemon, just exit the house!
                if 'HOUSE' in location_upper or 'BEDROOM' in location_upper:
                    if 'BEDROOM' in location_upper or '2F' in location_upper:
                        # In bedroom - go downstairs first
                        return self._get_directive_info('PROCEED', 'EXIT_HOUSE_GO_DOWNSTAIRS', 10)
                    else:
                        # On 1F - exit house
                        return self._get_directive_info('PROCEED', 'EXIT_HOUSE_GO_DOWN', 10)
            else:
                # No progress yet - need to set clock first
                if 'CLOCK_SET' not in implied_milestones:
                    # Player's house: PLAYER in name OR Brendan's for male OR May's for female
                    is_in_player_house = (
                        'PLAYER' in location_upper or
                        (gender != "Girl" and 'BRENDAN' in location_upper) or  # Male: Brendan's = player's
                        (gender == "Girl" and 'MAY' in location_upper)  # Female: May's = player's
                    )
                    if is_in_player_house and ('HOUSE' in location_upper or 'BEDROOM' in location_upper):
                        if 'BEDROOM' in location_upper or '2F' in location_upper:
                            return self._get_directive_info('PROCEED', 'SET_CLOCK_IN_BEDROOM', 10)
                        else:
                            return self._get_directive_info('PROCEED', 'GO_UPSTAIRS_TO_BEDROOM', 10)
            
            # Exit any house
            if 'HOUSE' in location_upper or 'BEDROOM' in location_upper:
                return self._get_directive_info('PROCEED', 'EXIT_HOUSE_GO_DOWN', 10)
            return self._get_directive_info('PROCEED', 'NO_PARTY_DATA')
        
        # PRIORITY 0.5: RIVAL ARC (Starter Get -> Pokedex Get)
        # User requested specific railroading: Littleroot -> 101 -> Oldale -> 103 -> Rival -> Lab -> Pokedex
        # NO SKIPPING STEPS! Must battle rival BEFORE going to lab for Pokedex!
        # CRITICAL: Use has_starter which checks milestones too!
        # ALSO CHECK: If no rival battle yet, we're still in early game regardless of RECEIVED_POKEDEX!
        rival_battled = 'FIRST_RIVAL_BATTLE' in implied_milestones
        still_in_early_game = has_starter and (not rival_battled or 'RECEIVED_POKEDEX' not in implied_milestones)
        
        if still_in_early_game:
             
             # Check if rival battle completed
             rival_battled = 'FIRST_RIVAL_BATTLE' in implied_milestones
             
             # EMERGENCY HEAL CHECK: If HP is low, go to PokeCenter/Mom first!
             if party_state and len(party_state) > 0:
                 party_hp = sum(p.get('hp', 0) for p in party_state)
                 party_max = sum(p.get('max_hp', 1) for p in party_state) or 1
                 hp_ratio = party_hp / party_max
             else:
                 hp_ratio = 0.0  # No party data = assume need healing!
             
             if hp_ratio < 0.50:  # HP below 50% - need to heal!
                 if 'ROUTE' in location_upper and '103' in location_upper:
                     return self._get_directive_info('HEAL', 'GO_SOUTH_TO_OLDALE_POKECENTER_EMERGENCY', 10)
                 if 'ROUTE' in location_upper and '101' in location_upper:
                     return self._get_directive_info('HEAL', 'GO_NORTH_TO_OLDALE_POKECENTER_EMERGENCY', 10)
                 if 'OLDALE' in location_upper and 'POKECENTER' not in location_upper:
                     return self._get_directive_info('HEAL', 'VISIT_OLDALE_POKECENTER_EMERGENCY', 10)
                 if 'POKECENTER' in location_upper:
                     return self._get_directive_info('HEAL', 'TALK_TO_NURSE_JOY', 10)
                 if 'LITTLEROOT' in location_upper and 'HOUSE' not in location_upper:
                     return self._get_directive_info('HEAL', 'GO_TO_MOM_HOUSE', 10)
                 if 'PLAYER' in location_upper or 'BRENDAN' in location_upper:
                     return self._get_directive_info('HEAL', 'TALK_TO_MOM', 10)
             
             # Phase 1: BEFORE RIVAL BATTLE - Go north to Route 103 to find rival
             if not rival_battled:
                  # CRITICAL: Check if we need to visit rival bedroom first!
                  rival_bedroom_visited = 'RIVAL_BEDROOM' in implied_milestones
                  
                  # If in rival's house but haven't visited bedroom yet - GO UPSTAIRS!
                  if not rival_bedroom_visited:
                       if 'RIVAL' in location_upper and 'HOUSE' in location_upper:
                            # Inside rival house but not bedroom yet
                            if '2F' in location_upper or 'BEDROOM' in location_upper:
                                 # Already in bedroom - exit and continue to Route 103
                                 return self._get_directive_info('PROCEED', 'EXIT_RIVAL_HOUSE_GO_DOWN', 10)
                            else:
                                 # On 1F - go upstairs to bedroom first!
                                 return self._get_directive_info('PROCEED', 'GO_UPSTAIRS_RIVAL_HOUSE', 10)
                       # Outside rival house in Littleroot - enter it first!
                       elif 'LITTLEROOT' in location_upper and 'RIVAL_HOUSE' in implied_milestones:
                            # Already touched rival house but didn't go upstairs - re-enter!
                            return self._get_directive_info('PROCEED', 'ENTER_RIVAL_HOUSE', 10)
                  
                  # If in lab during Phase 1, exit immediately - don't get Pokedex yet!
                  if 'LAB' in location_upper or 'PROFESSOR' in location_upper:
                       return self._get_directive_info('PROCEED', 'EXIT_LAB_GO_FIND_RIVAL', 10)
                  # If in player's house, exit first
                  if 'PLAYER' in location_upper or 'BRENDAN' in location_upper:
                       return self._get_directive_info('PROCEED', 'EXIT_HOUSE_GO_DOWN', 10)
                  if 'LITTLEROOT' in location_upper:
                       return self._get_directive_info('PROCEED', 'GO_NORTH_TO_ROUTE_101', 10)
                  if 'ROUTE' in location_upper and '101' in location_upper:
                       return self._get_directive_info('PROCEED', 'GO_NORTH_TO_OLDALE', 10)
                  if 'OLDALE' in location_upper:
                       # Visit PokeCenter first to update respawn point (CRITICAL!)
                       pokecenter_visited = 'OLDALE_POKECENTER' in implied_milestones or 'POKECENTER' in location_upper
                       if not pokecenter_visited:
                           return self._get_directive_info('HEAL', 'VISIT_OLDALE_POKECENTER_FIRST', 10)
                       else:
                           return self._get_directive_info('PROCEED', 'GO_NORTH_TO_ROUTE_103', 10)
                  if 'ROUTE' in location_upper and '103' in location_upper:
                       return self._get_directive_info('BATTLE', 'FIND_RIVAL_AND_FIGHT', 10)
             
             # Phase 2: AFTER RIVAL BATTLE - Return south to Lab for Pokedex
             # BUT FIRST: Verify rival is actually beaten by checking Route 103!
             else: 
                 # EMERGENCY HEAL CHECK: If HP is low, heal before getting Pokedex!
                 if party_state and len(party_state) > 0:
                     party_hp = sum(p.get('hp', 0) for p in party_state)
                     party_max = sum(p.get('max_hp', 1) for p in party_state) or 1
                     hp_ratio = party_hp / party_max
                 else:
                     hp_ratio = 1.0  # No party = assume full HP
                 
                 if hp_ratio < 0.30:  # HP below 30% - CRITICAL! Heal first!
                     if 'ROUTE' in location_upper and '103' in location_upper:
                         return self._get_directive_info('HEAL', 'GO_SOUTH_TO_OLDALE_POKECENTER_EMERGENCY', 10)
                     if 'ROUTE' in location_upper and '101' in location_upper:
                         return self._get_directive_info('HEAL', 'GO_NORTH_TO_OLDALE_POKECENTER_EMERGENCY', 10)
                     if 'OLDALE' in location_upper and 'POKECENTER' not in location_upper:
                         return self._get_directive_info('HEAL', 'VISIT_OLDALE_POKECENTER_EMERGENCY', 10)
                     if 'POKECENTER' in location_upper:
                         return self._get_directive_info('HEAL', 'TALK_TO_NURSE_JOY', 10)
                     if 'LITTLEROOT' in location_upper and 'HOUSE' not in location_upper:
                         return self._get_directive_info('HEAL', 'GO_TO_MOM_HOUSE', 10)
                     if 'PLAYER' in location_upper or 'BRENDAN' in location_upper:
                         return self._get_directive_info('HEAL', 'TALK_TO_MOM', 10)
                 
                 # If in Oldale and haven't verified rival yet, go CHECK Route 103 first
                 # (More cost-effective than going all the way to lab if rival still there)
                 if 'OLDALE' in location_upper:
                      # Check if we've already verified Route 103 (by being there post-battle)
                      verified_route_103 = 'VERIFIED_ROUTE_103' in implied_milestones
                      if not verified_route_103:
                          return self._get_directive_info('PROCEED', 'GO_NORTH_TO_ROUTE_103_VERIFY', 10)
                      else:
                          return self._get_directive_info('PROCEED', 'GO_SOUTH_TO_ROUTE_101', 10)
                 
                 if 'ROUTE' in location_upper and '103' in location_upper:
                      # On Route 103 - check if rival is here
                      # If rival gone, mark verified and head back
                      implied_milestones.add('VERIFIED_ROUTE_103')
                      return self._get_directive_info('PROCEED', 'GO_SOUTH_TO_OLDALE', 10)
                 if 'ROUTE' in location_upper and '101' in location_upper:
                      return self._get_directive_info('PROCEED', 'GO_SOUTH_TO_LITTLEROOT', 10)
                 if 'LITTLEROOT' in location_upper and 'HOUSE' not in location_upper:
                      return self._get_directive_info('PROCEED', 'ENTER_LAB_GET_POKEDEX', 10)
                 if 'LAB' in location_upper:
                      return self._get_directive_info('INTERACT', 'TALK_TO_BIRCH', 10)
        
        # POST-POKEDEX: Continue journey north (Route 101 → Oldale → Route 102 → Petalburg)
        if has_pokedex and party_state and len(party_state) > 0:
            # Exit lab if still inside
            if 'LAB' in location_upper:
                return self._get_directive_info('PROCEED', 'EXIT_LAB_GO_NORTH', 10)
            # Exit house if in one
            if 'HOUSE' in location_upper or 'BEDROOM' in location_upper:
                return self._get_directive_info('PROCEED', 'EXIT_HOUSE_GO_DOWN', 10)
            # In Littleroot Town, go north to Route 101
            if 'LITTLEROOT' in location_upper:
                return self._get_directive_info('PROCEED', 'GO_NORTH_ROUTE_101', 10)
            # In Littleroot, go north to Route 101
            if 'LITTLEROOT' in location_upper:
                return self._get_directive_info('PROCEED', 'GO_NORTH_TO_ROUTE_101_CONTINUE', 10)
            # On Route 101, go north to Oldale
            if 'ROUTE' in location_upper and '101' in location_upper:
                return self._get_directive_info('PROCEED', 'GO_NORTH_TO_OLDALE', 10)
            # In Oldale, go west to Route 102
            if 'OLDALE' in location_upper:
                return self._get_directive_info('PROCEED', 'GO_WEST_TO_ROUTE_102', 10)
        
        # Calculate party stats
        party_levels = [p.get('level', 0) for p in party_state if p]
        
        # Calculate average HP ratio (used for healing priority)
        if party_state and len(party_state) > 0:
            total_hp = sum(p.get('hp', 0) for p in party_state if p)
            total_max = sum(p.get('max_hp', 1) for p in party_state if p) or 1
            avg_hp = total_hp / total_max
        else:
            avg_hp = 1.0  # Default to full HP if no party data
        
        # Calculate gym-related variables (used later for gym preparation)
        next_milestone = self._get_next_milestone(implied_milestones) if hasattr(self, '_get_next_milestone') else ''
        avg_level = sum(party_levels) / len(party_levels) if party_levels else 1
        required_level = 15  # Default level for early game
        level_deficit = required_level - avg_level if avg_level else 0
        
        # POST-HEAL: If HP is full and in player's house, EXIT immediately!
        # This prevents the loop of going up after healing at Mom's
        if avg_hp >= 0.95 and ('PLAYER' in location_upper or 'MOM' in location_upper) and 'HOUSE' in location_upper:
            return self._get_directive_info('PROCEED', 'EXIT_HOUSE_AFTER_HEAL', 10)
        
        # If healed (full HP) in player's house (1F or 2F), EXIT immediately!
        if avg_hp >= 0.95 and 'PLAYER' in location_upper:
            if 'BEDROOM' in location_upper or '2F' in location_upper or 'HOUSE' in location_upper or '1F' in location_upper:
                return self._get_directive_info('PROCEED', 'EXIT_HOUSE_AFTER_HEAL', 10)
        
        # PRIORITY 1: Critical HP - MUST heal (raised threshold for earlier healing)
        if avg_hp < 0.40:
            return self._transition_to('HEAL', 'CRITICAL_HP', required_level)
        
        # PRIORITY 2: Post-blackout recovery
        if self.just_blackouted:
            self.just_blackouted = False
            return self._transition_to('POST_BLACKOUT', 'RECOVERY_FROM_BLACKOUT', required_level)
        
        # PRIORITY 3: Low HP but not critical
        if avg_hp < 0.4:
            return self._transition_to('HEAL', 'LOW_HP', required_level)
        
        # PRIORITY 4: Near gym - final preparation
        if self._is_near_gym(location, next_milestone):
            if level_deficit >= 1:
                return self._transition_to('PREPARE_GYM', 'UNDER_LEVELED_FOR_GYM', required_level)
            elif avg_hp < 0.6:
                return self._transition_to('HEAL', 'HEAL_BEFORE_GYM', required_level)
            else:
                return self._transition_to('PROCEED', 'READY_FOR_GYM', required_level)
        
        # PRIORITY 5: Check if in gym battle
        if 'GYM' in location.upper() and 'CITY' not in location.upper():
            return self._transition_to('GYM_BATTLE', 'IN_GYM', required_level)
        
        # PRIORITY 6: Level check for milestone progression
        if level_deficit >= 3:
            # Significantly under-leveled - MUST grind
            return self._transition_to('GRIND', f'NEED_{int(level_deficit)}_LEVELS', required_level)
        
        elif level_deficit >= 1:
            # Slightly under-leveled
            if self.current_directive == 'GRIND':
                # Already grinding, continue
                return self._get_directive_info('GRIND', 'CONTINUE_GRINDING', required_level)
            else:
                # Optional grinding
                return self._transition_to('OPTIONAL_GRIND', 'SLIGHT_DEFICIT', required_level)
        
        else:
            # Strong enough - proceed to milestone
            if self.current_directive in ['GRIND', 'OPTIONAL_GRIND', 'POST_BLACKOUT']:
                # Training complete!
                return self._transition_to('PROCEED', 'TRAINING_COMPLETE', required_level)
            else:
                story_directive = self._story_waypoint_directive(milestones_completed, location_upper, required_level)
                if story_directive:
                    # OVERRIDE: If in Lab and directed to go elsewhere, explicit EXIT command
                    if ('LAB' in location_upper or 'PROFESSOR' in location_upper) and 'BIRCH' not in story_directive.get('reason', ''):
                        story_directive['reason'] = 'EXIT_LAB_GO_DOWN'
                    return story_directive
                
                return self._get_directive_info('PROCEED', 'STRONG_ENOUGH', required_level)
    
    def _transition_to(self, new_directive, reason, target_level):
        """Transition to new directive with logging and feedback tracking."""
        should_transition = (new_directive != self.current_directive)
        
        if should_transition:
            old_directive = self.current_directive
            
            # Log transition
            self.directive_history.append({
                'from': old_directive,
                'to': new_directive,
                'reason': reason,
                'step': self.step_count
            })
            
            logger.info(f"🎯 STRATEGIC: {old_directive} → {new_directive} ({reason})")
            
            # Initialize new directive state
            if new_directive == 'GRIND':
                self.grinding_session_start_level = 0  # Will be set in update
                self.grinding_session_start_step = self.step_count
                self.grinding_session_battles = 0
                self.levels_gained_this_session = 0
            
            elif new_directive == 'PROCEED' and old_directive in ['GRIND', 'POST_BLACKOUT']:
                # Calculate grinding effectiveness
                steps_elapsed = self.step_count - self.grinding_session_start_step
                if steps_elapsed > 0:
                    effectiveness = (self.levels_gained_this_session * 1000.0) / steps_elapsed
                    logger.info(f"✅ Grinding session complete: {self.levels_gained_this_session} levels in {steps_elapsed} steps ({effectiveness:.2f} levels/1k steps)")
            
            self.current_directive = new_directive
            self.last_update_step = self.step_count
        
        return self._get_directive_info(new_directive, reason, target_level, should_transition)
    
    def _get_directive_info(self, directive, reason, target_level=10, should_transition=False):
        """Return directive information dict."""
        return {
            'directive': directive,
            'reason': reason,
            'target_level': target_level,
            'should_transition': should_transition
        }
    
    def record_level_gain(self, new_level):
        """Track level gains during grinding sessions."""
        if self.current_directive in ['GRIND', 'POST_BLACKOUT']:
            if self.grinding_session_start_level == 0:
                self.grinding_session_start_level = new_level
            else:
                self.levels_gained_this_session = new_level - self.grinding_session_start_level
    
    def record_battle(self):
        """Track battles during grinding sessions."""
        if self.current_directive in ['GRIND', 'POST_BLACKOUT']:
            self.grinding_session_battles += 1
    
    def record_blackout(self):
        """Mark that blackout occurred for next update."""
        self.just_blackouted = True
    
    def check_grinding_progress(self, current_level):
        """
        Monitor grinding progress and detect ineffective grinding.
        Returns feedback for potential strategy change.
        """
        if self.current_directive not in ['GRIND', 'POST_BLACKOUT']:
            return None
        
        steps_elapsed = self.step_count - self.grinding_session_start_step
        
        # Check if grinding is ineffective (no progress after 1000 steps)
        if steps_elapsed > 1000 and self.levels_gained_this_session < 1:
            logger.warning(f"⚠️ GRINDING INEFFECTIVE: {steps_elapsed} steps, {self.levels_gained_this_session} levels")
            logger.warning(f"   → Possible issues: Wrong location, avoiding battles, need evolution")
            return {
                'ineffective': True,
                'recommendation': 'CHANGE_LOCATION_OR_STRATEGY'
            }
        
        # Check if making good progress
        elif self.levels_gained_this_session >= 1:
            return {
                'ineffective': False,
                'levels_gained': self.levels_gained_this_session,
                'battles_fought': self.grinding_session_battles
            }
        
        return None
    
    def _get_next_milestone(self, completed_milestones):
        """Determine next milestone based on completed ones."""
        milestone_order = [
            'STARTER_CHOSEN',
            'FIRST_RIVAL_BATTLE',
            'RECEIVED_POKEDEX',
            'OLDALE_TOWN',
            'RUSTBORO_GYM',
            'ROXANNE_DEFEATED',
            'BRAWLY_GYM',
            'WATTSON_GYM',
            'FLANNERY_GYM',
            'NORMAN_GYM',
            'WINONA_GYM',
            'TATE_LIZA_GYM',
            'WALLACE_GYM',
            'ELITE_FOUR'
        ]
        
        for milestone in milestone_order:
            if milestone not in completed_milestones:
                return milestone
        
        return 'ELITE_FOUR'
    
    def _is_near_gym(self, location, next_milestone):
        """Check if agent is near gym entrance."""
        location_upper = location.upper()
        
        # Check if location matches milestone gym
        if 'RUSTBORO' in next_milestone and 'RUSTBORO' in location_upper:
            return True
        elif 'BRAWLY' in next_milestone and 'DEWFORD' in location_upper:
            return True
        elif 'WATTSON' in next_milestone and 'MAUVILLE' in location_upper:
            return True
        elif 'FLANNERY' in next_milestone and 'LAVARIDGE' in location_upper:
            return True
        elif 'NORMAN' in next_milestone and 'PETALBURG' in location_upper:
            return True
        
        return False
    
    def get_grinding_status(self):
        """Get current grinding session statistics."""
        return {
            'directive': self.current_directive,
            'is_grinding': self.current_directive in ['GRIND', 'POST_BLACKOUT', 'OPTIONAL_GRIND'],
            'session_start_step': self.grinding_session_start_step,
            'steps_elapsed': self.step_count - self.grinding_session_start_step,
            'battles_fought': self.grinding_session_battles,
            'levels_gained': self.levels_gained_this_session
        }

    def _story_waypoint_directive(self, milestones_completed, location_upper, target_level):
        """
        Provide explicit navigation waypoints for key early-game beats without extra overhead.
        """
        completed = set(milestones_completed or [])
        info = None
        if 'STARTER_CHOSEN' in completed and 'FIRST_RIVAL_BATTLE' not in completed:
            info = self._get_directive_info('PROCEED', 'SEEK_RIVAL_ROUTE103', target_level)
            info['current_target'] = 'ROUTE_103_RIVAL'
            info['objective'] = 'FIRST_RIVAL_BATTLE'
            return info
        
        if 'FIRST_RIVAL_BATTLE' in completed and 'RECEIVED_POKEDEX' not in completed:
            info = self._get_directive_info('PROCEED', 'RETURN_TO_BIRCH', target_level)
            info['current_target'] = 'BIRCH_LAB'
            info['objective'] = 'RECEIVE_POKEDEX'
            return info
        
        if 'RECEIVED_POKEDEX' in completed and 'PETALBURG_CITY' not in completed:
            info = self._get_directive_info('PROCEED', 'ADVANCE_TO_ROUTE102', target_level)
            info['current_target'] = 'ROUTE_102'
            info['objective'] = 'PETALBURG_CITY'
            return info
        
        return None
