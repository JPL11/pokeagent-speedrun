"""
Battle Engagement Strategy - Determines when to fight vs escape wild battles
Integrates with SNN/k-NN for adaptive Pokemon training
"""

from collections import deque
import random


class BattleEngagementStrategy:
    """
    Determines WHEN to fight wild Pokemon and WHEN to escape.
    Balances story progression with necessary training for gyms/bosses.
    
    Key Features:
    • Adaptive grinding mode (activated after blackouts or before gyms)
    • Level target awareness (knows when party is strong enough)
    • HP risk assessment (don't fight with low HP)
    • Location-based safety (grind in appropriate-level areas)
    """
    
    def __init__(self):
        self.grinding_mode = False
        self.grinding_mode_steps_remaining = 0
        self.target_level_for_next_milestone = 10
        
        # Track battle outcomes for learning
        self.recent_wild_battle_outcomes = deque(maxlen=20)
        self.battles_this_grinding_session = 0
        self.levels_gained_this_session = 0
        
        # Safe grinding locations (map_id or location_name → data)
        self.safe_grinding_locations = {
            'ROUTE_101': {
                'wild_levels': [2, 2, 2],
                'recommended_player_level_min': 3,
                'recommended_player_level_max': 8,
                'pokemon': ['POOCHYENA', 'ZIGZAGOON', 'WURMPLE']
            },
            'ROUTE_102': {
                'wild_levels': [3, 3, 4],
                'recommended_player_level_min': 5,
                'recommended_player_level_max': 10,
                'pokemon': ['POOCHYENA', 'ZIGZAGOON', 'LOTAD', 'SEEDOT']
            },
            'ROUTE_103': {
                'wild_levels': [3, 4, 4],
                'recommended_player_level_min': 5,
                'recommended_player_level_max': 10,
                'pokemon': ['POOCHYENA', 'ZIGZAGOON', 'WINGULL']
            },
            'ROUTE_104': {
                'wild_levels': [4, 5, 5],
                'recommended_player_level_min': 7,
                'recommended_player_level_max': 12,
                'pokemon': ['POOCHYENA', 'ZIGZAGOON', 'TAILLOW', 'WINGULL']
            },
            'ROUTE_116': {
                'wild_levels': [6, 7, 8],
                'recommended_player_level_min': 10,
                'recommended_player_level_max': 15,
                'pokemon': ['POOCHYENA', 'ZIGZAGOON', 'WHISMUR', 'SKITTY']
            }
        }
        
        # Milestone level requirements (when agent needs to be strong enough)
        self.milestone_level_requirements = {
            'FIRST_RIVAL_BATTLE': 5,
            'RUSTBORO_GYM': 13,
            'BRAWLY_GYM': 18,
            'WATTSON_GYM': 23,
            'FLANNERY_GYM': 28,
            'NORMAN_GYM': 33,
            'WINONA_GYM': 35,
            'TATE_LIZA_GYM': 43,
            'WALLACE_GYM': 45,
            'ELITE_FOUR': 50
        }
    
    def should_fight_wild_battle(self, location, party_state, opponent_level=None, in_battle=False):
        """
        Core decision function: Fight this wild Pokemon or escape?
        
        Args:
            location (str): Current location name
            party_state (list): Party Pokemon data
            opponent_level (int): Opponent's level (if known)
            in_battle (bool): Whether currently in battle
        
        Returns:
            dict: {
                'action': 'FIGHT' | 'ESCAPE',
                'reason': str,
                'reward_modifier': float  # Bonus for following advice
            }
        """
        if not party_state or len(party_state) == 0:
            return {
                'action': 'ESCAPE',
                'reason': 'NO_PARTY',
                'reward_modifier': 0.0
            }
        
        # Calculate party stats
        party_levels = [p.get('level', 0) for p in party_state if p]
        avg_level = sum(party_levels) / max(len(party_levels), 1)
        
        party_hp_ratios = [
            p.get('current_hp', 0) / max(p.get('max_hp', 1), 1) 
            for p in party_state if p
        ]
        avg_hp = sum(party_hp_ratios) / max(len(party_hp_ratios), 1)
        
        # CASE 1: HP too low - MUST avoid battles
        if avg_hp < 0.25:
            return {
                'action': 'ESCAPE',
                'reason': 'HP_CRITICAL_HEAL_IMMEDIATELY',
                'reward_modifier': 8.0  # Strong bonus for escaping
            }
        elif avg_hp < 0.4:
            return {
                'action': 'ESCAPE',
                'reason': 'HP_LOW_HEAL_SOON',
                'reward_modifier': 5.0
            }
        
        # CASE 2: Grinding mode - fight most battles
        if self.grinding_mode:
            # But not if opponent way too strong
            if opponent_level and opponent_level > avg_level + 7:
                return {
                    'action': 'ESCAPE',
                    'reason': 'OPPONENT_TOO_STRONG_FOR_GRINDING',
                    'reward_modifier': 3.0
                }
            
            # Check if in a safe grinding location
            location_key = self._get_location_key(location)
            if location_key in self.safe_grinding_locations:
                grinding_data = self.safe_grinding_locations[location_key]
                
                # Good grinding location
                if (grinding_data['recommended_player_level_min'] <= avg_level 
                    <= grinding_data['recommended_player_level_max'] + 3):
                    return {
                        'action': 'FIGHT',
                        'reason': f'GRINDING_MODE (Target Lv{self.target_level_for_next_milestone})',
                        'reward_modifier': 4.0  # Strong reward for grinding
                    }
            
            # Default grinding behavior
            return {
                'action': 'FIGHT',
                'reason': 'GRINDING_MODE_ACTIVE',
                'reward_modifier': 2.0
            }
        
        # CASE 3: Under-leveled for next milestone - need training
        level_deficit = self.target_level_for_next_milestone - avg_level
        
        if level_deficit >= 3:
            # Significantly under-leveled - should grind
            location_key = self._get_location_key(location)
            
            if location_key in self.safe_grinding_locations:
                grinding_data = self.safe_grinding_locations[location_key]
                
                # Check if appropriate level for this area
                if avg_level >= grinding_data['recommended_player_level_min']:
                    return {
                        'action': 'FIGHT',
                        'reason': f'NEED_LEVELS (Currently Lv{avg_level:.0f}, Target Lv{self.target_level_for_next_milestone})',
                        'reward_modifier': 3.0
                    }
        
        elif level_deficit >= 1:
            # Slightly under-leveled - fight occasionally
            if random.random() < 0.6:  # 60% fight rate
                return {
                    'action': 'FIGHT',
                    'reason': 'SLIGHT_TRAINING_NEEDED',
                    'reward_modifier': 2.0
                }
        
        # CASE 4: Already strong enough - minimize wild battles
        if avg_level >= self.target_level_for_next_milestone + 2:
            return {
                'action': 'ESCAPE',
                'reason': 'ALREADY_STRONG_ENOUGH',
                'reward_modifier': 2.0  # Reward not wasting time
            }
        
        # CASE 5: Default - balanced approach
        # Fight ~30% of encounters for general training
        if random.random() < 0.3:
            return {
                'action': 'FIGHT',
                'reason': 'BALANCED_TRAINING',
                'reward_modifier': 1.0
            }
        else:
            return {
                'action': 'ESCAPE',
                'reason': 'FOCUS_ON_STORY_PROGRESS',
                'reward_modifier': 0.5
            }
    
    def update_target_level(self, next_milestone):
        """
        Set target level based on upcoming story milestone.
        Called when milestones change.
        """
        if not next_milestone:
            return
        
        # Look up required level
        required_level = self.milestone_level_requirements.get(next_milestone, 10)
        
        # Add buffer (want to be slightly over-leveled for safety)
        self.target_level_for_next_milestone = required_level
    
    def activate_grinding_mode(self, duration_steps=500, reason='BLACKOUT_RECOVERY'):
        """
        Explicitly enter grinding mode (fight all appropriate wild battles).
        
        Args:
            duration_steps (int): How many steps to grind for
            reason (str): Why grinding was activated
        """
        self.grinding_mode = True
        self.grinding_mode_steps_remaining = duration_steps
        self.battles_this_grinding_session = 0
        self.levels_gained_this_session = 0
        self.grinding_reason = reason
    
    def deactivate_grinding_mode(self):
        """Manually exit grinding mode (e.g., reached target level)."""
        self.grinding_mode = False
        self.grinding_mode_steps_remaining = 0
    
    def step_update(self, party_state=None):
        """
        Call every environment step to update internal state.
        
        Args:
            party_state (list): Current party (to check if target reached)
        """
        # Decay grinding mode over time
        if self.grinding_mode and self.grinding_mode_steps_remaining > 0:
            self.grinding_mode_steps_remaining -= 1
            
            # Check if target level reached (early exit from grinding)
            if party_state and len(party_state) > 0:
                party_levels = [p.get('level', 0) for p in party_state if p]
                avg_level = sum(party_levels) / max(len(party_levels), 1)
                
                if avg_level >= self.target_level_for_next_milestone:
                    # Reached target - exit grinding mode
                    self.deactivate_grinding_mode()
            
            # Time-based exit
            if self.grinding_mode_steps_remaining <= 0:
                self.deactivate_grinding_mode()
    
    def record_battle_outcome(self, won: bool, exp_gained: int = 0):
        """
        Track battle results for adaptive learning.
        
        Args:
            won (bool): Whether battle was won
            exp_gained (int): Experience points gained
        """
        self.recent_wild_battle_outcomes.append({
            'won': won,
            'exp_gained': exp_gained
        })
        
        if self.grinding_mode:
            self.battles_this_grinding_session += 1
    
    def record_level_up(self, new_level: int):
        """Track level gains during grinding session."""
        if self.grinding_mode:
            self.levels_gained_this_session += 1
    
    def get_grinding_status(self):
        """
        Get current grinding mode status for VLM prompt.
        
        Returns:
            dict: Grinding mode info for VLM context
        """
        return {
            'active': self.grinding_mode,
            'steps_remaining': self.grinding_mode_steps_remaining,
            'target_level': self.target_level_for_next_milestone,
            'battles_this_session': self.battles_this_grinding_session,
            'levels_gained': self.levels_gained_this_session,
            'reason': getattr(self, 'grinding_reason', 'UNKNOWN')
        }
    
    def _get_location_key(self, location):
        """
        Extract route identifier from location string.
        
        Args:
            location (str): Raw location string from game
        
        Returns:
            str: Normalized location key
        """
        location_upper = location.upper()
        
        # Route detection
        if 'ROUTE_101' in location_upper or 'ROUTE101' in location_upper or '0x0010' in location:
            return 'ROUTE_101'
        elif 'ROUTE_102' in location_upper or 'ROUTE102' in location_upper:
            return 'ROUTE_102'
        elif 'ROUTE_103' in location_upper or 'ROUTE103' in location_upper or '0x0012' in location:
            return 'ROUTE_103'
        elif 'ROUTE_104' in location_upper or 'ROUTE104' in location_upper:
            return 'ROUTE_104'
        elif 'ROUTE_116' in location_upper or 'ROUTE116' in location_upper:
            return 'ROUTE_116'
        
        return 'UNKNOWN'
    
    def get_recommended_grinding_location(self, current_level):
        """
        Suggest best location for current level.
        
        Args:
            current_level (float): Party average level
        
        Returns:
            str: Recommended location key
        """
        for location_key, data in self.safe_grinding_locations.items():
            if (data['recommended_player_level_min'] <= current_level 
                <= data['recommended_player_level_max']):
                return location_key
        
        # Default to highest available
        return 'ROUTE_104'
