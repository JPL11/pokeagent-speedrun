"""
Battle Experience Tracker - Learns from failures and integrates strategy knowledge
Feeds temporal battle reasoning into SNN for meta-learning
"""

import json
from collections import defaultdict, deque
from battle_strategy_knowledge import BATTLE_STRATEGY_GUIDE, encode_battle_knowledge_for_snn


class BattleExperienceTracker:
    """
    Tracks battle history, blackouts, and learned patterns.
    Integrates Pokemon Emerald strategy knowledge into agent's decision-making.
    
    Key Features:
    • Blackout recovery protocol (what to do after all Pokemon faint)
    • Type effectiveness learning (Water beats Fire through experience)
    • Gym-specific strategy memory (Roxanne needs Water/Grass)
    • Temporal failure patterns (died to same trainer 3x → change strategy)
    """
    
    def __init__(self):
        # Blackout tracking
        self.blackout_count = 0
        self.blackouts_last_100_steps = 0
        self.recent_death_locations = deque(maxlen=10)
        self.last_blackout_step = 0
        
        # Type effectiveness learning
        self.type_effectiveness_wins = defaultdict(int)  # "WATER_vs_FIRE": 5
        self.type_effectiveness_losses = defaultdict(int)
        
        # Gym/boss victories
        self.gym_victories = set()  # {"ROXANNE", "BRAWLY", ...}
        self.boss_attempts = defaultdict(int)  # Track retry count
        
        # Battle history
        self.battle_history = []  # [(step, outcome, context), ...]
        self.recent_battles = deque(maxlen=20)
        
        # Item tracking
        self.potion_count = 0
        self.has_revive = False
        
        # Meta-knowledge state
        self.active_strategy = None  # Current strategy being applied
        self.strategy_effectiveness = defaultdict(float)  # How well strategies work
        
        # EVENT SPIKES for SNN temporal credit assignment
        # These are binary flags that fire when battle events occur
        self.current_battle_events = {
            'battle_started': False,
            'battle_won': False,
            'battle_lost': False,
            'pokemon_fainted': False,
            'gym_leader_battle': False,
            'gym_leader_won': False,
            'used_item': False,
            'super_effective_hit': False
        }
    
    def record_blackout(self, step, location, party_state):
        """
        Record a blackout event (all Pokemon fainted).
        Triggers recovery protocol and failure analysis.
        """
        self.blackout_count += 1
        self.blackouts_last_100_steps += 1
        self.last_blackout_step = step
        self.recent_death_locations.append(location)
        
        # Analyze failure
        failure_pattern = self._analyze_failure_pattern(location, party_state)
        
        # Log for SNN temporal learning
        self.battle_history.append({
            'step': step,
            'event': 'BLACKOUT',
            'location': location,
            'pattern': failure_pattern,
            'party_state': party_state
        })
        
        return failure_pattern
    
    def _analyze_failure_pattern(self, location, party_state):
        """
        Determine WHY the agent failed (under-leveled, bad matchup, no items).
        Returns recommended recovery strategy.
        """
        pattern = {
            'reason': 'UNKNOWN',
            'recommendation': [],
            'priority': 'HIGH'
        }
        
        # Check for repeated failures at same location
        recent_deaths_here = [loc for loc in self.recent_death_locations if loc == location]
        if len(recent_deaths_here) >= 2:
            pattern['reason'] = 'REPEATED_FAILURE'
            pattern['recommendation'].append('CHANGE_STRATEGY')
            pattern['recommendation'].append('GRIND_LEVELS')
            pattern['priority'] = 'CRITICAL'
        
        # Check party HP before battle
        avg_hp = sum(p.get('hp', 0) / max(p.get('max_hp', 1), 1) for p in party_state) / max(len(party_state), 1)
        if avg_hp < 0.5:
            pattern['reason'] = 'LOW_HP_BEFORE_BATTLE'
            pattern['recommendation'].append('HEAL_BEFORE_FIGHT')
        
        # Check levels
        party_levels = [p.get('level', 0) for p in party_state]
        avg_level = sum(party_levels) / max(len(party_levels), 1)
        
        # Gym-specific analysis
        if 'RUSTBORO_GYM' in location or 'ROXANNE' in location:
            has_water_grass = any(
                'WATER' in str(p.get('types', [])) or 'GRASS' in str(p.get('types', []))
                for p in party_state
            )
            if not has_water_grass:
                pattern['reason'] = 'BAD_TYPE_MATCHUP'
                pattern['recommendation'].append('CATCH_WATER_OR_GRASS_TYPE')
            elif avg_level < 13:
                pattern['reason'] = 'UNDER_LEVELED'
                pattern['recommendation'].append('GRIND_TO_LEVEL_15')
        
        # Item check
        if self.potion_count == 0:
            pattern['recommendation'].append('BUY_POTIONS')
        
        return pattern
    
    def record_battle_victory(self, opponent_type, player_type, opponent_species, location):
        """
        Record a battle victory to learn type effectiveness and gym strategies.
        """
        # Learn type effectiveness
        if player_type and opponent_type:
            matchup_key = f"{player_type}_vs_{opponent_type}"
            self.type_effectiveness_wins[matchup_key] += 1
        
        # Gym leader victories
        if 'ROXANNE' in opponent_species or 'ROXANNE' in location:
            self.gym_victories.add('ROXANNE')
        elif 'BRAWLY' in opponent_species or 'BRAWLY' in location:
            self.gym_victories.add('BRAWLY')
        # ... add more gym leaders
        
        self.recent_battles.append({
            'outcome': 'WIN',
            'opponent': opponent_species,
            'player_type': player_type,
            'opponent_type': opponent_type
        })
    
    def get_strategy_for_location(self, location, party_state, player_level):
        """
        Query BATTLE_STRATEGY_GUIDE for recommended strategy at current location.
        Returns tactical advice based on game knowledge.
        """
        strategy = {
            'tactics': [],
            'warnings': [],
            'preparation_needed': False
        }
        
        # Route 103 - Rival battle
        if 'ROUTE_103' in location or 'ROUTE_0012' in location:
            rival_info = BATTLE_STRATEGY_GUIDE['early_game_strategy']['rival_battle_route_103']
            strategy['tactics'].append("RIVAL_BATTLE: Cannot escape, must win")
            strategy['tactics'].append(rival_info['win_condition'])
            
            # Check if prepared
            if player_level < 5:
                strategy['warnings'].append("Level too low for rival! Grind to Lv5+")
                strategy['preparation_needed'] = True
        
        # Rustboro Gym - Roxanne
        elif 'RUSTBORO_GYM' in location:
            roxanne_info = BATTLE_STRATEGY_GUIDE['early_game_strategy']['rustboro_gym_roxanne']
            strategy['tactics'].extend(roxanne_info['strategy'])
            
            # Check type matchup
            has_counter = any(
                'WATER' in str(p.get('types', [])) or 'GRASS' in str(p.get('types', []))
                for p in party_state
            )
            if not has_counter:
                strategy['warnings'].append("NO TYPE ADVANTAGE! Catch Lotad (Water/Grass) before fighting!")
                strategy['preparation_needed'] = True
            
            if player_level < 13:
                strategy['warnings'].append(f"Under-leveled ({player_level} < 13)! Grind 2-3 levels.")
                strategy['preparation_needed'] = True
            
            if self.potion_count < 3:
                strategy['warnings'].append("Low on Potions! Buy 3-5 before gym.")
                strategy['preparation_needed'] = True
        
        # General battle tactics
        general_tactics = BATTLE_STRATEGY_GUIDE['battle_tactics']
        for tactic in general_tactics['general_rules']:
            strategy['tactics'].append(tactic)
        
        return strategy
    
    def get_blackout_recovery_protocol(self):
        """
        After blackout, return step-by-step recovery plan.
        """
        protocol = BATTLE_STRATEGY_GUIDE['blackout_recovery_protocol']
        
        # Customize based on failure pattern
        last_failure = self.battle_history[-1] if self.battle_history else {}
        pattern = last_failure.get('pattern', {})
        
        recovery_plan = {
            'immediate_actions': protocol['after_blackout_actions'],
            'specific_recommendations': pattern.get('recommendation', []),
            'priority': pattern.get('priority', 'MEDIUM')
        }
        
        return recovery_plan
    
    def get_battle_event_spikes(self):
        """
        Return current battle event spikes as a vector for SNN input.
        
        Returns:
            8-dim binary vector: [battle_started, battle_won, battle_lost, 
                                  pokemon_fainted, gym_leader_battle, gym_leader_won,
                                  used_item, super_effective_hit]
        
        This lets SNN learn temporal patterns:
        - "When advisor says 'low safety' + we ignore it → battle_lost spike occurs → avoid that pattern"
        - "Certain move patterns + super_effective_hit spike → battle_won spike → reinforce"
        """
        spikes = [
            float(self.current_battle_events['battle_started']),
            float(self.current_battle_events['battle_won']),
            float(self.current_battle_events['battle_lost']),
            float(self.current_battle_events['pokemon_fainted']),
            float(self.current_battle_events['gym_leader_battle']),
            float(self.current_battle_events['gym_leader_won']),
            float(self.current_battle_events['used_item']),
            float(self.current_battle_events['super_effective_hit'])
        ]
        return spikes
    
    def reset_event_spikes(self):
        """Clear event spikes after they've been read by SNN."""
        for key in self.current_battle_events:
            self.current_battle_events[key] = False
    
    def fire_event(self, event_name: str):
        """Trigger a battle event spike."""
        if event_name in self.current_battle_events:
            self.current_battle_events[event_name] = True
    
    def encode_for_snn(self, battle_state=None):
        """
        Convert battle experience into 64-dim vector for SNN temporal reasoning.
        This is the key integration point: Past failures → SNN memory → Better decisions
        
        Format:
            [0-7]: Battle event spikes (8-dim)
            [8-63]: Battle strategy knowledge (56-dim)
        """
        # Get event spikes
        event_spikes = self.get_battle_event_spikes()
        
        # Get strategy knowledge
        agent_history = {
            'type_effectiveness_wins': dict(self.type_effectiveness_wins),
            'gym_victories': self.gym_victories,
            'blackouts_last_100_steps': self.blackouts_last_100_steps,
            'potion_count': self.potion_count,
            'has_revive': self.has_revive,
            'recent_death_locations': list(self.recent_death_locations)
        }
        
        strategy_vec = encode_battle_knowledge_for_snn(
            battle_state=battle_state,
            agent_history=agent_history,
            blackout_count=self.blackout_count
        )
        
        # Combine: [event_spikes (8-dim), strategy_knowledge (56-dim)] = 64-dim
        import numpy as np
        full_vec = np.concatenate([event_spikes, strategy_vec[:56]])
        
        return full_vec
    
    def decay_recent_failures(self, steps_since_last_blackout):
        """Decay blackout count over time (forgive old mistakes)"""
        if steps_since_last_blackout > 100:
            self.blackouts_last_100_steps = max(0, self.blackouts_last_100_steps - 1)
    
    def update_items(self, bag_state):
        """Track available healing items"""
        self.potion_count = bag_state.get('potions', 0) + bag_state.get('super_potions', 0)
        self.has_revive = bag_state.get('revive', 0) > 0 or bag_state.get('max_revive', 0) > 0
    
    def save_experience(self, filepath='battle_experience.json'):
        """Persist learned knowledge across training sessions"""
        data = {
            'blackout_count': self.blackout_count,
            'type_effectiveness_wins': dict(self.type_effectiveness_wins),
            'gym_victories': list(self.gym_victories),
            'boss_attempts': dict(self.boss_attempts),
            'battle_history': self.battle_history[-100:]  # Last 100 battles
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_experience(self, filepath='battle_experience.json'):
        """Load previous learned knowledge"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.blackout_count = data.get('blackout_count', 0)
            self.type_effectiveness_wins = defaultdict(int, data.get('type_effectiveness_wins', {}))
            self.gym_victories = set(data.get('gym_victories', []))
            self.boss_attempts = defaultdict(int, data.get('boss_attempts', {}))
            self.battle_history = data.get('battle_history', [])
            print(f"✅ Loaded battle experience: {self.blackout_count} blackouts, {len(self.gym_victories)} gyms beaten")
        except FileNotFoundError:
            print("ℹ️ No previous battle experience found, starting fresh")
