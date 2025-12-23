"""
Healing Item Manager - Tracks and recommends healing item usage
Integrates with battle tracker for smart resource management
"""


class HealingItemManager:
    """
    Tracks available healing items and recommends when to use them.
    Helps agent manage HP and avoid blackouts through item usage.
    
    Key Features:
    • In-battle healing recommendations (Potion when HP < 50%)
    • Out-of-battle preparation (heal before challenging battles)
    • Shopping recommendations (stock up before gyms)
    • Item usage history (learn optimal timing)
    """
    
    def __init__(self):
        # Item counts
        self.potion_count = 0
        self.super_potion_count = 0
        self.hyper_potion_count = 0
        self.max_potion_count = 0
        self.full_restore_count = 0
        
        # Revival items
        self.revive_count = 0
        self.max_revive_count = 0
        
        # Status healing
        self.antidote_count = 0
        self.paralyze_heal_count = 0
        self.awakening_count = 0
        self.burn_heal_count = 0
        self.ice_heal_count = 0
        self.full_heal_count = 0
        
        # Usage tracking
        self.items_used_this_session = 0
        self.last_item_use_step = 0
        self.item_use_history = []  # [(step, item_type, reason), ...]
        
        # Money tracking for shopping
        self.current_money = 0
    
    def update_from_bag(self, bag_state):
        """
        Parse bag state from game memory.
        
        Args:
            bag_state (dict): Bag items from game state
        """
        # Healing items (HP restoration)
        self.potion_count = bag_state.get('Potion', 0)
        self.super_potion_count = bag_state.get('Super Potion', 0)
        self.hyper_potion_count = bag_state.get('Hyper Potion', 0)
        self.max_potion_count = bag_state.get('Max Potion', 0)
        self.full_restore_count = bag_state.get('Full Restore', 0)
        
        # Revival items
        self.revive_count = bag_state.get('Revive', 0)
        self.max_revive_count = bag_state.get('Max Revive', 0)
        
        # Status healing
        self.antidote_count = bag_state.get('Antidote', 0)
        self.paralyze_heal_count = bag_state.get('Parlyz Heal', 0)
        self.awakening_count = bag_state.get('Awakening', 0)
        self.burn_heal_count = bag_state.get('Burn Heal', 0)
        self.ice_heal_count = bag_state.get('Ice Heal', 0)
        self.full_heal_count = bag_state.get('Full Heal', 0)
    
    def should_use_healing_item(self, pokemon_state, in_battle=False, opponent_level=None):
        """
        Decide if agent should use a healing item RIGHT NOW.
        
        Args:
            pokemon_state (dict): Current Pokemon's stats
            in_battle (bool): Whether currently in battle
            opponent_level (int): Opponent's level (for danger assessment)
        
        Returns:
            dict: {
                'use_item': bool,
                'item_type': str,  # 'POTION' | 'SUPER_POTION' | 'REVIVE' | etc
                'urgency': str,    # 'CRITICAL' | 'RECOMMENDED' | 'OPTIONAL'
                'reason': str,
                'reward_value': float  # Expected reward for using item
            }
        """
        if not pokemon_state:
            return self._no_action_response('NO_POKEMON_DATA')
        
        current_hp = pokemon_state.get('current_hp', 0)
        max_hp = pokemon_state.get('max_hp', 1)
        hp_ratio = current_hp / max(max_hp, 1)
        
        # CASE 1: Pokemon fainted - use Revive
        if current_hp == 0:
            if self.revive_count > 0:
                return {
                    'use_item': True,
                    'item_type': 'REVIVE',
                    'urgency': 'CRITICAL',
                    'reason': 'POKEMON_FAINTED',
                    'reward_value': 10.0
                }
            elif self.max_revive_count > 0:
                return {
                    'use_item': True,
                    'item_type': 'MAX_REVIVE',
                    'urgency': 'CRITICAL',
                    'reason': 'POKEMON_FAINTED',
                    'reward_value': 10.0
                }
            else:
                return {
                    'use_item': False,
                    'item_type': None,
                    'urgency': 'CRITICAL',
                    'reason': 'NO_REVIVE_AVAILABLE_MUST_HEAL_AT_CENTER',
                    'reward_value': 0.0
                }
        
        # CASE 2: In battle, HP critical (< 30%)
        if in_battle and hp_ratio < 0.3:
            # Use best available healing item
            item_choice = self._select_best_healing_item(max_hp, current_hp)
            
            if item_choice:
                return {
                    'use_item': True,
                    'item_type': item_choice,
                    'urgency': 'CRITICAL',
                    'reason': 'IN_BATTLE_CRITICAL_HP',
                    'reward_value': 8.0
                }
            else:
                return {
                    'use_item': False,
                    'item_type': None,
                    'urgency': 'CRITICAL',
                    'reason': 'NO_HEALING_ITEMS_MUST_ESCAPE',
                    'reward_value': 0.0
                }
        
        # CASE 3: In battle, HP low (30-50%)
        if in_battle and 0.3 <= hp_ratio < 0.5:
            item_choice = self._select_best_healing_item(max_hp, current_hp)
            
            if item_choice:
                return {
                    'use_item': True,
                    'item_type': item_choice,
                    'urgency': 'RECOMMENDED',
                    'reason': 'IN_BATTLE_LOW_HP',
                    'reward_value': 5.0
                }
        
        # CASE 4: In battle, HP moderate (50-70%) against strong opponent
        if in_battle and 0.5 <= hp_ratio < 0.7:
            if opponent_level and pokemon_state.get('level', 0) > 0:
                level_diff = opponent_level - pokemon_state.get('level', 0)
                
                # Opponent is stronger - heal proactively
                if level_diff >= 3:
                    item_choice = self._select_best_healing_item(max_hp, current_hp)
                    
                    if item_choice:
                        return {
                            'use_item': True,
                            'item_type': item_choice,
                            'urgency': 'RECOMMENDED',
                            'reason': 'STRONG_OPPONENT_HEAL_PROACTIVELY',
                            'reward_value': 4.0
                        }
        
        # CASE 5: Out of battle, HP < 40% - prepare for next battle
        if not in_battle and hp_ratio < 0.4:
            item_choice = self._select_best_healing_item(max_hp, current_hp)
            
            if item_choice:
                return {
                    'use_item': True,
                    'item_type': item_choice,
                    'urgency': 'RECOMMENDED',
                    'reason': 'PREPARE_FOR_NEXT_BATTLE',
                    'reward_value': 3.0
                }
        
        # CASE 6: Out of battle, HP < 60% before gym/boss
        # (This would require additional context about upcoming battle)
        
        # No action needed
        return self._no_action_response('HP_SUFFICIENT')
    
    def _select_best_healing_item(self, max_hp, current_hp):
        """
        Choose most appropriate healing item based on HP and availability.
        
        Strategy:
        - Full Restore: Any HP, any status (best item)
        - Max Potion: Full heal
        - Hyper Potion: 200 HP (for high max HP)
        - Super Potion: 50 HP (for mid max HP)
        - Potion: 20 HP (basic)
        
        Returns:
            str: Item type or None if no items available
        """
        hp_needed = max_hp - current_hp
        
        # Full Restore (best option if available)
        if self.full_restore_count > 0:
            return 'FULL_RESTORE'
        
        # Max Potion (full heal)
        if self.max_potion_count > 0:
            return 'MAX_POTION'
        
        # Hyper Potion (200 HP) - use if need > 100 HP
        if self.hyper_potion_count > 0 and hp_needed > 100:
            return 'HYPER_POTION'
        
        # Super Potion (50 HP) - use if need > 30 HP
        if self.super_potion_count > 0 and hp_needed > 30:
            return 'SUPER_POTION'
        
        # Potion (20 HP) - basic option
        if self.potion_count > 0:
            return 'POTION'
        
        return None
    
    def _no_action_response(self, reason):
        """Helper to return 'no action' response."""
        return {
            'use_item': False,
            'item_type': None,
            'urgency': 'NONE',
            'reason': reason,
            'reward_value': 0.0
        }
    
    def get_total_healing_items(self):
        """Count total healing items available."""
        return (self.potion_count + self.super_potion_count + 
                self.hyper_potion_count + self.max_potion_count + 
                self.full_restore_count)
    
    def get_shopping_recommendation(self, money_available, next_milestone=None):
        """
        Recommend buying healing items before important battles.
        
        Args:
            money_available (int): Current money
            next_milestone (str): Upcoming milestone (gym, rival, etc.)
        
        Returns:
            list: Shopping recommendations
        """
        recommendations = []
        
        # Potion: 200 each (basic item, always good to have)
        if self.potion_count < 5 and money_available >= 200:
            qty_needed = min(5 - self.potion_count, money_available // 200)
            if qty_needed > 0:
                recommendations.append({
                    'item': 'POTION',
                    'quantity': qty_needed,
                    'cost': qty_needed * 200,
                    'priority': 'HIGH',
                    'reason': 'Always keep 5 Potions for emergencies'
                })
        
        # Super Potion: 700 each (mid-game)
        if next_milestone in ['RUSTBORO_GYM', 'BRAWLY_GYM', 'WATTSON_GYM']:
            if self.super_potion_count < 3 and money_available >= 700:
                qty_needed = min(3 - self.super_potion_count, money_available // 700)
                if qty_needed > 0:
                    recommendations.append({
                        'item': 'SUPER_POTION',
                        'quantity': qty_needed,
                        'cost': qty_needed * 700,
                        'priority': 'MEDIUM',
                        'reason': f'Prepare for {next_milestone}'
                    })
        
        # Hyper Potion: 1200 each (late-game)
        if next_milestone in ['NORMAN_GYM', 'WINONA_GYM', 'TATE_LIZA_GYM', 'WALLACE_GYM', 'ELITE_FOUR']:
            if self.hyper_potion_count < 5 and money_available >= 1200:
                qty_needed = min(5 - self.hyper_potion_count, money_available // 1200)
                if qty_needed > 0:
                    recommendations.append({
                        'item': 'HYPER_POTION',
                        'quantity': qty_needed,
                        'cost': qty_needed * 1200,
                        'priority': 'HIGH',
                        'reason': f'Prepare for {next_milestone}'
                    })
        
        # Revive: 1500 each (essential for long battles)
        if self.revive_count < 2 and money_available >= 1500:
            qty_needed = min(2 - self.revive_count, money_available // 1500)
            if qty_needed > 0:
                recommendations.append({
                    'item': 'REVIVE',
                    'quantity': qty_needed,
                    'cost': qty_needed * 1500,
                    'priority': 'MEDIUM',
                    'reason': 'Revive fainted Pokemon during battles'
                })
        
        return recommendations
    
    def record_item_use(self, step, item_type, reason):
        """Track when items are used for analysis."""
        self.items_used_this_session += 1
        self.last_item_use_step = step
        self.item_use_history.append({
            'step': step,
            'item': item_type,
            'reason': reason
        })
    
    def get_item_summary(self):
        """
        Get summary of all available items for VLM context.
        
        Returns:
            str: Human-readable item summary
        """
        healing_total = self.get_total_healing_items()
        
        summary_parts = []
        
        if healing_total == 0:
            summary_parts.append("⚠️ NO HEALING ITEMS! Buy Potions at PokeMart.")
        else:
            if self.potion_count > 0:
                summary_parts.append(f"Potion×{self.potion_count}")
            if self.super_potion_count > 0:
                summary_parts.append(f"Super Potion×{self.super_potion_count}")
            if self.hyper_potion_count > 0:
                summary_parts.append(f"Hyper Potion×{self.hyper_potion_count}")
        
        if self.revive_count > 0:
            summary_parts.append(f"Revive×{self.revive_count}")
        
        if not summary_parts:
            return "No items"
        
        return ", ".join(summary_parts)
