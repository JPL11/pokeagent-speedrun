# POKEMON TRAINING & BATTLE MANAGEMENT SYSTEM
# Comprehensive battle strategy for story progression
# To be applied after current 2M training completes

## Problem Statement

The agent needs to:
1. **Train Pokemon** (level up party for gyms, Elite Four, rivals)
2. **Manage HP risk** (heal proactively, use items, avoid blackouts)
3. **Balance exploration vs grinding** (don't avoid battles, but grind strategically)
4. **Use healing resources** (Pokemon Centers, Mom, Potions, items)
5. **Learn from failures** (blackout → grind → retry with better strategy)

Current issues:
- Agent may be avoiding tall grass (fear of blackouts)
- Not training Pokemon enough for story progression
- No strategic battle engagement (run from all wilds vs selective grinding)
- Missing healing item usage (Potions during battle)

## Solution Architecture

### Component 1: Battle Engagement Strategy

**Concept**: Not all battles are equal. Some should be avoided, some should be fought.

```python
class BattleEngagementStrategy:
    """
    Determines WHEN to fight wild Pokemon and WHEN to escape.
    Integrates with SNN/k-NN for adaptive learning.
    """
    
    def __init__(self):
        self.grinding_mode = False  # Explicit training mode
        self.target_level_for_next_milestone = 10  # Dynamic based on story
        
        # Track battle outcomes for adaptive strategy
        self.recent_wild_battle_outcomes = deque(maxlen=20)
        self.battles_needed_for_next_level = 10  # Estimate
        
        # Grinding locations (safe areas with appropriate-level Pokemon)
        self.safe_grinding_locations = {
            'ROUTE_101': {'levels': [2, 2, 2], 'recommended_player_level': 5},
            'ROUTE_102': {'levels': [3, 3, 4], 'recommended_player_level': 7},
            'ROUTE_103': {'levels': [3, 4, 4], 'recommended_player_level': 7},
            'ROUTE_104': {'levels': [4, 5, 5], 'recommended_player_level': 10}
        }
    
    def should_fight_wild_battle(self, location, party_state, opponent_level=None):
        """
        Decision function: Fight this wild Pokemon or escape?
        
        Returns: {
            'action': 'FIGHT' | 'ESCAPE',
            'reason': str,
            'reward_modifier': float  # Bonus/penalty for following advice
        }
        """
        party_levels = [p.get('level', 0) for p in party_state if p]
        avg_level = sum(party_levels) / max(len(party_levels), 1)
        avg_hp = sum(
            p.get('current_hp', 0) / max(p.get('max_hp', 1), 1) 
            for p in party_state
        ) / max(len(party_state), 1)
        
        # CASE 1: HP too low - avoid all battles
        if avg_hp < 0.3:
            return {
                'action': 'ESCAPE',
                'reason': 'HP_CRITICAL_HEAL_FIRST',
                'reward_modifier': 5.0  # Strong bonus for escaping
            }
        
        # CASE 2: Grinding mode - fight everything
        if self.grinding_mode:
            # But not if opponent too strong
            if opponent_level and opponent_level > avg_level + 5:
                return {
                    'action': 'ESCAPE',
                    'reason': 'OPPONENT_TOO_STRONG',
                    'reward_modifier': 3.0
                }
            return {
                'action': 'FIGHT',
                'reason': 'GRINDING_MODE_ACTIVE',
                'reward_modifier': 2.0  # Reward grinding
            }
        
        # CASE 3: Under-leveled for next milestone - need grinding
        if avg_level < self.target_level_for_next_milestone:
            # Check if this is a safe grinding location
            location_key = self._get_location_key(location)
            if location_key in self.safe_grinding_locations:
                grinding_data = self.safe_grinding_locations[location_key]
                
                # Fight if opponent is appropriate level
                if not opponent_level or opponent_level <= avg_level + 2:
                    return {
                        'action': 'FIGHT',
                        'reason': f'NEED_LEVELS_FOR_MILESTONE (target: {self.target_level_for_next_milestone})',
                        'reward_modifier': 3.0
                    }
        
        # CASE 4: Already strong enough - avoid wild battles
        if avg_level >= self.target_level_for_next_milestone + 2:
            return {
                'action': 'ESCAPE',
                'reason': 'ALREADY_STRONG_ENOUGH',
                'reward_modifier': 2.0  # Reward not wasting time
            }
        
        # CASE 5: Default - fight occasionally for experience
        # Fight ~40% of wild encounters
        import random
        if random.random() < 0.4:
            return {
                'action': 'FIGHT',
                'reason': 'BALANCED_TRAINING',
                'reward_modifier': 1.0
            }
        else:
            return {
                'action': 'ESCAPE',
                'reason': 'SAVE_TIME_FOR_STORY',
                'reward_modifier': 0.5
            }
    
    def update_target_level(self, next_milestone):
        """
        Set target level based on upcoming story milestone.
        """
        milestone_level_requirements = {
            'FIRST_RIVAL_BATTLE': 5,    # Route 103, rival has Lv5
            'RUSTBORO_GYM': 13,          # Roxanne has Lv12-15
            'BRAWLY_GYM': 18,            # Dewford Gym
            'WATTSON_GYM': 23,           # Mauville Gym
            'FLANNERY_GYM': 28,          # Lavaridge Gym
            'NORMAN_GYM': 33,            # Petalburg Gym (Dad)
            'WINONA_GYM': 35,            # Fortree Gym
            'TATE_LIZA_GYM': 43,         # Mossdeep Gym
            'WALLACE_GYM': 45,           # Sootopolis Gym
            'ELITE_FOUR': 50             # Pokemon League
        }
        
        self.target_level_for_next_milestone = milestone_level_requirements.get(
            next_milestone, 10
        )
    
    def activate_grinding_mode(self, duration_steps=500):
        """
        Explicitly enter grinding mode (fight all wild battles).
        Used after blackout or before gym.
        """
        self.grinding_mode = True
        self.grinding_mode_steps_remaining = duration_steps
    
    def step_update(self):
        """Call every step to decay grinding mode."""
        if self.grinding_mode and hasattr(self, 'grinding_mode_steps_remaining'):
            self.grinding_mode_steps_remaining -= 1
            if self.grinding_mode_steps_remaining <= 0:
                self.grinding_mode = False
    
    def _get_location_key(self, location):
        """Extract route identifier from location string."""
        if 'ROUTE_101' in location or '0x0010' in location:
            return 'ROUTE_101'
        elif 'ROUTE_102' in location:
            return 'ROUTE_102'
        elif 'ROUTE_103' in location or '0x0012' in location:
            return 'ROUTE_103'
        elif 'ROUTE_104' in location:
            return 'ROUTE_104'
        return 'UNKNOWN'
```

### Component 2: Healing Item Management

**Concept**: Use Potions during battle when HP is critical.

```python
class HealingItemManager:
    """
    Tracks available items and recommends when to use them.
    Integrates with battle tracker for item usage spikes.
    """
    
    def __init__(self):
        self.potion_count = 0
        self.super_potion_count = 0
        self.hyper_potion_count = 0
        self.revive_count = 0
        
        # Track item usage history
        self.items_used_this_session = 0
        self.last_item_use_step = 0
    
    def update_from_bag(self, bag_state):
        """Parse bag state from game memory."""
        self.potion_count = bag_state.get('Potion', 0)
        self.super_potion_count = bag_state.get('Super Potion', 0)
        self.hyper_potion_count = bag_state.get('Hyper Potion', 0)
        self.revive_count = bag_state.get('Revive', 0)
    
    def should_use_healing_item(self, pokemon_state, in_battle=False):
        """
        Decide if agent should use a healing item.
        
        Returns: {
            'use_item': bool,
            'item_type': str,  # 'POTION' | 'SUPER_POTION' | 'REVIVE'
            'urgency': str,    # 'CRITICAL' | 'RECOMMENDED' | 'OPTIONAL'
            'reason': str
        }
        """
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
                    'reason': 'POKEMON_FAINTED'
                }
            else:
                return {
                    'use_item': False,
                    'item_type': None,
                    'urgency': 'NONE',
                    'reason': 'NO_REVIVE_AVAILABLE'
                }
        
        # CASE 2: In battle, HP < 30% - URGENT
        if in_battle and hp_ratio < 0.3:
            # Use best available healing item
            if self.hyper_potion_count > 0 and max_hp > 100:
                return {
                    'use_item': True,
                    'item_type': 'HYPER_POTION',
                    'urgency': 'CRITICAL',
                    'reason': 'IN_BATTLE_CRITICAL_HP'
                }
            elif self.super_potion_count > 0 and max_hp > 40:
                return {
                    'use_item': True,
                    'item_type': 'SUPER_POTION',
                    'urgency': 'CRITICAL',
                    'reason': 'IN_BATTLE_CRITICAL_HP'
                }
            elif self.potion_count > 0:
                return {
                    'use_item': True,
                    'item_type': 'POTION',
                    'urgency': 'CRITICAL',
                    'reason': 'IN_BATTLE_CRITICAL_HP'
                }
        
        # CASE 3: In battle, HP < 50% - RECOMMENDED
        if in_battle and hp_ratio < 0.5:
            if self.potion_count > 0:
                return {
                    'use_item': True,
                    'item_type': 'POTION',
                    'urgency': 'RECOMMENDED',
                    'reason': 'IN_BATTLE_LOW_HP'
                }
        
        # CASE 4: Out of battle, HP < 40% - heal before next battle
        if not in_battle and hp_ratio < 0.4:
            if self.potion_count > 0:
                return {
                    'use_item': True,
                    'item_type': 'POTION',
                    'urgency': 'RECOMMENDED',
                    'reason': 'PREPARE_FOR_NEXT_BATTLE'
                }
        
        return {
            'use_item': False,
            'item_type': None,
            'urgency': 'NONE',
            'reason': 'HP_SUFFICIENT'
        }
    
    def get_shopping_recommendation(self, money_available):
        """
        Recommend buying healing items before gym/important battle.
        """
        recommendations = []
        
        # Always stock up on Potions (200 each)
        if self.potion_count < 5 and money_available >= 1000:
            recommendations.append({
                'item': 'POTION',
                'quantity': 5 - self.potion_count,
                'cost': (5 - self.potion_count) * 200,
                'priority': 'HIGH'
            })
        
        # Super Potions for mid-game (700 each)
        if self.super_potion_count < 3 and money_available >= 2100:
            recommendations.append({
                'item': 'SUPER_POTION',
                'quantity': 3,
                'cost': 2100,
                'priority': 'MEDIUM'
            })
        
        return recommendations
```

### Component 3: Reward System Integration

**Key Changes to `train_rl_vlm_screen_aware.py`**:

```python
# In ServerAwareEnv.__init__():
from battle_engagement_strategy import BattleEngagementStrategy
from healing_item_manager import HealingItemManager

self.battle_engagement = BattleEngagementStrategy()
self.healing_items = HealingItemManager()

# Track training progress
self.battles_fought_this_episode = 0
self.wild_battles_won = 0
self.wild_battles_lost = 0

# In step() reward calculation:

# ============= BATTLE TRAINING REWARDS =============

# 1. LEVEL UP REWARD (INCREASED from 1.0 to 5.0)
try:
    if party and len(party) > 0:
        level_sum = sum(p.get('level', 0) for p in party)
        
        if not hasattr(self, 'max_level_sum'):
            self.max_level_sum = level_sum
        
        if level_sum > self.max_level_sum:
            level_gain = level_sum - self.max_level_sum
            level_reward = 5.0 * level_gain  # INCREASED from 1.0
            reward += level_reward
            info_parts.append(f"📈 LEVEL UP! +{level_reward:.1f}")
            self.max_level_sum = level_sum
            
            # Fire SNN spike for level up
            self.battle_tracker.fire_event('pokemon_leveled_up')
except:
    pass

# 2. WILD BATTLE WIN REWARD (NEW)
try:
    battle_info = state.get('game', {}).get('battle_info', {})
    if battle_info and battle_info.get('in_battle'):
        # Check if battle just ended (transition from in_battle to not in_battle)
        if hasattr(self, 'was_in_battle') and self.was_in_battle:
            # Battle ended - did we win?
            current_hp_sum = sum(p.get('current_hp', 0) for p in party)
            
            if current_hp_sum > 0:  # Party still alive = won battle
                battle_win_reward = 3.0
                reward += battle_win_reward
                info_parts.append(f"⚔️ Battle won! +{battle_win_reward:.1f}")
                
                self.wild_battles_won += 1
                self.battle_tracker.fire_event('battle_won')
                
                # Store in episodic memory
                self.episodic_memory.store_navigation_experience(
                    cnn_features=cnn_features,
                    story_vec=story_vec,
                    waypoint_hint=waypoint_hint,
                    action_taken=action,
                    reward_received=battle_win_reward,
                    milestone_reached='WILD_BATTLE_WON'
                )
    
    # Update battle state tracker
    self.was_in_battle = battle_info and battle_info.get('in_battle', False)
except:
    pass

# 3. BATTLE ENGAGEMENT STRATEGY REWARD (NEW)
try:
    battle_info = state.get('game', {}).get('battle_info', {})
    
    # Update target level based on next milestone
    next_milestone = self._get_next_milestone_name(milestones_completed)
    self.battle_engagement.update_target_level(next_milestone)
    
    if battle_info and battle_info.get('in_battle'):
        opp_pokemon = battle_info.get('opponent_pokemon', {})
        opp_level = opp_pokemon.get('level', 0)
        
        # Get engagement strategy
        strategy = self.battle_engagement.should_fight_wild_battle(
            location=location_name,
            party_state=party,
            opponent_level=opp_level
        )
        
        # Reward following the strategy
        # (This is a heuristic - VLM will ultimately decide action)
        if strategy['action'] == 'FIGHT':
            engagement_bonus = strategy['reward_modifier']
            reward += engagement_bonus
            if engagement_bonus >= 2.0:
                info_parts.append(f"🎯 Strategic battle +{engagement_bonus:.1f}")
except:
    pass

# 4. GRINDING MODE ACTIVATION (after blackout)
try:
    if hp_fraction == 0 and self.last_hp_fraction > 0:
        # Blackout occurred
        blackout_penalty = -10.0  # Keep at -10 for story priority
        reward += blackout_penalty
        info_parts.append(f"☠️ Blackout {blackout_penalty}")
        
        # Record in battle tracker
        self.battle_tracker.record_blackout(
            step=self.step_count,
            location=location_name,
            party_state=party
        )
        
        # ACTIVATE GRINDING MODE (fight all wild battles for 500 steps)
        self.battle_engagement.activate_grinding_mode(duration_steps=500)
        info_parts.append(f"🔄 GRINDING MODE: Next 500 steps")
        
        # Add to VLM context
        self.post_blackout_grinding = True
except:
    pass

# 5. HEALING ITEM USAGE REWARD (NEW)
try:
    # Check if agent used a healing item
    if hasattr(self, 'last_potion_count'):
        healing_items = self.healing_items
        healing_items.update_from_bag(state.get('game', {}).get('bag', {}))
        
        if healing_items.potion_count < self.last_potion_count:
            # Potion was used
            item_use_reward = 8.0  # Strong reward for smart item usage
            reward += item_use_reward
            info_parts.append(f"💊 Used Potion +{item_use_reward:.1f}")
            
            # Fire SNN spike
            self.battle_tracker.fire_event('used_item')
        
        self.last_potion_count = healing_items.potion_count
    else:
        self.last_potion_count = 0
except:
    pass

# 6. GRASS TRAVERSAL REWARD (encourage wild encounters)
try:
    map_tiles = state.get('map', {}).get('tiles', [])
    current_tile_behavior = None
    if map_tiles and len(map_tiles) > 0:
        center = len(map_tiles) // 2
        if 0 <= center < len(map_tiles) and 0 <= center < len(map_tiles[center]):
            tile = map_tiles[center][center]
            if len(tile) >= 2:
                current_tile_behavior = tile[1]
    
    if current_tile_behavior:
        bval = current_tile_behavior.value if hasattr(current_tile_behavior, 'value') else current_tile_behavior
        if isinstance(bval, int) and bval in [2, 3]:  # TALL_GRASS, LONG_GRASS
            # Reward depends on grinding mode
            if self.battle_engagement.grinding_mode:
                grass_reward = 2.0  # HIGHER during grinding
            else:
                grass_reward = 0.5  # Normal exploration
            
            reward += grass_reward
            if self.step_count % 50 == 0 and grass_reward >= 2.0:
                info_parts.append(f"🌿 Training in grass +{grass_reward:.1f}")
except:
    pass
```

### Component 4: VLM Prompt Enhancement

Add battle training context to VLM:

```python
# In _build_vlm_prompt_with_memory():

# Get battle engagement strategy
battle_strategy = self.battle_engagement.should_fight_wild_battle(
    location=location_name,
    party_state=party,
    opponent_level=None
)

# Get healing item recommendations
healing_recommendation = None
if party and len(party) > 0:
    active_pokemon = party[0]  # Lead Pokemon
    healing_recommendation = self.healing_items.should_use_healing_item(
        pokemon_state=active_pokemon,
        in_battle=battle_info and battle_info.get('in_battle', False)
    )

# Build battle training guidance
battle_guidance = ""

if self.battle_engagement.grinding_mode:
    battle_guidance = f"""
🔥 **GRINDING MODE ACTIVE** (Level up for next challenge)
Target Level: {self.battle_engagement.target_level_for_next_milestone}
Current Avg Level: {sum(p.get('level', 0) for p in party) / max(len(party), 1):.1f}

**INSTRUCTIONS:**
1. Walk through TALL GRASS (Route 101/102) to trigger wild battles
2. FIGHT wild Pokemon (don't run) to gain experience
3. Use Potions if HP < 50% during battle
4. Continue until party reaches target level

💡 Wild battles in Emerald give experience points (EXP) to level up Pokemon.
"""
elif healing_recommendation and healing_recommendation['use_item']:
    battle_guidance = f"""
⚠️ **HEALING NEEDED**
{healing_recommendation['reason']}
Available items: {self.healing_items.potion_count} Potions

**ACTION:**
{'In battle: Open BAG menu, select ITEM → POTION, use on active Pokemon' if battle_info.get('in_battle') else 'Use Potion from menu OR go to Pokemon Center'}
"""
else:
    # Normal battle strategy
    party_levels = [p.get('level', 0) for p in party if p]
    avg_level = sum(party_levels) / max(len(party_levels), 1)
    
    battle_guidance = f"""
⚔️ **BATTLE STRATEGY**
Party Avg Level: {avg_level:.1f} | Target: {self.battle_engagement.target_level_for_next_milestone}
HP Status: {sum(p.get('current_hp', 0)/max(p.get('max_hp', 1), 1) for p in party) / max(len(party), 1) * 100:.0f}%

{'🎯 Need more training! Fight wild battles to level up.' if avg_level < self.battle_engagement.target_level_for_next_milestone else '✅ Strong enough for next milestone.'}

**BATTLE RULES:**
• Fight wild Pokemon if HP > 50% and need levels
• Use Potions in battle when HP < 50%
• Escape from battles if HP < 30% (no items)
• After blackout: Grind 2-3 levels before retry
"""

# Add to prompt
prompt = f"""Pokemon Emerald Speedrun - Story Progress

{battle_guidance}

**📖 STORY CONTEXT:**
...
"""
```

### Component 5: Post-Blackout Recovery Protocol

```python
def _handle_blackout_recovery(self, state, milestones_completed):
    """
    After blackout, agent needs to:
    1. Identify why it failed (under-leveled, no items, bad matchup)
    2. Execute recovery plan (grind, buy items, catch counter-type)
    3. Retry the challenge
    """
    failure_pattern = self.battle_tracker.record_blackout(
        step=self.step_count,
        location=self.last_location,
        party_state=self.last_party_state
    )
    
    recovery_plan = []
    
    # Check recommendations
    if 'GRIND_LEVELS' in failure_pattern['recommendation']:
        recovery_plan.append("GRIND: Fight wild Pokemon in tall grass for 2-3 levels")
        self.battle_engagement.activate_grinding_mode(duration_steps=500)
    
    if 'BUY_POTIONS' in failure_pattern['recommendation']:
        recovery_plan.append("SHOP: Visit PokeMart, buy 5 Potions")
    
    if 'CATCH_WATER_OR_GRASS_TYPE' in failure_pattern['recommendation']:
        recovery_plan.append("CATCH: Find Lotad/Seedot on Route 102 (type advantage)")
    
    if 'HEAL_BEFORE_FIGHT' in failure_pattern['recommendation']:
        recovery_plan.append("HEAL: Visit Pokemon Center before next attempt")
    
    # Store recovery plan for VLM
    self.active_recovery_plan = recovery_plan
    
    return recovery_plan
```

## Expected Behavior Changes

### Before:
- Agent avoids wild battles (fear of blackouts)
- Doesn't level up Pokemon sufficiently
- Runs from all grass encounters
- No item usage strategy
- Reaches gyms under-leveled → blackout loop

### After:
- **Strategic grinding**: Fights wild Pokemon when under-leveled
- **Adaptive difficulty**: Grinding mode after blackout (500 steps of training)
- **Item management**: Uses Potions during critical battles
- **Level awareness**: Knows target level for each milestone
- **Battle rewards**: 
  - Level up: +5.0 (was +1.0)
  - Battle won: +3.0 (NEW)
  - Item use: +8.0 (NEW)
  - Grinding mode grass: +2.0 (was +1.0)

## Integration with Existing Architecture

### SNN Integration:
- **INPUT 9 (Battle Events)**: Event spikes for level_up, battle_won, used_item
- **Temporal Credit Assignment**: SNN learns "grinding → level up → gym win" pattern
- Battle event spikes feed into 608-neuron SNN for meta-learning

### k-NN Episodic Memory:
- Stores successful grinding locations: `(route_102, grass_tile) → (fight_action, +5.0 level_reward)`
- Recalls: "Last time I fought wilds here, I leveled up and got +5 reward"
- Suggests: "This location is good for grinding"

### VLM Feedback:
- VLM sees battle guidance in prompt
- Makes decisions based on HP%, level targets, grinding mode
- Agent learns "VLM suggested grinding → I leveled up → next milestone achieved"

### CNN Features:
- Recognizes tall grass visual patterns
- Associates grass tiles with potential battles
- Learns "grass = training opportunity" not "grass = danger"

## Reward Balance

Current reward structure (after changes):
- **Story Milestone**: +500 (HIGHEST - story priority preserved)
- **Level Up**: +5.0 (5x increase - encourage training)
- **Battle Won**: +3.0 (reward combat success)
- **Item Used**: +8.0 (reward smart resource management)
- **Grinding Mode Grass**: +2.0 (encourage training areas)
- **Blackout**: -10.0 (unchanged - balanced for story priority)
- **Healing**: +15.0 (from HEALING_SYSTEM_PLAN.md)

**Net effect**: Training Pokemon becomes attractive (5+3+2 = 10 per battle cycle) while story progression remains dominant (+500 milestone).

## Implementation Files

### New Files:
1. `agent/battle_engagement_strategy.py` - Battle decision logic
2. `agent/healing_item_manager.py` - Item usage strategy

### Modified Files:
1. `train_rl_vlm_screen_aware.py`:
   - Import new modules
   - Add battle training rewards
   - Update VLM prompt with battle guidance
   - Integrate grinding mode activation
   - Add item usage tracking

2. `battle_experience_tracker.py`:
   - Add `pokemon_leveled_up` event spike
   - Track grinding mode effectiveness
   - Store level-up patterns

## Testing Checklist

After applying changes:
1. ✅ Agent enters tall grass when grinding mode active
2. ✅ Level up events fire correctly (+5.0 reward)
3. ✅ Battle wins reward agent (+3.0 reward)
4. ✅ Grinding mode activates after blackout (500 steps)
5. ✅ VLM prompt shows grinding instructions
6. ✅ Agent uses Potions during battle (HP < 50%)
7. ✅ Target level updates based on next milestone
8. ✅ Party reaches Lv13+ before Rustboro Gym
9. ✅ SNN receives battle event spikes
10. ✅ k-NN stores successful grinding experiences

## Pokemon Emerald Story Progression Requirements

Key level milestones:
- **Route 103 Rival**: Level 5 (starter vs starter)
- **Rustboro Gym (Roxanne)**: Level 13-15 (Rock-type, need Water/Grass)
- **Dewford Gym (Brawly)**: Level 18-20 (Fighting-type)
- **Mauville Gym (Wattson)**: Level 23-25 (Electric-type)
- **Lavaridge Gym (Flannery)**: Level 28-30 (Fire-type)
- **Petalburg Gym (Norman/Dad)**: Level 33-35 (Normal-type)
- **Fortree Gym (Winona)**: Level 35-37 (Flying-type)
- **Mossdeep Gym (Tate & Liza)**: Level 43-45 (Psychic-type)
- **Sootopolis Gym (Wallace)**: Level 45-47 (Water-type)
- **Elite Four**: Level 50-55 (Champion)

This system ensures agent trains appropriately for each challenge.

## Notes for Future Enhancements

1. **Type Matchup Learning**: Track which types beat which (Water > Fire)
2. **Move Selection**: VLM learns to use super-effective moves
3. **Pokemon Catching**: Catch counter-types before gyms (Lotad for Roxanne)
4. **EV Training**: Advanced - optimize stat growth
5. **Team Composition**: Balanced party with type coverage
