# IMPLEMENTATION GUIDE - Pokemon Training System
# Apply after current 2M training completes (at ~2% progress)

## Files Created
1. ✅ `agent/battle_engagement_strategy.py` - Battle decision logic
2. ✅ `agent/healing_item_manager.py` - Item usage strategy
3. ✅ `POKEMON_TRAINING_SYSTEM.md` - Full design document

## Files to Modify
1. `train_rl_vlm_screen_aware.py` - Main training script

## Step-by-Step Implementation

### STEP 1: Add Imports (Line ~30, after other imports)

```python
# Pokemon training strategy modules
from agent.battle_engagement_strategy import BattleEngagementStrategy
from agent.healing_item_manager import HealingItemManager
```

### STEP 2: Initialize in ServerAwareEnv.__init__() (Line ~1050, after battle_tracker init)

```python
# 🎯 BATTLE ENGAGEMENT STRATEGY (Determines when to fight vs escape)
self.battle_engagement = BattleEngagementStrategy()

# 💊 HEALING ITEM MANAGER (Tracks items and recommends usage)
self.healing_items = HealingItemManager()

# Track battle state for rewards
self.was_in_battle = False
self.battles_fought_this_episode = 0
self.wild_battles_won = 0
self.last_potion_count = 0
self.post_blackout_grinding = False
```

### STEP 3: Modify Reward Calculation in step() (Line ~1690-1720)

Find the section with "# 4. PARTY LEVEL SUM (Reward training Pokemon)"

**REPLACE** the existing level reward section with:

```python
# 4. PARTY LEVEL SUM - INCREASED REWARD (Encourage training)
try:
    if party and len(party) > 0:
        level_sum = sum(p.get('level', 0) for p in party)
        
        if not hasattr(self, 'max_level_sum'):
            self.max_level_sum = level_sum
        
        if level_sum > self.max_level_sum:
            level_gain = level_sum - self.max_level_sum
            level_reward = 5.0 * level_gain  # INCREASED from 1.0 to 5.0
            reward += level_reward
            info_parts.append(f"📈 LEVEL UP! +{level_reward:.1f}")
            self.max_level_sum = level_sum
            
            # Record in battle engagement for grinding mode
            self.battle_engagement.record_level_up(level_sum)
            
            # Fire SNN event spike
            self.battle_tracker.fire_event('battle_won')  # Treat level-up as success
except:
    pass
```

### STEP 4: Add Battle Win Reward (NEW - insert after level reward)

```python
# 4b. WILD BATTLE WIN REWARD (NEW)
try:
    battle_info = state.get('game', {}).get('battle_info', {})
    currently_in_battle = battle_info and battle_info.get('in_battle', False)
    
    # Detect battle end (was in battle, now not)
    if hasattr(self, 'was_in_battle') and self.was_in_battle and not currently_in_battle:
        # Battle ended - check if we won (party still has HP)
        party_has_hp = any(p.get('current_hp', 0) > 0 for p in party)
        
        if party_has_hp:
            # WON BATTLE!
            battle_win_reward = 3.0
            reward += battle_win_reward
            info_parts.append(f"⚔️ Battle won! +{battle_win_reward:.1f}")
            
            self.wild_battles_won += 1
            self.battles_fought_this_episode += 1
            
            # Fire SNN event spike
            self.battle_tracker.fire_event('battle_won')
            
            # Record in engagement strategy
            self.battle_engagement.record_battle_outcome(won=True, exp_gained=0)
            
            # Store in episodic memory (successful battle)
            try:
                self.episodic_memory.store_navigation_experience(
                    cnn_features=cnn_features,
                    story_vec=story_vec,
                    waypoint_hint=waypoint_hint,
                    action_taken=action,
                    reward_received=battle_win_reward,
                    milestone_reached='WILD_BATTLE_WON'
                )
            except:
                pass
    
    # Update battle state
    self.was_in_battle = currently_in_battle
except:
    pass
```

### STEP 5: Modify Blackout Penalty to Activate Grinding Mode (Line ~1644)

**FIND** the blackout penalty section:
```python
# Penalize blackout (all Pokemon fainted)
elif hp_fraction == 0 and self.last_hp_fraction > 0:
    blackout_penalty = -10.0  # (reduced for story priority)
    reward += blackout_penalty
    info_parts.append(f"☠️ Blackout {blackout_penalty}")
```

**REPLACE WITH**:
```python
# Penalize blackout (all Pokemon fainted)
elif hp_fraction == 0 and self.last_hp_fraction > 0:
    blackout_penalty = -10.0  # (reduced for story priority)
    reward += blackout_penalty
    info_parts.append(f"☠️ Blackout {blackout_penalty}")
    
    # 🧠 BATTLE EXPERIENCE: Learn from failure
    try:
        failure_pattern = self.battle_tracker.record_blackout(
            step=self.step_count,
            location=location_name,
            party_state=party
        )
        
        # ACTIVATE GRINDING MODE (fight wild battles for 500 steps)
        self.battle_engagement.activate_grinding_mode(
            duration_steps=500,
            reason='BLACKOUT_RECOVERY'
        )
        self.post_blackout_grinding = True
        info_parts.append(f"🔄 GRINDING MODE: 500 steps")
        
        logger.warning(f"   ⚠️ Blackout at {location_name}")
        logger.warning(f"   🔄 Recovery: {', '.join(failure_pattern['recommendation'])}")
    except Exception as e:
        logger.error(f"   ⚠️ CRITICAL: Repeated failure detected!")
```

### STEP 6: Add Healing Item Usage Reward (NEW - after blackout section)

```python
# 5. HEALING ITEM USAGE REWARD (NEW)
try:
    # Update item manager from bag state
    bag_state = state.get('game', {}).get('bag', {})
    self.healing_items.update_from_bag(bag_state)
    
    # Detect item use (potion count decreased)
    current_potion_count = self.healing_items.get_total_healing_items()
    
    if hasattr(self, 'last_total_items'):
        if current_potion_count < self.last_total_items:
            # Item was used!
            items_used = self.last_total_items - current_potion_count
            item_use_reward = 8.0 * items_used  # Strong reward
            reward += item_use_reward
            info_parts.append(f"💊 Used item +{item_use_reward:.1f}")
            
            # Fire SNN event spike
            self.battle_tracker.fire_event('used_item')
            
            # Record in healing manager
            self.healing_items.record_item_use(
                step=self.step_count,
                item_type='HEALING_ITEM',
                reason='BATTLE_HEALING'
            )
    
    self.last_total_items = current_potion_count
except:
    pass
```

### STEP 7: Modify Grass Traversal Reward (Line ~1730)

**FIND** the grass reward section:
```python
if isinstance(bval, int) and bval in [2, 3]:  # TALL_GRASS, LONG_GRASS
    grass_reward = 1.0
    reward += grass_reward
    if self.step_count % 25 == 0:
        info_parts.append(f"🌿 Grass OK +{grass_reward:.1f}")
```

**REPLACE WITH**:
```python
if isinstance(bval, int) and bval in [2, 3]:  # TALL_GRASS, LONG_GRASS
    # Reward depends on grinding mode
    if self.battle_engagement.grinding_mode:
        grass_reward = 2.0  # INCREASED during grinding
    else:
        grass_reward = 0.5  # Normal exploration
    
    reward += grass_reward
    
    # Show message for grinding mode
    if self.battle_engagement.grinding_mode and self.step_count % 25 == 0:
        info_parts.append(f"🌿 Training +{grass_reward:.1f}")
```

### STEP 8: Update Battle Engagement Strategy Every Step (end of step())

**ADD** at the very end of the step() method, before `return`:

```python
# Update battle engagement strategy state
try:
    self.battle_engagement.step_update(party_state=party)
except:
    pass
```

### STEP 9: Modify VLM Prompt to Include Battle Guidance (Line ~850-920)

**FIND** the VLM prompt building section in ServerAPIOracle._build_vlm_prompt_with_memory()

**ADD** this section AFTER extracting party/battle_info but BEFORE building the main prompt:

```python
# ========== BATTLE TRAINING GUIDANCE ==========
battle_guidance = ""

try:
    # Get grinding status
    grinding_status = self.env.battle_engagement.get_grinding_status()
    
    # Get party stats
    party_levels = [p.get('level', 0) for p in party if p]
    avg_level = sum(party_levels) / max(len(party_levels), 1) if party_levels else 0
    
    party_hp_ratios = [
        p.get('current_hp', 0) / max(p.get('max_hp', 1), 1)
        for p in party if p
    ]
    avg_hp = sum(party_hp_ratios) / max(len(party_hp_ratios), 1) if party_hp_ratios else 0
    
    # Get healing item summary
    item_summary = self.env.healing_items.get_item_summary()
    
    # CASE 1: GRINDING MODE ACTIVE
    if grinding_status['active']:
        battle_guidance = f"""
🔥 **GRINDING MODE ACTIVE** - Train Pokemon for next challenge!
📊 Current Level: {avg_level:.1f} | Target: {grinding_status['target_level']}
⏱️ Steps Remaining: {grinding_status['steps_remaining']}
💪 Battles Won: {grinding_status['battles_this_session']} | Levels Gained: {grinding_status['levels_gained']}

**TRAINING INSTRUCTIONS:**
1. Walk through TALL GRASS (dark green tiles) to encounter wild Pokemon
2. FIGHT wild battles - don't run! (need experience to level up)
3. Use POTIONS if HP drops below 50% (open BAG → select POTION)
4. Continue grinding until reaching target level

💡 Emerald Grinding Strategy:
• Route 101-103: Good for Lv3-8 (Poochyena, Zigzagoon)
• Route 104: Good for Lv8-12 (Taillow, Wingull)
• Wild battles give EXP → level up → stronger for gym battles

Items: {item_summary}
"""
    
    # CASE 2: LOW HP - NEED HEALING
    elif avg_hp < 0.4:
        battle_guidance = f"""
⚠️ **LOW HP - HEALING NEEDED**
Party HP: {avg_hp*100:.0f}% (CRITICAL)
Items: {item_summary}

**URGENT ACTION REQUIRED:**
{'1. Open BAG menu, use POTION on low-HP Pokemon' if self.env.healing_items.get_total_healing_items() > 0 else '1. NO ITEMS! Go to Pokemon Center (Oldale/Petalburg)'}
2. Talk to Nurse Joy: "Please heal my Pokemon"
3. THEN continue exploring

💡 Don't explore with low HP - risk of blackout (lose progress & money)
"""
    
    # CASE 3: NEED MORE LEVELS FOR NEXT MILESTONE
    elif avg_level < grinding_status['target_level'] - 2:
        level_deficit = grinding_status['target_level'] - avg_level
        battle_guidance = f"""
⚔️ **TRAINING RECOMMENDED**
Current Level: {avg_level:.1f} | Needed: {grinding_status['target_level']}
Deficit: {level_deficit:.1f} levels

**RECOMMENDED:**
• Find tall grass area (Route 101, 102, or 103)
• Fight 5-10 wild Pokemon to gain levels
• Stock up on Potions at PokeMart if needed

💡 Next milestone requires stronger Pokemon. Quick grinding session recommended.
"""
    
    # CASE 4: READY FOR NEXT CHALLENGE
    else:
        battle_guidance = f"""
✅ **PARTY READY** - Strong enough for next challenge!
Level: {avg_level:.1f} | HP: {avg_hp*100:.0f}%
Items: {item_summary}

**FOCUS ON STORY PROGRESSION:**
• Continue to next milestone objective
• Fight wild battles occasionally (30% of encounters)
• Escape from weak wild Pokemon to save time
• Use items if HP drops below 50%
"""
    
except Exception as e:
    battle_guidance = ""

# ADD battle_guidance to the main prompt
```

**THEN** in the main prompt construction, add:

```python
prompt = f"""Pokemon Emerald Speedrun - Story Progress

{battle_guidance}

**📖 STORY CONTEXT:**
...
"""
```

### STEP 10: Helper Method for Next Milestone (add to ServerAwareEnv class)

```python
def _get_next_milestone_name(self, milestones_completed):
    """
    Determine which milestone is next based on completed milestones.
    
    Returns:
        str: Next milestone name for level target calculation
    """
    # Ordered list of story milestones
    milestone_order = [
        'STARTER_CHOSEN',
        'FIRST_RIVAL_BATTLE',
        'RECEIVED_POKEDEX',
        'OLDALE_TOWN',
        'RUSTBORO_GYM',
        'BRAWLY_GYM',
        'WATTSON_GYM',
        'FLANNERY_GYM',
        'NORMAN_GYM',
        'WINONA_GYM',
        'TATE_LIZA_GYM',
        'WALLACE_GYM',
        'ELITE_FOUR'
    ]
    
    # Find first uncompleted milestone
    for milestone in milestone_order:
        if milestone not in milestones_completed:
            return milestone
    
    return 'ELITE_FOUR'  # Default to final challenge
```

## Testing After Implementation

### 1. Verify Imports
```bash
cd /home/jpli/pokeagent-speedrun
python -c "from agent.battle_engagement_strategy import BattleEngagementStrategy; from agent.healing_item_manager import HealingItemManager; print('✅ Imports OK')"
```

### 2. Syntax Check
```bash
python -m py_compile train_rl_vlm_screen_aware.py
```

### 3. Short Test Run (1000 steps)
```bash
python train_rl_vlm_screen_aware.py --timesteps 1000
```

### 4. Monitor Training Logs
```bash
tail -f training_*.log | grep -E "Level up|Battle won|Grinding|Used item|Blackout"
```

Expected log patterns:
- `📈 LEVEL UP! +5.0` (increased from +1.0)
- `⚔️ Battle won! +3.0` (NEW)
- `🔄 GRINDING MODE: 500 steps` (after blackout)
- `💊 Used item +8.0` (NEW)
- `🌿 Training +2.0` (during grinding mode)

### 5. Check VLM Prompts
```bash
grep -A 10 "GRINDING MODE ACTIVE" training_*.log | head -30
```

Should see battle guidance in VLM prompts.

## Expected Behavioral Changes

### Phase 1: Initial Exploration (Steps 0-5000)
- **Before**: Agent avoids grass, runs from all battles
- **After**: Agent fights ~30% of wild battles, levels reach 7-8

### Phase 2: First Blackout (When it happens)
- **Before**: Agent confused, might blackout repeatedly
- **After**: Grinding mode activates → fights wild battles for 500 steps → levels up → retries challenge

### Phase 3: Gym Preparation (Before Rustboro Gym)
- **Before**: Attempts gym at Lv8-9 (under-leveled) → blackout
- **After**: Realizes needs Lv13, grinds in Route 102 → reaches Lv13+ → beats gym

### Phase 4: Smart Item Usage
- **Before**: Never uses Potions, relies only on Pokemon Center
- **After**: Uses Potions during battle when HP<50% → +8.0 reward → survives longer

## Reward Comparison Table

| Event | Before | After | Change |
|-------|--------|-------|--------|
| Level Up | +1.0 | +5.0 | 5x increase |
| Battle Won | 0 | +3.0 | NEW |
| Item Used | 0 | +8.0 | NEW |
| Grinding Grass | +1.0 | +2.0 | 2x during grind mode |
| Milestone | +500 | +500 | unchanged |
| Blackout | -10 | -10 + grinding mode | recovery added |

## Integration Verification Checklist

After implementation, verify:
- [ ] Level-up reward increased to +5.0
- [ ] Battle win detection working (+3.0 reward)
- [ ] Grinding mode activates after blackout
- [ ] Grass reward increases during grinding mode
- [ ] Item usage tracked and rewarded
- [ ] VLM prompt includes battle guidance
- [ ] Target level updates based on next milestone
- [ ] SNN receives battle event spikes
- [ ] k-NN stores successful battle experiences
- [ ] No syntax errors or crashes

## Rollback Plan

If issues occur:
1. Kill training process
2. Revert train_rl_vlm_screen_aware.py to backup
3. Report specific errors in logs
4. Can test individual modules separately

## Performance Expectations

After 50k steps with new system:
- Party average level should be 10-12 (vs 5-7 before)
- Wild battles won: 50-100 (vs ~0 before)
- Grinding mode activations: 1-3 (after blackouts)
- Item uses: 5-15 (vs ~0 before)
- Route 101 completion rate: Higher (agent has levels for Route 103 rival)

This system makes the agent **actively train Pokemon** rather than avoiding all battles, which is essential for story progression in Pokemon Emerald.
