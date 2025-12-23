# Pokemon Training Strategy Implementation

## Overview
Implemented three-tier strategic architecture to ensure agent trains Pokemon (instead of avoiding all battles) so story can progress through gyms, Elite Four, etc.

## Architecture

### Tier 1: VLM Strategic Directives (High-Level Goals)
- **Location**: VLM prompt enhancement (~line 590-680)
- **Directives**: PROCEED, GRIND, OPTIONAL_GRIND, HEAL, PREPARE_GYM, POST_BLACKOUT, GYM_BATTLE
- **Purpose**: VLM receives battle guidance with 4 cases:
  1. **GRINDING MODE** - Shows target level, progress, training instructions
  2. **LOW HP** - Urgent healing warnings with item availability
  3. **TRAINING RECOMMENDED** - Level deficit display
  4. **PARTY READY** - Focus on story progression

### Tier 2: SNN Temporal Learning (Pattern Recognition)
- **Integration**: Ready for INPUT 12 (32-dim strategic context vector)
- **Purpose**: Learn temporal patterns during grinding vs exploring
- **Status**: Architecture prepared, encoding to be added later

### Tier 3: Feedback Rewards (Context-Aware Reinforcement)
All rewards dynamically adjust based on strategic directive:

#### Level Reward (Line ~1720-1748)
- **GRIND**: 8.0x multiplier (highest priority)
- **PROCEED**: 3.0x multiplier (still valued)
- **Default**: 5.0x multiplier

#### Battle Win Reward (Line ~1750-1795)
- **GRIND**: +4.0 per win (encourage training)
- **PROCEED**: +1.0 per win (low priority)
- **Default**: +2.0 per win
- Fires `battle_won` event, records in strategic_manager

#### Healing Item Usage (Line ~1797-1810)
- **Reward**: +8.0 per item used
- Tracks Potions, Super Potions, Revives, etc.
- Fires `used_item` event

#### Strategic Goal Completion (Line ~1812-1827)
- **Reward**: +50.0 when training target reached
- Detects GRIND → PROCEED transitions
- Logs milestone achievements

#### Grass Rewards (Line ~1875-1909)
- **GRIND**: 2.5x (seek tall grass for encounters)
- **PROCEED**: 0.2x (avoid unnecessary battles)
- **Default**: 1.0x

## Key Modules

### agent/strategic_directive_manager.py (283 lines)
**Purpose**: High-level strategic goal coordination with feedback loops

**Key Methods**:
- `update_directive(party, milestones, location, step_count)` - Evaluates state, returns directive
- `record_blackout()` - Triggers POST_BLACKOUT recovery mode
- `check_grinding_progress()` - Detects ineffective grinding
- `get_grinding_status()` - Returns progress metrics

**Milestone Requirements**:
```python
MILESTONE_LEVEL_REQUIREMENTS = {
    'GOT_STARTER': 5,
    'REACHED_OLDALE': 7,
    'REACHED_PETALBURG': 10,
    'RUSTBORO_GYM': 13,
    'RIVAL_BATTLE_2': 16,
    'ELITE_FOUR': 50,
    # ... etc
}
```

### agent/battle_engagement_strategy.py (353 lines)
**Purpose**: Tactical battle decision support (when to fight vs escape)

**Key Methods**:
- `should_fight_wild_battle(party, location, in_grinding_mode)` - Returns (fight/escape, reward_modifier)
- `activate_grinding_mode(target_level)` - Enters training mode
- Safe grinding locations with level recommendations

### agent/healing_item_manager.py (323 lines)
**Purpose**: Healing resource management

**Key Methods**:
- `should_use_healing_item(party, items_dict)` - Recommends item usage with urgency
- `get_shopping_recommendation(party, money, next_milestone)` - What to buy before gyms
- Tracks Potions, Super Potions, Revives, Full Heals, etc.

## Integration Points

### 1. Module Initialization (Line ~1064)
```python
self.strategic_manager = StrategicDirectiveManager()
self.battle_engagement = BattleEngagementStrategy()
self.healing_items = HealingItemManager()
self.current_strategic_directive = StrategicDirective.PROCEED
```

### 2. Strategic Directive Update (Line ~1682-1691)
**When**: Every VLM call (~25 steps)
```python
directive_info = self.strategic_manager.update_directive(
    party, milestones_completed, map_name, step_count
)
self.current_strategic_directive = directive_info['directive']
if directive_info['should_transition']:
    logger.info(f"🎯 STRATEGIC DIRECTIVE UPDATED: {directive_info['directive'].name}")
```

### 3. Blackout Recovery (Line ~1776)
```python
self.strategic_manager.record_blackout()
```

### 4. VLM Battle Guidance (Line ~590-680)
Injects strategic context into VLM prompt based on current directive

## Feedback Loops

### Loop 1: Level Progress → Directive Update
```
Agent grinds → Levels up → Reaches target → GRIND → PROCEED transition → Rewards adjust
```

### Loop 2: Blackout Recovery
```
Battle lost → Blackout → POST_BLACKOUT directive → Healing emphasis → Gradual return to GRIND
```

### Loop 3: Ineffective Grinding Detection
```
Grinding for 500 steps → No levels gained → Warning logged → Human can intervene
```

## Testing Status

✅ **Completed**:
- Syntax check passed
- All imports resolve
- Modules instantiate correctly
- Battle decision logic tested with sample party

⏳ **Ready for Deployment**:
- Code compiles and is ready to run
- Will be deployed AFTER current 2M training completes (currently at 2.53%)

## Usage

### When Training Starts:
1. Strategic manager will evaluate party state every VLM call
2. If party is under-leveled for next milestone → GRIND directive
3. VLM receives grinding guidance with target levels
4. Rewards adjust to encourage grass exploration and battles
5. Agent learns: "During GRIND, tall grass = good, battles = good"
6. Once target reached → PROCEED directive
7. Rewards adjust to discourage unnecessary battles
8. Agent learns: "During PROCEED, avoid grass, focus on story"

### Monitoring:
Look for these log messages:
- `🎯 STRATEGIC DIRECTIVE UPDATED: GRIND` - Training mode activated
- `🔥 GRINDING MODE ACTIVE` - VLM received grinding instructions
- `⚔️ BATTLE WON! (+X.X reward during GRINDING)` - Context-aware battle reward
- `✅ STRATEGIC GOAL COMPLETE! Party reached target level` - Training target achieved

## Next Steps (Optional Enhancements)

### 1. SNN Strategic Context Encoding
Add INPUT 12 (32-dim vector) to SNN:
- Encode current directive (one-hot)
- Party state (avg level, HP, item count)
- Battle context (in_battle, wild vs trainer)
- Temporal feedback (steps in current directive)

### 2. Advanced Grinding Locations
Expand `BattleEngagementStrategy.SAFE_GRINDING_LOCATIONS` with:
- More routes
- Type effectiveness considerations
- Experience yield data

### 3. Gym Preparation System
Add `PREPARE_GYM` directive logic:
- Type advantage checking
- Item stocking recommendations
- Level verification

## Files Modified

1. **train_rl_vlm_screen_aware.py** (~3680 lines)
   - Added strategic module imports
   - Initialized managers in `__init__`
   - Added directive update at VLM call time
   - Enhanced VLM prompt with battle guidance
   - Made all relevant rewards context-aware
   - Integrated blackout recovery

2. **agent/strategic_directive_manager.py** (NEW, 283 lines)
3. **agent/battle_engagement_strategy.py** (NEW, 353 lines)
4. **agent/healing_item_manager.py** (NEW, 323 lines)

## Design Rationale

User correctly identified that battle strategy should be **high-level reasoning** (VLM decides "should I grind?") not **low-level rules** ("fight this specific Pokemon"). This implementation:

- **VLM Layer**: Sets strategic goals based on milestones
- **SNN Layer**: Learns temporal patterns (grinding vs exploring)
- **Reward Layer**: Provides feedback signals that adjust dynamically

This three-tier separation allows:
1. VLM to reason about long-term goals
2. SNN to learn efficient movement patterns
3. Rewards to guide learning without hardcoding behavior

## Training Philosophy

**Before**: Agent avoided all battles → never leveled up → failed gyms  
**After**: Agent adapts battle behavior to strategic needs:
- GRIND mode → Seek grass, fight battles, gain levels
- PROCEED mode → Focus on story, avoid unnecessary battles
- POST_BLACKOUT → Recover cautiously, heal frequently

This matches how human players train Pokemon: grind when needed, progress when ready.
