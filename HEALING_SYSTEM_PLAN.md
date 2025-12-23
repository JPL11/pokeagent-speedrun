# HEALING & EXPLORATION IMPROVEMENTS
# To be applied after current 2M training run
# Addresses: blackout discouragement, map revisit limits, healing strategy

## Analysis of Current Issues

### Problem 1: Agent Revisiting Same Maps (Brendan's House)
- Agent visited map 0101 (Brendan's House 2F) **22 times** in recent logs
- This is exploration inefficiency - should move to Route 101 instead
- Root cause: No penalty for revisiting completed map coordinates

### Problem 2: Blackout Penalty May Be Discouraging Exploration
- Current penalty: -10.0 (already reduced from -50)
- But agent needs to go through Route 101 tall grass (wild Pokemon encounters)
- Risk of blackout may still be preventing Route 101 exploration
- Early game: NO Pokemon Center unlocked yet, only Mom can heal

### Problem 3: No Healing Strategy
- Agent doesn't know about healing locations:
  - **Early game**: Mom at home (Brendan's House 2F, position ~(6,4))
  - **After Oldale**: Pokemon Center with Nurse Joy
- No reward for seeking healing when HP is low
- No penalty avoidance strategy (heal before blackout risk)

## Solution: Adaptive Healing & Exploration System

### Component 1: Map Revisit Limiter
**Concept**: After visiting same map coordinates N times, require map transition

```python
# Track visits per map coordinate
map_coord_visits = {}  # {(map_id, x, y): visit_count}

MAX_COORD_VISITS = 10  # Complete coords 10 times max

current_key = (map_id, x, y)
visit_count = map_coord_visits.get(current_key, 0) + 1
map_coord_visits[current_key] = visit_count

if visit_count > MAX_COORD_VISITS:
    # Strong penalty to force map transition
    revisit_penalty = -5.0 * (visit_count - MAX_COORD_VISITS)
    reward += revisit_penalty
    info_parts.append(f"🔁 Overvisited {revisit_penalty:.1f}")
    
    # Bonus for ANY map transition
    if map_id != last_map_id:
        transition_bonus = +10.0
        reward += transition_bonus
        info_parts.append(f"🗺️ Map change +{transition_bonus}")
```

### Component 2: Healing Strategy System

#### 2A: HP Risk Assessment
```python
def assess_hp_risk(party):
    """
    Returns: 'critical', 'low', 'moderate', 'healthy'
    """
    if not party or len(party) == 0:
        return 'critical'
    
    total_hp = sum(p.get('current_hp', 0) for p in party)
    max_hp = sum(p.get('max_hp', 1) for p in party)
    hp_ratio = total_hp / max(max_hp, 1)
    
    # Count fainted Pokemon
    fainted = sum(1 for p in party if p.get('current_hp', 0) == 0)
    
    if hp_ratio < 0.25 or fainted >= len(party) / 2:
        return 'critical'  # ☠️ High blackout risk
    elif hp_ratio < 0.5:
        return 'low'       # ⚠️ Should heal soon
    elif hp_ratio < 0.75:
        return 'moderate'  # 🟡 Monitor HP
    else:
        return 'healthy'   # ✅ Good to explore
```

#### 2B: Healing Location Tracker
```python
class HealingLocationTracker:
    def __init__(self):
        self.healing_locations = {
            'MOM_HOME': {
                'map_id': 0x0101,  # Brendan's House 2F
                'position': (6, 4),  # Approximate Mom position
                'available': True,   # Always available
                'type': 'mom',
                'name': "Mom's House"
            },
            'OLDALE_POKECENTER': {
                'map_id': 0x0201,  # Oldale Town Pokemon Center
                'position': (4, 4),  # Nurse Joy position
                'available': False,  # Unlocked after reaching Oldale
                'type': 'pokecenter',
                'name': 'Oldale Pokemon Center'
            },
            'PETALBURG_POKECENTER': {
                'map_id': 0x0301,  # Petalburg Pokemon Center
                'position': (4, 4),
                'available': False,
                'type': 'pokecenter',
                'name': 'Petalburg Pokemon Center'
            }
        }
        
        self.last_heal_location = None
        self.steps_since_heal = 0
    
    def get_nearest_healing_location(self, current_map, current_pos, story_progress):
        """Find closest available healing location"""
        # Update availability based on story progress
        if story_progress.get('OLDALE_TOWN_REACHED'):
            self.healing_locations['OLDALE_POKECENTER']['available'] = True
        if story_progress.get('PETALBURG_CITY_REACHED'):
            self.healing_locations['PETALBURG_POKECENTER']['available'] = True
        
        # Filter available locations
        available = [loc for loc in self.healing_locations.values() if loc['available']]
        
        if not available:
            return None
        
        # Find nearest (simple Manhattan distance as heuristic)
        # In real implementation, use SLAM pathfinding
        nearest = min(available, key=lambda loc: 
            abs(loc['map_id'] - current_map) * 1000 +  # Penalize map changes
            abs(loc['position'][0] - current_pos[0]) + 
            abs(loc['position'][1] - current_pos[1])
        )
        
        return nearest
    
    def record_heal(self, location_name):
        """Record that healing occurred"""
        self.last_heal_location = location_name
        self.steps_since_heal = 0
```

#### 2C: Healing Reward System
```python
# In step() function:

hp_risk = assess_hp_risk(party)

# CASE 1: Critical HP - STRONG incentive to heal
if hp_risk == 'critical':
    healing_loc = healing_tracker.get_nearest_healing_location(map_id, (x,y), milestones)
    
    if healing_loc:
        # Calculate distance to healing location
        heal_map = healing_loc['map_id']
        heal_pos = healing_loc['position']
        
        # If on same map, reward moving toward healing
        if map_id == heal_map:
            distance = abs(x - heal_pos[0]) + abs(y - heal_pos[1])
            
            if not hasattr(self, 'last_heal_distance'):
                self.last_heal_distance = distance
            
            # Reward getting closer to healing
            if distance < self.last_heal_distance:
                approach_reward = +3.0
                reward += approach_reward
                info_parts.append(f"🏥 → {healing_loc['name']} +{approach_reward}")
            
            # BIG reward for reaching healing location
            if distance <= 1:
                arrival_reward = +20.0
                reward += arrival_reward
                info_parts.append(f"🏥 AT {healing_loc['name']} +{arrival_reward}")
            
            self.last_heal_distance = distance
        
        # If on different map, STRONG penalty for exploring instead of healing
        else:
            risky_exploration_penalty = -8.0
            reward += risky_exploration_penalty
            info_parts.append(f"⚠️ Critical HP, should heal! {risky_exploration_penalty}")

# CASE 2: Healing occurred (HP increased)
if hp_fraction > self.last_hp_fraction and len(party) == self.last_party_size:
    heal_amount = hp_fraction - self.last_hp_fraction
    
    # INCREASED healing reward (was 2.0, now 15.0 for full heal)
    # Encourage healing behavior
    healing_reward = 15.0 * (heal_amount ** 2)
    reward += healing_reward
    info_parts.append(f"❤️ HEALED +{healing_reward:.1f}")
    
    # Record healing event
    healing_tracker.record_heal(location)
    
    # Fire battle tracker event
    self.battle_tracker.fire_event('used_item')  # Treat healing as strategic action

# CASE 3: Blackout occurred
elif hp_fraction == 0 and self.last_hp_fraction > 0:
    # Blackout penalty - keep at -10 for story priority
    blackout_penalty = -10.0
    reward += blackout_penalty
    info_parts.append(f"☠️ Blackout {blackout_penalty}")
    
    # But add LESSON: next time should heal
    # This is meta-learning for battle tracker
    self.battle_tracker.fire_event('battle_lost')
    self.battle_tracker.record_blackout(
        step=self.step_count,
        location=location,
        party_state=party
    )
```

### Component 3: VLM Healing Guidance

Add healing context to VLM prompts:

```python
# In VLM prompt generation:

hp_risk = assess_hp_risk(party)
healing_guidance = ""

if hp_risk in ['critical', 'low']:
    healing_loc = healing_tracker.get_nearest_healing_location(map_id, (x,y), milestones)
    
    if healing_loc:
        healing_guidance = f"""
⚠️ **HP STATUS: {hp_risk.upper()}**
Party HP: {hp_fraction*100:.0f}%
Nearest healing: {healing_loc['name']} (Map {healing_loc['map_id']:04x})

**URGENT: HEAL BEFORE EXPLORING**
1. Navigate to {healing_loc['name']}
2. Talk to {"Mom" if healing_loc['type'] == 'mom' else "Nurse Joy"}
3. Full heal happens automatically
4. THEN continue Route 101 exploration

💡 Pokemon Emerald strategy: Heal when HP < 50% to avoid blackout.
"""

prompt = f"""Pokemon Emerald Speedrun - Story Progress

{healing_guidance}

**📖 STORY CONTEXT:**
...
"""
```

### Component 4: Battle Strategy Integration

Update battle tracker to include healing recommendations:

```python
# In BattleExperienceTracker:

def should_heal_before_area(self, area_name, party_hp_ratio):
    """
    Check if healing is recommended before entering an area.
    Uses historical blackout data.
    """
    # Count blackouts in this area
    area_blackouts = [
        event for event in self.battle_history
        if event.get('location', '').startswith(area_name) and event['event'] == 'BLACKOUT'
    ]
    
    if len(area_blackouts) >= 2:
        # Area is dangerous - recommend healing threshold
        if party_hp_ratio < 0.7:
            return {
                'should_heal': True,
                'reason': f'Area dangerous: {len(area_blackouts)} previous blackouts',
                'recommended_hp': 0.8
            }
    
    # Standard threshold
    if party_hp_ratio < 0.4:
        return {
            'should_heal': True,
            'reason': 'HP critically low',
            'recommended_hp': 0.8
        }
    
    return {'should_heal': False}
```

## Implementation Plan (Apply After Training)

### Step 1: Add HealingLocationTracker class
Location: `agent/healing_tracker.py` (new file)

### Step 2: Update ServerAwareEnv.__init__()
Add:
```python
from agent.healing_tracker import HealingLocationTracker
self.healing_tracker = HealingLocationTracker()
self.map_coord_visits = {}
```

### Step 3: Update reward calculation in step()
Add all reward components from above:
- Map revisit limiter
- HP risk assessment
- Healing approach rewards
- Increased healing reward (2.0 → 15.0 multiplier)

### Step 4: Update VLM prompt generation
Add healing guidance section when HP is critical/low

### Step 5: Update BattleExperienceTracker
Add `should_heal_before_area()` method

### Step 6: Update episodic memory
Store healing experiences:
```python
if healing_occurred:
    episodic_memory.store_navigation_experience(
        cnn_features=cnn_features,
        story_vec=story_vec,
        waypoint_hint=waypoint_hint,
        action_taken=action,
        reward_received=healing_reward,
        milestone_reached='HEALING_OCCURRED'
    )
```

## Expected Behavior Changes

### Before:
- Agent revisits Brendan's House 22+ times
- Avoids Route 101 (fear of blackout)
- No healing strategy (continues with low HP)
- Gets stuck in safe zones

### After:
- **Map revisit limit**: After 10 visits to same coordinate, strong penalty → forces map transition
- **Healing awareness**: When HP < 50%, agent navigates to Mom/Nurse
- **Risk management**: Critical HP → seek heal, Healthy HP → explore Route 101
- **Meta-learning**: Blackouts in Route 101 → heal before re-entering
- **Efficient exploration**: Heal → explore → repeat cycle

## Reward Balance Check

Current rewards:
- Milestone: +500 (story progress priority)
- Blackout: -10 (reduced for story)
- Healing: +2.0 * amount² (was too low)
- Exploration: +0.2 per new coord

New rewards:
- **Healing (full)**: +15.0 (7.5x increase) - encourages healing behavior
- **Approach healing (critical HP)**: +3.0 per step closer
- **Reach healing location**: +20.0 (one-time)
- **Map overvisit**: -5.0 per excess visit
- **Map transition (after overvisit)**: +10.0

**Net effect**: 
- Healing becomes attractive (15-20 reward vs 2)
- Map cycling penalized (-5 per repeat)
- Story progress still dominates (+500 milestone)
- Blackout risk acceptable (-10) if leads to milestone

## Testing Checklist

After applying changes:
1. ✅ Agent navigates to Mom when HP < 50%
2. ✅ Full heal occurs at Mom's location
3. ✅ Agent returns to Route 101 after healing
4. ✅ Map revisit count decreases (Brendan's House < 10 visits)
5. ✅ Episodic memory stores healing patterns
6. ✅ VLM provides healing guidance in prompts
7. ✅ Battle tracker recommends healing for dangerous areas
8. ✅ Agent completes "LEAVE_LAB_TO_ROUTE_101" milestone

## Files to Create/Modify

1. **NEW**: `agent/healing_tracker.py` - HealingLocationTracker class
2. **MODIFY**: `train_rl_vlm_screen_aware.py`:
   - Import healing tracker
   - Add map_coord_visits tracking
   - Update reward calculation
   - Update VLM prompt generation
3. **MODIFY**: `battle_experience_tracker.py`:
   - Add should_heal_before_area() method
4. **MODIFY**: `agent/episodic_memory.py`:
   - Add healing experience storage (already has structure)

## Checkpoint Location

Current training saves to:
```
./checkpoints_server_aware/server_aware_model_10000_steps.zip
./checkpoints_server_aware/server_aware_model_20000_steps.zip
...
```

After 2M training completes, resume with:
```bash
python train_rl_vlm_screen_aware.py \
    --resume ./checkpoints_server_aware/server_aware_model_2000000_steps.zip \
    --timesteps 3000000
```

Then apply these healing/exploration improvements.
