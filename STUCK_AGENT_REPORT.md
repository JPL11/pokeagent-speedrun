# 🚨 CRITICAL TRAINING ISSUE REPORT

## Summary
**Agent has been stuck in May's House 1F for 448,000+ steps (6+ days / 148 hours runtime)**

---

## Current State

### Location & Stats
- **Location**: `LITTLEROOT TOWN MAYS HOUSE 1F`
- **Position**: `(9, 3)` to `(10, 7)` - wandering around inside
- **Pokemon**: Torchic Level 14 (41/41 HP - Full Health)
- **Money**: $5
- **Runtime**: 148:10:00 (over 6 days)
- **Total Steps**: 448,700+

### Milestones Completed (15 total)
✅ Last 5 completed:
1. STARTER_CHOSEN
2. BIRCH_LAB_VISITED  
3. OLDALE_TOWN
4. ROUTE_103
5. **RECEIVED_POKEDEX** ✅

**The agent has ALREADY completed the Pokedex milestone!**

### Training Metrics
- **Backtrack Ratio**: 78.1% (extremely high - agent constantly backtracking)
- **Explore Ratio**: 0.5% (almost zero exploration)
- **Error Rate**: 31.9%
- **Average Decision Time**: 1.3 seconds

---

## Root Cause Analysis

### Problem 1: VLM Hallucination 🤖💭
The VLM is **completely confused** about game state:

```
VLM Response (repeated 1000s of times):
"LOCATION: MAYS_HOUSE_1F - Birch is inside, looking out the window.
ACTION: DOWN
REASON: Birch is likely to be in the house..."
```

**Reality**: 
- Birch is NOT in May's house
- Agent already GOT the Pokedex from Birch
- Agent should be progressing to next milestone (likely PETALBURG_CITY or FIRST_GYM)

### Problem 2: Agent Never Left Town After Blackout 😵
Timeline:
1. Agent went to Route 103 ✅
2. Agent presumably battled and blacked out (lost all money: $5 remaining)
3. Agent respawned in Littleroot/Oldale Pokemon Center
4. Agent received Pokedex (milestone completed) ✅
5. **Agent got stuck in May's House and never progressed**

### Problem 3: Reward Function Not Penalizing Loops 🔄
- Agent is stuck in May's House for 448,000 steps
- Reward function gives small positive rewards for:
  - Dialog interactions (+2.0)
  - Novel views (+2.0)
  - VLM following (+5.0)
- **No strong penalty for being stuck in same location**

---

## Why VLM Is Confused

### Context The VLM Receives:
```python
**📖 STORY CONTEXT:**
⚡ **PRIORITY DIRECTIVE**: Story progress is TOP PRIORITY! Battles may cause 
blackouts, but exploring new areas and completing milestones is MORE IMPORTANT 
than avoiding risk. If you need to go through Route 101 to progress, DO IT 
even if you might faint.

**🎯 CURRENT MILESTONE GOAL: RECEIVED_POKEDEX**
Why: Birch will give you the Pokedex now that you've proven yourself
```

**The Problem**: 
- Milestone list shows `RECEIVED_POKEDEX` as completed
- But VLM prompt still says goal is `RECEIVED_POKEDEX`
- VLM doesn't understand it's already past this milestone
- VLM thinks Birch is somewhere nearby (hallucinating based on milestone name)

---

## Immediate Actions Required

### Option 1: Manual Teleport (RECOMMENDED) ⚡
Use `manual.py` to manually navigate agent out:

```bash
source .venv/bin/activate
python manual.py
```

Then manually:
1. Navigate to door (center-bottom of May's House)
2. Exit to Littleroot Town
3. Go north to Route 101
4. Go north through Route 101 to Oldale Town
5. Enter Oldale Town Pokemon Center
6. Save the game state

### Option 2: Load Earlier Savestate 💾
If you have a savestate from before the agent got stuck:

```bash
# Check for savestates
ls -lht Emerald-GBAdvance/test_states/*.state

# Load a savestate (replace with actual filename)
# Then restart training
```

### Option 3: Reset Training With Better Prompts 🔄
Stop current training and restart with fixes:

```bash
# Kill current training
kill $(cat .training.pid)

# Apply fixes to VLM prompt (see fixes below)
# Then restart
./start_enhanced_training.sh
```

---

## Code Fixes Needed

### Fix 1: Update Story Context Logic
**File**: `train_rl_vlm_screen_aware.py` line ~60-200

**Problem**: Story context shows wrong goal after milestone completed

**Fix**: Add logic to detect AFTER RECEIVED_POKEDEX:

```python
# After line 114, add:
if 'RECEIVED_POKEDEX' in completed:
    # Check next milestone
    if 'PETALBURG_CITY' not in completed:
        return {
            'story': "You have the Pokedex! Head west to Petalburg City to meet your father Norman (the Gym Leader).",
            'goal': "PETALBURG_CITY",
            'why': "Your father is the Gym Leader - he can teach you about gym battles"
        }
    elif 'DELIVERED_LETTER' not in completed:
        return {
            'story': "In Petalburg, deliver Birch's letter to your father Norman.",
            'goal': "DELIVERED_LETTER",
            'why': "Norman can give you guidance after receiving the letter"
        }
    # Add more progression logic...
```

### Fix 2: Add Stronger Stuck Detection Penalty
**File**: `train_rl_vlm_screen_aware.py` line ~1800-2000 (reward calculation)

**Add**:
```python
# Penalize being in same small room for too long
if map_id in [0x102, 0x103]:  # May/Brendan houses
    time_in_location = steps_since_map_change
    if time_in_location > 500:  # More than 500 steps in house
        reward -= 10.0  # Strong penalty
        logger.warning(f"🏠 STUCK IN HOUSE PENALTY: -{10.0} (been here {time_in_location} steps)")
```

### Fix 3: Add VLM Hallucination Detection
**File**: `train_rl_vlm_screen_aware.py` line ~700-800 (VLM response processing)

**Add**:
```python
# Detect if VLM is hallucinating NPCs not in current location
if "MAYS_HOUSE" in location_name or "BRENDANS_HOUSE" in location_name:
    if "BIRCH" in vlm_response.upper() and "BIRCH" not in location_name:
        logger.warning("🤖 VLM HALLUCINATION DETECTED: Birch not in this location!")
        # Override with rule-based guidance
        vlm_response = "LOCATION: House interior\\nOBJECTIVE: Exit to continue journey\\nACTION: DOWN\\nREASON: Need to leave house to progress"
```

---

## Prevention For Future

### 1. Add Milestone Progression Checker
Monitor if agent is stuck on wrong milestone goal:

```python
def check_milestone_confusion(completed, current_goal):
    """Detect if agent is working toward already-completed milestone"""
    if current_goal in completed:
        logger.error(f"❌ MILESTONE CONFUSION: Goal '{current_goal}' already completed!")
        return True
    return False
```

### 2. Add Location Diversity Reward
Reward agent for visiting different maps:

```python
# Track unique maps visited in last 1000 steps
recent_maps = set(last_1000_maps)
diversity_bonus = len(recent_maps) * 0.5
reward += diversity_bonus
```

### 3. Add Auto-Unstuck Intervention
If agent stuck in same location > 1000 steps:

```python
if steps_in_same_location > 1000:
    logger.error(f"🚨 AUTO-UNSTUCK: Agent stuck for {steps_in_same_location} steps")
    # Force a random exploration action
    action = random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])
```

---

## Recommendation

**🎯 IMMEDIATE ACTION**: Use Option 1 (Manual Teleport)

1. Run `python manual.py`
2. Manually navigate agent to Oldale Town
3. Let training resume from there

**Then, when training resumes:**
1. Monitor dashboard for 30 minutes
2. If agent leaves Littleroot ✅ → Let it continue
3. If agent returns to May's House ❌ → Stop and apply Code Fixes

**Long-term:**
- Apply all 3 code fixes above
- Add better milestone progression logic
- Improve VLM prompt to be context-aware
- Add auto-unstuck mechanisms

---

## Files to Review

1. `train_rl_vlm_screen_aware.py` lines 60-200 (story context)
2. `train_rl_vlm_screen_aware.py` lines 1800-2000 (reward calculation)
3. `train_rl_vlm_screen_aware.py` lines 700-800 (VLM response processing)
4. `submission.log` - last 1000 lines to see action patterns

---

## Next Steps Checklist

- [ ] Manual teleport agent out of May's House
- [ ] Verify agent in Oldale Town or Route 101
- [ ] Monitor for 30 minutes to ensure not stuck again
- [ ] If stuck again, stop training and apply code fixes
- [ ] Test fixes with manual.py before restarting long training
- [ ] Update VLM prompt to include "ALREADY COMPLETED" milestone context
- [ ] Add stuck detection with strong penalties
- [ ] Consider reducing VLM frequency if it's causing more confusion than help

---

Generated: 2025-11-21 19:12 UTC
Training Runtime: 148:10:00
Current Step: 448,730+
Status: 🚨 CRITICAL - MANUAL INTERVENTION REQUIRED
