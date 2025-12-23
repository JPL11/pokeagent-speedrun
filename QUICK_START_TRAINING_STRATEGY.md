# Quick Start: Pokemon Training Strategy

## What Changed?

Agent now has **strategic battle intelligence**:
- **Grinds** when party is under-leveled for next milestone
- **Progresses** story when party is strong enough
- **Heals** when HP is low
- **Recovers** after blackouts

## Three-Tier Architecture

```
┌─────────────────────────────────────────┐
│  VLM (High-Level Strategic Reasoning)   │ ← "Should I grind?"
│  Sets: GRIND / PROCEED / HEAL / etc.    │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│  SNN (Temporal Pattern Learning)        │ ← "How do I grind?"
│  Learns: Movement patterns, timing      │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│  Rewards (Context-Aware Feedback)       │ ← "Grass good during GRIND"
│  Adjusts: Based on current directive    │
└─────────────────────────────────────────┘
```

## Strategic Directives

| Directive | When Active | Agent Behavior |
|-----------|-------------|----------------|
| **GRIND** | Party level < milestone requirement | Seek grass, fight battles, gain levels |
| **PROCEED** | Party level >= milestone requirement | Focus on story, avoid unnecessary battles |
| **HEAL** | Party HP < 40% | Go to Pokemon Center or use items |
| **POST_BLACKOUT** | Just lost battle | Cautious exploration, frequent healing |
| **PREPARE_GYM** | Before gym battle | Stock items, verify levels |
| **GYM_BATTLE** | During gym | Full battle focus |
| **OPTIONAL_GRIND** | Over-leveled but can train | Balanced approach |

## Reward Modulation

### Level Up Rewards
- GRIND: **8.0x** (highest priority)
- PROCEED: **3.0x** (still valued)
- Default: **5.0x**

### Battle Win Rewards
- GRIND: **+4.0** (encourage training)
- PROCEED: **+1.0** (low priority)
- Default: **+2.0**

### Grass Tile Rewards
- GRIND: **+2.5** (seek encounters)
- PROCEED: **+0.2** (avoid unnecessary battles)
- Default: **+1.0**

### Special Rewards
- Healing item used: **+8.0**
- Strategic goal complete: **+50.0**

## Milestone Level Requirements

```python
GOT_STARTER        →  Level 5
REACHED_OLDALE     →  Level 7
REACHED_PETALBURG  →  Level 10
RUSTBORO_GYM       →  Level 13
RIVAL_BATTLE_2     →  Level 16
MAUVILLE_GYM       →  Level 22
...
ELITE_FOUR         →  Level 50
```

## Example Training Flow

```
Step 1-1000: Get starter (Lv5)
  → GRIND mode (target: Lv7)
  → Fight wild Pokemon in Route 101
  → Reach Lv7
  
Step 1001-3000: PROCEED mode
  → Navigate to Oldale Town
  → Complete REACHED_OLDALE milestone
  → GRIND mode (target: Lv10)
  
Step 3001-5000: Train in Route 102
  → Fight Poochyena, Zigzagoon
  → Reach Lv10
  → PROCEED mode
  
Step 5001-8000: Navigate to Petalburg
  → Story progression
  → GRIND mode (target: Lv13 for gym)
  
... continues ...
```

## VLM Prompt Enhancement

Agent now sees strategic guidance:

**During GRIND**:
```
🔥 **GRINDING MODE ACTIVE** - Train Pokemon for next challenge!
📊 Current Level: 6.0 | Target: 10
⏱️ Grinding Session: 250 steps
💪 Progress: 3 battles, 1 levels gained

**TRAINING INSTRUCTIONS:**
1. Walk through TALL GRASS to encounter wild Pokemon
2. FIGHT wild battles - don't run!
3. Use POTIONS if HP drops below 50%
```

**During PROCEED**:
```
✅ **PARTY READY** - Strong enough for next challenge!
Level: 10.2 | HP: 85%

**FOCUS ON STORY PROGRESSION:**
• Continue to next milestone objective
• Fight wild battles occasionally (30%)
• Escape from weak wild Pokemon
```

## Files Modified

- `train_rl_vlm_screen_aware.py` - Main training loop
- `agent/strategic_directive_manager.py` (NEW)
- `agent/battle_engagement_strategy.py` (NEW)
- `agent/healing_item_manager.py` (NEW)

## When to Deploy

✅ Code is ready NOW  
⏳ Wait for current 2M training to complete (currently 2.53%)  
🚀 Then start new training with this system

## How to Start New Training

```bash
# After current training completes
python train_rl_vlm_screen_aware.py --steps 2000000
```

## What to Monitor

Look for these in logs:
- `🎯 STRATEGIC DIRECTIVE UPDATED: GRIND`
- `🔥 GRINDING MODE ACTIVE`
- `⚔️ BATTLE WON! (+4.0 reward during GRINDING)`
- `✅ STRATEGIC GOAL COMPLETE!`

## Expected Results

Agent should:
1. Train Pokemon before challenging gyms
2. Complete milestones in order
3. Use healing items appropriately
4. Recover from blackouts intelligently

## Why This Architecture?

User correctly identified that **battle strategy = high-level reasoning**, not hardcoded rules.

**Before**: "Always avoid battles" (procedural)  
**After**: "Grind when needed, progress when ready" (strategic)

This matches how humans play Pokemon:
- Under-leveled? Grind in grass.
- Strong enough? Progress story.
- Low HP? Heal first.

## Design Philosophy

> "The VLM decides WHEN to grind (strategic goal).  
> The SNN learns HOW to grind efficiently (temporal patterns).  
> The rewards provide feedback signals (reinforcement)."

Three tiers, clean separation of concerns, feedback loops throughout.
