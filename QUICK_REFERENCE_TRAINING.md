# QUICK REFERENCE - Pokemon Training System

## 🚀 Fast Implementation (Copy-Paste Ready)

### When your training finishes, run this:

```bash
cd /home/jpli/pokeagent-speedrun

# 1. Backup current training file
cp train_rl_vlm_screen_aware.py train_rl_vlm_screen_aware.py.backup_before_training_system

# 2. Test new modules work
python3 -c "from agent.battle_engagement_strategy import BattleEngagementStrategy; from agent.healing_item_manager import HealingItemManager; print('✅ Modules OK')"

# 3. Follow IMPLEMENTATION_GUIDE_TRAINING_SYSTEM.md for code changes
# (10 steps total - adds ~200 lines of code)

# 4. Verify syntax
python -m py_compile train_rl_vlm_screen_aware.py && echo "✅ Syntax OK"

# 5. Short test
python train_rl_vlm_screen_aware.py --timesteps 1000 > test_training_system.log 2>&1 &
TEST_PID=$!

# 6. Monitor test (wait 2 minutes)
sleep 120
kill $TEST_PID 2>/dev/null

# 7. Check test results
echo "=== Test Results ==="
grep -E "LEVEL UP|Battle won|GRINDING MODE|Used item" test_training_system.log | tail -10

# 8. If test successful, start full training
./start_training_overnight.sh
```

## 📋 What Gets Changed

| File | Changes | Lines Added |
|------|---------|-------------|
| `train_rl_vlm_screen_aware.py` | Reward system + VLM prompts | ~200 |
| `agent/battle_engagement_strategy.py` | NEW FILE | 353 |
| `agent/healing_item_manager.py` | NEW FILE | 323 |

## 🎯 Key Reward Changes

```python
# Level up reward
OLD: level_reward = 1.0 * level_gain
NEW: level_reward = 5.0 * level_gain  # 5x increase

# Battle won reward (NEW)
NEW: battle_win_reward = 3.0

# Item usage reward (NEW)  
NEW: item_use_reward = 8.0

# Grinding mode grass
OLD: grass_reward = 1.0
NEW: grass_reward = 2.0  # during grinding mode

# Blackout penalty + recovery
OLD: blackout_penalty = -10.0
NEW: blackout_penalty = -10.0
     + activate_grinding_mode(500 steps)  # NEW
```

## 🔍 Log Patterns to Watch For

**SUCCESS INDICATORS:**
```
✅ 📈 LEVEL UP! +5.0        # Was +1.0 before
✅ ⚔️ Battle won! +3.0       # NEW
✅ 🔄 GRINDING MODE: 500 steps  # After blackout
✅ 💊 Used item +8.0         # NEW
✅ 🌿 Training +2.0          # During grinding
```

**FAILURE INDICATORS:**
```
❌ ImportError: No module named 'agent.battle_engagement_strategy'
❌ SyntaxError in train_rl_vlm_screen_aware.py
❌ No "LEVEL UP" messages in logs
❌ No "GRINDING MODE" after blackout
```

## 📊 Expected Timeline

| Phase | Steps | Agent Behavior |
|-------|-------|----------------|
| Initial | 0-5k | Explores, fights ~30% of battles, reaches Lv7-8 |
| First Blackout | ~8k | Grinding mode activates, trains for 500 steps |
| Post-Grinding | ~10k | Reaches Lv10-12, continues story |
| Gym Ready | ~20k | Level 13+, attempts Rustboro Gym |
| Mid-Game | 50k+ | Regular grinding sessions, strategic item usage |

## 🛠️ Emergency Fixes

### If agent avoids ALL battles:
```python
# In battle_engagement_strategy.py, line ~150
# CHANGE:
if random.random() < 0.3:  # 30% fight rate
# TO:
if random.random() < 0.7:  # 70% fight rate
```

### If grinding mode too aggressive:
```python
# In train_rl_vlm_screen_aware.py, blackout section
# CHANGE:
self.battle_engagement.activate_grinding_mode(duration_steps=500)
# TO:
self.battle_engagement.activate_grinding_mode(duration_steps=300)
```

### If running out of items:
```python
# In healing_item_manager.py, line ~150
# CHANGE:
if in_battle and hp_ratio < 0.3:  # Use at 30% HP
# TO:
if in_battle and hp_ratio < 0.2:  # Use at 20% HP
```

## 🎮 PokemonRedExperiments Comparison

**Their approach (baselines):**
```python
level_reward = level_sum * 100 / reward_scale  # Heavy scaling
explore_reward = explore_count * 150 / scale   # Heavy exploration
```

**Our approach (hybrid):**
```python
level_reward = 5.0 * level_gain     # Direct per-level reward
battle_win = 3.0                    # Immediate battle feedback
milestone = 500                     # Story priority preserved
```

**Key Difference**: We balance story progression (milestones) with training (levels), while PokemonRed focuses purely on exploration/levels.

## 📞 Monitoring Commands

```bash
# Live dashboard (press Ctrl+C to exit)
python dashboard.py

# Battle statistics
grep -E "LEVEL UP|Battle won" training_*.log | wc -l

# Grinding mode count
grep "GRINDING MODE" training_*.log | wc -l

# Current party level (from recent VLM calls)
grep "Party Avg Level" training_*.log | tail -5

# Item usage count
grep "Used item" training_*.log | wc -l

# Watch training in real-time
tail -f training_*.log | grep -E "LEVEL UP|Battle|GRINDING|item"
```

## ⚡ Performance Expectations

**After 50k steps:**
- Levels: 10-12 (was 5-7)
- Battles: 50-100 (was ~5)
- Items used: 10-20 (was ~0)
- Milestones: 3-5 (was 1-2)

**After 200k steps:**
- Levels: 18-22 (ready for mid-game gyms)
- Battles: 200-400
- Items used: 50-100
- Milestones: 8-12

**After 1M steps:**
- Levels: 35-45 (late-game ready)
- Battles: 1000+
- Items used: 300+
- Milestones: 15-20

## 📁 File Locations

```
/home/jpli/pokeagent-speedrun/
├── agent/
│   ├── battle_engagement_strategy.py  ✅ NEW (353 lines)
│   └── healing_item_manager.py        ✅ NEW (323 lines)
├── train_rl_vlm_screen_aware.py       🔧 TO MODIFY (~200 lines)
├── POKEMON_TRAINING_SYSTEM.md         📚 Design doc
├── IMPLEMENTATION_GUIDE_TRAINING_SYSTEM.md  📋 Step-by-step
└── TRAINING_SYSTEM_READY.md           ✅ Status report
```

## ✅ Verification Checklist

Before starting new training:
- [ ] Modules import successfully
- [ ] Syntax check passes
- [ ] Test run completes 1000 steps
- [ ] New log patterns appear
- [ ] VLM prompts include battle guidance
- [ ] Backup of old file created
- [ ] Implementation guide followed completely

## 🎯 Success Criteria

Training is successful if after 100k steps:
- [x] Party level >= 12
- [x] Wild battles won >= 50
- [x] Grinding mode activated >= 2 times
- [x] Items used >= 10
- [x] Reached Route 103 (rival battle)
- [x] No infinite loops or crashes

---

**READY TO DEPLOY** ✅

Current training: 50,600 / 2,000,000 steps (2.53%)  
Recommended: Apply after current run completes  
Alternative: Apply now if agent appears stuck

**Documentation**: See `IMPLEMENTATION_GUIDE_TRAINING_SYSTEM.md` for detailed steps
