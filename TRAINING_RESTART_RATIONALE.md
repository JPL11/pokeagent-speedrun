# Training Restart - Why and What Changed

**Date**: November 19, 2025  
**Old Run**: Started Nov 17, stuck in May's house for 40,000+ steps  
**Decision**: STOPPED - Restart with enhanced code

---

## Why Stop the Old Training?

### Problem: Agent Stuck in Loop
- **Location**: May's house (rival's house)
- **Duration**: 40,000+ steps in same location
- **VLM Issue**: Hallucinating that May might be there
- **Reality**: NPC (May's mom) explicitly says "May is on Route 103"
- **Progress**: Only 62,480 / 2,000,000 steps (3.12%)

### Root Causes Identified:
1. **No confidence signal**: k-NN provided recalls without indicating familiarity
2. **Placeholder zeros**: milestone and SLAM context were not real state
3. **No context awareness**: Mixed navigation/battle/story memories together
4. **VLM ignoring NPCs**: Strategic layer hallucinating despite dialogue

---

## What's New in Enhanced Training?

### 1. k-NN Confidence Gating ★★★
**Before**:
```python
augmentation = episodic_memory.get_episodic_augmentation(...)
# Policy has no idea if this recall is reliable
```

**After**:
```python
augmentation, confidence = episodic_memory.get_episodic_augmentation_with_confidence(...)
# confidence in [0, 1]: 1.0 = very familiar, 0.0 = novel
# Policy can down-weight uncertain recalls
```

**Impact**: Agent knows when it's in a familiar stuck state vs genuinely new situation

---

### 2. Real Milestone State (TODO #1 Fixed)
**Before**:
```python
story_vec = np.zeros(16)  # TODO: get from milestone tracker
```

**After**:
```python
story_vec = np.array(milestone_tracker.get_vector()[:16], dtype=np.float32)
# Real progress flags, level gates, quest states
```

**Impact**: k-NN matches consider actual story progress, not generic zeros

---

### 3. Real SLAM Waypoints (TODO #2/#3 Fixed)
**Before**:
```python
waypoint_hint = np.zeros(8)  # TODO: get from SLAM
```

**After**:
```python
waypoint_hint = get_waypoint_hint_from_slam(slam_mapper, map_id, x, y)
# [dx, dy, distance, on_path, path_exists, progress, at_goal, confidence]
```

**Impact**: k-NN considers spatial context - "stuck in house" vs "exploring route"

---

### 4. Context-Aware Memory Split
**Before**: Single memory buffer mixing everything

**After**: Separate buffers by domain
```python
self.nav_memory = deque(maxlen=10000)     # Navigation only
self.battle_memory = deque(maxlen=10000)   # Battle only  
self.quest_memory = deque(maxlen=10000)    # Story only
```

**Impact**: Cleaner retrieval, no mixing "walk around house" with "battle wild Pokemon"

---

### 5. SNN INPUT 11 Enhanced Structure
**Before**: 32-dim generic episodic recall

**After**: 32-dim with confidence at index [10]
```
[0-7]:   Action distribution from k nearest neighbors
[8]:     Average reward from similar states
[9]:     Milestone likelihood
[10]:    ★ k-NN confidence (0.0-1.0) NEW
[11-31]: Domain-specific features
```

**Impact**: Policy learns to weight k-NN suggestions by confidence

---

## Expected Improvements

### Short-term (First 100k steps):
- **Faster unstuck**: High confidence on repeated states → try different actions
- **Better exploration**: Low confidence → ignore k-NN, explore freely
- **Fewer hallucinations**: Real milestone state helps VLM stay grounded

### Medium-term (100k-500k steps):
- **Smarter routing**: Real SLAM waypoints improve navigation recalls
- **Domain separation**: Battle strategies don't pollute navigation memory
- **Milestone awareness**: k-NN suggests actions that worked at similar story points

### Long-term (500k-2M steps):
- **Graceful novelty handling**: Confidence gating enables smooth exploration
- **Faster learning**: Better recalls accelerate policy improvement
- **Publication ready**: Novel confidence-aware episodic memory system

---

## How to Start Enhanced Training

### Option 1: Interactive (recommended for monitoring)
```bash
./start_enhanced_training.sh
```

### Option 2: Background (for overnight runs)
```bash
nohup ./start_enhanced_training.sh > training.log 2>&1 &
```

### Option 3: Direct command
```bash
source .venv/bin/activate
python train_rl_vlm_screen_aware.py --timesteps 2000000 --vlm-frequency 25
```

---

## Monitoring the New Run

### Check training progress:
```bash
tail -f training_enhanced_*.log
```

### Watch TensorBoard (if logging enabled):
```bash
tensorboard --logdir=checkpoints_2M/tensorboard
```

### Check server state:
```bash
curl -s http://localhost:8000/state | python3 -c "import sys, json; d=json.load(sys.stdin); print('Map:', d['player']['map_id'], 'Pos:', d['player']['x'], d['player']['y'], 'Location:', d['player']['location'])"
```

### Monitor k-NN confidence (add this to training if desired):
```python
# In training loop:
if step % 100 == 0:
    print(f"k-NN confidence: {confidence:.3f}")
```

---

## Success Criteria

### ✅ Training is working well if:
- Agent leaves May's house within 1,000 steps
- Reaches Route 103 to find May (first rival battle)
- k-NN confidence increases over time (learning)
- No 40,000-step stuck loops

### ⚠️ May need adjustments if:
- Still stuck in houses after 5,000 steps
- k-NN confidence always near 0 (not learning from experience)
- VLM still hallucinating despite real milestone state

### ❌ Stop and debug if:
- Crashes or OOM errors
- k-NN confidence always 1.0 (overfitting to stuck states)
- Agent actively avoids Route 103 despite being told to go there

---

## Rollback Plan (if needed)

If enhanced training performs worse:

1. **Disable confidence gating**: Use old `get_episodic_augmentation()` API
2. **Revert TODOs**: Put back `np.zeros()` placeholders
3. **Merge memory buffers**: Single deque instead of 3 separate ones
4. **Restore from checkpoint**: Use Nov 14-15 checkpoints if needed

---

## Next Steps After This Training

1. **Ablation study**: Compare k-NN on vs off
2. **Confidence analysis**: Plot confidence trends vs milestone success
3. **Add TensorBoard logging**: Track k-NN metrics
4. **VLM prompt tuning**: Emphasize NPC dialogue more strongly
5. **Publication prep**: Document confidence-aware episodic memory

---

**Ready to start?** Run: `./start_enhanced_training.sh`
