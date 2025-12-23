# 🎮 How to Watch Your Agent Play

## Three Ways to Visualize

### Option 1: Current VLM System (Works Now!)

The server/client you already have shows graphics:

```bash
# Terminal 1: Start server
cd /home/jpli/pokeagent-speedrun
python -m server.app

# Terminal 2: Start client with graphics
python run.py --vlm_backend qwen
```

**What you see:**
- Game screen updates in real-time
- VLM reasoning + heuristics (not RL learning, but functional)
- Can watch it explore and play

---

### Option 2: Watch Trained RL Agent (After Training Completes)

Once RL training finishes, watch your trained agent:

```bash
# After training completes:
python run_rl_with_graphics.py --checkpoint .rl_checkpoints/pokemon_emerald_final.zip
```

**What you see:**
- Emulator window pops up
- Trained RL policy playing
- LEARNED behavior (not hard-coded!)
- Much smarter than VLM

---

### Option 3: Monitor Training Progress (No Graphics)

While training, you can't see graphics (too slow), but you can monitor:

```bash
# Watch training log
tail -f logs/training_*.log

# Watch tensorboard (graphs, metrics)
tensorboard --logdir .rl_checkpoints/tensorboard
# Open: http://localhost:6006
```

**What you see:**
- Episode rewards increasing over time
- Policy loss decreasing
- Value function improving
- Learning curves!

---

## 🎯 Recommended Flow

**Right Now (Before Training):**
```bash
# Watch current VLM system play
python -m server.app  # Terminal 1
python run.py --vlm_backend qwen  # Terminal 2
```

**During Training (6-12 hours):**
```bash
# Monitor learning progress
tail -f logs/training_*.log
tensorboard --logdir .rl_checkpoints/tensorboard
```

**After Training:**
```bash
# Watch TRAINED agent (much smarter!)
python run_rl_with_graphics.py --checkpoint .rl_checkpoints/pokemon_emerald_final.zip
```

---

## 📊 Tensorboard Metrics to Watch

When training, tensorboard shows:

- **rollout/ep_rew_mean** - Average episode reward (should increase!)
- **train/policy_loss** - Policy loss (should stabilize)
- **train/value_loss** - Value function loss (should decrease)
- **rollout/ep_len_mean** - Episode length

**Increasing rewards = Agent is learning!** 📈

---

## 🔍 What's the Difference?

| System | Learning? | Speed | Graphics |
|--------|-----------|-------|----------|
| **VLM + Heuristics** | ❌ No | Slow | ✅ Yes |
| **RL Training** | ✅ YES! | Fast | ❌ No (headless) |
| **Trained RL Agent** | Already learned | Fast | ✅ Yes |

**The best experience:** Watch the trained RL agent after training completes. It will be MUCH smarter because it learned optimal play!

---

Generated: $(date)
