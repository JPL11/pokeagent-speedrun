# 🚀 Quick Start: Real RL Learning

## What You Get

✅ **REAL learning** - Policy network that improves with experience  
✅ **No more repeated mistakes** - Gradient descent updates weights  
✅ **SNN encoder integration** - Temporal features for better decisions  
✅ **VLM guidance** - Optional high-level strategic advice  
✅ **Competition ready** - Fast inference, no API calls  

---

## Installation

```bash
# 1. Install RL dependencies
pip install -r requirements_rl.txt

# Or install manually:
pip install stable-baselines3 torch tensorboard
```

---

## Usage

### Option 1: Train from Scratch (Recommended)

```bash
# Basic training (6-12 hours on GPU)
python train_rl.py --timesteps 10000000

# With SNN features (default, better performance)
python train_rl.py --timesteps 10000000 --use_snn

# With VLM guidance (slower but smarter)
python train_rl.py --timesteps 10000000 --vlm_guidance --vlm_backend qwen
```

**What happens during training:**
- Agent explores the game randomly at first
- Receives rewards for new coordinates, events, dialogues
- Gets penalties for loops, sign spam, animation farming
- Policy network weights update via gradient descent
- After millions of steps, learns optimal behavior

**Monitor progress:**
```bash
tensorboard --logdir .rl_checkpoints/tensorboard
```

### Option 2: Run Pre-trained Agent

```bash
# If you have a trained checkpoint
python run_rl.py --checkpoint .rl_checkpoints/pokemon_emerald_final.zip

# With VLM reasoning overlay (hybrid mode)
python run_rl.py --checkpoint .rl_checkpoints/pokemon_emerald_final.zip --vlm_backend qwen

# Visualize the agent playing
python run_rl.py --checkpoint .rl_checkpoints/pokemon_emerald_final.zip --render
```

### Option 3: Resume Training

```bash
# Continue training from a checkpoint
python train_rl.py --resume .rl_checkpoints/pokemon_emerald_rl_500000_steps.zip --timesteps 5000000
```

---

## Key Differences from Old Approach

### ❌ Old (VLM + Heuristics)

```python
# Every step:
action = vlm.decide(frame)  # Expensive API call
                            # No learning
                            # Repeats same mistakes

# When stuck:
if at_town_sign and pressed_a:
    hardcoded_escape()  # Specific to this one case
```

**Problems:**
- Doesn't learn
- Expensive ($$ per API call)
- Hard-coded solutions
- Brittle

### ✅ New (RL + SNN + VLM)

```python
# Training (once):
for 10_000_000 steps:
    action = policy(state)
    reward = env.step(action)
    policy.learn(reward)  # UPDATES WEIGHTS

# Inference (fast):
action = policy(state)  # Learned, no API
```

**Benefits:**
- **Learns general principles**: "Signs → A spam → bad"
- **Fast**: No API calls
- **Adapts**: Works in new situations
- **Gets better**: Improves over time

---

## Example: Learning to Avoid Sign Spam

### Without Learning (Old):
```
Step 100:  Agent presses A on sign → Stuck → Manual escape
Step 500:  Agent presses A on SAME sign → Stuck again! (No learning)
Step 1000: Still making same mistake...
```

### With Learning (New):
```
Episode 1:   Press A on sign → -2.0 penalty → Update policy
Episode 10:  Try A on sign → -2.0 penalty → Update more
Episode 50:  Policy learns: "A near signs = bad, avoid!"
Episode 100: Never spam signs anymore! (Generalized!)
```

---

## Training Timeline

| Time | Progress |
|------|----------|
| 0 min | Install dependencies |
| 5 min | Start training |
| 1 hour | Agent starts avoiding obvious loops |
| 4 hours | Efficient exploration emerges |
| 8 hours | Near-optimal behavior |
| 12 hours | Training complete, save checkpoint |

**Total: ~12 hours** for a fully trained agent that **actually learns**!

---

## File Structure

```
pokeagent-speedrun/
├── agent/
│   └── rl_trainer.py          # RL environment + PPO training
├── train_rl.py                # Training script
├── run_rl.py                  # Run trained agent
├── requirements_rl.txt        # RL dependencies
├── .rl_checkpoints/           # Saved models (created during training)
│   ├── pokemon_emerald_rl_50000_steps.zip
│   ├── pokemon_emerald_rl_100000_steps.zip
│   └── pokemon_emerald_final.zip
└── RL_LEARNING_README.md      # Full documentation
```

---

## Monitoring Training

### Tensorboard

```bash
tensorboard --logdir .rl_checkpoints/tensorboard
```

Open http://localhost:6006 to see:
- Episode rewards (should increase)
- Policy loss (should stabilize)
- Value loss (should decrease)
- Exploration stats

### Console Output

```
Step 1000   | Reward: 45.2   | Location: LITTLEROOT | Action: UP
Step 2000   | Reward: 120.5  | Location: ROUTE_101  | Action: RIGHT
Step 3000   | Reward: 250.3  | Location: OLDALE     | Action: A
...
🎯 New milestone achieved! Reward: +10.0
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'stable_baselines3'"

```bash
pip install stable-baselines3
```

### "CUDA out of memory"

Train on CPU (slower but works):
```bash
# Force CPU
export CUDA_VISIBLE_DEVICES=""
python train_rl.py --timesteps 10000000
```

### "Training is slow"

- Use fewer timesteps for testing: `--timesteps 1000000`
- Disable VLM guidance: remove `--vlm_guidance`
- Use GPU if available

### "Agent not improving"

- Check tensorboard logs
- Tune reward weights in `RLConfig`
- Increase training timesteps
- Try different learning rates

---

## Next Steps

1. **Train the agent** (6-12 hours)
   ```bash
   python train_rl.py --timesteps 10000000 --use_snn
   ```

2. **Monitor progress**
   ```bash
   tensorboard --logdir .rl_checkpoints/tensorboard
   ```

3. **Test trained agent**
   ```bash
   python run_rl.py --checkpoint .rl_checkpoints/pokemon_emerald_final.zip
   ```

4. **Submit to competition**
   - The trained model is in `.rl_checkpoints/pokemon_emerald_final.zip`
   - No API dependencies!
   - Fast inference!

---

## 🎓 This is REAL Learning!

- **Policy network learns** through gradient descent
- **No more hard-coded heuristics**
- **Generalizes to new situations**
- **Gets better with more training**

**Stop making the same mistakes! Train once, play smart forever!** 🚀

