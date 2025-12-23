# 🚀 START TRAINING NOW - Complete Guide

## ⚠️ IMPORTANT: You Need a ROM First!

The RL training requires a Pokemon Emerald ROM file.

### Quick Setup:

```bash
cd /home/jpli/pokeagent-speedrun

# 1. Place your Pokemon Emerald ROM here
# Name it: emerald.gbc (or emerald.gba)

# 2. Verify ROM exists
ls -lh *.gbc *.gba 2>/dev/null

# 3. Start training overnight!
./train_rl_now.sh
```

---

## 🎮 If You Have the ROM:

### Option 1: Full Training (Overnight - 6-12 hours)

```bash
# Terminal 1: Start training
nohup python train_rl.py --timesteps 10000000 --use_snn --rom emerald.gbc > training.log 2>&1 &

# Terminal 2: Monitor progress
tail -f training.log

# Terminal 3: Watch tensorboard
tensorboard --logdir .rl_checkpoints/tensorboard
```

### Option 2: Test Run (Quick - 30 minutes)

```bash
# Test with fewer timesteps
python train_rl.py --timesteps 100000 --use_snn --rom emerald.gbc
```

### Option 3: Use Existing VLM System (Works Now!)

```bash
# This already works - uses VLM + heuristics
# (No learning, but demonstrates the game)

# Terminal 1: Start server
python -m server.app

# Terminal 2: Start client
python run.py --headless --vlm_backend qwen
```

---

## 📊 While Training, You Can:

1. **Monitor progress:**
   ```bash
   tail -f training.log
   ```

2. **View tensorboard:**
   ```bash
   tensorboard --logdir .rl_checkpoints/tensorboard
   # Open: http://localhost:6006
   ```

3. **Check checkpoints:**
   ```bash
   ls -lh .rl_checkpoints/
   ```

4. **Resume if interrupted:**
   ```bash
   python train_rl.py --resume .rl_checkpoints/pokemon_emerald_rl_500000_steps.zip --timesteps 10000000
   ```

---

## 🎯 What Happens During Training:

```
Hour 1:  Agent explores randomly, learns basics
Hour 2:  Starting to avoid obvious loops
Hour 4:  Efficient exploration emerges
Hour 6:  Seeking events and progress
Hour 12: Near-optimal behavior learned!
```

---

## 🔧 If You Don't Have a ROM:

You can still use the existing VLM system that's already working:

```bash
# Works without ROM (uses existing checkpoints)
python run.py --headless --vlm_backend qwen
```

This uses the VLM + heuristic system (not RL learning, but functional).

---

## ✅ After Training Completes:

```bash
# Run your trained agent!
python run_rl.py --checkpoint .rl_checkpoints/pokemon_emerald_final.zip

# Or with VLM reasoning overlay:
python run_rl.py --checkpoint .rl_checkpoints/pokemon_emerald_final.zip --vlm_backend qwen
```

---

## 📝 Training Output Example:

```
🚀 Starting RL training with PPO + SNN features
✅ PPO agent created with SNN feature extractor
🎓 Starting training...

Episode 1    | Reward: -20.5  | Steps: 1234
Episode 10   | Reward: 45.2   | Steps: 2341  ← Learning!
Episode 50   | Reward: 180.4  | Steps: 3452  ← Getting better!
Episode 100  | Reward: 350.1  | Steps: 4123  ← Learned!

💾 Checkpoint saved: .rl_checkpoints/pokemon_emerald_rl_50000_steps.zip
...
```

---

## 🎓 The Learning Process:

While training, the agent will:
1. **Explore** the game world
2. **Get rewards** for new coordinates, events, dialogue
3. **Get penalties** for loops, sign spam
4. **Update weights** via gradient descent
5. **Improve** over time (REAL LEARNING!)

---

## 🆘 Troubleshooting:

### "ROM not found"
→ Place your Pokemon Emerald ROM in the project directory

### "CUDA out of memory"
→ Use CPU training: `export CUDA_VISIBLE_DEVICES=""`

### "Training is slow"
→ Reduce timesteps for testing: `--timesteps 100000`

### "Want to see it running now"
→ Use existing VLM system: `python run.py --headless --vlm_backend qwen`

---

## 💤 Sleep Well!

The training will run overnight. When you wake up:

```bash
# Check results
cat training.log | tail -50

# Run trained agent
python run_rl.py --checkpoint .rl_checkpoints/pokemon_emerald_final.zip
```

**Your agent will have LEARNED from experience!** 🧠✨

---

Generated: $(date)

