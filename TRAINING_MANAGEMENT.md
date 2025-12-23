# Training Management Guide - Sleep-Safe Training 💤

## Quick Start

### Start Training (Background, Persistent)
```bash
./start_enhanced_training.sh
```
**This will:**
- ✅ Run training in background with `nohup`
- ✅ Continue even if you close terminal
- ✅ Continue even if you logout
- ✅ Continue even if you go to sleep
- ✅ Save PID to `.training.pid` for management

### Check Training Status
```bash
./check_training.sh
```
Shows: PID, memory usage, log file, last entries, uptime

### Monitor Training (Live Dashboard)
```bash
python dashboard.py
```
**New Dashboard Features:**
- 📊 Progress bar with ETA calculation
- 🎮 Recent actions sequence
- 📈 Training stability metrics
- 💾 Memory & log file size
- ⏱️  Uptime tracker
- 🧠 k-NN confidence (if logged)
- 🚫 Stuck detection count
- ⚡ FPS/speed metrics
- 🏆 Milestones, badges, party size

### Watch Logs Live
```bash
tail -f training_enhanced_*.log
```

### Stop Training
```bash
# Method 1: Using PID file
kill $(cat .training.pid)

# Method 2: Kill by name
pkill -f train_rl_vlm_screen_aware

# Method 3: Using check script to get PID, then kill it
./check_training.sh
kill <PID>
```

---

## Why Previous Training Stopped

**Problem:** When using `.sh` scripts in terminal, any input/interaction stops the process.

**Solutions Implemented:**

### 1. **nohup (No Hangup)**
```bash
nohup python train_rl_vlm_screen_aware.py ... > log.txt 2>&1 &
```
- `nohup`: Immune to hangup signal (SIGHUP)
- `&`: Runs in background
- `> log.txt 2>&1`: Redirects output to file
- Process continues even when terminal closes

### 2. **PID File Management**
- Saves process ID to `.training.pid`
- Prevents multiple training runs
- Easy to check/kill process later

### 3. **Detached from Terminal**
- Training runs as background job
- No stdin attachment (won't stop on input)
- Output goes to log file, not terminal

---

## Sleep-Safe Workflow

### Before Going to Sleep 😴

1. **Start Training:**
   ```bash
   ./start_enhanced_training.sh
   ```

2. **Verify It's Running:**
   ```bash
   ./check_training.sh
   ```
   Should show: ✅ Training is RUNNING

3. **Optional - Start Dashboard in Screen/Tmux:**
   ```bash
   # If you want to check dashboard later
   screen -S dashboard
   python dashboard.py
   # Press Ctrl+A then D to detach
   ```

4. **Close Terminal & Go to Sleep!**
   - Training continues in background
   - Server keeps running
   - No human interaction needed

### When You Wake Up ☀️

1. **Check Training Status:**
   ```bash
   ./check_training.sh
   ```

2. **View Progress:**
   ```bash
   python dashboard.py
   ```

3. **Check Recent Logs:**
   ```bash
   tail -100 training_enhanced_*.log
   ```

---

## Troubleshooting

### Training Not Running
```bash
# Check if process exists
ps aux | grep train_rl

# If PID file is stale
rm .training.pid

# Restart training
./start_enhanced_training.sh
```

### Server Stopped
```bash
# Check server
curl http://localhost:8000/state

# Restart if needed
source .venv/bin/activate
python -m server.app --port 8000 > server.log 2>&1 &
```

### Out of Memory
```bash
# Check memory usage
./check_training.sh

# If > 30GB, might need to kill and restart
kill $(cat .training.pid)

# Wait a moment, then restart
sleep 5
./start_enhanced_training.sh
```

### Training Stuck in Loop
```bash
# Check dashboard for stuck count
python dashboard.py

# If stuck for 1000+ steps in same location
kill $(cat .training.pid)

# Check why it's stuck (VLM hallucinating?)
tail -50 training_enhanced_*.log | grep -A5 "VLM"

# Restart
./start_enhanced_training.sh
```

---

## Advanced: Using Screen or Tmux

For ultimate peace of mind, run everything in screen/tmux sessions:

### Screen Example
```bash
# Create training session
screen -S training
./start_enhanced_training.sh
# Press Ctrl+A then D to detach

# Create dashboard session
screen -S dashboard
python dashboard.py
# Press Ctrl+A then D to detach

# List sessions
screen -ls

# Reattach to dashboard
screen -r dashboard

# Reattach to training (to see startup logs)
screen -r training
```

### Tmux Example
```bash
# Create training session
tmux new -s training
./start_enhanced_training.sh
# Press Ctrl+B then D to detach

# Create dashboard session  
tmux new -s dashboard
python dashboard.py
# Press Ctrl+B then D to detach

# List sessions
tmux ls

# Reattach
tmux attach -t dashboard
```

**With screen/tmux + nohup:** Triple protection! 🛡️🛡️🛡️

---

## Files Created

### Scripts
- `start_enhanced_training.sh` - Start training with nohup
- `check_training.sh` - Check training status
- `dashboard.py` - Enhanced live monitoring

### Runtime Files
- `.training.pid` - Current training process ID
- `training_enhanced_YYYYMMDD_HHMMSS.log` - Training logs
- `server.log` - Server logs

### Checkpoints
- `checkpoints_2M/` - Model checkpoints every 50k steps
- `.pokeagent_cache/` - Game state cache

---

## Dashboard Features (New!)

### Progress Section
- **Progress bar**: Visual 50-char bar
- **Percentage**: 3 decimal precision
- **ETA**: Estimated time remaining (days/hours)
- **Steps/hour**: Training speed

### Training Metrics
- **Average Reward**: Episode reward mean
- **Milestones**: X/Y completed
- **VLM Calls**: Count from last 1000 lines
- **Battle Events**: Won/lost/events
- **Episodic Memories**: k-NN stores
- **k-NN Confidence**: Average confidence (NEW!)
- **Stuck Detections**: Count of [STUCK] flags (NEW!)
- **FPS**: Training frames per second (NEW!)

### Agent State
- **Location**: Current map name
- **Position**: (x, y) coordinates
- **Party Size**: Number of Pokemon (NEW!)
- **Badges**: Badge count (NEW!)
- **Last Action**: Most recent button (NEW!)

### Recent Actions
- Shows last 10 actions in sequence (NEW!)
- Example: `UP → DOWN → A → A → RIGHT → DOWN → LEFT → UP → A → B`

### System Health
- **Training Process**: ✓ Running or ✗ Not running
- **Memory Usage**: MB consumed
- **Server API**: Response status
- **Log File**: Current file + size (NEW!)
- **Uptime**: Hours/minutes running (NEW!)

### Training Stability
- **Status**: Running smoothly or error details
- **Auto-detects**: Crashes, errors, anomalies

---

## Tips for Long Training Runs

1. **Monitor disk space:** 2M steps = ~5-10GB logs
   ```bash
   df -h .
   ```

2. **Set up email alerts** (optional):
   ```bash
   # Add to crontab for hourly checks
   0 * * * * /path/to/check_training.sh || echo "Training stopped!" | mail -s "Training Alert" you@email.com
   ```

3. **Backup checkpoints regularly:**
   ```bash
   # Every 6 hours
   rsync -av checkpoints_2M/ backup_location/
   ```

4. **Check for stuck loops:**
   ```bash
   # If same position for 30+ minutes, might be stuck
   tail -100 training_enhanced_*.log | grep "Position:"
   ```

5. **Use dashboard in tmux** for quick checks without SSH:
   ```bash
   tmux new -s dash
   python dashboard.py
   # Detach with Ctrl+B then D
   # Reattach anytime with: tmux attach -t dash
   ```

---

## Summary

✅ **Training now runs in background with nohup**  
✅ **Persists through terminal close, logout, sleep**  
✅ **PID file tracks process for easy management**  
✅ **Enhanced dashboard shows comprehensive metrics**  
✅ **Check script provides quick status**  
✅ **Safe to close terminal and go to sleep!** 💤

**Start training and forget about it - check back when you wake up!** 🌙➡️☀️
