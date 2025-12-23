#!/bin/bash
# Long-term training script with monitoring
# Trains for 2 million steps with periodic checkpoints and health checks

LOG_FILE="training_2M_steps.log"
CHECKPOINT_DIR="checkpoints_2M"
MONITOR_INTERVAL=300  # Check every 5 minutes

mkdir -p "$CHECKPOINT_DIR"

echo "========================================" | tee -a "$LOG_FILE"
echo "Starting 2M step training run" | tee -a "$LOG_FILE"
echo "Start time: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Kill any existing training
pkill -9 -f train_rl_vlm_screen_aware

# Start training in background
source .venv/bin/activate
nohup python train_rl_vlm_screen_aware.py \
    --timesteps 2000000 \
    --vlm-frequency 25 \
    > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "Training PID: $TRAIN_PID" | tee -a monitor.log

# Monitoring loop
while true; do
    sleep $MONITOR_INTERVAL
    
    # Check if training is still running
    if ! ps -p $TRAIN_PID > /dev/null 2>&1; then
        echo "[$(date)] ⚠️ Training process died! PID $TRAIN_PID not found" | tee -a monitor.log
        
        # Check if it completed successfully
        if tail -20 "$LOG_FILE" | grep -q "Training complete"; then
            echo "[$(date)] ✅ Training completed successfully!" | tee -a monitor.log
            break
        else
            echo "[$(date)] ❌ Training crashed. Last 50 lines:" | tee -a monitor.log
            tail -50 "$LOG_FILE" | tee -a monitor.log
            break
        fi
    fi
    
    # Log current progress
    CURRENT_STEP=$(tail -100 "$LOG_FILE" | grep -oP "Step \K[0-9]+" | tail -1)
    CURRENT_REWARD=$(tail -100 "$LOG_FILE" | grep -oP "ep_rew_mean.*?\K[-0-9.]+" | tail -1)
    
    echo "[$(date)] ✓ Training alive - Step: $CURRENT_STEP | Reward: $CURRENT_REWARD" | tee -a monitor.log
    
    # Check milestones
    curl -s http://localhost:8000/milestones 2>/dev/null | \
        python3 -c "import sys, json; d=json.load(sys.stdin); completed=sum(1 for v in d.values() if v==True); print(f'Milestones: {completed}')" \
        2>/dev/null | tee -a monitor.log
    
    # Check for errors
    ERROR_COUNT=$(tail -500 "$LOG_FILE" | grep -c "ERROR\|Exception\|Traceback")
    if [ "$ERROR_COUNT" -gt 10 ]; then
        echo "[$(date)] ⚠️ High error count: $ERROR_COUNT in last 500 lines" | tee -a monitor.log
    fi
    
    # Check episodic memory stats
    if tail -100 "$LOG_FILE" | grep -q "Episodic Memory"; then
        tail -100 "$LOG_FILE" | grep "Episodic Memory" | tail -1 | tee -a monitor.log
    fi
    
    # Memory usage check
    MEM_USAGE=$(ps -p $TRAIN_PID -o rss= 2>/dev/null | awk '{print $1/1024/1024 " GB"}')
    echo "[$(date)] Memory: $MEM_USAGE" | tee -a monitor.log
    
    echo "---" >> monitor.log
done

echo "[$(date)] Monitoring ended" | tee -a monitor.log
