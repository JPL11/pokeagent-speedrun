#!/bin/bash
# Start training with k-NN enhancements and TODO fixes
# Date: November 19, 2025
# 
# THIS SCRIPT RUNS TRAINING IN BACKGROUND WITH NOHUP
# Training will continue even if you close terminal or logout!

set -e

LOG_FILE="training_enhanced_$(date +%Y%m%d_%H%M%S).log"
PID_FILE=".training.pid"

echo "========================================================================"
echo "Starting Enhanced Training with k-NN Confidence Gating"
echo "========================================================================"
echo ""
echo "Enhancements in this run:"
echo "  ✅ k-NN confidence gating (policy knows when to trust recalls)"
echo "  ✅ Context-aware memory split (nav/battle/story)"
echo "  ✅ Real milestone state (not placeholder zeros)"
echo "  ✅ Real SLAM waypoints (not placeholder zeros)"
echo "  ✅ SNN INPUT 11 includes confidence at index [10]"
echo ""
echo "Training parameters:"
echo "  • Target: 2,000,000 timesteps"
echo "  • VLM frequency: Every 60 steps"
echo "  • Checkpoint: Every 50,000 steps"
echo "  • Log file: $LOG_FILE"
echo ""
echo "========================================================================"
echo ""

# Check if training is already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo "⚠️  Training already running with PID $OLD_PID"
        echo "   Kill it first with: kill $OLD_PID"
        echo "   Or run: pkill -f train_rl_vlm_screen_aware"
        exit 1
    fi
fi

# Activate virtual environment
source .venv/bin/activate

# Create checkpoints directory if it doesn't exist
mkdir -p checkpoints_2M

echo "🚀 Starting training in background with nohup..."
echo "   Training will continue even if you:"
echo "   • Close this terminal"
echo "   • Logout from SSH"
echo "   • Go to sleep 💤"
echo ""
echo "📋 Monitor with:"
echo "   tail -f $LOG_FILE"
echo "   python dashboard.py"
echo ""
echo "🛑 Stop training with:"
echo "   kill \$(cat .training.pid)"
echo "   or: pkill -f train_rl_vlm_screen_aware"
echo ""

# Start training with nohup (immune to hangup signal)
nohup python train_rl_vlm_screen_aware.py \
    --timesteps 2000000 \
    --vlm-frequency 60 \
    > "$LOG_FILE" 2>&1 &

# Save PID for monitoring
TRAINING_PID=$!
echo $TRAINING_PID > "$PID_FILE"

echo "✅ Training started successfully!"
echo "   PID: $TRAINING_PID"
echo "   Log: $LOG_FILE"
echo "   PID file: $PID_FILE"
echo ""
echo "========================================================================"
echo "🎉 You can now safely close this terminal or logout!"
echo "========================================================================"
echo ""
echo "Quick commands:"
echo "  • Watch logs:  tail -f $LOG_FILE"
echo "  • Dashboard:   python dashboard.py"
echo "  • Check alive: ps -p $TRAINING_PID"
echo "  • Stop it:     kill $TRAINING_PID"
echo ""
