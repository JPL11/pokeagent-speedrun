#!/bin/bash

# Monitor RL Training Progress

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  🎮 Pokemon Emerald RL Training Monitor                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check if training process is running (find actual Python process, not bash wrapper)
TRAIN_PID=$(ps aux | grep "python train_rl.py" | grep -v grep | grep -v bash | awk '{print $2}' | head -1)

if [ -z "$TRAIN_PID" ]; then
    echo "❌ Training process not found!"
    echo ""
    echo "To start training:"
    echo "  python train_rl.py --rom /home/jpli/pokeagent-speedrun/Emerald-GBAdvance/rom.gba --timesteps 500000"
    exit 1
fi

# Process info
echo "📊 Process Status:"
ps aux | grep "$TRAIN_PID" | grep -v grep | awk '{printf "   PID: %s\n   CPU: %s%%\n   Memory: %s MB\n", $2, $3, int($6/1024)}'
echo ""

# Training log
if [ -f "nohup.out" ]; then
    echo "📝 Latest Training Output:"
    echo "════════════════════════════════════════════════════════════════"
    tail -30 nohup.out
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    
    # Check for progress indicators
    if grep -q "rollout/" nohup.out; then
        echo "✅ Training is progressing!"
        echo ""
        echo "Latest stats:"
        grep -E "ep_rew_mean|total_timesteps|fps" nohup.out | tail -10
    elif grep -q "Using cuda device" nohup.out; then
        echo "⏳ Collecting first rollout (this takes 5-10 minutes)..."
        echo "   Be patient! Progress will appear soon."
    fi
else
    echo "❌ No training log found (nohup.out)"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "📈 To view real-time graphs:"
echo "   tensorboard --logdir .rl_checkpoints/tensorboard --port 6006"
echo ""
echo "📖 For full debug info:"
echo "   cat SNN_CNN_DEBUG_COMPLETE.md"
echo ""
echo "🔄 To check again:"
echo "   ./monitor_training.sh"
echo ""

