#!/bin/bash
# Start RL training in the background to run overnight

cd /home/jpli/pokeagent-speedrun

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  🚀 Starting RL Training - Will Run Overnight!               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check ROM exists
if [ ! -f "emerald.gba" ]; then
    echo "❌ ERROR: emerald.gba not found!"
    exit 1
fi

echo "✅ ROM found: emerald.gba (16MB)"
echo "✅ Dependencies installed: stable-baselines3, torch"
echo ""

# Create output directory
mkdir -p .rl_checkpoints
mkdir -p logs

# Start training in background with nohup
echo "📊 Starting training with:"
echo "   - Total timesteps: 10,000,000 (full training)"
echo "   - SNN features: ENABLED"
echo "   - Save frequency: Every 50,000 steps"
echo "   - Log file: logs/training_$(date +%Y%m%d_%H%M%S).log"
echo ""

LOG_FILE="logs/training_$(date +%Y%m%d_%H%M%S).log"

nohup python train_rl.py \
    --timesteps 10000000 \
    --use_snn \
    --save_freq 50000 \
    --rom emerald.gba \
    --checkpoint_dir .rl_checkpoints \
    > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!

echo "✅ Training started in background!"
echo "   PID: $TRAIN_PID"
echo "   Log: $LOG_FILE"
echo ""
echo "📊 Monitor progress with:"
echo "   tail -f $LOG_FILE"
echo ""
echo "📈 Watch tensorboard:"
echo "   tensorboard --logdir .rl_checkpoints/tensorboard"
echo "   (open http://localhost:6006)"
echo ""
echo "🛑 To stop training:"
echo "   kill $TRAIN_PID"
echo ""
echo "💤 Go to sleep! Training will complete in 6-12 hours."
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  When you wake up, check:                                    ║"
echo "║  - tail $LOG_FILE                                             "
echo "║  - ls .rl_checkpoints/                                        ║"
echo "║                                                              ║"
echo "║  Run trained agent:                                          ║"
echo "║  python run_rl.py --checkpoint .rl_checkpoints/pokemon_emerald_final.zip ║"
echo "╚══════════════════════════════════════════════════════════════╝"

