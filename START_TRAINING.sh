#!/bin/bash
# Start RL training overnight

cd /home/jpli/pokeagent-speedrun

echo "🚀 Starting RL Training..."

nohup python train_rl.py \
  --timesteps 10000000 \
  --use_snn \
  --save_freq 50000 \
  --rom Emerald-GBAdvance/rom.gba \
  --checkpoint_dir .rl_checkpoints \
  > logs/training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

PID=$!

sleep 3

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✅ RL TRAINING STARTED!                                      ║"
echo "║  💤 GO TO SLEEP - Training runs for 6-12 hours               ║"
echo "║                                                              ║"
echo "║  PID: $PID                                                    "
echo "║  Log: logs/training_*.log                                    ║"
echo "║                                                              ║"
echo "║  Monitor: tail -f logs/training_*.log                        ║"
echo "║  Check process: ps aux | grep train_rl.py                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

