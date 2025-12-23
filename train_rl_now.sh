#!/bin/bash
# Training script to run RL agent overnight

set -e

cd /home/jpli/pokeagent-speedrun

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  🚀 Starting RL Training Pipeline                             ║"
echo "║  Training will run overnight while you sleep!                 ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Check for ROM
if [ ! -f "emerald.gbc" ]; then
    echo "❌ ERROR: emerald.gbc not found!"
    echo "   Please place your Pokemon Emerald ROM in the project directory"
    exit 1
fi

echo "✅ ROM found: emerald.gbc"
echo ""

# Start training with progress logging
echo "📊 Starting training with:"
echo "   - Timesteps: 1,000,000 (test run)"
echo "   - SNN features: ENABLED"
echo "   - Logging: .rl_checkpoints/"
echo ""

# Run training (redirect output to log file)
python train_rl.py \
    --timesteps 1000000 \
    --use_snn \
    --save_freq 10000 \
    --rom emerald.gbc \
    2>&1 | tee training_log.txt

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  ✅ Training Complete!                                        ║"
echo "║                                                               ║"
echo "║  Check results:                                               ║"
echo "║  - training_log.txt (full log)                                ║"
echo "║  - .rl_checkpoints/ (model checkpoints)                       ║"
echo "║                                                               ║"
echo "║  Run trained agent:                                           ║"
echo "║  python run_rl.py --checkpoint .rl_checkpoints/pokemon_emerald_final.zip ║"
echo "╚═══════════════════════════════════════════════════════════════╝"

