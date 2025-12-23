#!/bin/bash
# Watch Your RL Agent Learn in Real-Time!
# This shows a live display window while training

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║  Watch Your RL Agent Learn Pokemon Emerald!               ║"
echo "║                                                            ║"
echo "║  This opens a display window showing the agent play        ║"
echo "║  while it learns with VLM guidance                         ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if there's already a checkpoint
CHECKPOINT_DIR=".rl_checkpoints"
LATEST_CHECKPOINT=""

if [ -d "$CHECKPOINT_DIR" ]; then
    LATEST_CHECKPOINT=$(ls -t $CHECKPOINT_DIR/pokemon_emerald_rl_*_steps.zip 2>/dev/null | head -1)
fi

if [ -n "$LATEST_CHECKPOINT" ]; then
    echo "📂 Found existing checkpoint: $LATEST_CHECKPOINT"
    echo ""
    read -p "Resume from this checkpoint? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        RESUME_FLAG="--resume $LATEST_CHECKPOINT"
    else
        RESUME_FLAG=""
    fi
else
    echo "ℹ️  No existing checkpoint found - will train from scratch"
    RESUME_FLAG=""
fi

# Check VLM setup
VLM_BACKEND="${VLM_BACKEND:-qwen}"
echo ""
echo "🧠 VLM Backend: $VLM_BACKEND"

if [ "$VLM_BACKEND" = "gpt" ] || [ "$VLM_BACKEND" = "openai" ]; then
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "⚠️  Warning: OPENAI_API_KEY not set - VLM features will be limited"
    fi
fi

# Training parameters
TIMESTEPS="${TIMESTEPS:-1000000}"  # Default 1M steps for watching
echo ""
echo "📋 Configuration:"
echo "   Timesteps: $TIMESTEPS"
echo "   Display: ENABLED 🎬"
echo "   VLM Guidance: ENABLED 🧠"
echo ""
echo "💡 Tips:"
echo "   - A window will open showing the game screen"
echo "   - You'll see the agent explore and learn in real-time"
echo "   - Watch for VLM detections in the terminal output"
echo "   - Training will be slower with display (but fun to watch!)"
echo "   - Press Ctrl+C to stop training"
echo ""

read -p "Ready to watch the agent learn? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "🚀 Starting VLM-guided RL training with live display..."
echo "   (Window will open shortly - be patient!)"
echo ""

# Run training with display enabled
python train_rl_vlm_guided.py \
    $RESUME_FLAG \
    --timesteps "$TIMESTEPS" \
    --vlm_backend "$VLM_BACKEND" \
    --display

echo ""
echo "Training session ended!"
