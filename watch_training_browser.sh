#!/bin/bash
# Watch Your Agent Train with Live Browser Feed
# This starts everything you need to see training in real-time

set -e

# Activate venv if not already activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "🔧 Activating virtual environment..."
    source .venv/bin/activate
fi

echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║  Watch RL Training in Your Browser!                       ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if servers are already running
if pgrep -f "server.app" > /dev/null; then
    echo "✅ Game server already running on port 8000"
else
    echo "🚀 Starting game server on port 8000..."
    python -m server.app --port 8000 &
    GAME_SERVER_PID=$!
    sleep 3
    echo "✅ Game server started (PID: $GAME_SERVER_PID)"
fi

if pgrep -f "server.frame_server" > /dev/null; then
    echo "✅ Frame server already running on port 8001"
else
    echo "🚀 Starting frame server on port 8001..."
    python -m server.frame_server --port 8001 &
    FRAME_SERVER_PID=$!
    sleep 2
    echo "✅ Frame server started (PID: $FRAME_SERVER_PID)"
fi

echo ""
echo "🎯 Servers ready! Now starting training..."
echo ""
echo "📺 Open your browser to: http://localhost:8001"
echo "   (You'll see the game screen update in real-time)"
echo ""

# Find latest checkpoint
LATEST_CHECKPOINT=$(ls -t .rl_checkpoints/pokemon_emerald_rl_*.zip 2>/dev/null | head -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "❌ No checkpoint found!"
    echo "   Please train a model first"
    exit 1
fi

echo "📂 Using checkpoint: $LATEST_CHECKPOINT"
echo ""
echo "🚀 Starting training (press Ctrl+C to stop)..."
echo ""

# Start training with server backend
python train_rl_server.py \
    --resume "$LATEST_CHECKPOINT" \
    --timesteps 100000 \
    --port 8000

echo ""
echo "Training stopped."
