#!/bin/bash
# Restart training from Littleroot Town savestate

echo "🛑 Stopping any running training..."
pkill -f train_rl_event_flags.py
pkill -f train_rl_server.py
sleep 2

echo "🎮 Starting server with quick_start savestate..."
cd /home/jpli/pokeagent-speedrun
source .venv/bin/activate

# Start server with savestate
python train_rl_server.py --load-state Emerald-GBAdvance/test_states/quick_start_save.state &
sleep 5

echo "🤖 Starting RL training..."
python train_rl_event_flags.py

