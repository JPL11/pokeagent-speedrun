#!/bin/bash
# Quick Start: Resume Your RL Training with VLM Guidance
# This adds VLM-based unstuck detection to your 6-hour trained agent

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║  Resume RL Training with VLM Unstuck Guidance              ║"
echo "║                                                            ║"
echo "║  This will add VLM reasoning to your existing checkpoint   ║"
echo "║  to reduce reliance on hard-coded heuristics               ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Find the latest checkpoint
CHECKPOINT_DIR=".rl_checkpoints"
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ No checkpoint directory found: $CHECKPOINT_DIR"
    echo "   Train a model first with: python train_rl.py"
    exit 1
fi

LATEST_CHECKPOINT=$(ls -t $CHECKPOINT_DIR/pokemon_emerald_rl_*_steps.zip 2>/dev/null | head -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "❌ No checkpoints found in $CHECKPOINT_DIR"
    echo "   Train a model first with: python train_rl.py"
    exit 1
fi

echo "📂 Found latest checkpoint: $LATEST_CHECKPOINT"
echo ""

# Check VLM backend
VLM_BACKEND="${VLM_BACKEND:-qwen}"
echo "🧠 VLM Backend: $VLM_BACKEND"
echo "   (Set VLM_BACKEND=claude for better reasoning but slower)"
echo ""

# Check API keys
if [ "$VLM_BACKEND" = "gpt" ] || [ "$VLM_BACKEND" = "openai" ]; then
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "⚠️  Warning: OPENAI_API_KEY not set!"
        echo "   Set it with: export OPENAI_API_KEY='your-key'"
        read -p "   Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
elif [ "$VLM_BACKEND" = "claude" ]; then
    if [ -z "$ANTHROPIC_API_KEY" ]; then
        echo "⚠️  Warning: ANTHROPIC_API_KEY not set!"
        echo "   Set it with: export ANTHROPIC_API_KEY='your-key'"
        read -p "   Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
elif [ "$VLM_BACKEND" = "gemini" ]; then
    if [ -z "$GOOGLE_API_KEY" ]; then
        echo "⚠️  Warning: GOOGLE_API_KEY not set!"
        echo "   Set it with: export GOOGLE_API_KEY='your-key'"
        read -p "   Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# Training parameters
TIMESTEPS="${TIMESTEPS:-5000000}"  # Additional 5M steps
VLM_UNSTUCK_FREQ="${VLM_UNSTUCK_FREQ:-30}"
VLM_GUIDANCE_FREQ="${VLM_GUIDANCE_FREQ:-100}"

echo "📋 Training Configuration:"
echo "   Checkpoint: $LATEST_CHECKPOINT"
echo "   Additional timesteps: $TIMESTEPS"
echo "   VLM unstuck check frequency: every $VLM_UNSTUCK_FREQ steps"
echo "   VLM guidance frequency: every $VLM_GUIDANCE_FREQ steps"
echo ""

echo "🎯 What VLM Will Do:"
echo "   ✓ Detect stuck patterns with reasoning (not just counting)"
echo "   ✓ Suggest exploration strategies when stuck"
echo "   ✓ Shape rewards based on game context"
echo "   ✓ Teach RL to handle dialogue naturally"
echo ""

read -p "Ready to start VLM-guided training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "🚀 Starting VLM-guided RL training..."
echo ""

# Run training with VLM guidance
python train_rl_vlm_guided.py \
    --resume "$LATEST_CHECKPOINT" \
    --timesteps "$TIMESTEPS" \
    --vlm_backend "$VLM_BACKEND" \
    --vlm_unstuck_freq "$VLM_UNSTUCK_FREQ" \
    --vlm_guidance_freq "$VLM_GUIDANCE_FREQ"

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║  VLM-Guided Training Complete!                             ║"
echo "║                                                            ║"
echo "║  Your agent should now be better at:                       ║"
echo "║  ✓ Handling dialogue (not just spamming A)                 ║"
echo "║  ✓ Escaping stuck patterns                                 ║"
echo "║  ✓ Purposeful exploration                                  ║"
echo "║                                                            ║"
echo "║  Test it with:                                             ║"
echo "║  python run_rl.py --checkpoint <new_checkpoint> --render   ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
