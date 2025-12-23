#!/bin/bash
# Cleanup Script: Archive redundant files for publication

echo "🧹 PokéAgent Repository Cleanup"
echo "================================"
echo ""

# Create archive directories
echo "Creating archive directories..."
mkdir -p experiments/training_variants
mkdir -p logs/archive
mkdir -p docs/archive

echo "✅ Directories created"
echo ""

# Archive training variants
echo "Archiving training variants..."
mv train_rl_vlm_screen_aware_*.py experiments/training_variants/ 2>/dev/null
mv train_rl_server.py experiments/training_variants/ 2>/dev/null
mv train_rl_milestone.py experiments/training_variants/ 2>/dev/null
mv train_rl_snn_enhanced.py experiments/training_variants/ 2>/dev/null
mv train_rl_complete.py experiments/training_variants/ 2>/dev/null
mv train_rl_event_flags.py experiments/training_variants/ 2>/dev/null
mv train_rl_server_vlm.py experiments/training_variants/ 2>/dev/null
mv train_rl_title_screen.py experiments/training_variants/ 2>/dev/null
mv train_rl_vlm_guided.py experiments/training_variants/ 2>/dev/null

echo "✅ Training variants archived"
echo ""

# Archive training logs
echo "Archiving training logs..."
mv training_*.log logs/archive/ 2>/dev/null

echo "✅ Training logs archived"
echo ""

# Archive documentation
echo "Archiving old documentation..."
mv COMPARISON_OLD_VS_NEW.md docs/archive/ 2>/dev/null
mv DEPLOYMENT_SUMMARY.md docs/archive/ 2>/dev/null
mv DIALOGUE_ARCHITECTURE_ENHANCED.md docs/archive/ 2>/dev/null
mv ENHANCED_TERRAIN_REASONING.md docs/archive/ 2>/dev/null
mv FEEDBACK_ARCHITECTURE.md docs/archive/ 2>/dev/null
mv HEALING_SYSTEM_PLAN.md docs/archive/ 2>/dev/null
mv HEURISTIC_VS_VLM_COMPARISON.md docs/archive/ 2>/dev/null
mv HYBRID_ARCHITECTURE.md docs/archive/ 2>/dev/null
mv IMPLEMENTATION_GUIDE_TRAINING_SYSTEM.md docs/archive/ 2>/dev/null
mv IMPROVEMENTS_SUMMARY.md docs/archive/ 2>/dev/null
mv KNN_SNN_INTEGRATION.md docs/archive/ 2>/dev/null
mv LAB_STUCK_ANALYSIS.md docs/archive/ 2>/dev/null
mv MEMORY_DRIVEN_QUESTS.md docs/archive/ 2>/dev/null
mv NOVELTY_DETECTOR.md docs/archive/ 2>/dev/null
mv POKEMON_TRAINING_SYSTEM.md docs/archive/ 2>/dev/null
mv QUEST_CHAIN_FIX.md docs/archive/ 2>/dev/null
mv REFINED_ARCHITECTURE_FEEDBACK_LOOPS.md docs/archive/ 2>/dev/null
mv RL_DEBUG_STATUS.md docs/archive/ 2>/dev/null
mv RL_LEARNING_README.md docs/archive/ 2>/dev/null
mv RL_PROJECT_STATUS.md docs/archive/ 2>/dev/null
mv RL_TRAINING_STATUS.md docs/archive/ 2>/dev/null
mv SLEEP_WELL_README.md docs/archive/ 2>/dev/null
mv SNN_CNN_DEBUG_COMPLETE.md docs/archive/ 2>/dev/null
mv SNN_ENCODER_README.md docs/archive/ 2>/dev/null
mv START_TRAINING_NOW.md docs/archive/ 2>/dev/null
mv STATUS_AND_NEXT_STEPS.md docs/archive/ 2>/dev/null
mv SUMMARY_RL_IMPLEMENTATION.md docs/archive/ 2>/dev/null
mv TRAINING_STATUS_2M.md docs/archive/ 2>/dev/null
mv TRAINING_SYSTEM_READY.md docs/archive/ 2>/dev/null
mv VLM_GUIDED_RL_README.md docs/archive/ 2>/dev/null
mv VLM_IMPLEMENTATION_SUMMARY.md docs/archive/ 2>/dev/null
mv VLM_SCREEN_AWARE_SOLUTION.md docs/archive/ 2>/dev/null
mv VLM_SERVER_SOLUTION.md docs/archive/ 2>/dev/null
mv WATCH_AGENT_PLAY.md docs/archive/ 2>/dev/null
mv ANSWERS.md docs/archive/ 2>/dev/null

echo "✅ Documentation archived"
echo ""

# Keep important docs
echo "Keeping core documentation:"
echo "  • README.md (main documentation)"
echo "  • ARCHITECTURE_ANALYSIS.md (this analysis)"
echo "  • ARCHITECTURE_DIAGRAM.md (system diagram)"
echo "  • TRAINING_STRATEGY_IMPLEMENTATION.md (recent changes)"
echo "  • PRE_DEPLOYMENT_CHECKLIST.md (deployment guide)"
echo "  • QUICK_START_RL.md (user guide)"
echo "  • QUICK_START_TRAINING_STRATEGY.md (quick reference)"
echo "  • QUICK_REFERENCE_TRAINING.md (training guide)"
echo "  • FINAL_RL_GUIDE.md (comprehensive guide)"
echo "  • POKEAGENT.md (project overview)"
echo "  • LICENSE (license file)"
echo ""

# Archive helper scripts
echo "Archiving helper scripts..."
mkdir -p experiments/helper_scripts
mv break_stair_loop_now.py experiments/helper_scripts/ 2>/dev/null
mv exit_house_now.py experiments/helper_scripts/ 2>/dev/null
mv unstuck_corner_now.py experiments/helper_scripts/ 2>/dev/null
mv trigger_clock_fix.py experiments/helper_scripts/ 2>/dev/null
mv create_start_savestate.py experiments/helper_scripts/ 2>/dev/null
mv check_game_screen.py experiments/helper_scripts/ 2>/dev/null
mv grab_map.py experiments/helper_scripts/ 2>/dev/null
mv analyze_lab_stuck.py experiments/helper_scripts/ 2>/dev/null

echo "✅ Helper scripts archived"
echo ""

# Show final structure
echo "📊 Repository Structure After Cleanup:"
echo "──────────────────────────────────────"
echo ""
echo "Core Training Files:"
find . -maxdepth 1 -name "train_rl*.py" -type f | sort
echo ""
echo "Agent Modules:"
ls -1 agent/*.py | grep -v __pycache__ | head -12
echo ""
echo "Archived Files:"
echo "  • $(ls experiments/training_variants/*.py 2>/dev/null | wc -l) training variants"
echo "  • $(ls logs/archive/*.log 2>/dev/null | wc -l) training logs"
echo "  • $(ls docs/archive/*.md 2>/dev/null | wc -l) documentation files"
echo "  • $(ls experiments/helper_scripts/*.py 2>/dev/null | wc -l) helper scripts"
echo ""

# Calculate LOC reduction
echo "📈 Code Statistics:"
echo "──────────────────"
TOTAL_LOC=$(find . -name "*.py" -not -path "./.venv/*" -not -path "./PokemonRedExperiments/*" -not -path "./__pycache__/*" -not -path "./experiments/*" -exec wc -l {} + | tail -1 | awk '{print $1}')
ARCHIVED_LOC=$(find experiments/ -name "*.py" -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')
ARCHIVED_LOC=${ARCHIVED_LOC:-0}

echo "  • Core codebase: ~$TOTAL_LOC lines"
echo "  • Archived code: ~$ARCHIVED_LOC lines"
echo ""

echo "✅ Cleanup complete!"
echo ""
echo "Next Steps:"
echo "1. Review archived files in experiments/, logs/archive/, docs/archive/"
echo "2. Delete archive directories if happy with cleanup: rm -rf experiments/ logs/archive/ docs/archive/"
echo "3. Commit changes: git add . && git commit -m 'Clean up redundant files for publication'"
echo ""
