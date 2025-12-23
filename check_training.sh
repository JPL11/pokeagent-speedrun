#!/bin/bash
# Quick script to check training status

PID_FILE=".training.pid"

echo "========================================================================"
echo "TRAINING STATUS CHECK"
echo "========================================================================"
echo ""

# Check PID file
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    echo "📋 PID File Found: $PID"
    
    # Check if process is running
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ Training is RUNNING"
        echo ""
        
        # Get process info
        echo "Process Details:"
        ps -p $PID -o pid,ppid,cmd,%cpu,%mem,etime
        echo ""
        
        # Get memory usage
        MEM_KB=$(ps -p $PID -o rss= | tr -d ' ')
        MEM_MB=$((MEM_KB / 1024))
        echo "Memory Usage: ${MEM_MB} MB"
        
        # Find latest log
        LATEST_LOG=$(ls -t training_*.log 2>/dev/null | head -1)
        if [ -n "$LATEST_LOG" ]; then
            echo "Latest Log:   $LATEST_LOG"
            LOG_SIZE=$(du -h "$LATEST_LOG" | cut -f1)
            echo "Log Size:     $LOG_SIZE"
            
            # Get last few lines
            echo ""
            echo "Last 3 log entries:"
            echo "---"
            tail -3 "$LATEST_LOG" 2>/dev/null || echo "  (unable to read log)"
        fi
        
        echo ""
        echo "Commands:"
        echo "  • Stop training:    kill $PID"
        echo "  • Watch logs:       tail -f $LATEST_LOG"
        echo "  • Live dashboard:   python dashboard.py"
        
    else
        echo "❌ Training is NOT RUNNING (stale PID file)"
        echo "   Old PID: $PID"
        echo "   Run: rm $PID_FILE"
    fi
else
    echo "❌ No PID file found ($PID_FILE)"
    
    # Check if any training process exists
    if pgrep -f "train_rl_vlm_screen_aware" > /dev/null; then
        echo "⚠️  Found training process without PID file:"
        pgrep -af "train_rl_vlm_screen_aware"
        echo ""
        echo "   Kill with: pkill -f train_rl_vlm_screen_aware"
    else
        echo "   No training processes found"
        echo ""
        echo "Start training with: ./start_enhanced_training.sh"
    fi
fi

echo ""
echo "========================================================================"
