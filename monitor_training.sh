#!/bin/bash

# Monitor training progress

echo "Monitoring training progress..."
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    clear
    echo "================================================================================"
    echo "TRAINING PROGRESS MONITOR"
    echo "================================================================================"
    echo ""

    # Check if process is running
    if pgrep -f "run_complete_pipeline.py" > /dev/null; then
        echo "Status: ✅ Training is running"
    else
        echo "Status: ⏹️  Training completed or not running"
        echo ""
        echo "Check pipeline_output.log for results"
        break
    fi

    echo ""
    echo "Latest output:"
    echo "--------------------------------------------------------------------------------"
    tail -30 pipeline_output.log 2>/dev/null || echo "No output file found"
    echo "================================================================================"

    sleep 10
done
