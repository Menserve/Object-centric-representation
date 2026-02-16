#!/bin/bash
# Monitor training completion and prepare for analysis

LOG_FILE="logs/video_stopgrad_200ep.log"
PID=$(ps aux | grep "[p]ython.*train_movi.py.*video_stopgrad" | awk '{print $2}')

echo "Monitoring training process (PID: $PID)"
echo "Started at: $(date)"
echo ""

# Wait for completion
while ps -p $PID > /dev/null 2>&1; do
    sleep 30
    CURRENT_EPOCH=$(tail -20 "$LOG_FILE" | grep "Epoch" | tail -1 | awk '{print $2}')
    CURRENT_LOSS=$(tail -20 "$LOG_FILE" | grep "Epoch" | tail -1 | awk '{print $5}')
    echo "[$(date +%H:%M:%S)] Progress: $CURRENT_EPOCH | Loss: $CURRENT_LOSS"
done

echo ""
echo "✅ Training completed at: $(date)"
echo ""
echo "Final results:"
echo "============================================================"
tail -30 "$LOG_FILE" | grep -E "Epoch [0-9]+/200|Best|Saved checkpoint"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. python src/analyze_video_masks.py (detailed collapse analysis)"
echo "  2. Compare with baseline: checkpoints/video_mode_8frames"
echo "  3. Proceed to DINOv1 + CLIP backbone comparison"
