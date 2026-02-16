#!/bin/bash
# Auto-start optimized backbone comparison after DINOv2 completes

cd /home/menserve/Object-centric-representation

# Wait for DINOv2 training to complete
echo "Waiting for DINOv2 training to complete..."
while ps -p 109385 > /dev/null 2>&1; do
    sleep 5
done

echo ""
echo "✅ DINOv2 training completed!"
echo ""
echo "Starting GPU-optimized backbone comparison..."
echo "============================================================"

# Analyze DINOv2 results
echo ""
echo "1. Analyzing DINOv2 (Stop-gradient) results..."
source .venv/bin/activate
python src/analyze_video_masks.py \
  --checkpoint checkpoints/video_stopgrad_200ep/dinov2_vits14/best_model.pt \
  > logs/dinov2_stopgrad_analysis.log 2>&1

echo "   ✓ Analysis complete: logs/dinov2_stopgrad_analysis.log"

# Start DINOv1 training (batch_size=8, GPU optimized)
echo ""
echo "2. Starting DINOv1 training (batch_size=8)..."
bash src/train_dinov1_optimized.sh > logs/dinov1_optimized.log 2>&1 &
DINOV1_PID=$!
echo "   ✓ DINOv1 training started (PID: $DINOV1_PID)"

# Monitor DINOv1 completion
echo ""
echo "3. Monitoring DINOv1 training..."
while ps -p $DINOV1_PID > /dev/null 2>&1; do
    sleep 30
    EPOCH=$(tail -20 logs/dinov1_optimized.log | grep "Epoch" | tail -1 | awk '{print $2}')
    GPU=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    echo "   [$(date +%H:%M:%S)] $EPOCH | GPU: ${GPU}%"
done

echo ""
echo "✅ DINOv1 training completed!"

# Start CLIP training (batch_size=6, GPU optimized)
echo ""
echo "4. Starting CLIP training (batch_size=6)..."
bash src/train_clip_optimized.sh > logs/clip_optimized.log 2>&1 &
CLIP_PID=$!
echo "   ✓ CLIP training started (PID: $CLIP_PID)"

# Monitor CLIP completion
echo ""
echo "5. Monitoring CLIP training..."
while ps -p $CLIP_PID > /dev/null 2>&1; do
    sleep 30
    EPOCH=$(tail -20 logs/clip_optimized.log | grep "Epoch" | tail -1 | awk '{print $2}')
    GPU=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    echo "   [$(date +%H:%M:%S)] $EPOCH | GPU: ${GPU}%"
done

echo ""
echo "✅ CLIP training completed!"

# Final comparison
echo ""
echo "6. Generating backbone comparison report..."
python src/compare_backbones.py > logs/backbone_comparison_report.log 2>&1

echo ""
echo "============================================================"
echo "✅ ALL TRAINING COMPLETE!"
echo "============================================================"
echo ""
echo "Results:"
echo "  - DINOv2: checkpoints/video_stopgrad_200ep/"
echo "  - DINOv1: checkpoints/dinov1_optimized_200ep/"
echo "  - CLIP:   checkpoints/clip_optimized_200ep/"
echo ""
echo "Logs:"
echo "  - Analysis: logs/dinov2_stopgrad_analysis.log"
echo "  - DINOv1:   logs/dinov1_optimized.log"
echo "  - CLIP:     logs/clip_optimized.log"
echo "  - Report:   logs/backbone_comparison_report.log"
echo ""
echo "Next step: Review comparison report for TA paper"
