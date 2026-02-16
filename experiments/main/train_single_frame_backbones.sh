#!/bin/bash
# Single-frame backbone comparison training
# Using the proven successful configuration: single frame + τ=0.5

set -e

EPOCHS=200
BATCH_SIZE=2
LR=0.0004
DIV_WEIGHT=0.1
MAX_FRAMES=1
TEMP=0.5

echo "============================================================"
echo "SINGLE-FRAME BACKBONE COMPARISON"
echo "Proven configuration: max_frames=1, τ=0.5"
echo "============================================================"

# DINOv2 (baseline - already trained)
echo ""
echo "1. DINOv2: Using existing temp_scaling_tau05 result"
echo "   (Already trained, mask_sim=0.558)"
if [ -f checkpoints/temp_scaling_tau05/dinov2_vits14/best_model.pt ]; then
    echo "   ✓ Found: checkpoints/temp_scaling_tau05/dinov2_vits14/"
else
    echo "   ⚠️  Not found - needs retraining"
fi

# DINOv1 (single frame)
echo ""
echo "2. Training DINOv1 (single frame)..."
python src/train_movi.py \
    --backbone dino_vits16 \
    --num_epochs $EPOCHS \
    --max_frames $MAX_FRAMES \
    --mask_temperature $TEMP \
    --batch_size 8 \
    --lr $LR \
    --diversity_weight $DIV_WEIGHT \
    --save_dir checkpoints/dinov1_singleframe_tau05 \
    2>&1 | tee logs/dinov1_singleframe.log

echo "✓ DINOv1 complete!"

# CLIP (single frame)
echo ""
echo "3. Training CLIP (single frame)..."
python src/train_movi.py \
    --backbone clip_vitb16 \
    --num_epochs $EPOCHS \
    --max_frames $MAX_FRAMES \
    --mask_temperature $TEMP \
    --batch_size 6 \
    --lr $LR \
    --diversity_weight $DIV_WEIGHT \
    --save_dir checkpoints/clip_singleframe_tau05 \
    2>&1 | tee logs/clip_singleframe.log

echo "✓ CLIP complete!"

echo ""
echo "============================================================"
echo "✅ ALL SINGLE-FRAME TRAINING COMPLETE!"
echo "============================================================"
echo ""
echo "Results:"
echo "  - DINOv2: checkpoints/temp_scaling_tau05/dinov2_vits14/"
echo "  - DINOv1: checkpoints/dinov1_singleframe_tau05/dino_vits16/"
echo "  - CLIP:   checkpoints/clip_singleframe_tau05/clip_vitb16/"
echo ""
echo "Next: python src/compare_single_frame_backbones.py"
