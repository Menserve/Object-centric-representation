#!/bin/bash
# DINOv1 Backbone Training with Optimized GPU Utilization
# Target: RTX 5090 (32GB) - 95%+ utilization

cd /home/menserve/Object-centric-representation
source .venv/bin/activate

echo "============================================================"
echo "DINOv1 Training (GPU Optimized)"
echo "============================================================"
echo "Configuration:"
echo "  - Backbone: DINOv1 ViT-S/16"
echo "  - Batch size: 8 (4x increase for GPU utilization)"
echo "  - Max frames: 8"
echo "  - Mask temperature: 0.5 (proven optimal)"
echo "  - Collapse prevention: Stop-gradient"
echo "  - Expected GPU usage: 95%+"
echo "  - Expected memory: ~15GB / 32GB"
echo ""

python -u src/train_movi.py \
  --backbone dino_vits16 \
  --data_dir data/movi_a_subset \
  --num_epochs 200 \
  --batch_size 8 \
  --num_slots 5 \
  --max_frames 8 \
  --lr 0.0004 \
  --diversity_weight 0.1 \
  --mask_temperature 0.5 \
  --use_stop_gradient \
  --save_dir checkpoints/dinov1_optimized_200ep

echo ""
echo "✅ Training complete!"
