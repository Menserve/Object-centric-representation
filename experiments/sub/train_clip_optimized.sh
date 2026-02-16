#!/bin/bash
# CLIP Backbone Training with Optimized GPU Utilization
# Target: RTX 5090 (32GB) - 95%+ utilization

cd /home/menserve/Object-centric-representation
source .venv/bin/activate

echo "============================================================"
echo "CLIP Training (GPU Optimized)"
echo "============================================================"
echo "Configuration:"
echo "  - Backbone: CLIP ViT-B/16"
echo "  - Batch size: 6 (CLIP is heavier, use 6 instead of 8)"
echo "  - Max frames: 8"
echo "  - Mask temperature: 0.5 (proven optimal)"
echo "  - Collapse prevention: Stop-gradient"
echo "  - Expected GPU usage: 95%+"
echo "  - Expected memory: ~18GB / 32GB"
echo ""

# Check if open_clip is installed
python -c "import open_clip" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing open_clip..."
    uv pip install open-clip-torch
fi

python -u src/train_movi.py \
  --backbone clip_vitb16 \
  --data_dir data/movi_a_subset \
  --num_epochs 200 \
  --batch_size 6 \
  --num_slots 5 \
  --max_frames 8 \
  --lr 0.0004 \
  --diversity_weight 0.1 \
  --mask_temperature 0.5 \
  --use_stop_gradient \
  --save_dir checkpoints/clip_optimized_200ep

echo ""
echo "✅ Training complete!"
