#!/bin/bash
# Slot Collapse対策: より強い Diversity Loss でリトレーニング

cd "$(dirname "$0")"
source ../.venv/bin/activate

echo "============================================================"
echo "Strategy A: Strong Diversity Loss (weight=0.3)"
echo "============================================================"

python train_movi.py \
  --backbone dinov2_vits14 \
  --save_dir ../checkpoints/strong_diversity_single_frame \
  --num_epochs 100 \
  --diversity_weight 0.3 \
  --max_frames 1 \
  --batch_size 2 \
  --lr 0.001 \
  2>&1 | tee ../logs/strong_diversity_single_frame_100ep.log

echo "✅ Training completed"
echo "Run: eog ../checkpoints/strong_diversity_single_frame/dinov2_vits14/movi_result.png"
