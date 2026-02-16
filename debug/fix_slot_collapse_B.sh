#!/bin/bash
# Slot Collapse対策: Slot数を増やして競合を強化

cd "$(dirname "$0")"
source ../.venv/bin/activate

echo "============================================================"
echo "Strategy B: More Slots (5 → 10)"
echo "============================================================"

python train_movi.py \
  --backbone dinov2_vits14 \
  --save_dir ../checkpoints/more_slots_single_frame \
  --num_epochs 100 \
  --num_slots 10 \
  --diversity_weight 0.1 \
  --max_frames 1 \
  --batch_size 2 \
  --lr 0.001 \
  2>&1 | tee ../logs/more_slots_single_frame_100ep.log

echo "✅ Training completed"
echo "Run: eog ../checkpoints/more_slots_single_frame/dinov2_vits14/movi_result.png"
