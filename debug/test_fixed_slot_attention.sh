#!/bin/bash
# Over-smoothing修正版のテスト

cd "$(dirname "$0")"
source ../.venv/bin/activate

echo "============================================================"
echo "Testing Fixed Slot Attention (Xavier init + Temperature)"
echo "============================================================"

# SlotAttentionFixedに置き換えたモデルで単一フレームテスト
python train_movi.py \
  --backbone dinov2_vits14 \
  --save_dir ../checkpoints/fixed_init_single_frame \
  --num_epochs 100 \
  --diversity_weight 0.1 \
  --max_frames 1 \
  --batch_size 2 \
  --lr 0.001 \
  2>&1 | tee ../logs/fixed_init_single_frame_100ep.log

echo ""
echo "✅ Training completed"
echo ""
echo "Next steps:"
echo "1. Check visualization: eog ../checkpoints/fixed_init_single_frame/dinov2_vits14/movi_result.png"
echo "2. Analyze dynamics: python analyze_iteration_dynamics.py --checkpoint ../checkpoints/fixed_init_single_frame/dinov2_vits14/best_model.pt"
