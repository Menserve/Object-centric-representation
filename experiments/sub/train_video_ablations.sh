#!/bin/bash
# Video Mode Ablation Study: Slot Predictor Collapse Prevention
# ==============================================================
# 3つの設定を比較：
# 1. Baseline: オリジナル（崩壊する）
# 2. Stop-gradient のみ
# 3. Stop-gradient + Refresh (推奨)

cd /home/menserve/Object-centric-representation
source .venv/bin/activate

COMMON_ARGS="--backbone dinov2_vits14 \
  --data_dir data/movi_a_subset \
  --num_epochs 50 \
  --batch_size 2 \
  --num_slots 5 \
  --max_frames 8 \
  --lr 0.0004 \
  --diversity_weight 0.1 \
  --mask_temperature 0.5"

echo "=================================================="
echo "Ablation 1/3: Baseline (no prevention)"
echo "=================================================="
python src/train_movi.py $COMMON_ARGS \
  --save_dir checkpoints/video_baseline \
  > logs/video_baseline_50ep.log 2>&1

echo ""
echo "=================================================="
echo "Ablation 2/3: Stop-gradient only"
echo "=================================================="
python src/train_movi.py $COMMON_ARGS \
  --save_dir checkpoints/video_stopgrad \
  --use_stop_gradient \
  > logs/video_stopgrad_50ep.log 2>&1

echo ""
echo "=================================================="
echo "Ablation 3/3: Stop-gradient + Refresh (every 4 frames)"
echo "=================================================="
python src/train_movi.py $COMMON_ARGS \
  --save_dir checkpoints/video_stopgrad_refresh4 \
  --use_stop_gradient \
  --refresh_interval 4 \
  > logs/video_stopgrad_refresh4_50ep.log 2>&1

echo ""
echo "✅ All ablations completed!"
echo "Check logs/video_*.log for results"
