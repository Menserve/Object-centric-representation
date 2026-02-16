#!/bin/bash
# Quick Ablation: 20 epochs to verify collapse prevention
cd /home/menserve/Object-centric-representation
source .venv/bin/activate

COMMON="--backbone dinov2_vits14 --data_dir data/movi_a_subset \
  --num_epochs 20 --batch_size 2 --num_slots 5 --max_frames 8 \
  --lr 0.0004 --diversity_weight 0.1 --mask_temperature 0.5"

echo "Training 3 configurations (20 epochs each, ~5 min total)..."
echo ""

# 1. Baseline (control)
echo "[1/3] Baseline..."
python src/train_movi.py $COMMON \
  --save_dir checkpoints/quick_baseline \
  > logs/quick_baseline.log 2>&1 &
PID1=$!

# 2. Stop-gradient only
echo "[2/3] Stop-gradient..."
python src/train_movi.py $COMMON \
  --save_dir checkpoints/quick_stopgrad \
  --use_stop_gradient \
  > logs/quick_stopgrad.log 2>&1 &
PID2=$!

# 3. Stop-gradient + Refresh
echo "[3/3] Stop-gradient + Refresh (every 4 frames)..."
python src/train_movi.py $COMMON \
  --save_dir checkpoints/quick_stopgrad_refresh4 \
  --use_stop_gradient \
  --refresh_interval 4 \
  > logs/quick_stopgrad_refresh4.log 2>&1 &
PID3=$!

echo ""
echo "Training in parallel (PIDs: $PID1, $PID2, $PID3)"
echo "Monitor progress: tail -f logs/quick_*.log"
echo ""

# Wait for all
wait $PID1 $PID2 $PID3

echo ""
echo "✅ All training completed!"
echo ""
echo "Compare results:"
echo "  python src/compare_quick_ablations.py"
