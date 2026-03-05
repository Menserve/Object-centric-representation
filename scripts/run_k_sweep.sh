#!/bin/bash
# =============================================================================
# K Sweep 実験: K={3,5,7,9,11,13} × 3 backbones
# 推定時間: 18 runs × ~5 min = ~90 min
# =============================================================================
set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

DATA_DIR="data/movi_a_v2"
SAVE_BASE="checkpoints/k_sweep"
EPOCHS=200
BATCH_SIZE=32
LR=0.001

BACKBONES=("dinov2_vits14" "dino_vits16" "clip_vitb16")
LOSS_TYPES=("mse" "channel_norm" "mse")  # per-backbone loss
K_VALUES=(3 5 7 9 11 13)

echo "============================================"
echo "K Sweep Experiment"
echo "Backbones: ${BACKBONES[*]}"
echo "K values: ${K_VALUES[*]}"
echo "Epochs: $EPOCHS, LR: $LR, BS: $BATCH_SIZE"
echo "============================================"

for i in "${!BACKBONES[@]}"; do
    bb="${BACKBONES[$i]}"
    loss="${LOSS_TYPES[$i]}"
    for K in "${K_VALUES[@]}"; do
        SAVE_DIR="${SAVE_BASE}/${bb}_K${K}"
        
        if [ -f "${SAVE_DIR}/${bb}/best_model.pt" ]; then
            echo "[SKIP] ${bb} K=${K} (already exists)"
            continue
        fi
        
        echo ""
        echo ">>> Training ${bb} K=${K} (loss=${loss})"
        echo "    Save: ${SAVE_DIR}"
        
        python src/train_movi.py \
            --backbone "$bb" \
            --data_dir "$DATA_DIR" \
            --save_dir "$SAVE_DIR" \
            --num_epochs $EPOCHS \
            --batch_size $BATCH_SIZE \
            --num_slots $K \
            --lr $LR \
            --diversity_weight 0.01 \
            --mask_temperature 0.5 \
            --loss_type "$loss" \
            --max_frames 1 \
            --num_workers 4
        
        echo "    ✓ ${bb} K=${K} done"
    done
done

echo ""
echo "============================================"
echo "K Sweep completed!"
echo "============================================"
