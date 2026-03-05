#!/bin/bash
# =============================================================================
# τ グリッドサーチ: DINOv2 K=11, image_size={224, 448}, τ={0.1, 0.3, 0.5, 0.7, 1.0}
# 推定時間: 10 runs × ~5 min = ~50 min 
# 448x448は4倍トークン → 訓練時間やや増加
# =============================================================================
set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

DATA_DIR="data/movi_a_v2"
SAVE_BASE="checkpoints/tau_sweep"
EPOCHS=200
BATCH_SIZE=32
LR=0.001
K=11

BACKBONE="dinov2_vits14"
LOSS_TYPE="mse"
IMAGE_SIZES=(224 448)
TAU_VALUES=(0.1 0.3 0.5 0.7 1.0)

echo "============================================"
echo "τ Grid Search Experiment"
echo "Backbone: $BACKBONE, K=$K"
echo "Image sizes: ${IMAGE_SIZES[*]}"
echo "τ values: ${TAU_VALUES[*]}"
echo "============================================"

for IMG_SIZE in "${IMAGE_SIZES[@]}"; do
    # Adjust batch size for 448 (4x more tokens = more memory)
    if [ "$IMG_SIZE" -eq 448 ]; then
        BS=8
    else
        BS=$BATCH_SIZE
    fi
    
    for TAU in "${TAU_VALUES[@]}"; do
        SAVE_DIR="${SAVE_BASE}/${BACKBONE}_${IMG_SIZE}_tau${TAU}"
        
        if [ -f "${SAVE_DIR}/${BACKBONE}/best_model.pt" ]; then
            echo "[SKIP] ${IMG_SIZE}px τ=${TAU} (already exists)"
            continue
        fi
        
        echo ""
        echo ">>> Training ${BACKBONE} ${IMG_SIZE}px τ=${TAU} (BS=${BS})"
        echo "    Save: ${SAVE_DIR}"
        
        python src/train_movi.py \
            --backbone "$BACKBONE" \
            --data_dir "$DATA_DIR" \
            --save_dir "$SAVE_DIR" \
            --num_epochs $EPOCHS \
            --batch_size $BS \
            --num_slots $K \
            --lr $LR \
            --diversity_weight 0.01 \
            --mask_temperature "$TAU" \
            --loss_type "$LOSS_TYPE" \
            --max_frames 1 \
            --image_size "$IMG_SIZE" \
            --num_workers 4
        
        echo "    ✓ ${IMG_SIZE}px τ=${TAU} done"
    done
done

echo ""
echo "============================================"
echo "τ Grid Search completed!"
echo "============================================"
