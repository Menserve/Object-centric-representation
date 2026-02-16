#!/bin/bash
# Phase 2: 3つのbackboneの学習を実行するスクリプト
# =========================================================
# 
# メイン機とサブ機で並列実行する場合の例:
#
# メイン機 (RTX5090):
#   - DINOv2 (既存のベースライン)
#   - DINOv1
#
# サブ機 (RTX4080 Super):
#   - CLIP
#

# 共通設定
DATA_DIR="../data/movi_a_subset"
NUM_EPOCHS=200
BATCH_SIZE=2
NUM_SLOTS=5
MAX_FRAMES=12
LR=0.001
DIVERSITY_WEIGHT=0.3

# 仮想環境の有効化
source ../.venv/bin/activate

echo "=========================================="
echo "Phase 2: Multi-Backbone Training"
echo "=========================================="
echo ""

# ========================================
# オプション1: 順次実行（1台のマシンで全て）
# ========================================
run_sequential() {
    echo "Running sequential training..."
    echo ""
    
    # DINOv2
    echo "[1/3] Training with DINOv2 ViT-S/14..."
    python train_movi.py \
        --backbone dinov2_vits14 \
        --data_dir ${DATA_DIR} \
        --save_dir ../checkpoints/dinov2_vits14 \
        --num_epochs ${NUM_EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --num_slots ${NUM_SLOTS} \
        --max_frames ${MAX_FRAMES} \
        --lr ${LR} \
        --diversity_weight ${DIVERSITY_WEIGHT}
    
    echo ""
    
    # DINOv1
    echo "[2/3] Training with DINOv1 ViT-S/16..."
    python train_movi.py \
        --backbone dino_vits16 \
        --data_dir ${DATA_DIR} \
        --save_dir ../checkpoints/dino_vits16 \
        --num_epochs ${NUM_EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --num_slots ${NUM_SLOTS} \
        --max_frames ${MAX_FRAMES} \
        --lr ${LR} \
        --diversity_weight ${DIVERSITY_WEIGHT}
    
    echo ""
    
    # CLIP
    echo "[3/3] Training with CLIP ViT-B/16..."
    python train_movi.py \
        --backbone clip_vitb16 \
        --data_dir ${DATA_DIR} \
        --save_dir ../checkpoints/clip_vitb16 \
        --num_epochs ${NUM_EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --num_slots ${NUM_SLOTS} \
        --max_frames ${MAX_FRAMES} \
        --lr ${LR} \
        --diversity_weight ${DIVERSITY_WEIGHT}
    
    echo ""
    echo "✅ Sequential training completed!"
}

# ========================================
# オプション2: 並列実行（複数GPU）
# ========================================
run_parallel() {
    echo "Running parallel training..."
    echo "Make sure you have multiple GPUs available!"
    echo ""
    
    # DINOv2 (GPU 0)
    CUDA_VISIBLE_DEVICES=0 python train_movi.py \
        --backbone dinov2_vits14 \
        --data_dir ${DATA_DIR} \
        --save_dir ../checkpoints/dinov2_vits14 \
        --num_epochs ${NUM_EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --num_slots ${NUM_SLOTS} \
        --max_frames ${MAX_FRAMES} \
        --lr ${LR} \
        --diversity_weight ${DIVERSITY_WEIGHT} \
        > ../logs/dinov2_training.log 2>&1 &
    
    PID1=$!
    echo "Started DINOv2 training (PID: ${PID1}, GPU: 0)"
    
    # 少し待ってから次を起動（メモリ確保のため）
    sleep 30
    
    # DINOv1 (GPU 0, バッチサイズを小さくして同じGPUで実行可能)
    # または別GPUがあれば CUDA_VISIBLE_DEVICES=1 に変更
    CUDA_VISIBLE_DEVICES=0 python train_movi.py \
        --backbone dino_vits16 \
        --data_dir ${DATA_DIR} \
        --save_dir ../checkpoints/dino_vits16 \
        --num_epochs ${NUM_EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --num_slots ${NUM_SLOTS} \
        --max_frames ${MAX_FRAMES} \
        --lr ${LR} \
        --diversity_weight ${DIVERSITY_WEIGHT} \
        > ../logs/dinov1_training.log 2>&1 &
    
    PID2=$!
    echo "Started DINOv1 training (PID: ${PID2}, GPU: 0)"
    
    echo ""
    echo "Training processes started in background."
    echo "Monitor progress with:"
    echo "  tail -f ../logs/dinov2_training.log"
    echo "  tail -f ../logs/dinov1_training.log"
    echo ""
    echo "Note: CLIP should be run on a separate machine (sub-machine)"
    echo ""
    
    # プロセスの完了を待つ
    wait ${PID1}
    echo "✓ DINOv2 training completed"
    
    wait ${PID2}
    echo "✓ DINOv1 training completed"
    
    echo ""
    echo "✅ Parallel training completed!"
}

# ========================================
# オプション3: サブ機用（CLIP単独）
# ========================================
run_clip_only() {
    echo "Running CLIP training only (for sub-machine)..."
    echo ""
    
    python train_movi.py \
        --backbone clip_vitb16 \
        --data_dir ${DATA_DIR} \
        --save_dir ../checkpoints/clip_vitb16 \
        --num_epochs ${NUM_EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --num_slots ${NUM_SLOTS} \
        --max_frames ${MAX_FRAMES} \
        --lr ${LR} \
        --diversity_weight ${DIVERSITY_WEIGHT}
    
    echo ""
    echo "✅ CLIP training completed!"
}

# ========================================
# オプション4: デバッグモード（少量エポックでテスト）
# ========================================
run_debug() {
    echo "Running debug mode (10 epochs)..."
    echo ""
    
    DEBUG_EPOCHS=10
    
    for BACKBONE in dinov2_vits14 dino_vits16 clip_vitb16; do
        echo "Testing ${BACKBONE}..."
        python train_movi.py \
            --backbone ${BACKBONE} \
            --data_dir ${DATA_DIR} \
            --save_dir ../checkpoints/debug_${BACKBONE} \
            --num_epochs ${DEBUG_EPOCHS} \
            --batch_size ${BATCH_SIZE} \
            --num_slots ${NUM_SLOTS} \
            --max_frames ${MAX_FRAMES} \
            --lr ${LR} \
            --diversity_weight ${DIVERSITY_WEIGHT}
        echo ""
    done
    
    echo "✅ Debug run completed!"
}

# ========================================
# メイン
# ========================================

# ログディレクトリの作成
mkdir -p ../logs

# コマンドライン引数でモードを選択
MODE=${1:-"help"}

case ${MODE} in
    sequential)
        run_sequential
        ;;
    parallel)
        run_parallel
        ;;
    clip)
        run_clip_only
        ;;
    debug)
        run_debug
        ;;
    *)
        echo "Usage: $0 {sequential|parallel|clip|debug}"
        echo ""
        echo "Modes:"
        echo "  sequential  - Train all 3 backbones one after another (single machine)"
        echo "  parallel    - Train DINOv2 and DINOv1 in parallel (main machine)"
        echo "  clip        - Train CLIP only (for sub-machine)"
        echo "  debug       - Quick test with 10 epochs for all backbones"
        echo ""
        echo "Examples:"
        echo "  # メイン機（順次実行）"
        echo "  ./run_phase2_training.sh sequential"
        echo ""
        echo "  # メイン機（並列実行）"
        echo "  ./run_phase2_training.sh parallel"
        echo ""
        echo "  # サブ機（CLIP単独）"
        echo "  ./run_phase2_training.sh clip"
        echo ""
        echo "  # デバッグ（全backbone 10エポック）"
        echo "  ./run_phase2_training.sh debug"
        exit 1
        ;;
esac
