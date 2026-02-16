# 単一フレームテスト結果

## 実施日: 2026-02-15

## 診断結果: 🔴 **完全にSlot Collapse**

### 可視化結果
- **全スロット単色** - 視覚的に全く分離していない
- **Attention map分析**:
  ```
  Coverage: 100% (全スロット)
  類似度: 1.0000 (完全同一)
  Max attention: 0.20 (= 1/5 = 完全均等)
  ```

### 根本原因: **対称性が全く破れていない**

全スロットが完全に同じAttention patternを持っています。これは：
1. 初期化が不十分（全スロットが同じ分布からサンプリング）
2. Diversity lossが弱すぎる（0.01 = 効果なし）
3. スロット競合メカニズムが機能していない

### Geminiの「divide and conquer」戦略の結論

**❌ ケースC: 完全失敗 → 問題は Spatial（空間軸）**

時間軸（GRU/Temporal）は無関係でした。Slot Attentionの基礎的な空間分離能力に問題があります。

## 次のアクション

### アプローチA: 強いDiversity Loss ⚡
```bash
cd src
bash fix_slot_collapse_A.sh  # diversity_weight=0.3
```

### アプローチB: Slot数を増やす 🔢
```bash
cd src
bash fix_slot_collapse_B.sh  # num_slots=5 → 10
```

### アプローチC: 初期化改善（実装必要）
- K-means clusteringベースの初期化
- Multi-head Slot Attention
- OCL frameworkの実装を参照

## 実験設定（失敗した設定）
```bash
python train_movi.py \
  --backbone dinov2_vits14 \
  --num_epochs 50 \
  --diversity_weight 0.01 \  # ← 弱すぎた
  --max_frames 1
```

## トレーニング結果
- **Final Loss: 1.276**
- **Epoch 2で爆発（57）** → その後回復
- 1エポック: ~0.5秒（非常に高速）
- 総時間: ~25秒

## 参考資料
- Attention maps: `checkpoints/single_frame_spatial/dinov2_vits14/attention_maps.png`
- Visualization: `checkpoints/single_frame_spatial/dinov2_vits14/movi_result.png`
- OCL Framework: `ocl_framework/ocl/perceptual_grouping.py` (Multi-head implementation)

