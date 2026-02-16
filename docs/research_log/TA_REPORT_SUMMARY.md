# TA報告：金属光沢物体の物体中心学習

**提出日**: 2026-02-15  
**テーマ**: DINOv2特徴量を用いた鏡面反射物体の物体中心分離

---

## 1. 研究目的

金属光沢（鏡面反射）を持つ物体に対する物体中心学習（Object-Centric Learning）の実現。従来のピクセルベースSlot Attentionでは色変化に脆弱であったため、DINOv2事前学習特徴量を活用。

---

## 2. 手法

### モデル構成
- **Backbone**: DINOv2 ViT-S/14（凍結）
- **Architecture**: DINOSAUR (DINOv2 + Slot Attention + U-Net Decoder)
- **Key Features**:
  * 2-layer MLP projection (384→384→64): 分散崩壊を防止
  * Temperature Scaling (τ=0.5): Mask競合を強化
  * Xavier初期化: 適切な初期分散を確保

### データセット
- **MOVi-A subset**: 60 samples (20 metal + 40 mixed)
- **Properties**: 24 frames/sample, 224×224 pixels
- **Objects**: Metal (鏡面反射) + Rubber/Plastic

### 訓練設定
```
Epochs: 200
Batch size: 2
Learning rate: 0.0004 (warmup 5 epochs + cosine decay)
Optimizer: Adam
Loss: Reconstruction + Diversity (weight=0.1)
```

---

## 3. 主要な技術的課題と解決策

### 課題1: スロット崩壊 (Slot Collapse)
**症状**: 5スロット中1つのみ活性化、残り無視  
**原因**: Single linear projection (384→64) による分散破壊

**解決策**: 2-layer MLP
```python
self.feature_projection = nn.Sequential(
    nn.LayerNorm(384),
    nn.Linear(384, 384),
    nn.ReLU(inplace=True),
    nn.Linear(384, 64)
)
```
**効果**: Mask Similarity 0.866 → 0.723 (16%改善)

---

### 課題2: マスクの均一化
**症状**: すべてのマスクが1/5に収束  
**原因**: mask_logitsの分散が極めて小さい (std=0.03)

**解決策**: Temperature Scaling
```python
mask_logits = mask_logits / self.mask_temperature  # τ=0.5
masks = torch.softmax(mask_logits, dim=1)
```

**τのアブレーション**:
| Temperature | Mask Similarity | 改善率 |
|-------------|-----------------|--------|
| 1.0 (baseline) | 0.723 | - |
| 0.7 | 0.642 | 11% |
| **0.5** | **0.558** | **23%** |
| 0.3 | 0.621 | 14% (過剰) |

**効果**: マスク分化が大幅に改善

---

## 4. 実験結果

### 定量的結果
```
Training Loss: 0.729 (200 epochs)
Test Loss: 1.710
Mask Similarity: 0.558 (目標: <0.5)
Training Time: ~10分 (RTX 5090)
```

### 視覚的結果
- ✅ 金属物体の輪郭を検出
- ✅ 複数スロットが分化（単一支配を回避）
- ⚠️ マスク境界のにじみ（16×16解像度の構造的限界）

**参考画像**: `checkpoints/temp_scaling_tau05/dinov2_vits14/movi_result.png`

---

## 5. 他のバックボーン（DINOv1, CLIP）の検証

### 動機
複数のViT事前学習手法を比較し、鏡面反射物体への適性を評価。

### 実験試行と結果

#### 試行1: 空間次元正規化
```python
mean = features.mean(dim=(2, 3), keepdim=True)
std = features.std(dim=(2, 3), keepdim=True)
features = (features - mean) / std
```
**結果**: ❌ 全モデルで空間構造が破壊（単一色塗りつぶし）

#### 試行2: Fixed Scaling
```python
# DINOv1: std 3.7 → 2.4 (×0.65)
# CLIP: std 0.5 → 2.4 (×4.8)
```
**結果**: ❌ 物体輪郭が消失、相対的特徴関係が破壊

#### 試行3: Baseline（正規化なし）
**結果**: 
- DINOv1: Loss 4.692 (DINOv2比 **6倍悪化**)
- CLIP: Loss -0.005 (**負の値、訓練不安定**)

### 特徴量スケール分析
| Backbone | Feature Std | 状態 |
|----------|-------------|------|
| DINOv2 | 2.41 | ✅ 安定 |
| DINOv1 | 3.74 (+55%) | ❌ 訓練発散 |
| CLIP | 0.47 (-80%) | ❌ 負のLoss |

### 結論
**バックボーン間の特徴量スケール差が、同一ハイパーパラメータでの訓練を不可能にする**。正規化/スケーリングは空間構造を破壊するため使用できない。

**DINOv2が最も適している理由**:
1. 適切な特徴量スケール（std ~2.4）
2. 豊富な空間情報を保持
3. 訓練の安定性

---

## 6. 重要な教訓

### 1. 定量指標と視覚的品質の乖離
- CLIP: Loss 0.006-0.022（数値的に最良）→ 視覚的に最悪（全塗りつぶし）
- **教訓**: Loss値だけでなく視覚的検証が必須

### 2. 特徴量正規化の危険性
- 平均/分散正規化 → 空間情報の損失
- スケーリング → 相対的関係の破壊
- **教訓**: 凍結バックボーンの特徴量は触らない

### 3. アーキテクチャ詳細の重要性
- Single linear vs 2-layer MLP: 150倍の分散差
- Temperature scaling: 23%の改善
- **教訓**: 論文の実装詳細を正確に再現

### 4. ハードウェア最適化 vs 品質
- Batch size 2→16: GPU 87%達成 → Loss 5倍悪化
- **教訓**: 効率と品質のトレードオフ慎重に

---

## 7. 構造的限界と今後の課題

### 解像度の限界
- **16×16パッチ**: 1パッチ = 14×14ピクセル = 196px²
- **問題**: 小物体のマスクにじみ（19-31%）
- **解決策**: CRF後処理、32×32解像度での訓練

### Video Modeの課題
- Slot Predictor実装 → フレーム進行で崩壊
- Stop-gradient有効だが、さらなる最適化が必要

### 定量評価の不足
- ARI, mIoU等の標準メトリクス未実装
- Ground-truthセグメンテーションとの比較が必要

---

## 8. 結論

**成果**:
- ✅ DINOv2 + Temperature Scaling でスロット崩壊を克服
- ✅ 金属光沢物体の輪郭検出に成功
- ✅ 2層MLP + τ=0.5 の有効性を実証

**限界**:
- ❌ 他のバックボーン（DINOv1, CLIP）は特徴量スケール問題で訓練不可
- ⚠️ 16×16解像度によるマスク境界のにじみ

**実用的示唆**:
- 物体中心学習には**バックボーンの選択が決定的**
- DINOv2の特徴量特性（スケール、空間情報）が最適
- 正規化/スケーリングは安易に適用すべきでない

---

## 9. 再現性

### コードベース
- **メインモデル**: `src/savi_dinosaur.py` (852行)
- **訓練スクリプト**: `src/train_movi.py`
- **チェックポイント**: `checkpoints/temp_scaling_tau05/dinov2_vits14/`

### 訓練コマンド
```bash
python src/train_movi.py \
  --backbone dinov2_vits14 \
  --data_dir data/movi_a_subset \
  --num_epochs 200 \
  --max_frames 1 \
  --mask_temperature 0.5 \
  --batch_size 2 \
  --lr 0.0004 \
  --diversity_weight 0.1 \
  --save_dir checkpoints/temp_scaling_tau05
```

### ハードウェア要件
- **GPU**: RTX 5090 (32GB VRAM) または同等
- **RAM**: 16GB以上
- **訓練時間**: 約10分（200 epochs）

---

## 10. 謝辞・参考文献

### 実装参考
- DINOSAUR論文: Singh et al., 2022
- OCL Framework: [ocl-framework GitHub](https://github.com/amazon-science/object-centric-learning-framework)
- MOVi Dataset: Greff et al., 2021

### デバッグ支援
- 膨大な試行錯誤の結果、最終的にDINOv2のみが成功

---

**提出ファイル**:
- 本レポート (`docs/TA_REPORT_SUMMARY.md`)
- 研究ログ (`docs/RESEARCH_LOG.md`)
- チェックポイント (`checkpoints/temp_scaling_tau05/`)
- 結果画像 (`checkpoints/temp_scaling_tau05/dinov2_vits14/movi_result.png`)

*最終更新: 2026-02-15 21:50 JST*
