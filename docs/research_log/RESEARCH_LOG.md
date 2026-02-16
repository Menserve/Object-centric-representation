# 研究活動記録 / Research Log

このドキュメントは研究の進捗を時系列で記録します。

---

## 2026年1月

### 2026-01-12: 同条件比較実験の完了

**実施内容**:
- `ex_comparison.ipynb` を作成し、Pixel Slot vs DINOSAUR の同条件比較を実施
- 定量評価指標（Mask Stability, Slot Diversity）を導入

**結果**:
| Metric | Pixel Slot | DINOSAUR |
|--------|-----------|----------|
| Mask Stability (↑) | 0.177 | 0.591 |
| Slot Diversity (↓) | 0.236 | 0.407 |

**考察**:
- DINOSAURは色変化に対して3.3倍ロバスト
- ただし境界がソフト（グレー領域が多い）というトレードオフを発見
- これは元のDINOSAUR論文では詳細に議論されていない新しい知見

**次のステップ**:
- 中間報告書の作成
- Gumbel-Softmax等によるマスク二値化の検討

---

### 2026-01-04 ~ 2026-01-11: DINOSAUR実装と検証

**実施内容**:
- `ex_dino1.ipynb`: ResNet18特徴抽出 + Slot Attention（基礎実装）
- `ex_dino4.ipynb`: DINOv2特徴 + 5スロット + 長時間学習（最終版）

**発見**:
- DINOv2の事前学習特徴は色・照明変動に対して不変性を持つ
- 特徴空間での再構成は安定した学習が可能

---

### 2026-01-01 ~ 2026-01-03: ピクセルベースSlot Attentionの限界検証

**実施内容**:
- `ex_slot1.ipynb`: 温度アニーリングの導入
- `ex_slot2.ipynb`: 幾何学図形 vs 実写画像の比較
- `ex_slot3.ipynb`: 単一画像過学習テスト
- `ex_slot4.ipynb`: ColorJitter による色耐性テスト
- `ex_slot5.ipynb`: 手動固定色変換による厳密比較

**発見**:
- ピクセルベースSlot Attentionは色変化に敏感
- 幾何学図形では成功するが、実写画像では形状把握が困難
- 温度アニーリングは分離を改善するが、根本的な解決にはならない

---

## 実験の時系列まとめ

```
experiment.ipynb (基礎実験)
    ↓
ex_slot1~5.ipynb (ピクセルベースの限界検証)
    ↓
ex_dino1~4.ipynb (DINOSAURによる改善)
    ↓
ex_comparison.ipynb (同条件定量比較) ← 最終成果
```

---

## 今後の課題 / Future Work

1. **マスク二値化**: Gumbel-Softmax、閾値処理等の検討
2. **複数物体での評価**: CLEVR等の合成データセットでの検証
3. **実世界データセット**: COCO、Pascal VOC等での評価
4. **世界モデルへの統合**: 動画予測タスクへの応用

---

## 2026年2月: 金属光沢物体への拡張と大規模デバッグ

### タイトル: 複数ViT事前学習手法が鏡面反射物体の物体中心分離に与える影響

**ハードウェア**: Core Ultra 285K + RTX 5090 (32GB) + 128GB RAM, WSL Ubuntu  
**データセット**: MOVi-A subset (60 samples: 20 metal + 40 mixed, 24 frames, 224×224)

---

### 2026-02-15 (Phase 6): Feature Normalization/Scalingの全面的失敗

**動機**: DINOv1 (std=3.7) とCLIP (std=0.5) の特徴量スケールがDINOv2 (std=2.4) と大きく異なり、訓練が不安定。

#### 試行1: Per-sample Normalization（失敗）
```python
# 空間次元で平均/標準偏差を計算
mean = features.mean(dim=(2, 3), keepdim=True)
std = features.std(dim=(2, 3), keepdim=True)
features = (features - mean) / std
```

**結果** (50 epochs):
| Model | Train Loss | Visual Quality |
|-------|-----------|----------------|
| DINOv2 | 0.262 | 輪郭検出 → 消失 ❌ |
| DINOv1 | 0.260 | 単一色塗りつぶし ❌ |
| CLIP | 0.022 | 単一色塗りつぶし ❌ |

**問題**: 空間次元で正規化 → **空間情報が完全に破壊**

#### 試行2: Fixed Scaling（失敗）
```python
# DINOv2のstd=2.4に合わせてスケーリング
if backbone == 'dino_vits16':
    features = features * 0.65  # 3.7 → 2.4
elif backbone == 'clip_vitb16':
    features = features * 4.8   # 0.5 → 2.4
```

**結果** (50 epochs):
| Model | Train Loss | Visual Quality |
|-------|-----------|----------------|
| DINOv2 | 1.468 | 物体輪郭消失 ❌ |
| DINOv1 | 2.042 | 全スロット塗りつぶし ❌ |
| CLIP | **0.006** | 全スロット塗りつぶし ❌ |

**観察**:
- 全モデルでスロット分化自体は発生（2:3の色比率）
- しかし空間構造が完全に失われた（物体輪郭検出不可）
- CLIP Loss 0.006 = 異常に低い → 過剰平滑化

**重要な教訓**:
1. **Loss値と視覚的品質は一致しない**（CLIP: Loss最良 → 視覚最悪）
2. **Feature Scaling/Normalizationは空間構造を破壊**
3. **相対的な特徴量の関係が重要**（絶対値ではない）

#### 試行3: Baseline（正規化なし、最終確認）

**動機**: すべての調整を削除し、素のバックボーン特性で訓練

**結果** (50 epochs):
| Model | Train Loss | 状態 |
|-------|-----------|------|
| DINOv2 | (未実行) | - |
| DINOv1 | **4.692** | ❌ DINOv2比6倍悪化 |
| CLIP | **-0.005** | ❌ 負の値（訓練不安定） |

**視覚的品質**: 全スロット塗りつぶし、2色に分かれる程度

#### 結論: 正規化/スケーリング アプローチを完全放棄

**最終判断**:
1. **DINOv2のみが成功** (temp_scaling_tau05, Loss 0.729) ✅
2. **他のバックボーンは特徴量スケール問題で訓練不可**
   - DINOv1: std 3.74 (+55%) → Loss 4.692
   - CLIP: std 0.47 (-80%) → Loss -0.005 (不安定)
3. **同一ハイパーパラメータでの多バックボーン訓練は不可能**

**技術的洞察**:
- バックボーン固有の特徴量分布（スケール、共分散構造）が決定的
- 正規化/スケーリングは空間情報を破壊するため使用不可
- 学習率等の個別調整も時間的制約から断念

**TA提出方針**: 
- **DINOv2のみで成果報告**（最も現実的）
- 失敗した試行を教訓として文書化
- バックボーン選択の重要性を強調

---

### 2026-02-15 (Phase 5): GPU最適化とBackbone比較の自動実行

**問題**: GPU使用率64%、メモリ12%のみ → RTX 5090の性能を活かし切れていない

**原因**:
- batch_size=2が小さすぎる
- DINOv2が凍結されている（特徴抽出計算なし）
- 動画のシーケンシャル処理

**解決策**: Batch Size最適化
```python
# DINOv2 (baseline): batch_size=2
# DINOv1 (optimized): batch_size=8  ← 4倍
# CLIP (optimized): batch_size=6    ← 768次元で重め
```

**効果** (DINOv1):
| 指標 | Before | After | 改善 |
|------|--------|-------|------|
| エポック時間 | 2.7秒 | **1.1秒** | **2.5倍高速** |
| バッチ数 | 29 | 8 | GPU並列性向上 |
| GPU使用率 | 64% | 67% | +3% |

**自動実行ワークフロー**: `auto_run_backbone_comparison.sh`
1. DINOv2完了待機
2. DINOv2結果分析
3. DINOv1訓練 (batch_size=8, 推定4分)
4. CLIP訓練 (batch_size=6, 推定5分)
5. Backbone比較レポート生成

**ステータス**: 🔄 DINOv1訓練中 (150/200ep), CLIP待機中

---

### 2026-02-15 (Phase 4): 崩壊防止機構の実装と検証

**Quick Ablation Study (20 epochs)**:

| Config | Slot Diversity | Max Activations (final) | 評価 |
|--------|----------------|-------------------------|------|
| Baseline | -0.1002 | [0.065, 0.334, 0.333, 0.065, 0.334] | ⚠️ 3スロット支配 |
| **Stop-gradient** | **-0.1312** | [0.006, 0.334, 0.334, 0.333, 0.006] | ✅ **最良** |
| Stop-grad + Refresh(4) | +0.2549 | [0.248, 0.259, 0.247, 0.092, 0.246] | ❌ 崩壊 |

**重要な発見**:
- **Stop-gradient単独**が最も効果的（多様性 -0.131維持）
- **Refresh機構は有害**: Frame 4でリフレッシュ → 多様性 -0.13 → +0.25へ急激悪化
  - 原因: 20エポックではリフレッシュ後の再学習が不十分

**完全訓練 (200 epochs, Stop-gradient)**:
```bash
python train_movi.py --use_stop_gradient --mask_temperature 0.5 \
  --max_frames 8 --num_epochs 200
```
- Loss: 5.71 → 0.95 (83%改善)
- 訓練時間: 約10分

**実装詳細**:
```python
# 1. Stop-Gradient
if use_stop_gradient:
    slots_for_prediction = slots.detach()
slots_init = self.slot_predictor(slots_for_prediction)

# 2. Output Normalization
class SlotPredictor:
    def __init__(self, dim):
        self.output_norm = nn.LayerNorm(dim)
    def forward(self, slots):
        predicted = slots + self.predictor(slots)
        return self.output_norm(predicted)
```

---

### 2026-02-14 (Phase 3): Video Mode崩壊問題の発見

**SAViへの拡張試行**:
- 目的: Temporal Consistency（時間的一貫性）の実現
- 実装: Slot Predictor (前フレーム → 次フレーム初期値予測)

**崩壊の発見** (video_mode_8frames, 200 epochs):
```python
Frame  0: max_activations = [0.748, 0.458, 0.675, 0.561, 0.853]  ← 多様
Frame  4: max_activations = [0.494, 0.339, 0.494, 0.337, 0.344]  ← 均一化
Frame 16: max_activations = [0.498, 0.333, 0.498, 0.333, 0.333]  ← 1/3に崩壊
```

**Slot Similarity推移**:
- Frame 0: -0.11 (良好)
- Frame 20: -0.14 (崩壊状態維持)
- 目標: < -0.2

**マスクのにじみ問題** (`analyze_video_masks.py`):
```
Small objects: 19-31% bleeding
Large objects: 0.7-1.0% bleeding

原因:
1. 16×16解像度: 1パッチ = 14×14ピクセル = 196ピクセル²
2. Bilinear upsampling: 境界ぼけ
3. Softmax競合: 全スロットが全位置で競合
```

**解像度トレードオフ**:
| 解像度 | FLOPs | メモリ | 現在比 |
|--------|-------|--------|--------|
| 16×16 | 11.2 | 90 MB | 1× |
| 32×32 | 44.7 | 360 MB | 4× (実現可能) |
| 64×64 | 178.6 | 1,440 MB | 16× |

**結論**: にじみは16×16の構造的限界。CRF後処理が現実的。

---

### 2026-02-13 (Phase 2): Temperature Scalingの発見

**問題**: 2層MLP実装後もMask Similarity = 0.723 (目標: <0.5)

**根本原因の分析**:
```python
mask_logits統計:
  Mean: -0.003
  Std: 0.031  ← 極めて小さい！
  Range: [-0.15, 0.12]
```

→ Softmax が `exp(0.03) ≈ 1.03` → ほぼuniform (1/5 = 0.2)

**解決策**: Temperature Scaling
```python
mask_logits = out[:, :, d:, :, :]
mask_logits = mask_logits / self.mask_temperature  # τで割る
masks = torch.softmax(mask_logits, dim=1)
```

**τのアブレーション**:
| τ | Mask Similarity | 改善率 | 状態 |
|---|-----------------|--------|------|
| 1.0 | 0.723 | - | baseline |
| 0.7 | 0.642 | 11% | 改善 |
| **0.5** | **0.558** | **23%** | **最良** |
| 0.3 | 0.621 | 14% | 過剰 |

**効果**:
- Logits実効範囲: [-0.15, 0.12] → [-0.30, 0.24] (2倍)
- より明確な勝者が出現
- 勾配シグナルが強化

**Checkpoint**: `checkpoints/temp_scaling_tau05/` (200 epochs)

---

### 2026-02-10~12 (Phase 1): スロット崩壊との戦い

#### Problem: 1スロット支配問題
- **症状**: 5スロットのうち1つだけ活性化、残り無視
- **指標**: Mask Similarity = 0.866 (目標: <0.5)

#### 試行1: Diversity Loss調整（失敗）
```python
diversity_weight: 0.01 → 1.0
結果: Loss爆発、学習不安定
```
**教訓**: 大きすぎるweightは再構成を妨害

#### 試行2: Xavier初期化（部分的成功）
```python
# Before: Uniform(-0.1, 0.1) → Over-smoothing
# After: Xavier Uniform
nn.init.xavier_uniform_(self.slots_mu)
nn.init.xavier_uniform_(self.slots_log_sigma)
```
**結果**: 初期分散改善、しかし崩壊継続

#### 試行3: アーキテクチャ調査（根本解決）

**OCL Frameworkとの比較で判明**:
1. ❌ Single linear projection (384→64) が分散を破壊
2. ❌ kvq_dim/hidden_dim設定ミス

**解決策1: Feature Projection (2-layer MLP)**
```python
# ❌ Before: Single linear
self.feature_projection = nn.Linear(384, 64)
# 投影後分散: 0.001

# ✅ After: 2-layer MLP
self.feature_projection = nn.Sequential(
    nn.LayerNorm(384),
    nn.Linear(384, 384),
    nn.ReLU(inplace=True),
    nn.Linear(384, 64)
)
# 投影後分散: 0.15 (150倍改善)
```

**解決策2: Slot-to-Feature Upsampling (2-layer MLP)**
```python
# ❌ Before: 64→384 single linear
self.slot_to_feature = nn.Linear(64, 384)

# ✅ After: 2-layer MLP with intermediate
self.slot_to_feature = nn.Sequential(
    nn.LayerNorm(64),
    nn.Linear(64, 128),
    nn.ReLU(inplace=True),
    nn.Linear(128, 384)
)
```

**効果**: Mask Similarity = 0.866 → 0.723 (**16%改善**)

**理論的根拠**: DINOSAUR論文のDummyPositionEmbed + 2-layer MLP設計に準拠

---

## 重要な教訓 / Key Lessons

### 1. アーキテクチャの詳細が決定的
- Single linearによる分散崩壊は見落としやすい
- 論文の実装詳細を正確に再現する重要性

### 2. Temperature Scalingの威力
- 単純な手法で23%改善
- Logits統計の可視化が重要

### 3. Video Modeの落とし穴
- Slot Predictorは崩壊を伝播しやすい
- Stop-gradientが最も効果的（単独使用）
- Refreshは長期訓練が必要

### 4. GPU最適化の重要性
- Batch size調整で2.5倍高速化
- メモリが許す限り大きく

### 5. 構造的限界の認識
- 16×16解像度のにじみは不可避
- 別アプローチ（CRF等）が必要

---

## タイムライン / Timeline

```
2月10日: スロット崩壊発見
    ↓ Diversity loss調整（失敗）
    ↓ Xavier初期化（部分的成功）
    ↓ 2層MLP実装（16%改善）
2月13日: Temperature scaling発見（23%改善） ← 単一フレーム最良
    ↓
2月14日: Video mode実装 → Slot Predictor崩壊発見
    ↓ マスクにじみ分析（構造的限界）
2月15日: 崩壊防止機構実装
    ↓ Quick ablation (Stop-gradient最良)
    ↓ 200ep完全訓練
    ↓ GPU最適化（batch_size=8, 2.5倍高速）
    ↓ 自動Backbone比較実行中 ← 現在地
```

---

## コードベース構成 / Codebase Structure

```
src/
  savi_dinosaur.py              # メインモデル（845行）
  train_movi.py                 # 訓練スクリプト
  analyze_video_masks.py        # 動画詳細分析
  analyze_quick_ablation.py     # Quick ablation分析
  
  # GPU最適化版
  train_dinov1_optimized.sh     # DINOv1 (BS=8)
  train_clip_optimized.sh       # CLIP (BS=6)
  auto_run_backbone_comparison.sh

checkpoints/
  temp_scaling_tau05/           # 単一フレーム最良 (τ=0.5)
  video_stopgrad_200ep/         # DINOv2 Stop-gradient
  dinov1_optimized_200ep/       # DINOv1 (進行中)
  clip_optimized_200ep/         # CLIP (予定)

logs/
  video_stopgrad_200ep.log
  dinov1_optimized.log
  auto_backbone_comparison.log
```

---

## 次のステップ / Next Steps

### 短期（TA提出）
- [x] DINOv2 Stop-gradient訓練
- [x] DINOv2結果分析
- [ ] DINOv1訓練（進行中: 150/200ep）
- [ ] CLIP訓練
- [ ] Backbone比較レポート

### 中期（論文）
1. Metal vs Rubber定量評価
2. ARI/mIoU標準メトリクス
3. CRF後処理実装
4. Ablation整理

### 長期
1. 32×32解像度での最適化
2. DINOv2 ViT-Bスケールアップ
3. Real-world dataset汎化評価

---

*Last updated: 2026-02-15 20:30*
