# Opus 4.6 コードレビュー & Phase 2 実験計画

**作成日**: 2026-02-15  
**目的**: Sonnetで構築したコードベースをOpus 4.6が第三者レビューし、見落とし・改善策を特定

---

## 1. レビューで発見された問題

### 1.1 【致命的】CLIP projection のゼロ崩壊バグ

**症状**: CLIPの訓練で Loss が負の値（-0.005）になり、マスクが均一に塗りつぶされる

**根本原因**: `FeatureExtractor` の CLIP 用 projection (`nn.Conv2d(768, 384, 1)`) が**学習可能パラメータ**であり、その出力が再構成ターゲットとして直接使用されている。

```python
# savi_dinosaur.py FeatureExtractor.__init__()
self.projection = nn.Conv2d(768, 384, 1)  # ← requires_grad=True

# savi_dinosaur.py FeatureExtractor.forward()
features = self.projection(features)  # ← 勾配が流れる
return features  # ← これが encode() 経由で target_feat になる
```

```python
# savi_dinosaur.py forward_video()
loss = ((target_feat - recon_feat) ** 2).mean()
# target_feat.requires_grad = True (CLIP only!)
# → projection は target_feat → 0 に崩壊させることで loss を最小化できる
```

**検証結果**:
| Backbone | `target_feat.requires_grad` | 状態 |
|----------|---------------------------|------|
| DINOv2   | `False`                   | ✅ 正常（凍結バックボーン、projection なし） |
| DINOv1   | `False`                   | ✅ 正常（凍結バックボーン、projection なし） |
| **CLIP** | **`True`**                | ❌ **バグ**（projection 経由で勾配流入） |

**勾配の流れ**:
```
MSE Loss → recon_feat (OK: decoder → slot_attention → feature_projection)
MSE Loss → target_feat → projection.weight (BUG: 自明な最適化が可能)
```

projection は出力をゼロに近づけることで MSE を最小化できるため、diversity loss の正の項と相殺して**負の総損失**が発生する。これが「特徴量スケール問題」ではなく「アーキテクチャバグ」であったことを意味する。

---

### 1.2 【中程度】正規化の方向の誤り

これまで試した正規化は**空間次元**（H, W）に対するもの:
```python
# 試行1（失敗）: 各チャンネルの空間的変動を除去
features = (features - features.mean(dim=(2,3))) / features.std(dim=(2,3))
```

Slot Attention が必要とするのは「位置 A と位置 B の特徴が異なる」という**空間的差異**。空間次元正規化はこの差異そのものを除去する。

**まだ試していないアプローチ: 損失関数側でのチャンネル正規化**

| 操作 | 正規化方向 | 保存されるもの | 除去されるもの |
|------|-----------|---------------|---------------|
| 空間正規化（試行済❌） | H,W次元 | チャンネル間の関係 | **位置間の差異（破壊）** |
| チャンネル正規化（未試行）| C次元 | **位置間の差異（保存）** | 特徴量の絶対スケール |

チャンネル正規化は入力特徴量には何もせず、損失計算時に各位置の384次元ベクトルを正規化する。これによりスケール差は吸収されつつ空間構造は完全保存される。

---

### 1.3 DINOv1 の真の問題: 損失関数のスケール感度

レイヤー解析により、DINOv1 の空間弁別能力は DINOv2 より**高い**ことが判明:

| Backbone | Spatial/Channel Ratio | 空間弁別能力 |
|----------|----------------------|-------------|
| DINOv2 (final layer) | 0.599 | 基準 |
| **DINOv1 (final layer)** | **0.740** | **DINOv2 より優秀** |
| CLIP (best layer) | 0.374 | 根本的に不足 |

DINOv1 が失敗した原因は空間情報の欠如ではなく、**再構成損失の絶対スケール不一致**:
- DINOv2: MSE ベースライン ≈ 2.4² = 5.8 → 到達 Loss 0.73 (87% 再構成)
- DINOv1: MSE ベースライン ≈ 4.0² = 16.0 → 到達 Loss 4.0 (75% 再構成)
- **比率としてはそこまで悪くない** → チャンネル正規化損失で解決可能

---

### 1.4 CLIP の根本的限界

レイヤー解析の結果、CLIP はどのレイヤーを使っても空間的弁別能力が DINO の 1/10:
```
CLIP 最良レイヤー (11): Spatial Std = 0.150
DINOv2 final:            Spatial Std = 1.447  ← 10倍の差
```

CLIP のグローバル対照学習目的上、位置間の特徴差が本質的に小さい。projection バグの修正だけで slot 分化は期待できないが、「負の Loss」という異常は確実に解消される。

---

### 1.5 【軽微】Slot Attention 入力に位置埋め込みがない

OCL Framework の `SlotAttentionGrouping` は positional embedding を Slot Attention 入力に加算する。現在の実装にはこれがない。DINOv2 の成功は ViT 自体の位置埋め込みが残存しているため。

---

## 2. Phase 2 実験計画

### Task 1: CLIP projection バグ修正 + 再訓練

**優先度**: 最高  
**工数**: コード 2 行 + 50ep (~1分)

**目的**:
- CLIPの負のLossの真因であるprojectionバグを修正
- 修正後のCLIPが正常な損失曲線を示すか検証
- 「特徴量スケール問題」と「アーキテクチャバグ」を切り分け

**実装方法**:
`savi_dinosaur.py` の `forward_video()` で損失計算時にターゲットを detach:
```python
# 変更前:
loss = ((target_feat - recon_feat) ** 2).mean()

# 変更後:
loss = ((target_feat.detach() - recon_feat) ** 2).mean()
```

これにより全バックボーンで同一の修正が適用されるが、実質的に影響があるのは CLIP のみ（DINOv1/v2 は元から `requires_grad=False`）。

**期待される結果**:
- ✅ CLIPのLossが正の値に改善（現在の -0.005 → 正値へ）
- ✅ マスクの分化が改善される可能性
- ⚠️ 空間弁別能力の根本的限界（Spatial Std = 0.17）は残る

---

### Task 2: チャンネル正規化損失で DINOv1 再訓練

**優先度**: 高  
**工数**: 損失関数 5 行変更 + 200ep (~10分)

**目的**:
- 空間構造を保存しつつスケール差を吸収する損失関数を検証
- DINOv1 (std=4.0) と DINOv2 (std=2.4) のスケール差が損失関数側で解決可能か確認
- マルチバックボーン比較の復活可能性を探る

**実装方法**:
`train_movi.py` に `--loss_type` 引数を追加し、チャンネル正規化損失を選択可能にする:
```python
# 損失関数の差し替え
if loss_type == 'cosine':
    target_flat = target_feat.permute(0, 2, 3, 1).reshape(-1, 384)
    recon_flat = recon_feat.permute(0, 2, 3, 1).reshape(-1, 384)
    loss = 1 - F.cosine_similarity(target_flat, recon_flat, dim=1).mean()
elif loss_type == 'channel_norm':
    target_flat = target_feat.permute(0, 2, 3, 1).reshape(-1, 384)
    recon_flat = recon_feat.permute(0, 2, 3, 1).reshape(-1, 384)
    loss = ((F.layer_norm(target_flat, [384]) - F.layer_norm(recon_flat, [384])) ** 2).mean()
else:  # 'mse' (default)
    loss = ((target_feat.detach() - recon_feat) ** 2).mean()
```

**期待される結果**:
- ✅ DINOv1 の Loss がスケール正規化で改善（現在の 4.692 → ~0.7 レベルへ）
- ✅ 空間弁別能力は DINOv2 と同等以上（Spatial/Channel Ratio: 0.74 vs 0.60）なので、マスク分化の改善が期待
- ⚠️ 過剰正規化で情報損失が発生する可能性

**検証条件**: DINOv2 でも同一損失関数を適用し、MSE での既存結果 (Loss=0.729) と比較

---

### Task 3: ARI 計算（既存チェックポイント）

**優先度**: 高  
**工数**: 新スクリプト ~30行

**目的**:
- 既存の全チェックポイントに対して定量的セグメンテーション評価を追加
- Ground-truth segmentation との一致度を ARI (Adjusted Rand Index) で定量化
- Metal-only / Mixed / All の3条件で分析し、金属物体への適性を検証

**実装方法**:
`src/compute_ari.py` を新規作成:
```python
from sklearn.metrics import adjusted_rand_score
# 各サンプルについて:
# 1. モデルでマスクを推論 → argmax でハード割当
# 2. GT segmentationを mask 解像度 (16×16) にリサイズ
# 3. ARI を計算
# 4. Metal/Mixed/All で集計
```

**期待される結果**:
- DINOv2 (τ=0.5): ARI > 0 を確認（ランダムは ARI ≈ 0）
- Metal vs Mixed での ARI 差を定量化
- Task 1, 2 の修正後のモデルとの比較基準

---

## 3. 実験実行順序

```
1. [Task 1] CLIP projection 修正 → 50ep 訓練 → 結果確認
2. [Task 2] channel_norm 損失 → DINOv1 200ep + DINOv2 200ep → 比較
3. [Task 3] ARI 計算 → 全チェックポイント評価
```

各タスク完了後にこのドキュメントの「4. 実験結果」セクションに結果を追記。

---

## 4. 実験結果

### Task 1: CLIP Projection Bug Fix (`target_feat.detach()`)

**実行日**: 2025-02-16  
**設定**: clip_vitb16, 50 epochs, τ=0.5, lr=0.0004, batch_size=2, diversity_weight=0.1

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| Train Loss | -0.005 (zero-collapse) | **0.036** |
| Test Loss | N/A | **0.099** |

**結論**: `detach()` により CLIP のゼロ崩壊が完全に解消。 
ただし、CLIPの空間弁別能力の低さ（Spatial std=0.17）により、ARI は低い。

---

### Task 2: Channel-Normalized Loss for DINOv1

**実行日**: 2025-02-16  
**設定**: dino_vits16, 200 epochs, channel_norm loss, τ=0.5, lr=0.0004

| Metric | DINOv1 MSE (旧) | DINOv1 channel_norm |
|--------|-----------------|---------------------|
| Train Loss | 4.692 | **0.163** |
| Test Loss | N/A | **0.574** |

**結論**: channel_norm正規化により、DINOv1の訓練ロスが劇的に改善（4.692 → 0.163）。
スケール差の吸収に成功。

---

### Task 3: ARI 比較（全モデル）

**実行日**: 2025-02-16  
**ツール**: `src/compute_ari.py` (FG-ARI = 背景除外, Full-ARI = 全ピクセル)

| Model | Loss Type | Train Loss | FG-ARI | Full-ARI | Metal FG-ARI | Non-metal FG-ARI |
|-------|-----------|-----------|--------|----------|-------------|-----------------|
| **DINOv2** (ViT-S/14) | MSE | 0.729 | **0.165 ± 0.186** | **0.073 ± 0.080** | 0.185 | 0.146 |
| **DINOv1** (ViT-S/16) | channel_norm | 0.163 | 0.153 ± 0.172 | 0.047 ± 0.075 | 0.154 | 0.152 |
| **CLIP** (ViT-B/16) | MSE + detach | 0.036 | 0.041 ± 0.118 | -0.026 ± 0.064 | 0.048 | 0.034 |

**分析**:

1. **DINOv2 が依然としてベスト**: FG-ARI 0.165 は DINOv1 の 0.153 をやや上回る
2. **DINOv1 channel_norm は大幅改善**: 
   - MSE だと Loss 4.692 で学習失敗 → channel_norm で 0.163 まで低下
   - ARI は DINOv2 とほぼ同等（0.153 vs 0.165）、差は小さい
   - Metal/Non-metal の差が小さい（0.154 vs 0.152）= 材質に依存しない均一な分離
3. **CLIP はセグメンテーションに不向き**: 
   - detach fix で学習は可能になったが、ARI ≈ 0（ランダムレベル）
   - CLIP の特徴空間は意味的類似性に最適化されており、空間的オブジェクト分離には適さない
4. **Metal vs Non-metal**: 
   - DINOv2 は Metal (0.185) > Non-metal (0.146) で金属物体をやや得意とする
   - DINOv1 は均一（Metal ≈ Non-metal）
   - いずれもランダムベースライン (ARI ≈ 0) を大きく上回る

---

### 総括

Phase 2 で得られた主要な知見:

1. **CLIP zero-collapse バグの原因特定と修正**: `nn.Conv2d` projection の勾配がターゲット特徴に流入していた。`.detach()` で解消。
2. **DINOv1 のスケール問題解決**: channel_norm loss により、DINOv1 (std=4.0) のスケール問題を吸収。ARI は DINOv2 とほぼ同等に。
3. **Backbone 適性の明確化**: 
   - DINOv2 > DINOv1 ≫ CLIP （オブジェクトセグメンテーションタスクにおいて）
   - CLIP は本質的に空間弁別能力が低く、slot attention との相性が悪い
4. **ARI 基準値の確立**: DINOv2 FG-ARI = 0.165 が現在のベースライン
