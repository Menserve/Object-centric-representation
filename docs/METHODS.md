# 手法の詳細 / Methods

## 1. 問題設定

物体中心表現学習（Object-Centric Learning）は、画像から個々の物体を教師なしで分離し、それぞれをスロットと呼ばれる表現ベクトルに割り当てる問題である。

### 入力
- 画像 $x \in \mathbb{R}^{H \times W \times 3}$

### 出力
- スロット表現 $\{s_k\}_{k=1}^{K}$ where $s_k \in \mathbb{R}^{D}$
- マスク $\{m_k\}_{k=1}^{K}$ where $m_k \in [0,1]^{H \times W}$
- 再構成画像 $\hat{x} = \sum_k m_k \odot \hat{x}_k$

---

## 2. 比較手法

### 2.1 ピクセルベース Slot Attention

**アーキテクチャ**:
```
入力画像 (224×224×3)
    ↓
CNN Encoder (+ 位置グリッド)
    ↓
特徴マップ (28×28×64)
    ↓
Slot Attention (5 slots, 5 iterations)
    ↓
Spatial Broadcast Decoder
    ↓
再構成画像 + マスク
```

**損失関数**:
$$\mathcal{L}_{\text{pixel}} = \mathbb{E}\left[\|x - \hat{x}\|^2\right]$$

**温度アニーリング**:
$$\text{attn} = \text{softmax}\left(\frac{QK^T}{\sqrt{d} \cdot \tau}\right)$$
where $\tau: 1.0 \to 0.01$ (linear decay)

### 2.2 DINOSAUR (Feature-based)

**アーキテクチャ**:
```
入力画像 (224×224×3)
    ↓
DINOv2 ViT-S/14 (frozen)
    ↓
パッチ特徴 (16×16×384)
    ↓
Slot Attention (5 slots, 5 iterations)
    ↓
Feature Decoder
    ↓
再構成特徴 + マスク
```

**損失関数**:
$$\mathcal{L}_{\text{feature}} = \mathbb{E}\left[\|f(x) - \hat{f}(x)\|^2\right]$$

where $f(x)$ は DINOv2 の特徴抽出器

---

## 3. Slot Attention の詳細

Slot Attention (Locatello et al., 2020) は、入力特徴から競合的にスロットを更新するメカニズム。

### アルゴリズム

```python
# 初期化
slots = randn(B, K, D)  # ランダム初期化

# イテレーション
for _ in range(num_iters):
    # Attention計算
    Q = linear_q(slots)      # (B, K, D)
    K = linear_k(inputs)     # (B, N, D)
    V = linear_v(inputs)     # (B, N, D)
    
    attn = softmax(Q @ K.T / sqrt(D), dim=slots)  # (B, K, N)
    
    # 加重平均
    updates = attn @ V / attn.sum(dim=-1)  # (B, K, D)
    
    # GRU更新
    slots = GRU(updates, slots)
    slots = slots + MLP(slots)
```

### 重要なポイント

- **Softmax の軸**: `dim=slots` (各入力位置がどのスロットに割り当てられるかを決定)
- **競合メカニズム**: 複数スロットが同じ領域を取り合う
- **イテレーション**: 5回程度で収束

---

## 4. 評価指標

### 4.1 Mask Stability（マスク安定性）

色変化に対するマスクの一貫性を測定。

$$\text{Stability} = \frac{1}{|P|} \sum_{(i,j) \in P} \sum_{k=1}^{K} \text{cos\_sim}(m_k^{(i)}, m_k^{(j)})$$

where $P$ は画像ペアの集合、$m_k^{(i)}$ は画像 $i$ のスロット $k$ のマスク。

**解釈**: 高いほど色変化に対してロバスト

### 4.2 Slot Diversity（スロット多様性）

スロット間の分離度を測定。

$$\text{Diversity} = \frac{1}{K(K-1)/2} \sum_{k < l} \text{cos\_sim}(m_k, m_l)$$

**解釈**: 低いほど各スロットが異なる領域を担当

---

## 5. 実験設定

| 設定 | 値 |
|------|-----|
| 入力解像度 | 224×224 |
| スロット数 | 5 |
| 学習ステップ | 2000 |
| バッチサイズ | 24 |
| 学習率 | 0.0004 |
| 乱数シード | 42 |

### テストデータ

同一画像に対する4種類の色変換:
1. **Blue**: 色相 +0.45, 彩度 ×1.5
2. **Dark**: 明度 ×0.2
3. **Purple**: 色相 -0.4, 彩度 ×1.5
4. **High Contrast**: コントラスト ×3.0

---

## 参考文献

1. Locatello, F., et al. (2020). "Object-Centric Learning with Slot Attention." *NeurIPS 2020*.
2. Seitzer, M., et al. (2023). "Bridging the Gap to Real-World Object-Centric Learning." *ICLR 2023*.
3. Oquab, M., et al. (2024). "DINOv2: Learning Robust Visual Features without Supervision." *TMLR 2024*.
