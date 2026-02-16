# Phase 2 実装プラン：複数ViT Backbone比較実験

**作成日**: 2026年2月15日  
**期限**: 提出まで40時間、発表まで1ヶ月

---

## 背景・動機

### TAからのフィードバック
> 「Slot AttentionとDINOSAURの比較のみでは、ViTを使うと良い、の検証に留まり、やや新規性が弱い気がします。複数のViTを使用した場合のメリット・デメリットなど比較できるとより良いのかと思います。」

### 現状の成果
- モデル: SAVi-DINOSAUR（DINOv2 ViT-S/14 + Slot Attention）
- データ: MOVi-A subset（metal×20, mixed×40, 各24フレーム）
- 主な結果: Metal IoU 0.660 vs Mixed IoU 0.670
- 発見: DINOv2が鏡面反射の外観変化を吸収している

### 研究の核心
**「金属光沢のある物体を認識できるようにしたい」** という問題意識に基づき、物体中心学習を解決手段として捉えている。

---

## 新規性のある研究テーマ

### タイトル案
**「事前学習手法が鏡面反射物体の物体中心分離に与える影響」**

*Impact of Pre-training Objectives on Specular Object Decomposition in Object-Centric Learning*

### リサーチクエスチョン
> **異なる事前学習目的で訓練されたViTは、鏡面反射を持つ物体のSlot分離にどのような影響を与えるか？**

- 単なる「ViTが良い」ではなく、「どの種類のViT表現が鏡面反射に最適か」を問う
- 既存結果（DINOv2が鏡面反射を吸収）を深掘りし、他の事前学習手法と比較
- MOVi-Aの材質メタデータ（metal/rubber）を活用した定量評価

---

## 比較する3つのBackbone

| Backbone | 事前学習手法 | feat_dim | 空間解像度 | 仮説 |
|----------|------------|---------|-----------|------|
| **DINOv2 ViT-S/14** (現行) | 自己蒸留 + 密な特徴学習 | 384 | 16×16 | 外観変化に最も不変 → Metal/Rubber差が最小 |
| **DINOv1 ViT-S/16** | 自己蒸留（初代） | 384 | 14×14 | v2ほど密でない → 鏡面反射で差が出る可能性 |
| **CLIP ViT-B/16** | 言語-画像対照学習 | 768 | 14×14 | 意味的だが、material-awareな表現？ |

### Backboneの選定理由

1. **事前学習の目的が異なる**
   - DINOv2: 密な自己蒸留（dense self-supervised learning）
   - DINOv1: Attention重視の自己蒸留
   - CLIP: 言語アライメント（language-aligned vision）

2. **同一のSlot Attentionフレームワーク**
   - 特徴抽出器だけを差し替え
   - 学習設定（LR, epoch, スロット数）を統一
   - → 純粋に「事前学習表現の質」を比較できる

3. **MOVi-Aの材質メタデータで定量評価可能**
   - Metal vs Rubber の分離性能差を測定
   - 鏡面反射に強いbackboneを特定できる

---

## タイムライン（全20h）

### Phase 1: Backbone実装・リファクタリング（3h）

**タスク**:
- `src/savi_dinosaur.py` の `DinoFeatureExtractor` を `FeatureExtractor` に一般化
- DINOv1, CLIP backboneの追加実装
- 特徴次元・空間解像度の差異を吸収する Projection層 の追加
- コマンドライン引数 `--backbone` でbackboneを切り替え可能に

**実装詳細**:
```python
class FeatureExtractor(nn.Module):
    def __init__(self, backbone: str = 'dinov2_vits14'):
        # 'dinov2_vits14', 'dino_vits16', 'clip_vitb16' をサポート
        # 出力は統一: (B, 384, 16, 16) に正規化
```

**環境セットアップ**:
```bash
# CLIP用のライブラリをインストール
uv pip install open-clip-torch
```

---

### Phase 2: 学習（4h）

**実行計画**:
- 3 backbone × 同一ハイパーパラメータ
- 並列実行可能:
  - RTX5090メイン機: DINOv2 (baseline) + DINOv1
  - RTX4080サブ機: CLIP

**学習設定**:
```python
NUM_EPOCHS = 200
BATCH_SIZE = 2
NUM_SLOTS = 5
MAX_FRAMES = 12
LR = 0.001
DIVERSITY_WEIGHT = 0.1
```

**予想所要時間**: 各60-90分（60サンプル、200epoch）

**学習コマンド**:
```bash
# DINOv2 (baseline, 既存)
python src/train_movi.py --backbone dinov2_vits14 --save_dir checkpoints/dinov2

# DINOv1
python src/train_movi.py --backbone dino_vits16 --save_dir checkpoints/dinov1

# CLIP
python src/train_movi.py --backbone clip_vitb16 --save_dir checkpoints/clip
```

---

### Phase 3: 評価・分析（4h）

#### 定量評価

**Metal vs Rubber分離性能**:

| 指標 | 説明 | 意味 |
|------|------|------|
| Metal IoU vs Rubber IoU | 材質別の時間的一貫性 | backboneによる「金属ペナルティ」の差 |
| Temporal Stability (metal) | 鏡面反射による時間的不安定さ | 低いほど追跡が安定 |
| Slot Assignment Change Rate | フレーム間でのスロット割り当て変化 | 低いほど一貫性が高い |
| Slot Diversity | スロット間の分離度 | 高いほど異なる領域を担当 |

**比較表の例**:

| Backbone | Metal IoU | Rubber IoU | IoU差分 | Stability (metal) | Change Rate |
|----------|-----------|------------|---------|-------------------|-------------|
| DINOv2 | 0.660 | 0.670 | -0.010 | 0.058 | 0.147 |
| DINOv1 | TBD | TBD | TBD | TBD | TBD |
| CLIP | TBD | TBD | TBD | TBD | TBD |

#### 定性評価（可視化）

1. **Attention Map比較**
   - Metal物体 vs Rubber物体でのマスク品質
   - 各backboneの成功例・失敗例を並べる

2. **失敗モード分析**
   - スロット崩壊（全スロットが背景に割り当て）
   - オーバーセグメンテーション（1物体を複数スロットに分割）
   - 背景・物体の混同

3. **時間的一貫性の可視化**
   - Metal物体でのスロット追跡の安定性を動画として保存

#### 追加分析（差別化要素）

1. **特徴空間の可視化**
   ```python
   # 各backboneの特徴空間でmetal/rubberパッチの分布を t-SNE で可視化
   # → backboneが材質をどう表現しているかを直接確認
   ```

2. **Slot表現ベクトルのクラスタ分析**
   ```python
   # Metal物体に割り当てられたスロット vs Rubber物体のスロット
   # → backboneによって「材質認識」がSlot表現に反映されているか
   ```

**分析スクリプト**:
```bash
python src/compare_backbones.py --checkpoints checkpoints/dinov2,checkpoints/dinov1,checkpoints/clip
```

---

### Phase 4: 論文執筆（8h）

| セクション | 時間 | 内容 |
|-----------|------|------|
| **Introduction** | 1h | ・物体中心学習の重要性<br>・鏡面反射物体の追跡が困難な理由<br>・「ViTが有効だが、どの事前学習が最適か未解明」という問題設定 |
| **Related Work** | 1h | ・Slot Attention / SAVi<br>・DINOSAUR<br>・DINOv1, DINOv2, CLIPの事前学習手法の特徴 |
| **Method** | 1.5h | ・SAVi-DINOSAURの共通フレームワーク<br>・3つのbackboneの説明<br>・評価指標の定義 |
| **Experiments** | 2.5h | ・定量結果の表作成<br>・Attention Mapの比較図<br>・t-SNE可視化図<br>・失敗例の分析 |
| **Discussion** | 1.5h | ・各backboneのメリット・デメリット<br>・Metal vs Rubberの結果の解釈<br>・鏡面反射に強いbackboneの特定<br>・Limitation（データ規模、合成データの限界） |
| **Conclusion** | 0.5h | ・まとめと今後の展望 |
| **推敲・調整** | 0.5h | ・フォーマット調整<br>・図表番号の確認<br>・参考文献の整理 |

#### 期待される考察の方向性

**仮説**:
1. **DINOv2** → 鏡面反射に最も強い。密な特徴学習が外観変化を吸収。Metal/Rubber差が最小
2. **DINOv1** → v2より鏡面反射への頑健性が低い。v1→v2の改善がspecular robustnessに寄与することを示せる
3. **CLIP** → 意味レベルでは物体を認識できるが、背景分離が粗い可能性。ただし「金属的な外観」を言語的にエンコードしている可能性

**仮説通りの場合**:
> 「物体中心学習において鏡面反射を扱うには、密な自己教師あり特徴（DINOv2）が最適であり、言語対照学習（CLIP）と自己蒸留v1（DINO）はそれぞれ異なるトレードオフを持つ」

**仮説と異なる場合**:
> 差異を分析することで「なぜその結果になったか」を考察すれば十分な論文になる

---

## 提出後の発展案（発表まで1ヶ月）

### 短期（~2週間）
1. **MAE backbone の追加**
   - ピクセル再構成タスクで訓練されたViT
   - 鏡面反射への影響を検証

2. **MOVi-Aデータの拡充**
   - 現在60サンプル → 300+サンプルに拡大
   - より統計的に有意な結果

3. **ARI (Adjusted Rand Index) の厳密な算出**
   - Ground-truthセグメンテーションとの比較
   - Object-centricタスクの標準的な評価指標

### 中期（~1ヶ月）
4. **鏡面反射の強度を制御した合成データ**
   - Kubricでカスタムデータセット生成
   - 材質のmetallic パラメータを系統的に変化
   - 「どの程度の鏡面反射まで各backboneが耐えられるか」を定量化

5. **実世界データでの検証**
   - COCO/Pascal VOCの金属物体（車、金属製品）
   - 実用性の検証

6. **Backboneのファインチューニング**
   - 現在は全てfrozen
   - LoRAなどで軽量にアダプトさせた場合の改善効果

---

## 実装チェックリスト

### Phase 1: Backbone実装（3h）
- [ ] `FeatureExtractor` クラスの一般化
- [ ] DINOv1 backboneの実装
- [ ] CLIP backboneの実装
- [ ] 特徴次元の統一（Projection層）
- [ ] `train_movi.py` に `--backbone` 引数追加
- [ ] 環境セットアップ（`uv pip install open-clip-torch`）
- [ ] 動作確認（各backboneで1サンプル推論）

### Phase 2: 学習（4h）
- [ ] DINOv2 baseline学習（既存チェックポイント再利用可）
- [ ] DINOv1 学習（メイン機）
- [ ] CLIP 学習（サブ機 or メイン機）
- [ ] 学習曲線の保存・可視化

### Phase 3: 評価・分析（4h）
- [ ] `compare_backbones.py` スクリプト作成
- [ ] Metal vs Rubber分離性能の表作成
- [ ] Attention Map比較図の生成
- [ ] t-SNE特徴空間可視化
- [ ] Slotクラスタ分析
- [ ] 失敗例の抽出と分析

### Phase 4: 論文執筆（8h）
- [ ] Introduction
- [ ] Related Work
- [ ] Method
- [ ] Experiments
- [ ] Discussion
- [ ] Conclusion
- [ ] 図表の作成と配置
- [ ] 参考文献の整理
- [ ] 全体の推敲

---

## 計算リソース配分

### メイン機（Core Ultra 285K + RTX5090 + 128GB RAM）
- DINOv2 baseline（既存）
- DINOv1 学習
- 評価・可視化スクリプト実行
- 論文執筆

### サブ機（Ryzen 3950X + RTX4080 Super + 32GB RAM）
- CLIP 学習
- バックアップ・データ処理

### ソフトウェア環境
- OS: WSL Ubuntu
- パッケージ管理: uv
- Python: 3.10
- PyTorch: 2.x
- 仮想環境: `/home/menserve/Object-centric-representation/.venv`

---

## 参考文献（追加分）

### ViT事前学習手法
- **DINO**: Caron, M., et al. (2021). "Emerging Properties in Self-Supervised Vision Transformers." *ICCV 2021*.
- **DINOv2**: Oquab, M., et al. (2024). "DINOv2: Learning Robust Visual Features without Supervision." *TMLR 2024*.
- **CLIP**: Radford, A., et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision." *ICML 2021*.

### 鏡面反射・材質認識
- Liang, Y., et al. (2022). "Material-Aware Neural Rendering." (材質と外観の関係)
- PHI調査論文（鏡面反射の物体認識への影響）

---

## リスクと対策

| リスク | 対策 |
|--------|------|
| 学習時間が予想より長い | エポック数を150に削減、またはサブセットで実験 |
| backboneの性能差が小さい | 差が小さいこと自体が知見。「DINOv2の優位性は限定的」として考察 |
| CLIPの実装が複雑 | torchvisionのCLIPで代用可能 |
| 論文執筆が間に合わない | 図表を優先、文章は簡潔に |

---

## まとめ

このプランにより：
1. ✅ TAの要求（複数ViT比較）を満たす
2. ✅ あなたの興味（金属光沢物体の認識）に直結
3. ✅ 既存の実験結果を深掘りし、新規性を高める
4. ✅ 40時間以内に実施可能（実装12h + 執筆8h）
5. ✅ 発表までの1ヶ月で発展的な検証も可能

**次のアクション**: Phase 1の実装開始 🚀
