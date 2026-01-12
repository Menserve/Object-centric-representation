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

*Last updated: 2026-01-12*
