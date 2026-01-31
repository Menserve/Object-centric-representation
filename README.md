# Object-Centric Representation Learning with Pre-trained Features

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 研究課題名 / Research Title

**時間的物体中心学習における鏡面反射物体の追跡安定性検証：SAVi-DINOSAURによるMetal vs Rubber比較**

*Temporal Stability of Specular Object Tracking in Object-Centric Learning: A Comparison of Metal vs Rubber using SAVi-DINOSAUR*

## 概要 / Abstract

### 日本語

物体中心表現学習は、画像や動画に映る物体を個別のスロットとして教師なしで分離する手法である。本研究では、鏡面反射（specular reflection）を持つ物体が時間的な追跡においてマット物体より不安定になるかを検証した。

SAVi（Slot Attention for Video）とDINOSAUR（DINOv2特徴ベース）を組み合わせた**SAVi-DINOSAUR**を実装し、MOVi-Aデータセット上でMetal物体とRubber物体の時間的一貫性を比較した。学習では、スロット崩壊を防ぐために**多様性損失（Diversity Loss）**を導入した。

実験の結果、Metal物体とMixed（Metal+Rubber）で時間的一貫性（IoU）に大きな差は見られなかった（0.66 vs 0.67）。ただし、Metalでは若干の不安定さ（+3%のスロット割り当て変化）が観察された。これは、DINOv2の事前学習特徴量が鏡面反射の見えの変化を吸収していることを示唆する。

### English

Object-centric representation learning aims to decompose visual scenes into individual object representations (slots) without supervision. This study investigates whether specular (reflective) objects are harder to track temporally compared to matte objects.

We implemented **SAVi-DINOSAUR**, combining SAVi (Slot Attention for Video) with DINOSAUR (DINOv2 feature-based approach), and compared temporal consistency between Metal and Rubber objects on the MOVi-A dataset. We introduced **Diversity Loss** to prevent slot collapse during training.

Our experiments show no significant difference in temporal consistency (IoU) between Metal and Mixed objects (0.66 vs 0.67). However, Metal showed slightly higher instability (+3% slot assignment changes). This suggests that DINOv2's pre-trained features absorb appearance variations caused by specular reflections.

## 主要な結果 / Key Results

### Metal vs Mixed 時間的一貫性比較

| 指標 | Metal | Mixed | 差 | 解釈 |
|------|-------|-------|-----|------|
| Mean IoU ↑ | 0.660 | 0.670 | -0.010 | ほぼ同等 |
| Stability ↓ | 0.058 | 0.045 | +0.013 | Metalがやや不安定 |
| Change Rate ↓ | 0.147 | 0.117 | +0.030 | Metalで割り当て変化多い |

### マスク品質（多様性損失追加後）

| Slot | Coverage |
|------|----------|
| 0 | 32.0% |
| 1 | 20.3% |
| 2 | 23.9% |
| 3 | 9.1% |
| 4 | 14.6% |
| **合計** | **99.9%** |

## プロジェクト構成 / Project Structure

\`\`\`
.
├── README.md                           # このファイル
├── notebooks/
│   ├── experiment_report.ipynb         # 実験レポート（画像付き）
│   └── experiment_2026_02_01/          # 実験画像
├── src/
│   ├── savi_dinosaur.py                # SAVi-DINOSAURモデル実装
│   ├── train_movi.py                   # 学習スクリプト（多様性損失含む）
│   ├── analyze_metal_vs_rubber.py      # Metal vs Mixed分析
│   ├── download_movi.py                # MOVi-Aデータダウンロード
│   ├── ex_movi_explore.ipynb           # MOVi-Aデータ探索
│   └── archive/                        # 旧実験ノートブック（Pixel vs DINOSAUR）
├── docs/
│   ├── RESEARCH_LOG.md                 # 研究活動記録
│   └── METHODS.md                      # 手法の詳細説明
├── checkpoints/                        # 学習済みモデル（git管理外）
├── data/                               # データセット（git管理外）
├── requirements.txt                    # 依存パッケージ
├── pyproject.toml                      # プロジェクト設定
└── LICENSE                             # ライセンス
\`\`\`

## セットアップ / Setup

\`\`\`bash
# リポジトリのクローン
git clone https://github.com/Menserve/Object-centric-representation.git
cd Object-centric-representation

# 仮想環境の作成と有効化（uv推奨）
uv venv
source .venv/bin/activate

# 依存パッケージのインストール
uv pip install -r requirements.txt

# MOVi-Aデータのダウンロード
python src/download_movi.py
\`\`\`

## 実行方法 / How to Run

### 1. 学習
\`\`\`bash
python src/train_movi.py
\`\`\`

### 2. Metal vs Mixed 分析
\`\`\`bash
python src/analyze_metal_vs_rubber.py
\`\`\`

### 3. 実験レポート確認
\`\`\`bash
jupyter notebook notebooks/experiment_report.ipynb
\`\`\`

## モデル構成 / Model Architecture

\`\`\`
入力動画 (B, T, 3, 224, 224)
    ↓
DINOv2 ViT-S/14 [凍結] → 特徴量 (B, T, 384, 16, 16)
    ↓
Slot Attention (K=5 slots, 5 iterations)
    ↓
Slot Predictor [次フレーム予測]
    ↓
Feature Decoder → 再構成特徴量 (B, T, 384, 16, 16)
\`\`\`

**パラメータ数:**
- 総パラメータ: 32,918,145
- 学習可能: 10,944,641
- 凍結（DINOv2）: 21,973,504

## 参考文献 / References

1. Kipf, T., et al. (2022). "Conditional Object-Centric Learning from Video." *ICLR 2022*. (SAVi)
2. Seitzer, M., et al. (2023). "Bridging the Gap to Real-World Object-Centric Learning." *ICLR 2023*. (DINOSAUR)
3. Oquab, M., et al. (2024). "DINOv2: Learning Robust Visual Features without Supervision." *TMLR 2024*.
4. Greff, K., et al. (2022). "Kubric: A Scalable Dataset Generator." *CVPR 2022*. (MOVi)

## ライセンス / License

MIT License - 詳細は [LICENSE](LICENSE) を参照
