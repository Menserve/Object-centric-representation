# Object-Centric Representation Learning with Pre-trained Features

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 研究課題名 / Research Title

**複数ViT事前学習手法が鏡面反射物体の物体中心分離に与える影響**

*Impact of Multiple ViT Pre-training Objectives on Specular Object Decomposition in Object-Centric Learning*

### Phase 2 拡張（2026年2月15日）

TAフィードバックに基づき、複数のViT backboneを比較する実験を追加実装：
- **DINOv2 ViT-S/14**: 密な自己蒸留学習（baseline）
- **DINOv1 ViT-S/16**: 初代DINO自己蒸留学習
- **CLIP ViT-B/16**: 言語-画像対照学習

詳細プランは [docs/IMPLEMENTATION_PLAN_PHASE2.md](docs/IMPLEMENTATION_PLAN_PHASE2.md) 参照。

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
├── src/                                # コアモジュール
│   ├── savi_dinosaur.py                # SAVi-DINOSAURモデル実装
│   ├── train_movi.py                   # 学習スクリプト
│   └── compute_ari.py                  # ARI評価
├── experiments/                        # 実験スクリプトと結果
│   ├── main/                           # 論文の主要実験
│   │   ├── download_movi.py            # データダウンロード
│   │   └── logs/                       # 論文引用ログ（Tier 1, 8ファイル）
│   ├── sub/                            # サブ実験・アブレーション
│   │   ├── alternative_diversity_loss.py
│   │   ├── analyze_video_masks.py      # 等々
│   │   └── logs/                       # 対照実験ログ（Tier 2, 10ファイル）
│   └── analysis/                       # 結果分析ツール
│       ├── analyze_metal_vs_rubber.py  # 材質別比較
│       ├── compare_single_frame_backbones.py
│       ├── visualize_checkpoint.py     # 等々
│       └── experiment_report.ipynb     # 実験レポート
├── debug/                              # デバッグ・開発ツール
│   ├── debug_features.py               # 特徴量デバッグ
│   ├── test_architecture.py            # 等々
│   └── README.md
├── notebooks/
│   └── phase1_exploration/             # Phase 1探索ノートブック
│       ├── ex_comparison.ipynb
│       └── ex_movi_explore.ipynb
├── docs/
│   ├── paper/                          # 論文ソース（LaTeX）
│   │   ├── paper.tex                   # 最終論文
│   │   ├── slide.tex                   # プレゼンテーション
│   │   ├── script.md                   # 発表原稿
│   │   └── figures/                    # 論文図版
│   └── research_log/                   # 研究記録
│       ├── RESEARCH_LOG.md             # 研究活動記録
│       ├── METHODS.md                  # 手法詳細
│       └── log_analysis_handoff.md     # ログ分析文書
├── checkpoints/                        # 学習済みモデル（git管理外, 12GB）
├── data/                               # データセット（git管理外）
├── requirements.txt                    # 依存パッケージ
├── pyproject.toml                      # プロジェクト設定
└── LICENSE                             # ライセンス
\`\`\`

**ディレクトリ整理方針（2026-02-16）:**
- **論文直結の実験** (`experiments/main/`) と **補助実験** (`experiments/sub/`) を明確に分離
- **開発過程のデバッグスクリプト** は `debug/` に集約
- **論文ソース** (`docs/paper/`) と **研究記録** (`docs/research_log/`) を分離
- 詳細は [docs/research_log/log_analysis_handoff.md](docs/research_log/log_analysis_handoff.md) 参照

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
python experiments/main/download_movi.py
\`\`\`

## 実行方法 / How to Run

### 1. 学習（複数Backbone対応）

**単一Backboneの学習**:
\`\`\`bash
# DINOv2 (baseline)
python src/train_movi.py --backbone dinov2_vits14 --save_dir checkpoints/dinov2_vits14

# DINOv1
python src/train_movi.py --backbone dino_vits16 --save_dir checkpoints/dino_vits16

# CLIP
python src/train_movi.py --backbone clip_vitb16 --save_dir checkpoints/clip_vitb16
\`\`\`

**複数Backboneの一括学習**:
\`\`\`bash
cd src

# 順次実行（1台のマシンで全て）
./run_phase2_training.sh sequential

# 並列実行（メイン機: DINOv2 + DINOv1）
./run_phase2_training.sh parallel

# サブ機用（CLIP単独）
./run_phase2_training.sh clip

# デバッグモード（10エポックでテスト）
./run_phase2_training.sh debug
\`\`\`

詳細オプション:
\`\`\`bash
python src/train_movi.py --help
\`\`\`

### 2. Metal vs Mixed 分析
\`\`\`bash
python experiments/analysis/analyze_metal_vs_rubber.py
\`\`\`

### 3. 実験レポート確認
\`\`\`bash
jupyter notebook experiments/analysis/experiment_report.ipynb
\`\`\`

## モデル構成 / Model Architecture

\`\`\`
入力動画 (B, T, 3, 224, 224)
    ↓
ViT Feature Extractor [凍結] → 特徴量 (B, T, 384, 16, 16)
│   ├─ DINOv2 ViT-S/14 (dense self-supervised)
│   ├─ DINOv1 ViT-S/16 (self-distillation)
│   └─ CLIP ViT-B/16 (language-aligned)
    ↓
Slot Attention (K=5 slots, 5 iterations)
    ↓
Slot Predictor [次フレーム予測]
    ↓
Feature Decoder → 再構成特徴量 (B, T, 384, 16, 16)
\`\`\`

**パラメータ数（DINOv2 baseline）:**
- 総パラメータ: 32,918,145
- 学習可能: 10,944,641
- 凍結（DINOv2）: 21,973,504

**Backbone比較:**

| Backbone | 事前学習手法 | feat_dim | 総パラメータ | 学習可能 |
|----------|------------|---------|-------------|---------|
| DINOv2 ViT-S/14 | 密な自己蒸留 | 384 | 33.0M | 10.9M |
| DINOv1 ViT-S/16 | 自己蒸留 | 384 | 32.6M | 10.9M |
| CLIP ViT-B/16 | 言語対照学習 | 768→384 | 160.9M | 11.2M |

## 参考文献 / References

### Object-Centric Learning
1. Kipf, T., et al. (2022). "Conditional Object-Centric Learning from Video." *ICLR 2022*. (SAVi)
2. Seitzer, M., et al. (2023). "Bridging the Gap to Real-World Object-Centric Learning." *ICLR 2023*. (DINOSAUR)
3. Greff, K., et al. (2022). "Kubric: A Scalable Dataset Generator." *CVPR 2022*. (MOVi)

### Vision Transformers & Pre-training
4. Oquab, M., et al. (2024). "DINOv2: Learning Robust Visual Features without Supervision." *TMLR 2024*.
5. Caron, M., et al. (2021). "Emerging Properties in Self-Supervised Vision Transformers." *ICCV 2021*. (DINO)
6. Radford, A., et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision." *ICML 2021*. (CLIP)

## ライセンス / License

MIT License - 詳細は [LICENSE](LICENSE) を参照
