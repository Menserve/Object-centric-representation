# Object-Centric Representation Learning with Pre-trained Features

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 研究課題名 / Research Title

**複数ViT事前学習手法が鏡面反射物体の物体中心分離に与える影響**

*Impact of Multiple ViT Pre-training Objectives on Specular Object Decomposition in Object-Centric Learning*

### 研究完了（2026年3月7日）

3バックボーン (DINOv2 / DINOv1 / CLIP) × スロット数 K∈{3,5,7,9,11,13} の網羅的評価を MOVi-A 300シーンで実施。
論文 (9ページ)、A1ポスター、発表スライドを完成。

詳細は [docs/research_log/RESEARCH_LOG.md](docs/research_log/RESEARCH_LOG.md) 参照。

## 概要 / Abstract

### 日本語

物体中心表現学習は、画像や動画に映る物体を個別のスロットとして教師なしで分離する手法である。本研究では、凍結ViT特徴量とSlot Attentionを組み合わせた**SAVi-DINOSAUR**を用いて、鏡面反射（specular reflection）を持つ金属物体の教師なしセグメンテーションにおける構造的限界を検証した。

MOVi-Aデータセット300シーンを対象に、3種のViTバックボーン（DINOv2・DINOv1・CLIP）×スロット数K∈{3,5,7,9,11,13}の計18モデルを学習し、FG-ARI・mBOで定量評価した。

主要な知見:
- **DINOv2がK=9で最高性能**（FG-ARI 52.1%）、DINOv1（24.1%）・CLIP（14.6%）を大幅に上回る
- **Goldilocks特性**: K=9付近でFG-ARIが最大化後に飽和（過少・過多でともに低下）
- エントロピー分析で余剰スロットが崩壊の定量的指標となることを確認
- 時系列追跡（40シーン×24フレーム）でDINOv2スロットが10フレーム以上物体IDを保持

### English

Object-centric representation learning decomposes visual scenes into individual object slots without supervision. This study investigates structural limitations in unsupervised segmentation of specular metallic objects using **SAVi-DINOSAUR** (frozen ViT features + Slot Attention).

We trained 18 models across 3 ViT backbones (DINOv2/DINOv1/CLIP) × K∈{3,5,7,9,11,13} slots on 300 MOVi-A scenes, evaluating with FG-ARI and mBO.

Key findings:
- **DINOv2 at K=9 achieves best performance** (FG-ARI 52.1%), significantly outperforming DINOv1 (24.1%) and CLIP (14.6%)
- **Goldilocks property**: FG-ARI peaks near K=9 and saturates (both under- and over-slotting degrade performance)
- Entropy analysis reveals surplus slots as a quantitative indicator of slot collapse
- Temporal tracking (40 scenes × 24 frames) shows DINOv2 slots maintain object ID for 10+ frames

## 主要な結果 / Key Results

### K スイープ定量評価（300シーン、3バックボーン）

| Backbone | K=3 | K=5 | K=7 | K=9 | K=11 | K=13 |
|----------|-----|-----|-----|-----|------|------|
| **FG-ARI (%)** | | | | | | |
| DINOv2 | 28.4 | 41.2 | 47.6 | **52.1** | 51.8 | 50.3 |
| DINOv1 | 12.1 | 18.4 | 22.3 | 24.1 | 23.8 | 23.2 |
| CLIP | 8.3 | 11.2 | 13.4 | 14.6 | 14.1 | 13.8 |
| **mBO (%)** | | | | | | |
| DINOv2 | 31.2 | 38.4 | 42.1 | **44.8** | 44.2 | 43.1 |
| DINOv1 | 18.3 | 22.1 | 25.4 | 26.8 | 26.2 | 25.7 |
| CLIP | 14.2 | 17.8 | 19.4 | 20.3 | 19.9 | 19.4 |

- DINOv2 が全設定で DINOv1・CLIP を大幅に上回る（FG-ARI 約2倍）
- K=9 が全バックボーンで最適（Goldilocks 特性）
- バックボーン間のギャップは素材・K によらず一貫

## プロジェクト構成 / Project Structure

```text
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
├── scripts/                            # 図表生成・スイープ実行スクリプト
│   ├── run_k_sweep.sh                  # K sweep 一括実行
│   ├── run_tau_sweep.sh                # τ sweep 一括実行
│   ├── evaluate_sweeps.py              # sweep 結果評価
│   ├── temporal_analysis.py            # 時系列追跡分析
│   ├── generate_architecture_fig.py    # アーキテクチャ図生成
│   └── generate_paper_slot_figures.py  # 論文用スロット可視化図
├── notebooks/
│   ├── day2_analysis.ipynb             # 統計的検証
│   ├── day3_retrain_analysis.ipynb     # 300サンプル再学習分析
│   ├── day4_failure_analysis.ipynb     # 3バックボーン失敗解析
│   ├── day4_resolution_experiment.ipynb # 解像度448 vs 224
│   ├── day6_sweep_experiments.ipynb    # K sweep / τ sweep
│   ├── day7_analysis_report.ipynb      # エントロピー・時系列・最終分析
│   └── phase1_exploration/             # Phase 1探索ノートブック
├── docs/
│   ├── paper/                          # 論文ソース（LaTeX）
│   │   ├── paper.tex                   # 最終論文（9ページ、jsai-acl）
│   │   ├── poster.tex                  # A1ポスター（594×841mm）
│   │   ├── slide.tex                   # プレゼンテーション
│   │   ├── script.md                   # 発表原稿
│   │   └── figures/                    # 論文図版
│   └── research_log/                   # 研究記録
│       ├── RESEARCH_LOG.md             # 研究活動記録（2025-11〜2026-03）
│       ├── METHODS.md                  # 手法詳細
│       └── log_analysis_handoff.md     # ログ分析文書
├── checkpoints/                        # 学習済みモデル（git管理外, 12GB）
├── data/                               # データセット（git管理外）
├── requirements.txt                    # 依存パッケージ
├── pyproject.toml                      # プロジェクト設定
└── LICENSE                             # ライセンス
```

**ディレクトリ整理方針:**
- **論文直結の実験** (`experiments/main/`) と **補助実験** (`experiments/sub/`) を明確に分離
- **開発過程のデバッグスクリプト** は `debug/` に集約
- **論文ソース** (`docs/paper/`) と **研究記録** (`docs/research_log/`) を分離
- **図表生成・sweep実行** は `scripts/` に集約
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
# K sweep 一括実行
bash scripts/run_k_sweep.sh

# τ sweep 一括実行
bash scripts/run_tau_sweep.sh
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
Slot Attention (K slots, 5 iterations; K=9 optimal, tested K∈{3,5,7,9,11,13})
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
