# ログファイル分析 & リポジトリ整理 引き継ぎドキュメント

> **作成**: 2025-02 Opus による深い分析  
> **目的**: Sonnet への引き継ぎ用。ログの取捨選択判断と、リポジトリ整理の実行計画を記録する。

---

## 1. ログファイルの分類

### 全50ファイルの棚卸し

ルートに3つ (`train.log`, `train_500.log`, `train_diversity.log`) + `logs/` 配下に47ファイル。

---

### Tier 1: 論文に直結するログ（必ず保持）

論文 `paper.tex` に引用された数値の一次証拠となるファイル。

| ファイル | 内容 | 論文での引用箇所 |
|---------|------|-----------------|
| `logs/ari_dinov2_mse.log` | FG-ARI 0.1654±0.1859, Metal 0.1848, Non-metal 0.1460 | Table 1, Table 2 |
| `logs/ari_dinov1_channel_norm.log` | FG-ARI 0.1527±0.1717, Metal 0.1539, Non-metal 0.1515 | Table 1, Table 2 |
| `logs/ari_clip_detach.log` | FG-ARI 0.0410±0.1179, Metal 0.0478, Non-metal 0.0342 | Table 1, Table 2 |
| `logs/temp_scaling_tau05_200ep.log` | Best loss 0.729 (DINOv2 τ=0.5, 本流モデル) | §4.1 訓練損失 |
| `logs/dinov1_channel_norm.log` | Best loss 0.163 (ch-norm適用後) | §4.1「損失は0.163まで低下」 |
| `logs/clip_detach_fix.log` | Best loss 0.036 (detach修正後) | §4.4「損失は0.036に改善」 |
| `logs/twolayer_mlp_200ep.log` | Best loss 0.685 (2層MLP、Table 4) | §4.3 デコーダ比較 |
| `logs/temp_scaling_tau02_200ep.log` | Best loss 0.783 (τ=0.3, 過剰先鋭化) | Table 3 温度比較 |

**計8ファイル**。これらは `experiments/main/logs/` に移動し、git で追跡する（.gitignore から除外）。

---

### Tier 2: 論文の議論を支持する補助ログ（保持推奨）

直接引用されていないが、論文の考察（§6）を裏付けるコントラスト実験。

| ファイル | 内容 | 役割 |
|---------|------|------|
| `logs/clip_baseline_50ep.log` | Best loss −0.005（detach前、負の損失） | detach bugfix の before/after 対照 |
| `logs/dinov1_baseline_50ep.log` | Best loss 4.692（MSE発散） | ch-norm が必要だった証拠 |
| `logs/dinov1_normalized.log` | Best loss 0.260（L2正規化、ch-normより劣る） | ch-norm 選定の根拠 |
| `logs/clip_normalized.log` | Best loss 0.022 | CLIP L2正規化の試行 |
| `logs/fixed_slot_to_feature_200ep.log` | Best loss 0.711 | アーキテクチャ改善過程 |
| `logs/fixed_upsampling_mlp_200ep.log` | Best loss 0.784 | Upsampling MLP 試行 |
| `logs/video_mode_8frames_200ep.log` | Best loss 0.778 | ビデオモードの実験 |
| `logs/video_stopgrad_200ep.log` | Best loss 0.833 | stop-grad + ビデオモード |
| `logs/deeper_decoder_200ep.log` | Best loss 1.238 | deeper decoder の失敗記録 |
| `logs/dinov2_singleframe_final.log` | Best loss 0.733 | single-frame baseline |

**計10ファイル**。`experiments/sub/logs/` に移動。

---

### Tier 3: 開発過程ログ（削除可能）

初期デバッグ・短期テスト・中断実験。論文にもアブレーションにも不要。

| ファイル | 理由 |
|---------|------|
| `logs/architecture_fix_training.log` | 50ep, loss 1.50, 初期アーキテクチャ修正 |
| `logs/clip_optimized.log` | detach前の最適化（負の損失 −0.015）→ 無効な実験 |
| `logs/clip_singleframe.log` | 76epで中断 |
| `logs/clip_singleframe_final.log` | 102epで中断 |
| `logs/clip_singleframe_optimized.log` | 負の損失 −0.014 → detach前 |
| `logs/corrected_architecture_100ep.log` | loss 1.04, 過渡期 |
| `logs/dinov1_optimized.log` | loss 3.10, MSEで発散 |
| `logs/dinov1_singleframe.log` | 68epで中断 |
| `logs/dinov1_singleframe_final.log` | 89epで中断 |
| `logs/dinov1_singleframe_optimized.log` | loss 3.90, MSEで発散 |
| `logs/dinov2_normalized.log` | loss 0.262, 正規化テスト |
| `logs/fixed_dinov2_10ep.log` | 10epのみ, loss 1.97 |
| `logs/fixed_slot_attention_100ep.log` | loss 1.29, 過渡期 |
| `logs/improved_dinov2_30ep.log` | 30ep, loss 1.82 |
| `logs/quick_baseline.log` | 20ep, loss 1.65 |
| `logs/quick_stopgrad.log` | 20ep, loss 1.65 |
| `logs/quick_stopgrad_refresh4.log` | 20ep, loss 1.66 |
| `logs/scaled_clip_50ep.log` | loss 0.000 → ほぼゼロ（前処理テスト） |
| `logs/scaled_dinov1_50ep.log` | loss 2.04, スケーリングテスト |
| `logs/scaled_dinov2_50ep.log` | loss 1.47, スケーリングテスト |
| `logs/single_frame_50ep.log` | loss 1.28, 初期テスト |
| `logs/stable_decoder_30ep.log` | 30ep, loss 1.90 |
| `logs/symmetry_break_50ep.log` | loss 2.07, 対称性破壊テスト |
| `logs/xavier_init_single_frame_50ep.log` | loss 1.37, Xavier初期化テスト |
| `logs/auto_backbone_comparison.log` | 自動比較スクリプトのログ（結果なし） |
| `logs/backbone_comparison_report.log` | エラー1行のみ（ファイル不在） |
| `logs/dinov2_stopgrad_analysis.log` | FileNotFoundError で失敗 |
| `logs/image_analysis.log` | KeyboardInterrupt で中断 |
| `train.log` | Phase 1 初期実験（100ep, backbone指定なし） |
| `train_500.log` | Phase 1 初期実験（500ep, backbone指定なし） |
| `train_diversity.log` | Phase 1 初期実験（200ep, backbone指定なし） |

**計31ファイル**。削除して構わない。git history にも残す必要なし。

---

## 2. 判断根拠の要約

### なぜ Tier 1 を保持するか
- 論文に記載された定量結果（FG-ARI, 訓練損失, 温度比較）の **再現性証拠**
- 査読・卒論審査で「この数値はどこから来たか」と問われたときの一次ソース
- ARI ログ3つは特に重要：Table 1 & Table 2 の全数値がここから直接取得されている

### なぜ Tier 2 を保持するか
- 「なぜ ch-norm を選んだか」「なぜ detach が必要だったか」の **対照実験**
- 論文 §6（考察）の議論を裏付ける実験的根拠
- ファイルサイズが小さい（各 10-30KB）ため、保持コストが低い

### なぜ Tier 3 を削除するか
- 中断された実験、10-30ep の短期テスト、エラーで失敗したログ
- 論文のどの主張にも関連しない
- Phase 1 ルートログは backbone/loss_type パラメータすら持たない（現フレームワーク以前の実験）

---

## 3. 実行計画（Sonnet 向け）

### Step 1: ディレクトリ作成

```bash
mkdir -p experiments/main/logs
mkdir -p experiments/sub/logs
mkdir -p experiments/analysis
mkdir -p debug
mkdir -p docs/paper/figures
mkdir -p docs/research_log
mkdir -p notebooks/phase1_exploration
```

### Step 2: Tier 1 ログの移動

```bash
# 論文直結ログ → experiments/main/logs/
mv logs/ari_dinov2_mse.log experiments/main/logs/
mv logs/ari_dinov1_channel_norm.log experiments/main/logs/
mv logs/ari_clip_detach.log experiments/main/logs/
mv logs/temp_scaling_tau05_200ep.log experiments/main/logs/
mv logs/dinov1_channel_norm.log experiments/main/logs/
mv logs/clip_detach_fix.log experiments/main/logs/
mv logs/twolayer_mlp_200ep.log experiments/main/logs/
mv logs/temp_scaling_tau02_200ep.log experiments/main/logs/
```

### Step 3: Tier 2 ログの移動

```bash
# 補助ログ → experiments/sub/logs/
mv logs/clip_baseline_50ep.log experiments/sub/logs/
mv logs/dinov1_baseline_50ep.log experiments/sub/logs/
mv logs/dinov1_normalized.log experiments/sub/logs/
mv logs/clip_normalized.log experiments/sub/logs/
mv logs/fixed_slot_to_feature_200ep.log experiments/sub/logs/
mv logs/fixed_upsampling_mlp_200ep.log experiments/sub/logs/
mv logs/video_mode_8frames_200ep.log experiments/sub/logs/
mv logs/video_stopgrad_200ep.log experiments/sub/logs/
mv logs/deeper_decoder_200ep.log experiments/sub/logs/
mv logs/dinov2_singleframe_final.log experiments/sub/logs/
```

### Step 4: Tier 3 ログの削除

```bash
# 不要ログの削除（ルート + logs/ 残り全部）
rm -f train.log train_500.log train_diversity.log
rm -f logs/*.log  # Tier 1 & 2 は既に移動済み
rmdir logs/       # 空ディレクトリ削除
```

### Step 5: .gitignore の更新

```gitignore
# 以下を .gitignore に追加/修正:
# ログは experiments/*/logs/ 配下のみ保持し、追跡する
# *.log のグローバル除外は削除し、代わりに:
!experiments/main/logs/*.log
!experiments/sub/logs/*.log
```

### Step 6: ソースファイルの移動

（この文書のスコープ外。別途 Sonnet に指示する。）

---

## 4. 注意事項

1. **checkpoints/ (12GB)** は .gitignore 済み。git LFS も未使用。削除は要ユーザー確認。
2. **data/ (631MB)** も .gitignore 済み。
3. **ocl_framework/** は外部ライブラリ（git submodule 候補）。現在は丸ごとコピーされている。
4. ログの移動前に `git status` で untracked 状態を確認すること。現在ほぼ全ファイルが untracked。
