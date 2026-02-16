# Sub-Experiments & Ablation Studies

論文の議論を支持する補助実験とアブレーションスタディ。

## Scripts

### Ablation Analysis
- `analyze_quick_ablation.py` - クイックアブレーション分析
- `compare_quick_ablations.py` - アブレーション比較
- `analyze_iteration_dynamics.py` - イタレーション動態分析
- `analyze_xavier_dynamics.py` - Xavier初期化の動態
- `compare_xavier_fix.py` - Xavier修正の比較

### Video Mode Experiments
- `analyze_video_masks.py` - ビデオモードのマスク分析
- `compare_video_ablations.py` - ビデオモードアブレーション比較
- `diagnose_video_failure.py` - ビデオモード失敗の診断

### Alternative Approaches
- `alternative_diversity_loss.py` - 多様性損失の代替実装

## Logs (`logs/`)

論文§6（考察）を裏付ける対照実験：

**Before/After Comparisons:**
- `clip_baseline_50ep.log` - CLIP detach前 (Best loss: -0.005, 負の損失)
- `dinov1_baseline_50ep.log` - DINOv1 MSE発散 (Best loss: 4.692)

**Normalization Comparisons:**
- `dinov1_normalized.log` - DINOv1 L2正規化 (Best loss: 0.260)
- `clip_normalized.log` - CLIP L2正規化 (Best loss: 0.022)

**Architecture Variations:**
- `fixed_slot_to_feature_200ep.log` - Slot-to-feature改善 (Best loss: 0.711)
- `fixed_upsampling_mlp_200ep.log` - Upsampling MLP (Best loss: 0.784)
- `deeper_decoder_200ep.log` - Deeper decoder失敗 (Best loss: 1.238)

**Video Mode:**
- `video_mode_8frames_200ep.log` - 8フレームビデオモード (Best loss: 0.778)
- `video_stopgrad_200ep.log` - stop-grad適用 (Best loss: 0.833)

**Baselines:**
- `dinov2_singleframe_final.log` - Single-frame baseline (Best loss: 0.733)

## Role

これらの実験は、論文での主要な設計選択（ch-norm、detach、τ=0.5）の妥当性を示す根拠となっています。
