# Main Experiments

論文の主要な実験とその結果を含むディレクトリ。

## Contents

### Scripts
- `download_movi.py` - MOVi-Aデータセットのダウンロードと準備

### Logs (`logs/`)
論文 (Table 1, 2, 3) に直接引用された実験ログ：

**ARI評価結果:**
- `ari_dinov2_mse.log` - DINOv2 FG-ARI: 0.165 ± 0.186
- `ari_dinov1_channel_norm.log` - DINOv1 FG-ARI: 0.153 ± 0.172
- `ari_clip_detach.log` - CLIP FG-ARI: 0.041 ± 0.118

**主要訓練ログ:**
- `temp_scaling_tau05_200ep.log` - DINOv2 τ=0.5 (Best loss: 0.729)
- `dinov1_channel_norm.log` - DINOv1 ch-norm (Best loss: 0.163)
- `clip_detach_fix.log` - CLIP detach修正後 (Best loss: 0.036)

**比較実験:**
- `twolayer_mlp_200ep.log` - 2層MLPデコーダ (Best loss: 0.685)
- `temp_scaling_tau02_200ep.log` - τ=0.3 温度比較 (Best loss: 0.783)

## Corresponding Checkpoints

これらのログに対応するチェックポイントは `checkpoints/` にあります：
- `checkpoints/temp_scaling_tau05/`
- `checkpoints/dinov1_channel_norm/`
- `checkpoints/clip_detach_fix/`
- `checkpoints/twolayer_mlp_200ep/`
- `checkpoints/temp_scaling_tau02/`

## Paper Citation

これらの実験結果は「DINOSAUR による鏡面反射物体の教師なしセグメンテーション」論文の定量的根拠となっています。
