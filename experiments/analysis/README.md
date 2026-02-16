# Analysis Tools & Results

実験結果の分析とレポート生成。

## Analysis Scripts

### Result Analysis
- `analyze_metal_vs_rubber.py` - 材質別（Metal vs Non-metal）性能比較
- `analyze_result_images.py` - 出力画像の定性的分析
- `detailed_mask_analysis.py` - マスクの詳細分析
- `compare_single_frame_backbones.py` - バックボーン（DINOv2/DINOv1/CLIP）比較

### Model Inspection
- `visualize_checkpoint.py` - チェックポイントの可視化
- `verify_ground_truth.py` - Ground truthデータの検証
- `verify_old_checkpoint.py` - 旧チェックポイントの検証

### Training Monitoring
- `monitor_training.py` - 訓練プロセスのリアルタイム監視

## Notebooks
- `experiment_report.ipynb` - 実験結果の総合レポート（2026-02-02作成）

## Purpose

これらのツールは、論文のFigure生成、Table作成、定性的分析に使用されました。特に：
- Table 1, 2: バックボーン間の定量比較
- Figure 2: 3バックボーンの出力マスク比較
- §5 定性分析: マスクの可視化と解釈
