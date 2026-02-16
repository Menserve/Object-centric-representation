# Debug & Development Tools

開発過程でのデバッグ、テスト、問題診断用スクリプト。

## Debug Scripts

### Feature & Activation Debugging
- `debug_features.py` - 特徴量のデバッグ
- `debug_feature_saturation.py` - 特徴量飽和の調査
- `debug_intermediate_activations.py` - 中間活性化の検査

### Attention & Decoder Debugging
- `debug_attention_maps.py` - Attentionマップの可視化
- `debug_decoder_logits.py` - デコーダLogitsの調査

### Spatial & Architecture Debugging
- `debug_patch_boundaries.py` - パッチ境界の検証
- `slot_attention_fixed.py` - Slot Attention修正版

## Test Scripts

### Architecture Tests
- `test_architecture.py` - アーキテクチャの動作確認
- `test_backbones.py` - バックボーンの動作テスト
- `test_spatial_order.py` - 空間順序の検証
- `test_collapse_prevention.py` - スロット崩壊防止機構のテスト

## Usage

これらのスクリプトは開発・デバッグ用途であり、論文の実験結果には直接貢献していません。アーキテクチャの問題診断、実装の検証、バグ修正に使用されました。

**Note:** これらは最終的な実験には使用されていませんが、開発履歴として保持しています。
