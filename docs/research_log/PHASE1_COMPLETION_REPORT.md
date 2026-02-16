# Phase 1 実装完了レポート

**日付**: 2026年2月15日  
**所要時間**: 約2.5時間  
**ステータス**: ✅ 完了

---

## 実装内容

### 1. FeatureExtractor の一般化
- `DinoFeatureExtractor` → `FeatureExtractor` にリファクタリング
- 3つのbackboneをサポート：
  - `dinov2_vits14`: DINOv2 ViT-S/14
  - `dino_vits16`: DINOv1 ViT-S/16  
  - `clip_vitb16`: CLIP ViT-B/16

### 2. 特徴次元・空間解像度の統一
- 全てのbackboneから `(B, 384, 16, 16)` の特徴量を出力
- CLIPの768次元 → 384次元への projection層を追加
- DINOv1/CLIPの14×14 → 16×16へのupsampling

### 3. train_movi.py の拡張
- `--backbone` 引数でbackboneを選択可能
- その他のハイパーパラメータもコマンドライン引数化
- `diversity_weight` 引数を追加

### 4. 自動実行スクリプト
- `run_phase2_training.sh` を作成
- 4つのモード：
  - `sequential`: 順次実行
  - `parallel`: 並列実行（メイン機）
  - `clip`: CLIP単独（サブ機用）
  - `debug`: デバッグモード（10エポック）

### 5. テストスクリプト
- `test_backbones.py` で全backboneの動作確認
- 全てのテストがパス ✅

---

## 動作確認結果

| Backbone | テスト結果 | 出力形状 | 総パラメータ | 学習可能 |
|----------|-----------|---------|-------------|---------|
| DINOv2 ViT-S/14 | ✅ PASSED | (1, 384, 16, 16) | 33.0M | 10.9M |
| DINOv1 ViT-S/16 | ✅ PASSED | (1, 384, 16, 16) | 32.6M | 10.9M |
| CLIP ViT-B/16 | ✅ PASSED | (1, 384, 16, 16) | 160.9M | 11.2M |

---

## 追加されたファイル

```
src/
├── test_backbones.py          # Backbone動作確認スクリプト
└── run_phase2_training.sh     # 学習自動実行スクリプト

docs/
└── IMPLEMENTATION_PLAN_PHASE2.md  # 実装プラン詳細
```

## 更新されたファイル

```
src/
├── savi_dinosaur.py           # FeatureExtractor の一般化
└── train_movi.py              # argparse 追加、複数backbone対応

README.md                       # Usage更新、backbone比較表追加
requirements.txt                # (open-clip-torch 追加済み)
```

---

## 次のステップ（Phase 2: 学習）

### 実行方法

#### オプション1: デバッグモードで動作確認
```bash
cd src
./run_phase2_training.sh debug
```
→ 各backboneで10エポック学習（約30分）

#### オプション2: 本番学習（メイン機）
```bash
cd src
./run_phase2_training.sh sequential  # 順次実行（約3-4時間）
# または
./run_phase2_training.sh parallel    # 並列実行（約2時間）
```

#### オプション3: サブ機でCLIP学習
```bash
# サブ機（RTX4080 Super）で実行
cd src
./run_phase2_training.sh clip        # CLIP単独（約60-90分）
```

### 予想所要時間

| Backbone | エポック数 | 予想時間（RTX5090） | 予想時間（RTX4080 Super） |
|----------|-----------|------------------|----------------------|
| DINOv2 | 200 | 60-90分 | 90-120分 |
| DINOv1 | 200 | 60-90分 | 90-120分 |
| CLIP | 200 | 60-90分 | 90-120分 |

**並列実行時の合計時間**: 約2-3時間（メイン機で2つ並列）

---

## 学習中のモニタリング

### ログファイルの確認
```bash
# リアルタイムで進捗確認
tail -f ../logs/dinov2_training.log
tail -f ../logs/dinov1_training.log
```

### GPUメモリ使用状況
```bash
watch -n 1 nvidia-smi
```

### チェックポイントの確認
```bash
ls -lh ../checkpoints/dinov2_vits14/
ls -lh ../checkpoints/dino_vits16/
ls -lh ../checkpoints/clip_vitb16/
```

各ディレクトリに以下が保存される：
- `best_model.pt`: ベストモデルのチェックポイント
- `movi_result.png`: サンプル動画の可視化
- `training_history.png`: 学習曲線

---

## トラブルシューティング

### メモリ不足エラー
```bash
# バッチサイズを小さくする
python src/train_movi.py --backbone dinov2_vits14 --batch_size 1

# フレーム数を減らす
python src/train_movi.py --backbone dinov2_vits14 --max_frames 8
```

### 学習が不安定
```bash
# 学習率を下げる
python src/train_movi.py --backbone dinov2_vits14 --lr 0.0005

# 多様性損失を調整
python src/train_movi.py --backbone dinov2_vits14 --diversity_weight 0.05
```

### CLIP読み込みエラー
```bash
# open-clip-torch を再インストール
uv pip install --upgrade open-clip-torch
```

---

## Phase 2 完了後の確認事項

- [ ] 3つのbackboneの学習が全て完了
- [ ] 各backboneのベストモデルが保存されている
- [ ] 学習曲線が正常（損失が減少）
- [ ] 可視化画像が生成されている
- [ ] チェックポイントファイルのサイズが妥当（100-500MB程度）

---

## Phase 3 への準備

Phase 2が完了したら、Phase 3（評価・分析）に進みます：

1. **比較評価スクリプトの作成**
   - `src/compare_backbones.py`
   - Metal vs Rubber分離性能の比較表
   - Attention Mapの可視化

2. **特徴空間の分析**
   - t-SNE可視化
   - Slotクラスタ分析

3. **結果のまとめ**
   - 定量評価表の作成
   - 失敗例の抽出
   - 考察ポイントの整理

詳細は [docs/IMPLEMENTATION_PLAN_PHASE2.md](IMPLEMENTATION_PLAN_PHASE2.md) 参照。
