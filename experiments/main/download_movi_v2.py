"""
MOVi-A データセット v2: 正しい material_label を取得する改訂版
==============================================================

修正点 (2026-03-01, Phase 3):
  - instances/material_label (0=metal, 1=rubber) を TFDS レコードから直接取得
  - ランダムラベル (np.random.rand() < 0.3) を完全廃止
  - per-object で materials リストを保存
  - 3分類: metal_only / rubber_only / mixed
  - 300+ サンプルを新ディレクトリに保存（旧データと混在しない）

MOVi-A TFDS スキーマ:
  sample['instances']['material_label']  → per-object, 0=metal, 1=rubber
  sample['instances']['shape_label']     → per-object, 0=cube, 1=cylinder, 2=sphere
  sample['instances']['color_label']     → per-object, 0-7 (blue/brown/cyan/gray/green/purple/red/yellow)
  sample['instances']['size_label']      → per-object, 0=small, 1=large
  sample['instances']['friction']        → per-object, float (metal=0.4, rubber=0.8)
  sample['instances']['visibility']      → per-object, (T,) uint16, pixel count per frame
"""

import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
from collections import Counter

# TensorFlow のログを抑制
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow_datasets as tfds

# MOVi-A material labels
MATERIAL_NAMES = {0: 'metal', 1: 'rubber'}
SHAPE_NAMES = {0: 'cube', 1: 'cylinder', 2: 'sphere'}
COLOR_NAMES = {0: 'blue', 1: 'brown', 2: 'cyan', 3: 'gray',
               4: 'green', 5: 'purple', 6: 'red', 7: 'yellow'}
SIZE_NAMES = {0: 'small', 1: 'large'}


def classify_scene(materials: list) -> str:
    """
    シーンの材質分類を返す
    
    Args:
        materials: per-object の材質リスト ['metal', 'rubber', ...]
    Returns:
        'metal_only' / 'rubber_only' / 'mixed'
    """
    unique = set(materials)
    if unique == {'metal'}:
        return 'metal_only'
    elif unique == {'rubber'}:
        return 'rubber_only'
    else:
        return 'mixed'


def download_movi_a_v2(
    save_dir: str = "data/movi_a_v2",
    num_samples: int = 300,
    split: str = "train",
):
    """
    MOVi-A から正しい material_label 付きでサンプルをダウンロード

    Args:
        save_dir: 保存先ディレクトリ
        num_samples: ダウンロードするサンプル数
        split: TFDS split ('train' or 'validation')
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"=== MOVi-A v2 Download (正しい material_label) ===")
    print(f"Target: {num_samples} samples from '{split}' split")
    print(f"Save to: {save_dir}")
    print()

    # GCS からロード
    gcs_path = "gs://kubric-public/tfds/movi_a/128x128/1.0.0"
    print(f"Loading from {gcs_path} ...")

    try:
        builder = tfds.builder_from_directory(gcs_path)
        ds = builder.as_dataset(split=split)
        print("Successfully loaded!")
    except Exception as e:
        print(f"ERROR: GCS load failed: {e}")
        print("Falling back to re-labeling existing data...")
        relabel_existing_data(save_dir)
        return

    # 統計カウンタ
    scene_counts = Counter()  # metal_only / rubber_only / mixed
    material_counts = Counter()  # 全オブジェクトの material 分布
    objects_per_scene = []

    saved = 0

    for sample in tqdm(ds, total=num_samples, desc="Downloading"):
        if saved >= num_samples:
            break

        # --- 動画・セグメンテーション ---
        video = sample['video'].numpy()  # (T, H, W, 3) uint8
        segmentation = sample['segmentations'].numpy()  # (T, H, W, 1) uint8

        # --- per-object メタデータ ---
        instances = sample['instances']
        n_objects = len(instances['material_label'])

        materials = []
        shapes = []
        colors = []
        sizes = []

        for i in range(n_objects):
            mat_idx = instances['material_label'][i].numpy()
            shp_idx = instances['shape_label'][i].numpy()
            col_idx = instances['color_label'][i].numpy()
            siz_idx = instances['size_label'][i].numpy()

            materials.append(MATERIAL_NAMES.get(mat_idx, f'unknown_{mat_idx}'))
            shapes.append(SHAPE_NAMES.get(shp_idx, f'unknown_{shp_idx}'))
            colors.append(COLOR_NAMES.get(col_idx, f'unknown_{col_idx}'))
            sizes.append(SIZE_NAMES.get(siz_idx, f'unknown_{siz_idx}'))

        # シーン分類
        scene_type = classify_scene(materials)
        metal_count = materials.count('metal')
        rubber_count = materials.count('rubber')

        # 統計更新
        scene_counts[scene_type] += 1
        material_counts.update(materials)
        objects_per_scene.append(n_objects)

        # --- friction/restitution (material の cross-check 用) ---
        friction = instances['friction'].numpy().tolist()
        restitution = instances['restitution'].numpy().tolist()

        # --- テンソル変換 ---
        video_tensor = torch.from_numpy(video).float() / 255.0
        video_tensor = video_tensor.permute(0, 3, 1, 2)  # (T, 3, H, W)
        seg_tensor = torch.from_numpy(segmentation[:, :, :, 0]).long()  # (T, H, W)

        # --- 保存 ---
        data = {
            'video': video_tensor,
            'segmentation': seg_tensor,
            'num_instances': n_objects,
            # 正しい材質メタデータ
            'materials': materials,
            'shapes': shapes,
            'colors': colors,
            'sizes': sizes,
            'has_metal': metal_count > 0,
            'metal_count': metal_count,
            'rubber_count': rubber_count,
            'scene_type': scene_type,
            # cross-check 用物理パラメータ
            'friction': friction,
            'restitution': restitution,
        }

        filename = f"scene_{saved:04d}.pt"
        torch.save(data, os.path.join(save_dir, filename))
        saved += 1

    # --- 統計レポート ---
    print(f"\n{'=' * 60}")
    print(f"Download Complete: {saved} samples")
    print(f"{'=' * 60}")
    print(f"\nScene type distribution:")
    for stype in ['metal_only', 'rubber_only', 'mixed']:
        cnt = scene_counts[stype]
        pct = cnt / saved * 100 if saved > 0 else 0
        print(f"  {stype:15s}: {cnt:4d} ({pct:5.1f}%)")

    print(f"\nObject material distribution:")
    total_objects = sum(material_counts.values())
    for mat, cnt in material_counts.most_common():
        pct = cnt / total_objects * 100 if total_objects > 0 else 0
        print(f"  {mat:10s}: {cnt:5d} ({pct:5.1f}%)")

    print(f"\nObjects per scene: mean={np.mean(objects_per_scene):.1f}, "
          f"min={min(objects_per_scene)}, max={max(objects_per_scene)}")

    # 統計を JSON で保存
    stats = {
        'num_samples': saved,
        'split': split,
        'scene_counts': dict(scene_counts),
        'material_counts': dict(material_counts),
        'objects_per_scene_mean': float(np.mean(objects_per_scene)),
        'objects_per_scene_min': int(min(objects_per_scene)),
        'objects_per_scene_max': int(max(objects_per_scene)),
    }
    stats_path = os.path.join(save_dir, 'dataset_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to: {stats_path}")


def relabel_existing_data(new_save_dir: str):
    """
    GCS が使えない場合、既存の60サンプルを再ラベリングする fallback
    注: material_label を推定するため friction 値を使用（metal=0.4, rubber=0.8）
    既存の .pt に friction が含まれていない場合はこの方法は使えない
    """
    old_dir = "data/movi_a_subset"
    if not os.path.exists(old_dir):
        print(f"ERROR: {old_dir} not found. Cannot relabel.")
        return

    os.makedirs(new_save_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(old_dir) if f.endswith('.pt')])
    print(f"Re-labeling {len(files)} existing samples...")
    print("WARNING: 既存データには正しい material_label が含まれていません")
    print("         GCS からの再ダウンロードを強く推奨します")

    for i, f in enumerate(files):
        data = torch.load(os.path.join(old_dir, f), weights_only=False)
        # 既存データには materials フィールドがないため unknown とする
        data['materials'] = data.get('materials', ['unknown'] * data.get('num_instances', 3))
        data['scene_type'] = 'unknown'
        data['has_metal'] = None  # 不明
        data['metal_count'] = -1
        data['rubber_count'] = -1

        filename = f"scene_{i:04d}.pt"
        torch.save(data, os.path.join(new_save_dir, filename))

    print(f"Saved {len(files)} samples to {new_save_dir} (labels=unknown)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Download MOVi-A v2 with correct material labels')
    parser.add_argument('--save_dir', type=str, default='data/movi_a_v2',
                        help='Save directory')
    parser.add_argument('--num_samples', type=int, default=300,
                        help='Number of samples to download')
    parser.add_argument('--split', type=str, default='train',
                        help='TFDS split (train/validation)')
    args = parser.parse_args()

    download_movi_a_v2(
        save_dir=args.save_dir,
        num_samples=args.num_samples,
        split=args.split,
    )
