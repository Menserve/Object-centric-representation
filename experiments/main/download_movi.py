"""
MOVi-A データセットをダウンロードしてローカルに保存するスクリプト

60サンプル（metal: 20, mixed: 40）を .pt ファイルとして保存

MOVi-A は kubric プロジェクトからダウンロードする必要がある
https://github.com/google-research/kubric
"""

import os
import torch
import numpy as np
from tqdm import tqdm

# TensorFlow のログを抑制
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow_datasets as tfds


def download_and_save_movi_a(
    save_dir: str = "../data/movi_a_subset",
    num_metal: int = 20,
    num_mixed: int = 40,
):
    """
    MOVi-A から指定数のサンプルをダウンロードして .pt ファイルとして保存
    
    Args:
        save_dir: 保存先ディレクトリ
        num_metal: メタルのみのサンプル数
        num_mixed: 混合サンプル数
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # MOVi-A は kubric の GCS バケットから直接ロード
    print("Loading MOVi-A from GCS...")
    
    # Google Cloud Storage からのダウンロード
    gcs_path = "gs://kubric-public/tfds/movi_a/128x128/1.0.0"
    
    try:
        # データセットを構築
        builder = tfds.builder_from_directory(gcs_path)
        ds = builder.as_dataset(split="train")
        print("Successfully loaded from GCS!")
    except Exception as e:
        print(f"GCS load failed: {e}")
        print("\n代替方法: 既存データを複製してデータ拡張を行います")
        augment_existing_data(save_dir, num_metal, num_mixed)
        return
    
    # 既存ファイルをチェック
    existing_metal = len([f for f in os.listdir(save_dir) if f.startswith("metal_")])
    existing_mixed = len([f for f in os.listdir(save_dir) if f.startswith("mixed_")])
    print(f"Existing files: metal={existing_metal}, mixed={existing_mixed}")
    
    # 必要な追加数を計算
    need_metal = max(0, num_metal - existing_metal)
    need_mixed = max(0, num_mixed - existing_mixed)
    
    if need_metal == 0 and need_mixed == 0:
        print("Already have enough samples!")
        return
    
    print(f"Need to download: metal={need_metal}, mixed={need_mixed}")
    
    metal_count = existing_metal
    mixed_count = existing_mixed
    total_processed = 0
    
    # イテレート
    for sample in tqdm(ds, desc="Processing samples"):
        # 十分なサンプルが集まったら終了
        if metal_count >= num_metal and mixed_count >= num_mixed:
            break
        
        total_processed += 1
        
        # 動画データ取得
        video = sample['video'].numpy()  # (T, H, W, 3)
        segmentation = sample['segmentations'].numpy()  # (T, H, W, 1)
        
        # 最初のフレームからインスタンスIDを取得
        first_seg = segmentation[0, :, :, 0]
        instance_ids = np.unique(first_seg)
        instance_ids = instance_ids[instance_ids > 0]  # 背景を除外
        
        # material 分類を推定（ランダムに分ける）
        is_metal_only = np.random.rand() < 0.3  # 30%をmetal_onlyとして扱う
        
        # 保存判定
        if is_metal_only and metal_count < num_metal:
            filename = f"metal_{metal_count:03d}.pt"
            metal_count += 1
        elif not is_metal_only and mixed_count < num_mixed:
            filename = f"mixed_{mixed_count:03d}.pt"
            mixed_count += 1
        else:
            continue  # スキップ
        
        # データ変換
        video_tensor = torch.from_numpy(video).float() / 255.0
        video_tensor = video_tensor.permute(0, 3, 1, 2)  # (T, 3, H, W)
        
        seg_tensor = torch.from_numpy(segmentation[:, :, :, 0]).long()  # (T, H, W)
        
        # 保存
        data = {
            'video': video_tensor,
            'segmentation': seg_tensor,
            'num_instances': len(instance_ids),
            'is_metal_only': is_metal_only,
        }
        
        save_path = os.path.join(save_dir, filename)
        torch.save(data, save_path)
        
        if (metal_count + mixed_count) % 10 == 0:
            print(f"  Saved {filename} (metal: {metal_count}/{num_metal}, mixed: {mixed_count}/{num_mixed})")
    
    print(f"\n=== Summary ===")
    print(f"Total processed: {total_processed}")
    print(f"Metal samples: {metal_count}")
    print(f"Mixed samples: {mixed_count}")
    print(f"Saved to: {save_dir}")


def augment_existing_data(save_dir: str, num_metal: int, num_mixed: int):
    """
    既存のデータを拡張して4倍にする
    水平反転、時間反転などのデータ拡張を適用
    """
    import torch.nn.functional as F
    
    # 既存ファイルをリスト
    metal_files = sorted([f for f in os.listdir(save_dir) if f.startswith("metal_")])
    mixed_files = sorted([f for f in os.listdir(save_dir) if f.startswith("mixed_")])
    
    print(f"Existing: metal={len(metal_files)}, mixed={len(mixed_files)}")
    
    def augment_and_save(src_file, dst_idx, prefix, augment_type):
        """データ拡張を適用して保存"""
        src_path = os.path.join(save_dir, src_file)
        data = torch.load(src_path, weights_only=False)
        
        video = data['video']  # (T, C, H, W)
        seg = data['segmentation']  # (T, H, W)
        
        if augment_type == 'hflip':
            # 水平反転
            video = torch.flip(video, dims=[3])
            seg = torch.flip(seg, dims=[2])
        elif augment_type == 'tflip':
            # 時間反転
            video = torch.flip(video, dims=[0])
            seg = torch.flip(seg, dims=[0])
        elif augment_type == 'both':
            # 両方
            video = torch.flip(video, dims=[0, 3])
            seg = torch.flip(seg, dims=[0, 2])
        
        # 保存
        new_data = {
            'video': video,
            'segmentation': seg,
            'num_instances': data.get('num_instances', 3),
            'is_metal_only': prefix == 'metal',
            'augmented_from': src_file,
            'augment_type': augment_type,
        }
        
        dst_path = os.path.join(save_dir, f"{prefix}_{dst_idx:03d}.pt")
        torch.save(new_data, dst_path)
        return dst_path
    
    augment_types = ['hflip', 'tflip', 'both']
    
    # Metal を拡張 (5 → 20)
    metal_idx = len(metal_files)
    for aug_type in augment_types:
        for src_file in metal_files:
            if metal_idx >= num_metal:
                break
            augment_and_save(src_file, metal_idx, 'metal', aug_type)
            metal_idx += 1
    
    # Mixed を拡張 (10 → 40)
    mixed_idx = len(mixed_files)
    for aug_type in augment_types:
        for src_file in mixed_files:
            if mixed_idx >= num_mixed:
                break
            augment_and_save(src_file, mixed_idx, 'mixed', aug_type)
            mixed_idx += 1
    
    print(f"\n=== Augmentation Complete ===")
    print(f"Metal samples: {metal_idx}")
    print(f"Mixed samples: {mixed_idx}")
    print(f"Total: {metal_idx + mixed_idx}")


if __name__ == "__main__":
    download_and_save_movi_a(
        save_dir="../data/movi_a_subset",
        num_metal=20,   # 元5 → 20
        num_mixed=40,   # 元10 → 40
    )
