"""
MOVi-A データセットを使った SAVi-DINOSAUR 学習スクリプト
==========================================================

保存済みの .pt ファイルを読み込んで学習する。
TensorFlow は不要。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
import time

from savi_dinosaur import SAViDinosaur


# ==========================================
# 1. MOVi-A Dataset (PyTorch .pt files)
# ==========================================
class MoviDataset(Dataset):
    """
    保存済み .pt ファイルから MOVi-A データを読み込む
    
    各 .pt ファイルの構造:
        - video: (T, 3, H, W) float32, 0-1 正規化済み
        - segmentation: (T, H, W) uint8, 物体ID
        - materials: List[str], 各物体の材質
        - shapes: List[str], 各物体の形状
        - colors: List[str], 各物体の色
        - has_metal: bool, 金属物体を含むか
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'all',  # 'all', 'metal', 'mixed'
        target_size: Tuple[int, int] = (224, 224),
        max_frames: Optional[int] = None
    ):
        """
        Args:
            data_dir: .pt ファイルがあるディレクトリ
            split: 'all' (全て), 'metal' (金属のみ), 'mixed' (混合のみ)
            target_size: リサイズ後のサイズ (DINOv2用に224x224)
            max_frames: 使用する最大フレーム数 (Noneなら全フレーム)
        """
        self.data_dir = data_dir
        self.target_size = target_size
        self.max_frames = max_frames
        
        # ファイル一覧を取得
        all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt')])
        
        # splitでフィルタリング
        if split == 'metal':
            self.files = [f for f in all_files if f.startswith('metal_')]
        elif split == 'mixed':
            self.files = [f for f in all_files if f.startswith('mixed_')]
        else:
            self.files = all_files
        
        print(f"MoviDataset: Found {len(self.files)} samples (split={split})")
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with:
                - video: (T, 3, H, W) リサイズ済み動画
                - segmentation: (T, H, W) リサイズ済みセグメンテーション
                - materials: List[str]
                - has_metal: bool
        """
        path = os.path.join(self.data_dir, self.files[idx])
        sample = torch.load(path, weights_only=False)
        
        video = sample['video']  # (T, 3, 256, 256)
        segmentation = sample['segmentation']  # (T, 256, 256)
        
        # フレーム数を制限
        if self.max_frames is not None:
            video = video[:self.max_frames]
            segmentation = segmentation[:self.max_frames]
        
        # リサイズ (256x256 → 224x224)
        t, c, h, w = video.shape
        if (h, w) != self.target_size:
            video = torch.nn.functional.interpolate(
                video, size=self.target_size, mode='bilinear', align_corners=False
            )
            segmentation = torch.nn.functional.interpolate(
                segmentation.unsqueeze(1).float(),
                size=self.target_size,
                mode='nearest'
            ).squeeze(1).long()
        
        return {
            'video': video,
            'segmentation': segmentation,
            'materials': sample.get('materials', []),
            'has_metal': sample.get('has_metal', sample.get('is_metal_only', False)),
            'filename': self.files[idx]
        }


def collate_fn(batch: List[dict]) -> dict:
    """カスタム collate 関数（材質情報などを保持）"""
    videos = torch.stack([item['video'] for item in batch])
    segmentations = torch.stack([item['segmentation'] for item in batch])
    
    return {
        'video': videos,
        'segmentation': segmentations,
        'materials': [item['materials'] for item in batch],
        'has_metal': [item['has_metal'] for item in batch],
        'filename': [item['filename'] for item in batch]
    }


# ==========================================
# 2. 学習ループ
# ==========================================
def train_on_movi(
    model: SAViDinosaur,
    train_loader: DataLoader,
    num_epochs: int = 50,
    lr: float = 0.0004,
    device: str = 'cuda',
    log_interval: int = 5,
    save_dir: Optional[str] = None
) -> dict:
    """
    MOVi-A データセットで SAVi-DINOSAUR を学習
    
    Args:
        model: SAViDinosaur モデル
        train_loader: DataLoader
        num_epochs: エポック数
        lr: 学習率
        device: デバイス
        log_interval: ログ出力間隔（バッチ数）
        save_dir: モデル保存先（Noneなら保存しない）
    
    Returns:
        dict with training history
    """
    model = model.to(device)
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr
    )
    
    # 学習率スケジューラ
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    history = {
        'epoch_losses': [],
        'batch_losses': [],
        'lr': []
    }
    
    best_loss = float('inf')
    
    print(f"\n{'='*60}")
    print(f"Training SAVi-DINOSAUR on MOVi-A")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {lr}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Dataset size: {len(train_loader.dataset)}")
    print(f"{'='*60}\n")
    
    # 多様性損失の重み
    diversity_weight = 0.1
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            video = batch['video'].to(device)  # (B, T, 3, H, W)
            
            optimizer.zero_grad()
            
            # Forward pass（マスクも取得）
            result = model.forward_video(video, return_all_masks=True)
            recon_loss = result['total_loss']
            
            # 多様性損失：スロット間のマスクが異なることを促進
            # all_masks: (B, T, K, 1, H, W)
            masks = result['all_masks']  # (B, T, K, 1, H, W)
            b, t, k, _, h, w = masks.shape
            masks_flat = masks.view(b * t, k, h * w)  # (B*T, K, H*W)
            
            # 各スロット対間のコサイン類似度を計算
            masks_norm = F.normalize(masks_flat, dim=-1)  # (B*T, K, H*W)
            similarity = torch.bmm(masks_norm, masks_norm.transpose(1, 2))  # (B*T, K, K)
            
            # 対角成分（自己類似度=1）を除いた非対角成分の平均を最小化
            mask_diag = torch.eye(k, device=device).unsqueeze(0).expand(b*t, -1, -1)
            off_diag = similarity * (1 - mask_diag)
            diversity_loss = off_diag.sum() / (b * t * k * (k - 1))
            
            # 総損失
            loss = recon_loss + diversity_weight * diversity_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_losses.append(loss.item())
            history['batch_losses'].append(loss.item())
            
            if batch_idx % log_interval == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}: Loss = {loss.item():.6f}")
        
        # Epoch 終了
        scheduler.step()
        avg_loss = np.mean(epoch_losses)
        history['epoch_losses'].append(avg_loss)
        history['lr'].append(scheduler.get_last_lr()[0])
        
        elapsed = time.time() - start_time
        print(f"\nEpoch {epoch+1}/{num_epochs} | "
              f"Loss: {avg_loss:.6f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f} | "
              f"Time: {elapsed:.1f}s\n")
        
        # ベストモデル保存
        if save_dir and avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
            print(f"  ✓ Saved best model (loss={avg_loss:.6f})")
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"{'='*60}")
    
    return history


# ==========================================
# 3. 評価・可視化
# ==========================================
def evaluate_model(
    model: SAViDinosaur,
    test_loader: DataLoader,
    device: str = 'cuda'
) -> dict:
    """モデル評価"""
    model.eval()
    
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in test_loader:
            video = batch['video'].to(device)
            result = model.forward_video(video, return_all_masks=False)
            total_loss += result['total_loss'].item()
            num_batches += 1
    
    return {
        'avg_loss': total_loss / num_batches
    }


def visualize_movi_results(
    model: SAViDinosaur,
    sample: dict,
    device: str = 'cuda',
    save_path: Optional[str] = None,
    num_frames: int = 8
):
    """MOVi-A サンプルのスロット可視化"""
    model.eval()
    
    video = sample['video'].unsqueeze(0).to(device)  # (1, T, 3, H, W)
    
    with torch.no_grad():
        result = model.forward_video(video, return_all_masks=True)
    
    all_masks = result['all_masks'][0]  # (T, K, 1, H, W)
    t, k, _, h, w = all_masks.shape
    
    # フレームを間引く
    frame_indices = np.linspace(0, t-1, min(num_frames, t), dtype=int)
    
    # マスクをリサイズ
    masks_resized = torch.nn.functional.interpolate(
        all_masks.view(t * k, 1, h, w),
        size=(224, 224),
        mode='bilinear'
    ).view(t, k, 224, 224)
    
    # 可視化
    fig, axes = plt.subplots(len(frame_indices), k + 2, figsize=(3 * (k + 2), 3 * len(frame_indices)))
    
    for i, t_idx in enumerate(frame_indices):
        # 入力フレーム
        frame = sample['video'][t_idx].permute(1, 2, 0).cpu().numpy().clip(0, 1)
        axes[i, 0].imshow(frame)
        axes[i, 0].set_title(f"Frame {t_idx}")
        axes[i, 0].axis('off')
        
        # セグメンテーション GT
        seg = sample['segmentation'][t_idx].cpu().numpy()
        axes[i, 1].imshow(seg, cmap='tab10')
        axes[i, 1].set_title("GT Seg")
        axes[i, 1].axis('off')
        
        # 各スロットのマスク
        for slot_idx in range(k):
            mask = masks_resized[t_idx, slot_idx].cpu().numpy()
            axes[i, slot_idx + 2].imshow(mask, cmap='viridis', vmin=0, vmax=1)
            axes[i, slot_idx + 2].set_title(f"Slot {slot_idx}")
            axes[i, slot_idx + 2].axis('off')
    
    plt.suptitle(f"Materials: {sample['materials']}", fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_training_history(history: dict, save_path: Optional[str] = None):
    """学習曲線をプロット"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Epoch loss
    axes[0].plot(history['epoch_losses'], 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss per Epoch')
    axes[0].grid(True, alpha=0.3)
    
    # Batch loss (smoothed)
    batch_losses = history['batch_losses']
    window = min(10, len(batch_losses) // 10)
    if window > 1:
        smoothed = np.convolve(batch_losses, np.ones(window)/window, mode='valid')
        axes[1].plot(smoothed, 'g-', linewidth=1, alpha=0.8)
    else:
        axes[1].plot(batch_losses, 'g-', linewidth=1, alpha=0.8)
    axes[1].set_xlabel('Batch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training Loss per Batch (smoothed)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


# ==========================================
# 4. メイン
# ==========================================
def main():
    """MOVi-A での SAVi-DINOSAUR 学習デモ"""
    
    # 設定
    DATA_DIR = "../data/movi_a_subset"
    SAVE_DIR = "../checkpoints"
    NUM_EPOCHS = 200  # 多様性損失を追加したので短めに
    BATCH_SIZE = 2
    NUM_SLOTS = 5  # 物体数に近い数に（3〜5物体が多い）
    MAX_FRAMES = 12  # メモリ節約のため
    LR = 0.001  # 学習率を上げる
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # データセット
    print("\n1. Loading MOVi-A dataset...")
    dataset = MoviDataset(
        data_dir=DATA_DIR,
        split='all',
        target_size=(224, 224),
        max_frames=MAX_FRAMES
    )
    
    # Train/Test split (簡易版: 最後の2サンプルをテスト用)
    train_indices = list(range(len(dataset) - 2))
    test_indices = list(range(len(dataset) - 2, len(dataset)))
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # モデル作成
    print("\n2. Creating SAVi-DINOSAUR model...")
    model = SAViDinosaur(num_slots=NUM_SLOTS)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 学習
    print("\n3. Training...")
    history = train_on_movi(
        model=model,
        train_loader=train_loader,
        num_epochs=NUM_EPOCHS,
        lr=LR,
        device=device,
        log_interval=10,
        save_dir=SAVE_DIR
    )
    
    # 評価
    print("\n4. Evaluating...")
    eval_result = evaluate_model(model, test_loader, device)
    print(f"Test Loss: {eval_result['avg_loss']:.6f}")
    
    # 可視化
    print("\n5. Visualizing results...")
    test_sample = dataset[test_indices[0]]
    visualize_movi_results(
        model, test_sample, device,
        save_path=os.path.join(SAVE_DIR, "movi_result.png")
    )
    
    # 学習曲線
    plot_training_history(
        history,
        save_path=os.path.join(SAVE_DIR, "training_history.png")
    )
    
    print("\n✅ Training completed!")


if __name__ == "__main__":
    main()
