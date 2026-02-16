"""
特徴量抽出とSlot Attentionのデバッグスクリプト
====================================================

以下を可視化：
1. DINOv2の特徴量マップ（空間情報が保たれているか）
2. Slot Attentionの各イテレーションでのマスク変化
3. 各Slotの多様性（コサイン類似度）
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from savi_dinosaur import SAViDinosaur


def visualize_feature_map(model, video_path: str, save_dir: str):
    """特徴量マップを可視化"""
    
    # データ読み込み
    data = torch.load(video_path, weights_only=False)
    video = data['video'][:1]  # 1フレームだけ
    video = F.interpolate(video, size=(224, 224), mode='bilinear', align_corners=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    video = video.to(device)
    
    # 特徴量抽出
    with torch.no_grad():
        features = model.feature_extractor(video[0:1])  # (1, 384, 16, 16)
    
    print(f"Feature map shape: {features.shape}")
    print(f"Feature map min: {features.min().item():.4f}")
    print(f"Feature map max: {features.max().item():.4f}")
    print(f"Feature map mean: {features.mean().item():.4f}")
    print(f"Feature map std: {features.std().item():.4f}")
    
    # 最初の16チャンネルを可視化
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    features_np = features[0].cpu().numpy()
    
    for i in range(16):
        ax = axes[i // 4, i % 4]
        feat = features_np[i]
        im = ax.imshow(feat, cmap='viridis')
        ax.set_title(f'Channel {i}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle('DINOv2 Feature Maps (first 16 channels)', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/feature_maps.png', dpi=150, bbox_inches='tight')
    print(f"Saved feature maps to {save_dir}/feature_maps.png")
    plt.close()
    
    # 入力画像も保存
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    img = video[0].permute(1, 2, 0).cpu().numpy()
    ax.imshow(img.clip(0, 1))
    ax.set_title('Input Image')
    ax.axis('off')
    plt.savefig(f'{save_dir}/input_image.png', dpi=150, bbox_inches='tight')
    print(f"Saved input image to {save_dir}/input_image.png")
    plt.close()


def analyze_slot_diversity(model, video_path: str, save_dir: str):
    """Slotの多様性を分析"""
    
    # データ読み込み
    data = torch.load(video_path, weights_only=False)
    video = data['video'][:1]
    video = F.interpolate(video, size=(224, 224), mode='bilinear', align_corners=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()
    video = video.to(device)
    
    with torch.no_grad():
        recon_feat, target_feat, masks, slots = model.forward_image(video[0:1])
    
    # Slotベクトルの類似度行列
    slots_norm = F.normalize(slots[0], dim=-1)  # (K, D)
    similarity_matrix = torch.mm(slots_norm, slots_norm.t())  # (K, K)
    
    print("\n=== Slot Similarity Matrix ===")
    print(similarity_matrix.cpu().numpy())
    print(f"Off-diagonal mean: {(similarity_matrix.sum() - similarity_matrix.trace()).item() / (similarity_matrix.numel() - similarity_matrix.size(0)):.4f}")
    
    # マスクの可視化
    fig, axes = plt.subplots(2, model.num_slots + 1, figsize=(3 * (model.num_slots + 1), 6))
    
    # 入力画像
    img = video[0].permute(1, 2, 0).cpu().numpy()
    axes[0, 0].imshow(img.clip(0, 1))
    axes[0, 0].set_title('Input')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    
    # 各Slotのマスク
    masks_np = masks[0].cpu().numpy()  # (K, 1, H, W)
    
    for i in range(model.num_slots):
        mask = masks_np[i, 0]
        
        # リサイズして表示
        mask_resized = F.interpolate(
            torch.from_numpy(mask).unsqueeze(0).unsqueeze(0),
            size=(224, 224),
            mode='bilinear'
        ).squeeze().numpy()
        
        axes[0, i + 1].imshow(mask_resized, cmap='viridis', vmin=0, vmax=1)
        axes[0, i + 1].set_title(f'Slot {i}')
        axes[0, i + 1].axis('off')
        
        # ヒストグラム
        axes[1, i + 1].hist(mask.flatten(), bins=50, range=(0, 1))
        axes[1, i + 1].set_title(f'Slot {i} hist')
        axes[1, i + 1].set_xlabel('Mask value')
        axes[1, i + 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/slot_analysis.png', dpi=150, bbox_inches='tight')
    print(f"Saved slot analysis to {save_dir}/slot_analysis.png")
    plt.close()
    
    # マスクの統計
    print("\n=== Mask Statistics ===")
    for i in range(model.num_slots):
        mask = masks_np[i, 0]
        print(f"Slot {i}: min={mask.min():.4f}, max={mask.max():.4f}, mean={mask.mean():.4f}, std={mask.std():.4f}")
    
    # 各ピクセルでの最大Slotを確認
    argmax_mask = masks[0].argmax(dim=0).squeeze().cpu().numpy()  # (H, W)
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    im = ax.imshow(argmax_mask, cmap='tab10', vmin=0, vmax=model.num_slots-1)
    ax.set_title('Slot Assignment (argmax)')
    ax.axis('off')
    plt.colorbar(im, ax=ax, ticks=range(model.num_slots))
    plt.savefig(f'{save_dir}/slot_assignment.png', dpi=150, bbox_inches='tight')
    print(f"Saved slot assignment to {save_dir}/slot_assignment.png")
    plt.close()
    
    # 各Slotのカバレッジ
    print("\n=== Slot Coverage ===")
    for i in range(model.num_slots):
        coverage = (argmax_mask == i).sum() / argmax_mask.size
        print(f"Slot {i}: {coverage * 100:.2f}%")


def check_reshape_order():
    """Reshapeの順序が正しいか検証"""
    
    print("\n=== Reshape Order Verification ===")
    
    # テストパターンを作成（位置情報が分かりやすいように）
    B, N, D = 1, 16, 4  # 簡略化: 4×4のパッチ
    
    # パッチ位置を示す値（左上=0, 右下=15）
    test_tokens = torch.arange(N).float().unsqueeze(0).unsqueeze(-1).repeat(1, 1, D)
    print(f"Original tokens shape: {test_tokens.shape}")
    print(f"First 4 tokens (first channel): {test_tokens[0, :4, 0].tolist()}")
    
    # permute + reshape
    h = w = int(N ** 0.5)
    test_reshaped = test_tokens.permute(0, 2, 1).reshape(B, D, h, w)
    print(f"After permute+reshape: {test_reshaped.shape}")
    print(f"First row: {test_reshaped[0, 0, 0, :].tolist()}")
    print(f"Second row: {test_reshaped[0, 0, 1, :].tolist()}")
    
    # 期待値: [0, 1, 2, 3], [4, 5, 6, 7], ...
    expected_first_row = list(range(w))
    expected_second_row = list(range(w, 2*w))
    
    is_correct = (test_reshaped[0, 0, 0, :].tolist() == expected_first_row and
                  test_reshaped[0, 0, 1, :].tolist() == expected_second_row)
    
    if is_correct:
        print("✓ Reshape order is CORRECT")
    else:
        print("✗ Reshape order is WRONG!")
        print(f"  Expected first row: {expected_first_row}")
        print(f"  Expected second row: {expected_second_row}")


def main():
    """デバッグ実行"""
    
    print("="*60)
    print("Feature & Slot Attention Debugging")
    print("="*60)
    
    # Reshape順序の検証
    check_reshape_order()
    
    # モデル読み込み
    checkpoint_dir = Path('../checkpoints/fixed_dinov2/dinov2_vits14')
    if not checkpoint_dir.exists():
        print(f"\nCheckpoint not found: {checkpoint_dir}")
        print("Please train a model first.")
        return
    
    checkpoint_path = checkpoint_dir / 'best_model.pt'
    if not checkpoint_path.exists():
        print(f"\nCheckpoint file not found: {checkpoint_path}")
        return
    
    print(f"\nLoading model from {checkpoint_path}...")
    model = SAViDinosaur(num_slots=5, backbone='dinov2_vits14')
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # サンプル動画
    video_path = '../data/movi_a_subset/metal_000.pt'
    if not Path(video_path).exists():
        print(f"\nVideo not found: {video_path}")
        return
    
    # 保存先ディレクトリ
    save_dir = '../checkpoints/debug_visualizations'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n1. Visualizing feature maps...")
    visualize_feature_map(model, video_path, save_dir)
    
    print(f"\n2. Analyzing slot diversity...")
    analyze_slot_diversity(model, video_path, save_dir)
    
    print(f"\n{'='*60}")
    print(f"Debugging completed!")
    print(f"Check results in: {save_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
