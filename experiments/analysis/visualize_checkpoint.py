#!/usr/bin/env python3
"""
チェックポイントから可視化を再生成するスクリプト
"""
import sys
import torch
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from train_movi import visualize_movi_results
from savi_dinosaur import SAViDinosaur

def main():
    checkpoint_path = '../checkpoints/single_frame_spatial/dinov2_vits14/best_model.pt'
    data_dir = '../data/movi_a_subset'
    
    # チェックポイントのロード
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    print(f"Loaded checkpoint: Best loss = {checkpoint['loss']:.6f}")
    
    # モデルの再構築
    model = SAViDinosaur(
        backbone='dinov2_vits14',
        num_slots=5,
        slot_dim=64
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # データの読み込み（テストセット1サンプル）
    from train_movi import MoviDataset
    from torch.utils.data import DataLoader
    
    dataset = MoviDataset(data_dir, split='test', max_frames=1)
    print(f"Test dataset: {len(dataset)} samples")
    
    # 可視化
    sample = dataset[0]
    
    # 可視化保存
    save_dir = Path(checkpoint_path).parent
    save_path = save_dir / 'movi_result.png'
    
    print(f"\nVisualizing to {save_path}...")
    visualize_movi_results(model, sample, device='cpu', num_frames=1, save_path=str(save_path))
    print(f"✅ Saved to {save_path}")

if __name__ == '__main__':
    main()
