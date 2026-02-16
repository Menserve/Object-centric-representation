"""
空間情報の順序を徹底的にテスト
====================================

1. torch.meshgridの動作確認
2. Reshapeの順序確認
3. Decoderのgrid生成を可視化
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def test_meshgrid_indexing():
    """torch.meshgridの動作を確認"""
    print("=" * 60)
    print("Test 1: torch.meshgrid indexing")
    print("=" * 60)
    
    h, w = 4, 4
    y = torch.arange(h, dtype=torch.float32)  # [0, 1, 2, 3]
    x = torch.arange(w, dtype=torch.float32)  # [0, 1, 2, 3]
    
    # indexing='ij' (matrix indexing)
    grid_y_ij, grid_x_ij = torch.meshgrid(y, x, indexing='ij')
    
    print("\nWith indexing='ij':")
    print("grid_y (should be same values in each row):")
    print(grid_y_ij)
    print("\ngrid_x (should be same values in each column):")
    print(grid_x_ij)
    
    # indexing='xy' (Cartesian indexing)
    grid_x_xy, grid_y_xy = torch.meshgrid(x, y, indexing='xy')
    
    print("\n\nWith indexing='xy':")
    print("grid_x (should be same values in each column):")
    print(grid_x_xy)
    print("\ngrid_y (should be same values in each row):")
    print(grid_y_xy)
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    axes[0, 0].imshow(grid_y_ij, cmap='viridis')
    axes[0, 0].set_title("grid_y with indexing='ij'\n(Y coordinate, vertical)")
    for i in range(h):
        for j in range(w):
            axes[0, 0].text(j, i, f'{grid_y_ij[i, j]:.0f}', 
                           ha='center', va='center', color='white')
    
    axes[0, 1].imshow(grid_x_ij, cmap='viridis')
    axes[0, 1].set_title("grid_x with indexing='ij'\n(X coordinate, horizontal)")
    for i in range(h):
        for j in range(w):
            axes[0, 1].text(j, i, f'{grid_x_ij[i, j]:.0f}', 
                           ha='center', va='center', color='white')
    
    axes[1, 0].imshow(grid_x_xy, cmap='viridis')
    axes[1, 0].set_title("grid_x with indexing='xy'\n(X coordinate, horizontal)")
    for i in range(h):
        for j in range(w):
            axes[1, 0].text(j, i, f'{grid_x_xy[i, j]:.0f}', 
                           ha='center', va='center', color='white')
    
    axes[1, 1].imshow(grid_y_xy, cmap='viridis')
    axes[1, 1].set_title("grid_y with indexing='xy'\n(Y coordinate, vertical)")
    for i in range(h):
        for j in range(w):
            axes[1, 1].text(j, i, f'{grid_y_xy[i, j]:.0f}', 
                           ha='center', va='center', color='white')
    
    plt.tight_layout()
    plt.savefig('../checkpoints/test_meshgrid.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved to ../checkpoints/test_meshgrid.png")
    plt.close()


def test_reshape_order():
    """Reshapeの順序をテスト"""
    print("\n" + "=" * 60)
    print("Test 2: Reshape order")
    print("=" * 60)
    
    h, w = 4, 4
    
    # パッチ番号を明示的に割り当て
    # パッチ0=top-left, パッチ1=その右, ..., パッチ15=bottom-right
    patches = torch.arange(h * w, dtype=torch.float32)  # [0, 1, 2, ..., 15]
    
    print(f"\nFlat patches (1D): {patches}")
    
    # Reshapeして2D化
    patches_2d = patches.reshape(h, w)
    
    print(f"\nAfter reshape({h}, {w}):")
    print(patches_2d)
    print("\nExpected: Top row [0,1,2,3], Second row [4,5,6,7], etc.")
    
    # 可視化
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    im = ax.imshow(patches_2d, cmap='viridis')
    ax.set_title(f"Patch numbering after reshape({h}, {w})")
    
    for i in range(h):
        for j in range(w):
            ax.text(j, i, f'{int(patches_2d[i, j])}', 
                   ha='center', va='center', color='white', fontsize=16)
    
    plt.colorbar(im, ax=ax)
    plt.savefig('../checkpoints/test_reshape_order.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved to ../checkpoints/test_reshape_order.png")
    plt.close()


def test_decoder_grid():
    """Decoderのgrid生成をテスト"""
    print("\n" + "=" * 60)
    print("Test 3: Decoder grid generation")
    print("=" * 60)
    
    from savi_dinosaur import FeatureDecoder
    
    decoder = FeatureDecoder(feat_dim=384, resolution=(16, 16))
    grid = decoder.build_grid(batch_size=1, device='cpu')[0]  # (2, 16, 16)
    
    grid_x = grid[0]  # (16, 16)
    grid_y = grid[1]  # (16, 16)
    
    print(f"\nGrid X shape: {grid_x.shape}")
    print(f"Grid Y shape: {grid_y.shape}")
    print(f"\nGrid X range: [{grid_x.min():.2f}, {grid_x.max():.2f}]")
    print(f"Grid Y range: [{grid_y.min():.2f}, {grid_y.max():.2f}]")
    
    # Top-left corner (should be (-1, -1))
    print(f"\nTop-left (0, 0): grid_x={grid_x[0, 0]:.2f}, grid_y={grid_y[0, 0]:.2f}")
    # Top-right corner (should be (+1, -1))
    print(f"Top-right (0, 15): grid_x={grid_x[0, 15]:.2f}, grid_y={grid_y[0, 15]:.2f}")
    # Bottom-left corner (should be (-1, +1))
    print(f"Bottom-left (15, 0): grid_x={grid_x[15, 0]:.2f}, grid_y={grid_y[15, 0]:.2f}")
    # Bottom-right corner (should be (+1, +1))
    print(f"Bottom-right (15, 15): grid_x={grid_x[15, 15]:.2f}, grid_y={grid_y[15, 15]:.2f}")
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im0 = axes[0].imshow(grid_x, cmap='viridis', vmin=-1, vmax=1)
    axes[0].set_title("Grid X (horizontal coordinate)\nShould increase left→right")
    plt.colorbar(im0, ax=axes[0])
    
    # Corner annotations
    axes[0].text(0, 0, 'TL', ha='center', va='center', color='red', fontsize=12, weight='bold')
    axes[0].text(15, 0, 'TR', ha='center', va='center', color='red', fontsize=12, weight='bold')
    axes[0].text(0, 15, 'BL', ha='center', va='center', color='red', fontsize=12, weight='bold')
    axes[0].text(15, 15, 'BR', ha='center', va='center', color='red', fontsize=12, weight='bold')
    
    im1 = axes[1].imshow(grid_y, cmap='viridis', vmin=-1, vmax=1)
    axes[1].set_title("Grid Y (vertical coordinate)\nShould increase top→bottom")
    plt.colorbar(im1, ax=axes[1])
    
    # Corner annotations
    axes[1].text(0, 0, 'TL', ha='center', va='center', color='red', fontsize=12, weight='bold')
    axes[1].text(15, 0, 'TR', ha='center', va='center', color='red', fontsize=12, weight='bold')
    axes[1].text(0, 15, 'BL', ha='center', va='center', color='red', fontsize=12, weight='bold')
    axes[1].text(15, 15, 'BR', ha='center', va='center', color='red', fontsize=12, weight='bold')
    
    plt.tight_layout()
    plt.savefig('../checkpoints/test_decoder_grid.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved to ../checkpoints/test_decoder_grid.png")
    plt.close()
    
    # 問題の診断
    print("\n" + "=" * 60)
    print("Diagnostic Results:")
    print("=" * 60)
    
    # 期待値との比較
    expected_tl_x, expected_tl_y = -1, -1
    expected_tr_x, expected_tr_y = +1, -1
    expected_bl_x, expected_bl_y = -1, +1
    expected_br_x, expected_br_y = +1, +1
    
    tl_ok = (abs(grid_x[0, 0] - expected_tl_x) < 0.1 and abs(grid_y[0, 0] - expected_tl_y) < 0.1)
    tr_ok = (abs(grid_x[0, 15] - expected_tr_x) < 0.1 and abs(grid_y[0, 15] - expected_tr_y) < 0.1)
    bl_ok = (abs(grid_x[15, 0] - expected_bl_x) < 0.1 and abs(grid_y[15, 0] - expected_bl_y) < 0.1)
    br_ok = (abs(grid_x[15, 15] - expected_br_x) < 0.1 and abs(grid_y[15, 15] - expected_br_y) < 0.1)
    
    print(f"✓ Top-left OK: {tl_ok}")
    print(f"✓ Top-right OK: {tr_ok}")
    print(f"✓ Bottom-left OK: {bl_ok}")
    print(f"✓ Bottom-right OK: {br_ok}")
    
    if all([tl_ok, tr_ok, bl_ok, br_ok]):
        print("\n✅ Grid generation is CORRECT!")
    else:
        print("\n❌ Grid generation has ISSUES!")
        print("    This might explain the 'inverted L' artifact pattern.")


if __name__ == "__main__":
    Path('../checkpoints').mkdir(exist_ok=True)
    
    test_meshgrid_indexing()
    test_reshape_order()
    test_decoder_grid()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
