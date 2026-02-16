#!/usr/bin/env python3
"""Xavier初期化後のAttention map解析"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from analyze_iteration_dynamics import analyze_iteration_dynamics

if __name__ == '__main__':
    checkpoint_path = '../checkpoints/xavier_init_single_frame/dinov2_vits14/best_model.pt'
    data_dir = '../data/movi_a_subset'
    
    print("Analyzing Xavier-initialized model...")
    analyze_iteration_dynamics(checkpoint_path, data_dir, sample_idx=0)
