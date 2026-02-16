"""
Monitor training by checking checkpoint files instead of logs.
"""

import os
import time
from pathlib import Path
from datetime import datetime

def check_checkpoint(checkpoint_dir, model_name):
    """Check if checkpoint exists and get modification time."""
    backbone_dirs = list(Path(checkpoint_dir).glob("*/best_model.pt"))
    
    if not backbone_dirs:
        return None, "No checkpoint yet"
    
    checkpoint_path = backbone_dirs[0]
    stat = checkpoint_path.stat()
    size_mb = stat.st_size / (1024 * 1024)
    mod_time = datetime.fromtimestamp(stat.st_mtime)
    age_seconds = time.time() - stat.st_mtime
    
    # Check if there's a result image
    result_dir = checkpoint_path.parent
    result_png = result_dir / "movi_result.png"
    training_png = result_dir / "training_history.png"
    
    has_result = result_png.exists()
    has_training = training_png.exists()
    
    status = "✅ COMPLETE" if (has_result and has_training) else "🔄 Training..."
    
    return {
        'model': model_name,
        'checkpoint': str(checkpoint_path),
        'size_mb': size_mb,
        'modified': mod_time.strftime("%H:%M:%S"),
        'age_sec': int(age_seconds),
        'has_result': has_result,
        'has_training': has_training,
        'status': status
    }, None

def main():
    """Monitor all training runs."""
    
    configs = [
        ("checkpoints/temp_scaling_tau05", "DINOv2 (baseline)"),
        ("checkpoints/dinov1_singleframe_tau05", "DINOv1"),
        ("checkpoints/clip_singleframe_tau05", "CLIP"),
    ]
    
    print("=" * 80)
    print("SINGLE-FRAME BACKBONE TRAINING MONITOR")
    print("=" * 80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    results = []
    
    for checkpoint_dir, model_name in configs:
        info, error = check_checkpoint(checkpoint_dir, model_name)
        if error:
            print(f"{model_name:20} ❌ {error}")
        else:
            results.append(info)
            print(f"{info['model']:20} {info['status']:15} | "
                  f"Checkpoint: {info['size_mb']:.0f}MB | "
                  f"Modified: {info['modified']} ({info['age_sec']}s ago)")
    
    print("\n" + "=" * 80)
    
    # Summary
    complete = sum(1 for r in results if r['status'] == "✅ COMPLETE")
    training = sum(1 for r in results if r['status'] == "🔄 Training...")
    
    print(f"Status: {complete}/{len(configs)} complete, {training} training")
    
    if complete == len(configs):
        print("\n🎉 ALL TRAINING COMPLETE!")
        print("\nNext step: Compare results")
        print("  python src/compare_single_frame_backbones.py")
    else:
        print(f"\n⏳ Training in progress... ({training} remaining)")
        print(f"   Estimated completion: ~5-10 minutes for 200 epochs")

if __name__ == "__main__":
    main()
