#!/bin/bash
# Full 200-epoch training with Stop-gradient collapse prevention

cd /home/menserve/Object-centric-representation
source .venv/bin/activate

echo "Starting 200-epoch training with Stop-gradient..."
echo "Configuration:"
echo "  - Backbone: DINOv2 ViT-S/14"
echo "  - Max frames: 8"
echo "  - Collapse prevention: Stop-gradient"
echo "  - Mask temperature: 0.5"
echo "  - Epochs: 200 (~30 minutes)"
echo ""

python -u src/train_movi.py \
  --backbone dinov2_vits14 \
  --data_dir data/movi_a_subset \
  --num_epochs 200 \
  --batch_size 2 \
  --num_slots 5 \
  --max_frames 8 \
  --lr 0.0004 \
  --diversity_weight 0.1 \
  --mask_temperature 0.5 \
  --use_stop_gradient \
  --save_dir checkpoints/video_stopgrad_200ep \
  > logs/video_stopgrad_200ep.log 2>&1

echo ""
echo "Training complete! Check logs/video_stopgrad_200ep.log for details."
