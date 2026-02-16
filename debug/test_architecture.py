#!/usr/bin/env python3
"""
修正したアーキテクチャの動作確認
"""
import torch
import sys
sys.path.insert(0, '/home/menserve/Object-centric-representation/src')

from savi_dinosaur import SAViDinosaur

def test_architecture():
    print("="*60)
    print("Testing Corrected SAVi-DINOSAUR Architecture")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # Create model
    model = SAViDinosaur(num_slots=5, backbone='dinov2_vits14', slot_dim=64)
    model = model.to(device).eval()
    
    print("Model components:")
    print(f"  Feature extractor: {model.feature_extractor.backbone_name}")
    print(f"  Feature dim (DINOv2): {model.feat_dim}")
    print(f"  Slot dim (Slot Attention): {model.slot_dim}")
    print(f"  Num slots: {model.num_slots}")
    
    # Test forward pass
    print("\nTesting forward pass:")
    batch_size = 2
    img = torch.randn(batch_size, 3, 224, 224).to(device)
    
    with torch.no_grad():
        # Encode
        features_projected, target_feat = model.encode(img)
        print(f"  DINOv2 features: {target_feat.shape}")
        print(f"  Projected features: {features_projected.shape}")
        
        # Slot Attention
        slots = model.slot_attention(features_projected)
        print(f"  Slots (low-dim): {slots.shape}")
        
        # Upsample to feature space
        slots_upsampled = model.slot_to_feature(slots)
        print(f"  Slots (high-dim): {slots_upsampled.shape}")
        
        # Decoder
        recon_feat, pred_feats, masks = model.decoder(slots_upsampled, model.num_slots)
        print(f"  Reconstructed features: {recon_feat.shape}")
        print(f"  Masks: {masks.shape}")
        
        # Full forward
        recon, target, masks_full, slots_out = model.forward_image(img)
        print(f"\nFull forward pass:")
        print(f"  Reconstruction: {recon.shape}")
        print(f"  Target: {target.shape}")
        print(f"  Masks: {masks_full.shape}")
        print(f"  Slots: {slots_out.shape}")
    
    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {trainable_params:,}")
    
    print("\n" + "="*60)
    print("✅ Architecture test passed!")
    print("="*60)

if __name__ == "__main__":
    test_architecture()
