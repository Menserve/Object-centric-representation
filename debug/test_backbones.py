"""
複数のViT backboneの動作確認スクリプト
===========================================

各backboneが正しく読み込まれ、特徴抽出できるかテスト
"""

import torch
import sys
from savi_dinosaur import FeatureExtractor, SAViDinosaur

def test_backbone(backbone_name: str):
    """指定されたbackboneをテスト"""
    print(f"\n{'='*60}")
    print(f"Testing: {backbone_name}")
    print(f"{'='*60}")
    
    try:
        # 特徴抽出器を作成
        extractor = FeatureExtractor(backbone=backbone_name)
        
        # ダミー画像を作成 (1, 3, 224, 224)
        dummy_img = torch.randn(1, 3, 224, 224)
        
        # 特徴抽出
        with torch.no_grad():
            features = extractor(dummy_img)
        
        print(f"✓ Input shape: {dummy_img.shape}")
        print(f"✓ Output shape: {features.shape}")
        print(f"✓ Expected: (1, 384, 16, 16)")
        
        # 形状チェック
        assert features.shape == (1, 384, 16, 16), f"Unexpected shape: {features.shape}"
        
        # SAViDinosaurモデルの作成テスト
        model = SAViDinosaur(num_slots=5, backbone=backbone_name)
        print(f"✓ SAViDinosaur model created successfully")
        
        # パラメータ数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")
        
        # Forward pass テスト
        with torch.no_grad():
            recon_feat, target_feat, masks, slots = model.forward_image(dummy_img)
        
        print(f"✓ Forward pass successful")
        print(f"  - Recon features: {recon_feat.shape}")
        print(f"  - Target features: {target_feat.shape}")
        print(f"  - Masks: {masks.shape}")
        print(f"  - Slots: {slots.shape}")
        
        print(f"\n✅ {backbone_name} test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ {backbone_name} test FAILED!")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """全てのbackboneをテスト"""
    print("="*60)
    print("Multi-Backbone Feature Extractor Test")
    print("="*60)
    
    backbones = ['dinov2_vits14', 'dino_vits16', 'clip_vitb16']
    results = {}
    
    for backbone in backbones:
        results[backbone] = test_backbone(backbone)
    
    # 結果サマリ
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for backbone, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{backbone:20s}: {status}")
    
    # 全て成功したかチェック
    all_passed = all(results.values())
    if all_passed:
        print("\n✅ All tests passed! Ready for training.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
