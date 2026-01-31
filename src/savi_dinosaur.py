"""
SAVi-DINOSAUR: Slot Attention for Video with DINOv2 Features
==============================================================

DINOSAURを動画（時系列）に拡張したモデル。
前フレームのスロットを次フレームの初期値として使用することで、
物体の時間的一貫性（Temporal Consistency）を実現する。

主な変更点:
1. SlotAttention に slots_init 引数を追加（外部から初期スロットを渡せる）
2. SlotPredictor を追加（前フレームのスロットから次フレームの初期値を予測）
3. SAViDinosaur モデルで動画全体を処理

参考文献:
- SAVi: Kipf et al., "Conditional Object-Centric Learning from Video", ICLR 2022
- DINOSAUR: Seitzer et al., "Bridging the Gap to Real-World Object-Centric Learning", ICLR 2023
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from typing import Optional, Tuple, List


# ==========================================
# 1. DINOv2 Feature Extractor
# ==========================================
class DinoFeatureExtractor(nn.Module):
    """DINOv2 ViT-S/14 による特徴抽出（frozen）"""
    
    def __init__(self):
        super().__init__()
        print("Loading DINOv2 model...")
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        for param in self.dino.parameters():
            param.requires_grad = False
        self.dino.eval()
        self.feat_dim = 384  # ViT-S の次元

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) 入力画像
        Returns:
            features: (B, 384, 16, 16) パッチ特徴量
        """
        with torch.no_grad():
            features = self.dino.forward_features(x)["x_norm_patchtokens"]
        b, n, d = features.shape
        h = w = int(n**0.5)
        features = features.permute(0, 2, 1).reshape(b, d, h, w)
        return features


# ==========================================
# 2. Slot Attention（SAVi対応版）
# ==========================================
class SlotAttentionSAVi(nn.Module):
    """
    SAVi対応のSlot Attention
    
    通常のSlot Attentionとの違い:
    - slots_init 引数で外部から初期スロットを渡せる
    - 渡されない場合はランダム初期化（静止画モード）
    """
    
    def __init__(
        self,
        num_slots: int,
        dim: int,
        iters: int = 5,
        hidden_dim: int = 512,
        eps: float = 1e-8
    ):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.scale = dim ** -0.5
        self.eps = eps
        
        # 学習可能なスロット初期化パラメータ（ランダム初期化時に使用）
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim) * 0.1)
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, dim))
        
        # Attention layers
        self.norm_features = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)
        
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )

    def forward(
        self,
        inputs: torch.Tensor,
        slots_init: Optional[torch.Tensor] = None,
        num_slots: Optional[int] = None
    ) -> torch.Tensor:
        """
        Args:
            inputs: (B, N, D) 入力特徴量（N=16*16=256）
            slots_init: (B, K, D) 初期スロット（SAVi用、Noneならランダム初期化）
            num_slots: スロット数（Noneならself.num_slotsを使用）
        Returns:
            slots: (B, K, D) 出力スロット
        """
        inputs = self.norm_features(inputs)
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots
        
        # ★ SAVi の核心: 初期スロットを外部から渡せる
        if slots_init is not None:
            # 前フレームから予測されたスロットを使用
            slots = slots_init
        else:
            # 最初のフレーム: 学習可能なパラメータからサンプリング
            mu = self.slots_mu.expand(b, n_s, -1)
            sigma = self.slots_log_sigma.exp().expand(b, n_s, -1)
            slots = mu + sigma * torch.randn_like(mu)
        
        k = self.to_k(inputs)
        v = self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.to_q(slots)
            
            # Attention: dots (B, K, N)
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            
            # Softmax over slots (競合メカニズム)
            attn = dots.softmax(dim=1) + self.eps
            
            # Weighted mean
            attn_sum = attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum('bjd,bij->bid', v, attn / attn_sum)
            
            # GRU update
            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )
            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))
        
        return slots


# ==========================================
# 3. Slot Predictor（次フレーム予測）
# ==========================================
class SlotPredictor(nn.Module):
    """
    前フレームのスロットから次フレームの初期スロットを予測
    
    これにより、スロットが物体を「追跡」できるようになる。
    単純なMLPで実装（TransformerやLSTMも可能）。
    """
    
    def __init__(self, dim: int, hidden_dim: int = 512):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slots: (B, K, D) 前フレームのスロット
        Returns:
            predicted_slots: (B, K, D) 次フレームの初期スロット
        """
        # 残差接続: 物体が大きく動かない場合に有効
        return slots + self.predictor(slots)


# ==========================================
# 4. Feature Decoder
# ==========================================
class FeatureDecoder(nn.Module):
    """スロットからDINO特徴量とマスクを復元"""
    
    def __init__(self, feat_dim: int = 384, resolution: Tuple[int, int] = (16, 16)):
        super().__init__()
        self.feat_dim = feat_dim
        self.resolution = resolution
        
        self.decoder = nn.Sequential(
            nn.Conv2d(feat_dim + 2, 384, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, feat_dim + 1, 3, padding=1)  # feat + mask
        )
    
    def build_grid(self, batch_size: int, device: torch.device) -> torch.Tensor:
        h, w = self.resolution
        x = torch.linspace(-1, 1, w, device=device)
        y = torch.linspace(-1, 1, h, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=0)
        return grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    
    def forward(
        self,
        slots: torch.Tensor,
        num_slots: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            slots: (B, K, D) スロット
            num_slots: スロット数
        Returns:
            recon_combined: (B, D, H, W) 再構成特徴量
            pred_feats: (B, K, D, H, W) スロットごとの特徴量
            masks: (B, K, 1, H, W) スロットごとのマスク
        """
        b, k, d = slots.shape
        h, w = self.resolution
        
        # Spatial Broadcast
        slots_2d = slots.view(b * k, d, 1, 1).expand(-1, -1, h, w)
        grid = self.build_grid(b * k, slots.device)
        
        decode_in = torch.cat([slots_2d, grid], dim=1)
        out = self.decoder(decode_in)
        out = out.view(b, k, d + 1, h, w)
        
        pred_feats = out[:, :, :d, :, :]
        masks = torch.softmax(out[:, :, d:, :, :], dim=1)
        
        recon_combined = torch.sum(pred_feats * masks, dim=1)
        
        return recon_combined, pred_feats, masks


# ==========================================
# 5. SAVi-DINOSAUR Model
# ==========================================
class SAViDinosaur(nn.Module):
    """
    SAVi + DINOSAUR: 動画対応の物体中心学習モデル
    
    使い方:
    1. 静止画モード: forward_image() を使用
    2. 動画モード: forward_video() を使用（フレーム間でスロットを引き継ぐ）
    """
    
    def __init__(self, num_slots: int = 5, feat_dim: int = 384):
        super().__init__()
        self.num_slots = num_slots
        self.feat_dim = feat_dim
        
        # Components
        self.feature_extractor = DinoFeatureExtractor()
        self.pos_emb = nn.Parameter(torch.randn(1, 16, 16, feat_dim) * 0.05)
        self.slot_attention = SlotAttentionSAVi(num_slots, feat_dim, iters=5, hidden_dim=512)
        self.slot_predictor = SlotPredictor(feat_dim, hidden_dim=512)
        self.decoder = FeatureDecoder(feat_dim, resolution=(16, 16))
    
    def encode(self, img: torch.Tensor) -> torch.Tensor:
        """画像をDINO特徴量に変換"""
        features = self.feature_extractor(img)  # (B, D, H, W)
        b, c, h, w = features.shape
        features_perm = features.permute(0, 2, 3, 1)  # (B, H, W, D)
        features_pos = features_perm + self.pos_emb
        features_flat = features_pos.reshape(b, -1, c)  # (B, N, D)
        return features_flat, features
    
    def forward_image(
        self,
        img: torch.Tensor,
        slots_init: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        静止画モード（1フレーム処理）
        
        Args:
            img: (B, 3, 224, 224) 入力画像
            slots_init: (B, K, D) 初期スロット（オプション）
        Returns:
            recon_feat: (B, D, H, W) 再構成特徴量
            target_feat: (B, D, H, W) ターゲット特徴量
            masks: (B, K, 1, H, W) マスク
            slots: (B, K, D) 出力スロット
        """
        features_flat, target_feat = self.encode(img)
        slots = self.slot_attention(features_flat, slots_init=slots_init)
        recon_feat, _, masks = self.decoder(slots, self.num_slots)
        
        return recon_feat, target_feat, masks, slots
    
    def forward_video(
        self,
        video: torch.Tensor,
        return_all_masks: bool = True
    ) -> dict:
        """
        動画モード（複数フレーム処理）
        
        ★ SAVi の核心: 前フレームのスロットを次フレームの初期値として使用
        
        Args:
            video: (B, T, 3, 224, 224) 動画（Tフレーム）
            return_all_masks: 全フレームのマスクを返すか
        Returns:
            dict with:
                - total_loss: スカラー
                - all_masks: (B, T, K, 1, H, W) マスク
                - all_slots: (B, T, K, D) スロット
        """
        b, t, c, h, w = video.shape
        device = video.device
        
        all_losses = []
        all_masks = []
        all_slots = []
        
        slots = None  # 最初はNone（ランダム初期化）
        
        for frame_idx in range(t):
            frame = video[:, frame_idx]  # (B, 3, H, W)
            
            # ★ SAVi: 前フレームのスロットがあれば、予測して初期値にする
            if slots is not None:
                slots_init = self.slot_predictor(slots)
            else:
                slots_init = None
            
            # このフレームを処理
            recon_feat, target_feat, masks, slots = self.forward_image(
                frame, slots_init=slots_init
            )
            
            # 損失計算
            loss = ((target_feat - recon_feat) ** 2).mean()
            all_losses.append(loss)
            
            if return_all_masks:
                all_masks.append(masks)
                all_slots.append(slots)
        
        # 結果をまとめる
        total_loss = torch.stack(all_losses).mean()
        
        result = {
            'total_loss': total_loss,
            'frame_losses': all_losses,
        }
        
        if return_all_masks:
            result['all_masks'] = torch.stack(all_masks, dim=1)  # (B, T, K, 1, H, W)
            result['all_slots'] = torch.stack(all_slots, dim=1)  # (B, T, K, D)
        
        return result
    
    def forward(self, x: torch.Tensor, **kwargs):
        """自動モード判定"""
        if x.dim() == 4:
            return self.forward_image(x, **kwargs)
        elif x.dim() == 5:
            return self.forward_video(x, **kwargs)
        else:
            raise ValueError(f"Expected 4D (image) or 5D (video) tensor, got {x.dim()}D")


# ==========================================
# 6. 動画データセット（テスト用）
# ==========================================
class RotatingDogDataset(Dataset):
    """
    テスト用: 犬画像を回転させて擬似的な動画を生成
    
    これにより「物体は動くが、形状は同じ」という条件をシミュレート。
    金属光沢のテストには、色相を変化させることで「反射の動き」を模倣。
    """
    
    def __init__(
        self,
        num_videos: int = 100,
        frames_per_video: int = 8,
        resolution: Tuple[int, int] = (224, 224),
        mode: str = 'rotate'  # 'rotate', 'hue_shift', 'both'
    ):
        self.num_videos = num_videos
        self.frames_per_video = frames_per_video
        self.resolution = resolution
        self.mode = mode
        self.raw_img = self._load_dog_image()
        self.to_tensor = transforms.ToTensor()
    
    def _load_dog_image(self) -> Image.Image:
        url = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
        try:
            response = requests.get(url, timeout=5)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            return img.resize(self.resolution)
        except:
            print("Using noise (No Internet).")
            return Image.fromarray(np.uint8(np.random.rand(*self.resolution, 3) * 255))
    
    def __len__(self) -> int:
        return self.num_videos
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns:
            video: (T, 3, H, W) 動画テンソル
        """
        frames = []
        
        # ランダムな開始条件
        np.random.seed(idx)
        start_angle = np.random.uniform(0, 360)
        start_hue = np.random.uniform(-0.5, 0.5)
        
        for t in range(self.frames_per_video):
            img = self.raw_img.copy()
            
            if self.mode in ['rotate', 'both']:
                angle = start_angle + t * (360 / self.frames_per_video)
                img = TF.rotate(img, angle)
            
            if self.mode in ['hue_shift', 'both']:
                # 色相をゆっくり変化（金属光沢の反射が動くシミュレーション）
                hue = start_hue + t * 0.1
                hue = (hue + 0.5) % 1.0 - 0.5  # [-0.5, 0.5] に収める
                img = TF.adjust_hue(img, hue)
            
            frames.append(self.to_tensor(img))
        
        return torch.stack(frames)  # (T, 3, H, W)


# ==========================================
# 7. 学習・評価関数
# ==========================================
def train_savi_dinosaur(
    model: SAViDinosaur,
    dataloader: DataLoader,
    num_epochs: int = 10,
    lr: float = 0.0004,
    device: str = 'cuda'
) -> List[float]:
    """SAVi-DINOSAURの学習"""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        for batch_idx, video in enumerate(dataloader):
            video = video.to(device)
            
            optimizer.zero_grad()
            result = model.forward_video(video, return_all_masks=False)
            loss = result['total_loss']
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}")
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f}")
    
    return losses


def evaluate_temporal_consistency(
    model: SAViDinosaur,
    video: torch.Tensor,
    device: str = 'cuda'
) -> dict:
    """
    時間的一貫性（Temporal Consistency）を評価
    
    評価指標:
    1. Slot Cosine Similarity: 連続フレーム間のスロットの類似度
    2. Mask IoU: 連続フレーム間のマスクの重なり
    """
    model.eval()
    video = video.to(device)
    
    with torch.no_grad():
        result = model.forward_video(video.unsqueeze(0), return_all_masks=True)
    
    all_slots = result['all_slots'][0]  # (T, K, D)
    all_masks = result['all_masks'][0]  # (T, K, 1, H, W)
    
    t, k, d = all_slots.shape
    
    # Slot Cosine Similarity
    slot_similarities = []
    for t_idx in range(t - 1):
        slots_t = all_slots[t_idx]  # (K, D)
        slots_t1 = all_slots[t_idx + 1]  # (K, D)
        
        # 各スロットの類似度
        sim = torch.nn.functional.cosine_similarity(slots_t, slots_t1, dim=-1)
        slot_similarities.append(sim.mean().item())
    
    avg_slot_similarity = np.mean(slot_similarities)
    
    # Mask IoU
    mask_ious = []
    for t_idx in range(t - 1):
        masks_t = (all_masks[t_idx] > 0.5).float()  # (K, 1, H, W)
        masks_t1 = (all_masks[t_idx + 1] > 0.5).float()
        
        intersection = (masks_t * masks_t1).sum(dim=(-1, -2, -3))
        union = ((masks_t + masks_t1) > 0).float().sum(dim=(-1, -2, -3))
        iou = (intersection / (union + 1e-8)).mean().item()
        mask_ious.append(iou)
    
    avg_mask_iou = np.mean(mask_ious)
    
    return {
        'slot_similarity': avg_slot_similarity,
        'mask_iou': avg_mask_iou,
        'slot_similarities_per_frame': slot_similarities,
        'mask_ious_per_frame': mask_ious
    }


def visualize_video_slots(
    model: SAViDinosaur,
    video: torch.Tensor,
    device: str = 'cuda',
    save_path: Optional[str] = None
):
    """動画のスロット可視化"""
    model.eval()
    video = video.to(device)
    
    with torch.no_grad():
        result = model.forward_video(video.unsqueeze(0), return_all_masks=True)
    
    all_masks = result['all_masks'][0]  # (T, K, 1, H, W)
    t, k, _, h, w = all_masks.shape
    
    # マスクをリサイズ
    masks_resized = torch.nn.functional.interpolate(
        all_masks.view(t * k, 1, h, w),
        size=(224, 224),
        mode='bilinear'
    ).view(t, k, 224, 224)
    
    # 可視化
    fig, axes = plt.subplots(t, k + 1, figsize=(3 * (k + 1), 3 * t))
    
    for t_idx in range(t):
        # 入力フレーム
        frame = video[t_idx].permute(1, 2, 0).cpu().numpy().clip(0, 1)
        axes[t_idx, 0].imshow(frame)
        axes[t_idx, 0].set_title(f"Frame {t_idx}")
        axes[t_idx, 0].axis('off')
        
        # 各スロットのマスク
        for slot_idx in range(k):
            mask = masks_resized[t_idx, slot_idx].cpu().numpy()
            axes[t_idx, slot_idx + 1].imshow(mask, cmap='gray', vmin=0, vmax=1)
            axes[t_idx, slot_idx + 1].set_title(f"Slot {slot_idx}")
            axes[t_idx, slot_idx + 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


# ==========================================
# 8. メイン: デモ実行
# ==========================================
def main():
    """SAVi-DINOSAURのデモ"""
    
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # モデル作成
    print("\n1. Creating SAVi-DINOSAUR model...")
    model = SAViDinosaur(num_slots=5)
    print(f"   - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   - Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # テスト用動画データセット
    print("\n2. Creating test video dataset...")
    dataset = RotatingDogDataset(
        num_videos=50,
        frames_per_video=8,
        mode='hue_shift'  # 色相変化で金属光沢をシミュレート
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 学習
    print("\n3. Training SAVi-DINOSAUR...")
    losses = train_savi_dinosaur(
        model, dataloader,
        num_epochs=3,  # デモ用に短く
        lr=0.0004,
        device=device
    )
    
    # 評価
    print("\n4. Evaluating temporal consistency...")
    test_video = dataset[0]  # (T, 3, H, W)
    metrics = evaluate_temporal_consistency(model, test_video, device)
    print(f"   - Slot Similarity: {metrics['slot_similarity']:.4f}")
    print(f"   - Mask IoU: {metrics['mask_iou']:.4f}")
    
    # 可視化
    print("\n5. Visualizing slots over time...")
    visualize_video_slots(model, test_video, device, save_path="savi_result.png")
    
    print("\n✅ Demo completed!")
    return model, metrics


if __name__ == "__main__":
    model, metrics = main()
