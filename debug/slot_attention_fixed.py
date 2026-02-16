"""
Slot Collapse修正版: OCL Framework準拠のSlot Attention実装

修正内容:
1. Xavier初期化（分散を適切に設定）
2. Temperature調整可能なSoftmax
3. Multi-head Slot Attention（オプション）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np


class SlotAttentionFixed(nn.Module):
    """
    Over-smoothing対策版 Slot Attention
    
    主な修正:
    - Slot初期化にXavier初期化を使用（OCL Framework準拠）
    - Softmax temperatureを調整可能に
    - より強い初期化分散
    """
    def __init__(
        self,
        num_slots: int,
        dim: int,
        iters: int = 5,
        hidden_dim: int = 512,
        eps: float = 1e-8,
        temperature: float = 1.0  # NEW: Softmax temperature
    ):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.scale = (dim ** -0.5) / temperature  # temperatureで制御
        self.eps = eps
        
        # ★ 修正1: Xavier初期化（OCL Framework準拠）
        self.slots_mu = nn.Parameter(torch.zeros(1, num_slots, dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        
        with torch.no_grad():
            # Xavier uniform初期化
            nn.init.xavier_uniform_(self.slots_mu)
            nn.init.xavier_uniform_(self.slots_logsigma)
        
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
            inputs: (B, N, D) 入力特徴量
            slots_init: (B, K, D) 初期スロット（オプション）
            num_slots: スロット数（オプション）
        Returns:
            slots: (B, K, D) 出力スロット
        """
        inputs = self.norm_features(inputs)
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots
        
        if slots_init is not None:
            slots = slots_init
        else:
            # ★ 修正後: Xavier初期化により適切な分散
            mu = self.slots_mu.expand(b, n_s, -1)
            sigma = self.slots_logsigma.exp().expand(b, n_s, -1)
            slots = mu + sigma * torch.randn_like(mu)
        
        k = self.to_k(inputs)
        v = self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.to_q(slots)
            
            # ★ 修正後: temperatureでスケール調整
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            
            # Softmax over slots
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


class MultiHeadSlotAttention(nn.Module):
    """
    Multi-head版Slot Attention（OCL Framework準拠）
    
    各headが異なる特徴に注目することでOver-smoothing軽減
    """
    def __init__(
        self,
        num_slots: int,
        dim: int,
        n_heads: int = 4,  # Multi-head
        iters: int = 5,
        hidden_dim: int = 512,
        eps: float = 1e-8
    ):
        super().__init__()
        self.num_slots = num_slots
        self.n_heads = n_heads
        self.iters = iters
        self.eps = eps
        
        if dim % n_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by n_heads ({n_heads})")
        
        self.dims_per_head = dim // n_heads
        self.scale = self.dims_per_head ** -0.5  # Head単位のスケール
        
        # Xavier初期化
        self.slots_mu = nn.Parameter(torch.zeros(1, num_slots, dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        
        with torch.no_grad():
            nn.init.xavier_uniform_(self.slots_mu)
            nn.init.xavier_uniform_(self.slots_logsigma)
        
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
        inputs = self.norm_features(inputs)
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots
        
        if slots_init is not None:
            slots = slots_init
        else:
            mu = self.slots_mu.expand(b, n_s, -1)
            sigma = self.slots_logsigma.exp().expand(b, n_s, -1)
            slots = mu + sigma * torch.randn_like(mu)
        
        # Multi-head projection
        k = self.to_k(inputs).view(b, n, self.n_heads, self.dims_per_head)  # (B, N, H, D/H)
        v = self.to_v(inputs).view(b, n, self.n_heads, self.dims_per_head)

        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.to_q(slots).view(b, n_s, self.n_heads, self.dims_per_head)  # (B, K, H, D/H)
            
            # Multi-head attention
            dots = torch.einsum('bihd,bjhd->bihj', q, k) * self.scale  # (B, K, H, N)
            
            # Softmax over slots AND heads
            attn = dots.flatten(1, 2).softmax(dim=1)  # (B, K*H, N)
            attn = attn.view(b, n_s, self.n_heads, n) + self.eps
            
            # Weighted mean
            attn_sum = attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum('bjhd,bihj->bihd', v, attn / attn_sum)  # (B, K, H, D/H)
            updates = updates.reshape(b, n_s, d)
            
            # GRU update
            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )
            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))
        
        return slots
