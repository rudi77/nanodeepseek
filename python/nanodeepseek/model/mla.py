"""
Multi-Head Latent Attention (MLA) implementation for NanoDeepSeek.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import RoPE


class MLA(nn.Module):
    """
    Multi-Head Latent Attention as described in DeepSeek-V2
    """
    
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int = None,
        head_dim: int = None,
        latent_dim: int = None,
        max_seq_len: int = 2048,
        bias: bool = False
    ):
        super().__init__()
        
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.head_dim = head_dim if head_dim is not None else dim // n_heads
        self.latent_dim = latent_dim if latent_dim is not None else dim // 2
        
        # Query, Key, Value projections
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=bias)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        
        # Latent projections
        self.q_latent = nn.Linear(dim, self.latent_dim, bias=bias)
        self.kv_latent = nn.Linear(dim, self.latent_dim, bias=bias)
        
        # Output projection
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=bias)
        
        # RoPE
        self.rope = RoPE(self.head_dim, max_seq_len)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, start_pos: int = 0):
        batch_size, seq_len, _ = x.shape
        
        # Standard Q, K, V projections
        q = self.wq(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        # Latent projections
        q_latent = self.q_latent(x)  # [batch, seq, latent_dim]
        kv_latent = self.kv_latent(x)  # [batch, seq, latent_dim]
        
        # Apply RoPE to queries and keys
        q, k = self.rope(q, k, start_pos)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch, n_heads, seq, head_dim]
        k = k.transpose(1, 2)  # [batch, n_kv_heads, seq, head_dim]
        v = v.transpose(1, 2)  # [batch, n_kv_heads, seq, head_dim]
        
        # Grouped Query Attention if n_kv_heads < n_heads
        if self.n_kv_heads < self.n_heads:
            k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
            v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        
        # Transpose back and reshape
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # Latent attention component
        latent_scores = torch.matmul(q_latent, kv_latent.transpose(-2, -1)) * (self.latent_dim ** -0.5)
        if mask is not None:
            latent_scores = latent_scores.masked_fill(mask.squeeze(1).squeeze(1) == 0, float('-inf'))
        
        latent_weights = F.softmax(latent_scores, dim=-1)
        latent_out = torch.matmul(latent_weights, kv_latent)
        
        # Combine standard and latent attention
        # This is a simplified combination - the actual DeepSeek-V2 may use a different approach
        combined_out = out + latent_out.unsqueeze(1).expand(-1, self.n_heads, -1, -1).contiguous().view(batch_size, seq_len, -1)
        
        # Final output projection
        return self.wo(combined_out)