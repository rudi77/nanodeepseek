"""
Transformer Block implementation for NanoDeepSeek.
"""

import torch
import torch.nn as nn
from .rmsnorm import RMSNorm
from .attention import MLA
from .swiglu import SwiGLU


class TransformerBlock(nn.Module):
    """
    Transformer block with Multi-Head Latent Attention and SwiGLU
    """
    
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int = None,
        head_dim: int = None,
        latent_dim: int = None,
        hidden_dim: int = None,
        max_seq_len: int = 2048,
        norm_eps: float = 1e-6,
        bias: bool = False
    ):
        super().__init__()
        
        self.dim = dim
        hidden_dim = hidden_dim if hidden_dim is not None else 4 * dim

        # norm1   
        self.attention_norm = RMSNorm(dim, eps=norm_eps)

        # Multi-Head Latent Attention
        self.attention = MLA(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            latent_dim=latent_dim,
            max_seq_len=max_seq_len,
            bias=bias
        )

        # norm2
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

        # Feed Forward Network with SwiGLU
        self.feed_forward = SwiGLU(
            dim_in=dim,
            dim_hidden=hidden_dim,
            bias=bias
        )
    
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, start_pos: int = 0):
        # Attention with residual connection
        h = x + self.attention(self.attention_norm(x), mask, start_pos)
        
        # Feed forward with residual connection
        out = h + self.feed_forward(self.ffn_norm(h))
        
        return out