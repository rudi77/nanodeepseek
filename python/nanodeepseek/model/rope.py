"""
Rotary Position Embedding (RoPE) implementation for NanoDeepSeek.
"""

import torch
import torch.nn as nn
import math


class RoPE(nn.Module):
    """
    Rotary Position Embedding as described in:
    "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    """
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute the frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos and sin for efficiency
        self._precompute_freqs_cis(max_seq_len)
    
    def _precompute_freqs_cis(self, seq_len: int):
        """Precompute cos and sin frequencies"""
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        self.register_buffer('freqs_cis', freqs_cis)
    
    def apply_rotary_emb(self, x: torch.Tensor, start_pos: int = 0):
        """Apply rotary embedding to input tensor"""
        seq_len = x.shape[1]
        freqs_cis = self.freqs_cis[start_pos:start_pos + seq_len]
        
        # Convert to complex for rotation
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        
        # Apply rotation
        x_rotated = x_complex * freqs_cis.unsqueeze(0).unsqueeze(0)
        
        # Convert back to real
        x_out = torch.view_as_real(x_rotated).flatten(-2)
        return x_out.type_as(x)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, start_pos: int = 0):
        """Apply RoPE to query and key tensors"""
        q_rotated = self.apply_rotary_emb(q, start_pos)
        k_rotated = self.apply_rotary_emb(k, start_pos)
        return q_rotated, k_rotated