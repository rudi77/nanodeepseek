"""
Rotary Position Embedding (RoPE) implementation for NanoDeepSeek.

How this implementation works:
==============================

1. Vector to Complex (Transformation)
   The input vector x is taken pairwise (even and odd indices) and converted into 
   complex numbers. From [x1, x2, x3, x4] becomes [x1 + ix2, x3 + ix4].
   
   Code: torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

2. Rotation Factors (Preparation)
   Instead of separate cos and sin tensors, the function _precompute_freqs_cis 
   computes the rotation factors directly in complex form (torch.polar). 
   A rotation factor R for an angle θ is: R = cos(θ) + i·sin(θ).

3. Rotation (Multiplication)
   Rotation in the complex number space is simply a multiplication:
   (a + bi) · (cos(θ) + i·sin(θ))
   
   The result of this multiplication is exactly (y₁ + iy₂), where y₁ and y₂ 
   correspond to the rotation formulas mentioned above.
   
   Code: x_rotated = x_complex * freqs_cis.unsqueeze(0).unsqueeze(0)

4. Complex to Vector (Back-transformation)
   The result is decomposed back into its real and imaginary parts and combined 
   into a flat vector.
   
   Code: torch.view_as_real(x_rotated).flatten(-2)

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
        # freqs_cis shape: [seq_len, head_dim/2]
        # x_complex shape: [batch, seq_len, n_heads, head_dim/2]
        # unsqueeze(0) adds batch dim, unsqueeze(2) adds heads dim for broadcasting
        x_rotated = x_complex * freqs_cis.unsqueeze(0).unsqueeze(2)
        
        # Convert back to real
        x_out = torch.view_as_real(x_rotated).flatten(-2)
        return x_out.type_as(x)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, start_pos: int = 0):
        """Apply RoPE to query and key tensors"""
        q_rotated = self.apply_rotary_emb(q, start_pos)
        k_rotated = self.apply_rotary_emb(k, start_pos)
        return q_rotated, k_rotated


if __name__ == "__main__":
    # Simpler example with smaller dimensions
    print("\n--- Simpler Example ---")
    rope_simple = RoPE(dim=4, max_seq_len=8)
    x_simple = torch.randn(1, 3, 4)  # (batch=1, seq_len=3, dim=4)
    print("Input shape:", x_simple.shape)
    print("Input:\n", x_simple)
    
    q_rot, k_rot = rope_simple(x_simple, x_simple)
    print("\nRotated Q shape:", q_rot.shape)
    print("Rotated Q:\n", q_rot)