import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import RoPE


class MLACache:
    """
    Stores latent KV-representations per layer
    latents: [B, T_cached, d_latent]
    """

    def __init__(self):
        self.latents = None


    def append(self, c_kv_new: torch.Tensor):
        """Append new latent KV representations
        c_kv_new: [B, T_new, d_latent]
        """
        if self.latents is None:
            self.latents = c_kv_new
        else:
            self.latents = torch.cat([self.latents, c_kv_new], dim=1)

    @property
    def seq_len(self):
        return 0 if self.latents is None else self.latents.size(1)
    

class MLA(nn.Module):
    """
    Multi-Head Latent Attention (MLA) - V2 (KV latent-only, cachefähig)

    Shapes:
      x:      [B, T_new, D]
      q:      [B, T_new, H, Dh]
      c_kv:   [B, T_new, d_latent]         (das wird gecached)
      c_all:  [B, T_total, d_latent]
      kv_up:  [B, T_total, 2*H_kv*Dh]
      k,v:    [B, T_total, H_kv, Dh] -> ggf. auf H expandiert (GQA)
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
        assert n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.head_dim = head_dim if head_dim is not None else dim // n_heads
        assert self.head_dim * n_heads == dim, "dim must be divisible by n_heads"

        self.latent_dim = latent_dim if latent_dim is not None else dim // 2

        # Q stays in original dimension
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=bias)

        # KV projected to latent dimension (Down-projection)
        self.w_down = nn.Linear(dim, self.latent_dim, bias=bias)

        # KV up-projection: from latent -> (K,V) for kv_heads
        # Output: 2 * (n_kv_heads * head_dim)
        self.w_up = nn.Linear(self.latent_dim, 2 * self.n_kv_heads * self.head_dim, bias=bias)

        # output projection
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=bias)

        # RoPE head_dim
        self.rope = RoPE(self.head_dim, max_seq_len=max_seq_len)

        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,                 # [B, T_new, D]
        mask: torch.Tensor = None,       # optional (siehe unten)
        cache: MLACache = None,
        use_cache: bool = False,
    ):
        B, T_new, D = x.shape

        # ---- 1) Q ----
        q = self.wq(x).view(B, T_new, self.n_heads, self.head_dim)  # [B,T_new,H,Dh]

        # ---- 2) latent KV (Down) ----
        c_kv = self.w_down(x)  # [B,T_new,d_latent]

        if use_cache:
            if cache is None:
                cache = MLACache()
            start_pos = cache.seq_len   # Position der neuen Tokens im Gesamtstrom
            cache.append(c_kv)
            c_all = cache.latents       # [B,T_total,d_latent]
        else:
            start_pos = 0
            c_all = c_kv                # [B,T_new,d_latent]

        T_total = c_all.size(1)

        # ---- 3) latent -> K,V (Up) ----
        kv_up = self.w_up(c_all)  # [B,T_total, 2*H_kv*Dh]
        kv_up = kv_up.view(B, T_total, self.n_kv_heads, 2 * self.head_dim)  # [B,T_total,H_kv,2Dh]
        k, v = kv_up.split(self.head_dim, dim=-1)  # beide [B,T_total,H_kv,Dh]

        # ---- 4) RoPE auf Q und K ----
        # Q nur für die neuen Tokens: start_pos ... start_pos+T_new-1
        q = self.rope.apply_rotary_emb(q, start_pos=start_pos)   # [B,T_new,H,Dh]

        # K für alle Keys ab Position 0..T_total-1
        # (weil k repräsentiert den gesamten Cache)
        k = self.rope.apply_rotary_emb(k, start_pos=0)           # [B,T_total,H_kv,Dh]

        # ---- 5) GQA: kv-heads auf heads expandieren (falls nötig) ----
        if self.n_kv_heads < self.n_heads:
            repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat, dim=2)  # [B,T_total,H,Dh]
            v = v.repeat_interleave(repeat, dim=2)  # [B,T_total,H,Dh]

        # ---- 6) Attention MatMul ----
        # in [B,H,T,Dh] für MatMul
        q = q.permute(0, 2, 1, 3)  # [B,H,T_new,Dh]
        k = k.permute(0, 2, 1, 3)  # [B,H,T_total,Dh]
        v = v.permute(0, 2, 1, 3)  # [B,H,T_total,Dh]

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,T_new,T_total]

        # ---- 7) Causal Masking ----
        # Wenn du keinen Mask-Tensor übergibst, bauen wir eine korrekte kausale Maske,
        # die auch bei caching stimmt.
        if mask is None:
            # Token i (0..T_new-1) darf bis Position start_pos + i attendieren.
            i = torch.arange(T_new, device=x.device).unsqueeze(-1)           # [T_new,1]
            j = torch.arange(T_total, device=x.device).unsqueeze(0)          # [1,T_total]
            causal = j <= (start_pos + i)                                     # [T_new,T_total]
            scores = scores.masked_fill(~causal.unsqueeze(0).unsqueeze(0), float("-inf"))
        else:
            # Falls du eine additive Maske nutzt, sollte sie auf scores broadcasten:
            # z.B. [1,1,T_new,T_total] oder [B,1,T_new,T_total]
            # Wenn du eine 0/1 Maske nutzt, wandle sie vorher passend um.
            scores = scores + mask

        attn = F.softmax(scores, dim=-1)              # [B,H,T_new,T_total]
        out = torch.matmul(attn, v)                   # [B,H,T_new,Dh]

        out = out.permute(0, 2, 1, 3).contiguous()    # [B,T_new,H,Dh]
        out = out.view(B, T_new, self.n_heads * self.head_dim)  # [B,T_new,D]
        out = self.wo(out)                            # [B,T_new,D]

        return out, cache        