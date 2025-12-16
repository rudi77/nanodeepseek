"""
SwiGLU activation function implementation for NanoDeepSeek.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU activation function as described in:
    "GLU Variants Improve Transformer" (https://arxiv.org/abs/2002.05202)
    """
    
    def __init__(self, dim_in: int, dim_hidden: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim_in, dim_hidden, bias=bias)
        self.w2 = nn.Linear(dim_hidden, dim_in, bias=bias)
        self.w3 = nn.Linear(dim_in, dim_hidden, bias=bias)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    

if __name__ == "__main__":
    # Simple test
    swiglu = SwiGLU(dim_in=4, dim_hidden=8)
    x = torch.randn(2, 4)
    y = swiglu(x)
    print("Input:", x)
    print("Output:", y)