"""
RMSNorm implementation for NanoDeepSeek.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    
    Detailed Description:
    RMSNorm normalizes the input tensor based on the root mean square of its elements. 
    It scales the normalized output by a learnable weight parameter.

    Reference:
    "Root Mean Square Layer Normalization" (https://arxiv.org/abs/1910.07467)
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        
        # epsilon to avoid division by zero
        self.eps = eps

        # Learnable weight parameter, set to ones initially
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: [B, T, D] oder [*, D]
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x_norm = x / rms
        return x_norm * self.weight
    

if __name__ == "__main__":
    # Simple test
    rmsnorm = RMSNorm(dim=4)
    x = torch.randn(2, 4)
    y = rmsnorm(x)
    print("Input:", x)
    print("Output:", y)