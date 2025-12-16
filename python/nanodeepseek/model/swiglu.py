"""
SwiGLU activation function implementation for NanoDeepSeek.

"GLU Variants Improve Transformer" (https://arxiv.org/abs/2002.05202)

How does SwiGLU work?
======================
SwiGLU is essentially a "Gated Linear Unit" (GLU) that uses a gating mechanism to 
dynamically control which information is allowed to flow through the network.

The functionality can be summarized in the following steps:

1. Three Linear Transformations
   Unlike conventional FFNs that often use only two linear layers, SwiGLU employs 
   three separate linear transformations of the input vector. These are represented 
   by trainable weight matrices (W, V) and bias vectors (b, c).

2. The Gating Mechanism
   One of the transformations is passed through the Swish activation function (also 
   known as SiLU). The Swish function (Swish(x) = x * σ(x), where σ is the sigmoid 
   function) is a smoother, non-monotonic function compared to the binary nature of 
   a simple sigmoid or ReLU function.

3. Element-wise Multiplication
   The output of the Swish function (the "gate") is then element-wise multiplied 
   with the output of another linear transformation. The gate thus determines how 
   much of the information from the main path is allowed through or blocked, similar 
   to a selective filter.

4. Dimension Reduction
   The result of this multiplication is finally projected back to the original or 
   another desired model dimension through a third linear transformation (down-projection).

Advantages over other activation functions:
===========================================
- Better Performance: SwiGLU leads to lower loss and better performance on tasks 
  like language modeling and machine translation compared to GeLU or ReLU.
  
- Dynamic Feature Selection: The gating mechanism allows the network to dynamically 
  select relevant features and suppress irrelevant ones, leading to more efficient 
  pattern representation.
  
- Smoother Gradients: The smoothness of the Swish function compared to ReLU helps 
  avoid the problem of "dying neurons" and ensures more stable and better gradients 
  during training.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU activation function as described in:
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