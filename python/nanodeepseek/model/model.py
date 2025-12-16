"""
NanoDeepSeek model implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .rmsnorm import RMSNorm
from .block import TransformerBlock


class NanoDeepSeek(nn.Module):
    """
    NanoDeepSeek: A simplified version of DeepSeek model
    """
    
    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        n_kv_heads: int = None,
        head_dim: int = None,
        latent_dim: int = None,
        hidden_dim: int = None,
        max_seq_len: int = 2048,
        norm_eps: float = 1e-6,
        bias: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        # Token embeddings
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                max_seq_len=max_seq_len,
                norm_eps=norm_eps,
                bias=bias
            ) for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.norm = RMSNorm(dim, eps=norm_eps)
        
        # Output projection
        self.output = nn.Linear(dim, vocab_size, bias=bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        start_pos: int = 0
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len = tokens.shape
        
        # Token embeddings
        h = self.tok_embeddings(tokens)
        
        if self.dropout is not None:
            h = self.dropout(h)
        
        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=tokens.device))
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        # Pass through transformer blocks
        for layer in self.layers:
            h = layer(h, mask, start_pos)
        
        # Final layer norm
        h = self.norm(h)
        
        # Compute logits
        if targets is not None:
            # Training mode: compute logits for all positions
            logits = self.output(h)
        else:
            # Inference mode: only compute logits for the last position
            logits = self.output(h[:, -1:, :])
        
        loss = None
        if targets is not None:
            # Compute cross-entropy loss
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """
        Generate tokens using the model
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Get the last max_seq_len tokens
            tokens_cond = tokens if tokens.size(1) <= self.max_seq_len else tokens[:, -self.max_seq_len:]
            
            # Forward pass
            logits, _ = self(tokens_cond)
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to the sequence
            tokens = torch.cat([tokens, next_token], dim=1)
        
        return tokens
    
    def get_num_params(self) -> int:
        """Return the number of parameters in the model"""
        return sum(p.numel() for p in self.parameters())