"""
Attention mechanisms.

Implementations of Self-Attention and Multi-Head Attention.
"""

from typing import Optional, List, Tuple
import math
from quantml.tensor import Tensor
from quantml import ops
from quantml.models.linear import Linear


class SelfAttention:
    """
    Scaled Dot-Product Self-Attention.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

    Attributes:
        embed_dim: Dimension of embeddings
        num_heads: Number of attention heads (default: 1)
        dropout: Dropout probability

    Note:
        Supports both 2D inputs (seq_len x embed_dim) and 3D batched inputs
        (batch x seq_len x embed_dim) with full autograd support.

    Examples:
        >>> attn = SelfAttention(64)
        >>> x = Tensor([[1.0] * 64] * 10)  # seq x dim (autograd supported)
        >>> out = attn.forward(x)
    """
    
    def __init__(self, embed_dim: int, dropout: float = 0.0):
        self.embed_dim = embed_dim
        self.dropout = dropout
        
        # Projections
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)
        
        # Scaling factor
        self.scale = 1.0 / math.sqrt(embed_dim)
        
    def forward(
        self, 
        x: Tensor, 
        mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch x seq_len x embed_dim)
            mask: Optional mask (batch x seq_len x seq_len)
        
        Returns:
            Output tensor (batch x seq_len x embed_dim)
        """
        # Linear projections
        # Each is (batch x seq_len x embed_dim)
        q = self.q_proj.forward(x)
        k = self.k_proj.forward(x)
        v = self.v_proj.forward(x)
        
        # Calculate attention scores: Q @ K^T
        # Transpose k for matmul
        # Q: (B, S, D), K: (B, S, D) -> K^T: (B, D, S)
        # We need an explicit transpose op for 3D tensors which we don't have yet
        # So we'll implement a simplified version assuming batch=1 or flattened
        
        # NOTE: Full multi-head attention with batching requires reshaping and 3D matmul
        # which depends on robust tensor ops. This is a simplified implementation.
        
        # Fallback to pure python loop for batch support
        data = x.data
        if isinstance(data[0][0], list): # 3D
            batch_size = len(data)
            out_batches = []

            for b in range(batch_size):
                q_b = ops.select(q, b, dim=0)
                k_b = ops.select(k, b, dim=0)
                v_b = ops.select(v, b, dim=0)
                mask_b = ops.select(mask, b, dim=0) if mask is not None else None

                out_b = self._attention_2d(q_b, k_b, v_b, mask_b)
                out_batches.append(out_b)

            attn_out = ops.stack(out_batches, dim=0)
        else:
            # 2D case
            attn_out = self._attention_2d(q, k, v, mask)
            
        # Output projection
        return self.out_proj.forward(attn_out)
        
    def _attention_2d(self, An: Tensor, K: Tensor, V: Tensor, mask: Optional[Tensor]) -> Tensor:
        """Compute attention for single sample (seq_len x dim)."""
        # K^T
        # K is (S, D), we need (D, S) manually transposed
        K_T = ops.transpose(K)
        
        # Scores: (S, D) @ (D, S) -> (S, S)
        scores = ops.matmul(An, K_T)
        
        # Scale
        scaled_scores = ops.mul(scores, self.scale)
        
        # Mask
        if mask is not None:
            # Assume mask is additive (0 for keep, -inf for mask)
            scaled_scores = ops.add(scaled_scores, mask)
            
        # Softmax over last dim (rows)
        attn_weights = ops.softmax(scaled_scores, axis=-1)
        
        # Output: (S, S) @ (S, D) -> (S, D)
        output = ops.matmul(attn_weights, V)
        
        return output

    def parameters(self) -> List[Tensor]:
        """Get parameters."""
        return (self.q_proj.parameters() +
                self.k_proj.parameters() + 
                self.v_proj.parameters() + 
                self.out_proj.parameters())
                
    def zero_grad(self) -> None:
        """Zero gradients."""
        for p in self.parameters():
            p.zero_grad()


class MultiHeadAttention:
    """
    Multi-Head Attention.

    Splits embedding into multiple heads, applies attention independently,
    and concatenates results.

    Attributes:
        embed_dim: Model dimension
        num_heads: Number of heads

    Note:
        Currently operates as a single-head attention (placeholder until
        reshape/permute ops are added). Supports both 2D and 3D batched
        inputs with full autograd support.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}")
            
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # In typical implementation, we project to (num_heads * head_dim)
        # which is same as embed_dim. So we can use one big Linear layer
        # and reshape/split logic.
        
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Simple implementation: Loop over heads (inefficient but clear).
        True parallel implementation requires robust reshape/transpose ops.
        """
        batch_size = len(x.data) if isinstance(x.data[0], list) and isinstance(x.data[0][0], list) else 1
        
        # 1. Projections
        q = self.q_proj.forward(x)
        k = self.k_proj.forward(x)
        v = self.v_proj.forward(x)
        
        # 2. Split heads and generic attention logic is complex without proper tensor reshaping
        # For this library's current state, we will implement a simplified single-head equivalent
        # that mathematically matches but doesn't actually split distinct subspaces without reshaping.
        
        # For now, we'll delegate to a simpler attention mechanism that treats it as one big head
        # This is a PLACEHOLDER until complex tensor manipulation functions (view, permute) are added.
        
        # We'll re-use the SelfAttention logic which effectively does 1 head
        # To support multi-head properly, we need ops.reshape / ops.transpose(permute)
        
        return self._fallback_single_head(q, k, v, mask)

    def _fallback_single_head(self, q, k, v, mask):
        data = q.data
        if isinstance(data[0][0], list): # 3D
            batch_size = len(data)
            out_batches = []
            for b in range(batch_size):
                q_b = ops.select(q, b, dim=0)
                k_b = ops.select(k, b, dim=0)
                v_b = ops.select(v, b, dim=0)
                mask_b = ops.select(mask, b, dim=0) if mask is not None else None

                K_T = ops.transpose(k_b)
                scores = ops.matmul(q_b, K_T)
                scaled_scores = ops.mul(scores, self.scale)
                if mask_b is not None:
                    scaled_scores = ops.add(scaled_scores, mask_b)
                attn_weights = ops.softmax(scaled_scores, axis=-1)
                out_b = ops.matmul(attn_weights, v_b)

                out_batches.append(out_b)
            attn_out = ops.stack(out_batches, dim=0)
        else:
            K_T = ops.transpose(k)
            scores = ops.matmul(q, K_T)
            scaled_scores = ops.mul(scores, self.scale)
            if mask is not None:
                scaled_scores = ops.add(scaled_scores, mask)
            attn_weights = ops.softmax(scaled_scores, axis=-1)
            attn_out = ops.matmul(attn_weights, v)

        return self.out_proj.forward(attn_out)

    def parameters(self) -> List[Tensor]:
        return (self.q_proj.parameters() + 
                self.k_proj.parameters() + 
                self.v_proj.parameters() + 
                self.out_proj.parameters())
    
    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()
