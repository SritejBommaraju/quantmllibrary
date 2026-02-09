"""
Dropout regularization.

Implements Dropout layer for preventing overfitting.
"""

from typing import Optional, List
import random
from quantml.tensor import Tensor
from quantml import ops


class Dropout:
    """
    Dropout layer.
    
    Randomly zeroes some elements of the input tensor with probability p
    using samples from a Bernoulli distribution.
    
    During training, outputs are scaled by 1/(1-p).
    During evaluation, does nothing.
    
    Attributes:
        p: Probability of an element being zeroed
        inplace: If True, do operation in-place (not supported yet)
    
    Examples:
        >>> dropout = Dropout(p=0.5)
        >>> x = Tensor([[1.0, 2.0, 3.0]])
        >>> y = dropout(x)
    """
    
    def __init__(self, p: float = 0.5, inplace: bool = False):
        if p < 0 or p > 1:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        self.p = p
        self.inplace = inplace
        self.training = True
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
        
        Returns:
            Tensor with dropout applied
        """
        if not self.training or self.p == 0:
            return x

        # Create a mask tensor and multiply — this preserves the autograd graph
        scale = 1.0 / (1.0 - self.p)
        mask_data = self._generate_mask_data(x.data, scale)
        mask = Tensor(mask_data, requires_grad=False)

        return ops.mul(x, mask)
    
    def _generate_mask_data(self, data, scale):
        """Recursively generate dropout mask."""
        if isinstance(data, list):
            return [self._generate_mask_data(item, scale) for item in data]
        else:
            return scale if random.random() > self.p else 0.0
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def train(self, mode: bool = True) -> 'Dropout':
        """Set training mode."""
        self.training = mode
        return self
    
    def eval(self) -> 'Dropout':
        """Set evaluation mode."""
        return self.train(False)
    
    def parameters(self) -> List[Tensor]:
        """Get trainable parameters (none for dropout)."""
        return []
        
    def zero_grad(self) -> None:
        """Clear gradients (no-op)."""
        pass
