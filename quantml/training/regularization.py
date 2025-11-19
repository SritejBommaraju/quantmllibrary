"""
Regularization utilities for quant models.

Provides dropout and other regularization techniques.
"""

from typing import Optional
from quantml.tensor import Tensor
from quantml import ops
import random

# Try to import NumPy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


class Dropout:
    """
    Dropout layer for regularization.
    
    Randomly sets a fraction of inputs to zero during training.
    """
    
    def __init__(self, p: float = 0.5):
        """
        Initialize dropout layer.
        
        Args:
            p: Probability of dropping out (0.0 to 1.0)
        """
        self.p = p
        self.training = True
        self.mask = None
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through dropout.
        
        Args:
            x: Input tensor
        
        Returns:
            Output tensor with dropout applied
        """
        if not self.training or self.p == 0.0:
            return x
        
        # Create dropout mask
        if HAS_NUMPY:
            try:
                import numpy as np
                x_arr = x.numpy if x.numpy is not None else np.array(x.data, dtype=np.float64)
                mask = (np.random.random(x_arr.shape) > self.p).astype(np.float64)
                mask = mask / (1.0 - self.p)  # Scale to maintain expected value
                self.mask = mask
                out_arr = x_arr * mask
                return Tensor(out_arr.tolist(), requires_grad=x.requires_grad)
            except:
                pass
        
        # Fallback to list-based
        if isinstance(x.data[0], list):
            mask = [[1.0 if random.random() > self.p else 0.0 for _ in row] for row in x.data]
            scale = 1.0 / (1.0 - self.p)
            mask = [[m * scale for m in row] for row in mask]
            self.mask = mask
            out_data = [[x.data[i][j] * mask[i][j] for j in range(len(x.data[i]))] 
                       for i in range(len(x.data))]
        else:
            mask = [1.0 if random.random() > self.p else 0.0 for _ in x.data]
            scale = 1.0 / (1.0 - self.p)
            mask = [m * scale for m in mask]
            self.mask = mask
            out_data = [x.data[i] * mask[i] for i in range(len(x.data))]
        
        return Tensor(out_data, requires_grad=x.requires_grad)
    
    def eval(self):
        """Set to evaluation mode (no dropout)."""
        self.training = False
    
    def train(self):
        """Set to training mode (with dropout)."""
        self.training = True

