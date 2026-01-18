"""
Linear (fully connected) layer implementation.

This module provides a simple linear layer suitable for quant trading models.
"""

from typing import Optional
import math
from quantml.tensor import Tensor
from quantml import ops


class Linear:
    """
    Linear (fully connected) layer: y = xW^T + b
    
    Attributes:
        in_features: Number of input features
        out_features: Number of output features
        weight: Weight matrix (out_features x in_features)
        bias: Bias vector (out_features,)
    
    Examples:
        >>> layer = Linear(10, 5)
        >>> x = Tensor([[1.0] * 10])
        >>> y = layer.forward(x)  # Shape: (1, 5)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_init: Optional[Tensor] = None,
        bias_init: Optional[Tensor] = None
    ):
        """
        Initialize linear layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            bias: Whether to include bias term
            weight_init: Optional initial weight tensor
            bias_init: Optional initial bias tensor
        """
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        
        # Initialize weights
        if weight_init is not None:
            self.weight = weight_init
        else:
            # Xavier/Glorot initialization
            limit = math.sqrt(6.0 / (in_features + out_features))
            weight_data = [[(2.0 * limit * (i * out_features + j) / (in_features * out_features) - limit)
                           for j in range(in_features)]
                          for i in range(out_features)]
            self.weight = Tensor(weight_data, requires_grad=True)
        
        # Initialize bias
        if bias:
            if bias_init is not None:
                self.bias_param = bias_init
            else:
                bias_data = [[0.0] for _ in range(out_features)]
                self.bias_param = Tensor(bias_data, requires_grad=True)
        else:
            self.bias_param = None
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass: y = xW^T + b
        
        Args:
            x: Input tensor (batch_size x in_features)
        
        Returns:
            Output tensor (batch_size x out_features)
        """
        # x: (batch, in_features)
        # weight: (out_features, in_features)
        # We need: x @ weight.T
        # For simplicity, we'll do: (x @ weight.T) which is matmul(x, weight.T)
        
        # Transpose weight: (in_features, out_features)
        weight_T = ops.transpose(self.weight)
        
        # Matrix multiply: x @ weight_T
        out = ops.matmul(x, weight_T)
        
        # Add bias if present
        if self.bias and self.bias_param is not None:
            # Broadcast bias to match output shape
            out = ops.add(out, self.bias_param)
        
        return out
    
    def parameters(self) -> list:
        """Get all trainable parameters."""
        params = [self.weight]
        if self.bias and self.bias_param is not None:
            params.append(self.bias_param)
        return params
    
    def zero_grad(self):
        """Clear gradients for all parameters."""
        self.weight.zero_grad()
        if self.bias_param is not None:
            self.bias_param.zero_grad()

