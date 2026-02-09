"""
Temporal Convolutional Network (TCN) implementation.

TCN uses causal convolutions with dilation for sequence modeling,
making it suitable for time-series prediction in quant trading.
"""

from typing import List, Optional
import math
from quantml.tensor import Tensor
from quantml import ops


class TCNBlock:
    """
    A single TCN block with causal convolution and residual connection.
    
    TCN blocks use dilated convolutions to capture long-range dependencies
    while maintaining causality (no future information leakage).
    
    Attributes:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        dilation: Dilation rate
        stride: Stride (usually 1)
    
    Examples:
        >>> block = TCNBlock(10, 20, kernel_size=3, dilation=1)
        >>> x = Tensor([[1.0] * 10])
        >>> out = block.forward(x)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        stride: int = 1
    ):
        """
        Initialize TCN block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolution kernel
            dilation: Dilation rate for causal convolution
            stride: Stride (typically 1)
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        
        # Initialize weights as 2D (out_channels, in_channels)
        # This is a simplified linear transformation approach.
        # A full TCN implementation would use proper causal dilated convolution.
        limit = math.sqrt(1.0 / in_channels)
        weight_data = [[(2.0 * limit * (i * in_channels + j) /
                          (in_channels * out_channels) - limit)
                        for j in range(in_channels)]
                       for i in range(out_channels)]

        self.weight = Tensor(weight_data, requires_grad=True)

        # Bias: (1, out_channels) for broadcasting with (batch, out_channels)
        bias_data = [[0.0] * out_channels]
        self.bias = Tensor(bias_data, requires_grad=True)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through TCN block.
        
        Implements causal convolution with optional residual connection.
        
        Args:
            x: Input tensor (batch_size x seq_len x in_channels)
        
        Returns:
            Output tensor (batch_size x seq_len x out_channels)
        """
        # For simplicity, we'll implement a basic version
        # A full TCN would implement proper causal dilated convolution
        
        # Ensure 2D input
        x_data = x.data if isinstance(x.data[0], list) else [x.data]
        x_2d = Tensor(x_data)

        # Linear transformation: x @ weight^T + bias
        weight_T = ops.transpose(self.weight)
        out = ops.matmul(x_2d, weight_T)

        # Add bias
        out = ops.add(out, self.bias)

        # Apply activation (ReLU)
        out = ops.relu(out)

        # Residual connection if dimensions match
        if self.in_channels == self.out_channels:
            out = ops.add(out, x_2d)

        return out
    
    def parameters(self) -> list:
        """Get all trainable parameters."""
        return [self.weight, self.bias]
    
    def zero_grad(self):
        """Clear gradients for all parameters."""
        self.weight.zero_grad()
        self.bias.zero_grad()


class TCN:
    """
    Full TCN model with multiple stacked blocks.
    
    A TCN consists of multiple TCNBlock layers stacked together,
    with increasing dilation rates to capture multi-scale patterns.
    
    Attributes:
        blocks: List of TCN blocks
        input_size: Input feature size
        output_size: Output feature size
    
    Examples:
        >>> tcn = TCN(input_size=10, hidden_sizes=[20, 20], output_size=1)
        >>> x = Tensor([[1.0] * 10])
        >>> y = tcn.forward(x)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        kernel_size: int = 3
    ):
        """
        Initialize TCN model.
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output features
            kernel_size: Convolution kernel size
        """
        self.input_size = input_size
        self.output_size = output_size
        
        # Build TCN blocks
        self.blocks = []
        in_channels = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            dilation = 2 ** i  # Exponential dilation
            block = TCNBlock(in_channels, hidden_size, kernel_size, dilation)
            self.blocks.append(block)
            in_channels = hidden_size
        
        # Output layer
        from quantml.models.linear import Linear
        self.output_layer = Linear(in_channels, output_size)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through TCN.
        
        Args:
            x: Input tensor (batch_size x seq_len x input_size)
        
        Returns:
            Output tensor (batch_size x output_size)
        """
        # Pass through TCN blocks
        out = x
        for block in self.blocks:
            out = block.forward(out)

        # Output layer: blocks produce (batch, hidden), pass directly
        output = self.output_layer.forward(out)
        return output
    
    def parameters(self) -> list:
        """Get all trainable parameters."""
        params = []
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend(self.output_layer.parameters())
        return params
    
    def zero_grad(self):
        """Clear gradients for all parameters."""
        for block in self.blocks:
            block.zero_grad()
        self.output_layer.zero_grad()

