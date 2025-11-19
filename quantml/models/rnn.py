"""
Simple RNN (Recurrent Neural Network) implementation.

This module provides a basic RNN cell suitable for time-series prediction
in quantitative trading.
"""

from typing import Optional
import math
from quantml.tensor import Tensor
from quantml import ops
from quantml.models.linear import Linear


class SimpleRNN:
    """
    Simple RNN cell for sequence processing.
    
    The RNN maintains a hidden state and processes sequences one step at a time:
    h_t = tanh(x_t @ W_xh + h_{t-1} @ W_hh + b)
    
    Attributes:
        input_size: Size of input features
        hidden_size: Size of hidden state
        weight_ih: Input-to-hidden weights
        weight_hh: Hidden-to-hidden weights
        bias: Bias term
    
    Examples:
        >>> rnn = SimpleRNN(10, 20)
        >>> x = Tensor([[1.0] * 10])
        >>> h = rnn.forward(x)  # Initial hidden state
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True
    ):
        """
        Initialize RNN cell.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden state
            bias: Whether to include bias term
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        
        # Initialize weights
        # Input-to-hidden: (hidden_size, input_size)
        limit_ih = math.sqrt(1.0 / input_size)
        weight_ih_data = [[(2.0 * limit_ih * (i * input_size + j) / (input_size * hidden_size) - limit_ih)
                          for j in range(input_size)]
                         for i in range(hidden_size)]
        self.weight_ih = Tensor(weight_ih_data, requires_grad=True)
        
        # Hidden-to-hidden: (hidden_size, hidden_size)
        limit_hh = math.sqrt(1.0 / hidden_size)
        weight_hh_data = [[(2.0 * limit_hh * (i * hidden_size + j) / (hidden_size * hidden_size) - limit_hh)
                           for j in range(hidden_size)]
                          for i in range(hidden_size)]
        self.weight_hh = Tensor(weight_hh_data, requires_grad=True)
        
        # Bias
        if bias:
            bias_data = [[0.0] for _ in range(hidden_size)]
            self.bias_param = Tensor(bias_data, requires_grad=True)
        else:
            self.bias_param = None
        
        # Hidden state (initialized to zeros)
        self.hidden = None
    
    def forward(self, x: Tensor, hidden: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through RNN cell.
        
        Args:
            x: Input tensor (batch_size x input_size)
            hidden: Optional previous hidden state (batch_size x hidden_size)
        
        Returns:
            New hidden state (batch_size x hidden_size)
        """
        # Initialize hidden state if not provided
        if hidden is None:
            if self.hidden is None:
                # Create zero hidden state
                batch_size = len(x.data) if isinstance(x.data[0], list) else 1
                hidden_data = [[0.0] * self.hidden_size for _ in range(batch_size)]
                hidden = Tensor(hidden_data)
            else:
                hidden = self.hidden
        
        # Ensure x is 2D
        x_data = x.data if isinstance(x.data[0], list) else [x.data]
        x_2d = Tensor(x_data)
        
        # Input-to-hidden: x @ W_ih^T
        weight_ih_T = self._transpose(self.weight_ih)
        ih = ops.matmul(x_2d, weight_ih_T)
        
        # Hidden-to-hidden: h @ W_hh^T
        weight_hh_T = self._transpose(self.weight_hh)
        hh = ops.matmul(hidden, weight_hh_T)
        
        # Combine
        combined = ops.add(ih, hh)
        
        # Add bias
        if self.bias and self.bias_param is not None:
            combined = ops.add(combined, self.bias_param)
        
        # Apply tanh activation
        new_hidden = ops.tanh(combined)
        
        # Store hidden state
        self.hidden = new_hidden
        
        return new_hidden
    
    def _transpose(self, t: Tensor) -> Tensor:
        """Transpose a 2D tensor."""
        if not isinstance(t.data[0], list):
            data = [t.data]
        else:
            data = t.data
        
        transposed = [[data[j][i] for j in range(len(data))] 
                     for i in range(len(data[0]))]
        return Tensor(transposed, requires_grad=t.requires_grad)
    
    def reset_hidden(self):
        """Reset hidden state to zeros."""
        self.hidden = None
    
    def parameters(self) -> list:
        """Get all trainable parameters."""
        params = [self.weight_ih, self.weight_hh]
        if self.bias and self.bias_param is not None:
            params.append(self.bias_param)
        return params
    
    def zero_grad(self):
        """Clear gradients for all parameters."""
        self.weight_ih.zero_grad()
        self.weight_hh.zero_grad()
        if self.bias_param is not None:
            self.bias_param.zero_grad()

