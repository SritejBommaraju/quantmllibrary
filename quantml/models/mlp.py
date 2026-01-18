"""
Multi-Layer Perceptron (MLP) implementation.

Provides a flexible MLP builder for creating feedforward neural networks
with configurable layers, activations, and regularization.
"""

from typing import List, Optional, Union, Callable
import math
from quantml.tensor import Tensor
from quantml import ops
from quantml.models.linear import Linear


class MLP:
    """
    Multi-Layer Perceptron (feedforward neural network).
    
    A flexible MLP builder that creates a sequence of linear layers
    with configurable activations and dropout.
    
    Attributes:
        layers: List of Linear layers
        activations: Activation function(s) between layers
        dropout_rate: Dropout probability
    
    Examples:
        >>> # Simple 3-layer MLP
        >>> mlp = MLP([10, 64, 32, 1], activation='relu')
        >>> x = Tensor([[1.0] * 10])
        >>> y = mlp.forward(x)
        
        >>> # MLP with different activations per layer
        >>> mlp = MLP([10, 64, 1], activation=['relu', None])
    """
    
    def __init__(
        self,
        layer_sizes: List[int],
        activation: Union[str, List[Optional[str]], Callable, None] = 'relu',
        dropout: float = 0.0,
        bias: bool = True,
        final_activation: Optional[str] = None
    ):
        """
        Initialize MLP.
        
        Args:
            layer_sizes: List of layer sizes [input_size, hidden1, ..., output_size]
            activation: Activation function(s). Can be:
                - String: 'relu', 'tanh', 'sigmoid', 'leaky_relu', 'gelu', 'swish', None
                - List of strings (one per layer transition)
                - Callable function
            dropout: Dropout probability (0.0 = no dropout)
            bias: Whether to include bias in linear layers
            final_activation: Optional activation for the final layer
        
        Raises:
            ValueError: If layer_sizes has fewer than 2 elements
        """
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least 2 elements (input and output)")
        
        self.layer_sizes = layer_sizes
        self.dropout_rate = dropout
        self.training = True
        
        # Build layers
        self.layers: List[Linear] = []
        for i in range(len(layer_sizes) - 1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i + 1]
            self.layers.append(Linear(in_size, out_size, bias=bias))
        
        # Parse activations
        num_transitions = len(layer_sizes) - 1
        if isinstance(activation, list):
            if len(activation) != num_transitions:
                raise ValueError(f"activation list must have {num_transitions} elements")
            self.activations = activation
        else:
            # Same activation for all hidden layers, final_activation for last
            self.activations = [activation] * (num_transitions - 1) + [final_activation]
        
        # Convert string activations to functions
        self.activation_fns = [self._get_activation_fn(a) for a in self.activations]
    
    def _get_activation_fn(self, name: Optional[str]) -> Optional[Callable]:
        """Get activation function by name."""
        if name is None:
            return None
        if callable(name):
            return name
        
        activations = {
            'relu': ops.relu,
            'tanh': ops.tanh,
            'sigmoid': ops.sigmoid,
            'leaky_relu': ops.leaky_relu,
            'gelu': ops.gelu,
            'swish': ops.swish,
            'softmax': ops.softmax,
        }
        
        if name.lower() not in activations:
            raise ValueError(f"Unknown activation: {name}. Available: {list(activations.keys())}")
        
        return activations[name.lower()]
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through MLP.
        
        Args:
            x: Input tensor (batch_size x input_size)
        
        Returns:
            Output tensor (batch_size x output_size)
        """
        out = x
        
        for i, (layer, activation_fn) in enumerate(zip(self.layers, self.activation_fns)):
            # Linear transformation
            out = layer.forward(out)
            
            # Apply activation
            if activation_fn is not None:
                out = activation_fn(out)
            
            # Apply dropout (except on last layer)
            if self.training and self.dropout_rate > 0 and i < len(self.layers) - 1:
                out = self._apply_dropout(out)
        
        return out
    
    def _apply_dropout(self, x: Tensor) -> Tensor:
        """Apply dropout during training."""
        if not self.training or self.dropout_rate <= 0:
            return x
        
        # Simple dropout: randomly zero out elements
        import random
        
        data = x.data
        if isinstance(data[0], list):
            dropped = [
                [val if random.random() > self.dropout_rate 
                 else 0.0 for val in row]
                for row in data
            ]
            # Scale by 1/(1-p) to maintain expected value
            scale = 1.0 / (1.0 - self.dropout_rate)
            dropped = [[val * scale for val in row] for row in dropped]
        else:
            dropped = [
                val if random.random() > self.dropout_rate else 0.0
                for val in data
            ]
            scale = 1.0 / (1.0 - self.dropout_rate)
            dropped = [val * scale for val in dropped]
        
        return Tensor(dropped, requires_grad=x.requires_grad)
    
    def train(self, mode: bool = True) -> 'MLP':
        """
        Set training mode.
        
        Args:
            mode: If True, enables training mode (dropout active)
        
        Returns:
            self
        """
        self.training = mode
        return self
    
    def eval(self) -> 'MLP':
        """
        Set evaluation mode (disables dropout).
        
        Returns:
            self
        """
        return self.train(False)
    
    def parameters(self) -> List[Tensor]:
        """Get all trainable parameters."""
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    
    def zero_grad(self) -> None:
        """Clear gradients for all parameters."""
        for layer in self.layers:
            layer.zero_grad()
    
    def __repr__(self) -> str:
        """String representation."""
        layers_str = " -> ".join(str(s) for s in self.layer_sizes)
        activations_str = ", ".join(
            str(a) if a is not None else "None" for a in self.activations
        )
        return f"MLP({layers_str}, activations=[{activations_str}], dropout={self.dropout_rate})"


class ResidualMLP(MLP):
    """
    MLP with residual connections.
    
    Adds skip connections between layers where dimensions match,
    improving gradient flow for deeper networks.
    
    Examples:
        >>> mlp = ResidualMLP([10, 64, 64, 64, 1], activation='relu')
        >>> x = Tensor([[1.0] * 10])
        >>> y = mlp.forward(x)
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with residual connections.
        
        Adds residual connections where input and output dimensions match.
        """
        out = x
        
        for i, (layer, activation_fn) in enumerate(zip(self.layers, self.activation_fns)):
            # Store for residual
            residual = out
            
            # Linear transformation
            out = layer.forward(out)
            
            # Apply activation
            if activation_fn is not None:
                out = activation_fn(out)
            
            # Residual connection if dimensions match
            if i > 0 and i < len(self.layers) - 1:
                # Check if dimensions match
                out_shape = out.shape if hasattr(out, 'shape') else len(out.data[0]) if isinstance(out.data[0], list) else len(out.data)
                res_shape = residual.shape if hasattr(residual, 'shape') else len(residual.data[0]) if isinstance(residual.data[0], list) else len(residual.data)
                
                if out_shape == res_shape:
                    out = ops.add(out, residual)
            
            # Apply dropout (except on last layer)
            if self.training and self.dropout_rate > 0 and i < len(self.layers) - 1:
                out = self._apply_dropout(out)
        
        return out


def create_mlp(
    input_size: int,
    output_size: int,
    hidden_sizes: List[int],
    activation: str = 'relu',
    output_activation: Optional[str] = None,
    dropout: float = 0.0
) -> MLP:
    """
    Convenience function to create an MLP.
    
    Args:
        input_size: Input feature dimension
        output_size: Output dimension
        hidden_sizes: List of hidden layer sizes
        activation: Activation for hidden layers
        output_activation: Activation for output layer
        dropout: Dropout probability
    
    Returns:
        Configured MLP instance
    
    Examples:
        >>> mlp = create_mlp(10, 1, [64, 32], activation='relu')
    """
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    return MLP(
        layer_sizes, 
        activation=activation, 
        final_activation=output_activation,
        dropout=dropout
    )
