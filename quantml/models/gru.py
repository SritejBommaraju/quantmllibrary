"""
Gated Recurrent Unit (GRU) implementation.

GRU is a simplified variant of LSTM that uses fewer gates
while maintaining similar performance for sequence modeling.
"""

from typing import Optional, Tuple, List
import math
from quantml.tensor import Tensor
from quantml import ops


class GRUCell:
    """
    A single GRU cell.
    
    Implements the GRU equations:
        r_t = sigmoid(x_t @ W_xr + h_{t-1} @ W_hr + b_r)  # reset gate
        z_t = sigmoid(x_t @ W_xz + h_{t-1} @ W_hz + b_z)  # update gate
        n_t = tanh(x_t @ W_xn + r_t * (h_{t-1} @ W_hn) + b_n)  # candidate
        h_t = (1 - z_t) * n_t + z_t * h_{t-1}  # hidden state
    
    Attributes:
        input_size: Size of input features
        hidden_size: Size of hidden state
    
    Examples:
        >>> cell = GRUCell(10, 20)
        >>> x = Tensor([[1.0] * 10])
        >>> h = cell.forward(x)
    """
    
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        """
        Initialize GRU cell.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden state
            bias: Whether to include bias terms
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        
        # Initialize weights using Xavier/Glorot initialization
        # Combined weights for 3 gates: [r, z, n]
        combined_size = 3 * hidden_size
        limit_ih = math.sqrt(6.0 / (input_size + hidden_size))
        
        # Input-to-hidden weights: (3 * hidden_size, input_size)
        weight_ih_data = [
            [(2.0 * limit_ih * ((i * input_size + j) % 1000) / 1000 - limit_ih)
             for j in range(input_size)]
            for i in range(combined_size)
        ]
        self.weight_ih = Tensor(weight_ih_data, requires_grad=True)
        
        # Hidden-to-hidden weights: (3 * hidden_size, hidden_size)
        limit_hh = math.sqrt(6.0 / (hidden_size + hidden_size))
        weight_hh_data = [
            [(2.0 * limit_hh * ((i * hidden_size + j) % 1000) / 1000 - limit_hh)
             for j in range(hidden_size)]
            for i in range(combined_size)
        ]
        self.weight_hh = Tensor(weight_hh_data, requires_grad=True)
        
        # Biases for all 3 gates
        if bias:
            self.bias_ih = Tensor([[0.0] for _ in range(combined_size)], requires_grad=True)
            self.bias_hh = Tensor([[0.0] for _ in range(combined_size)], requires_grad=True)
        else:
            self.bias_ih = None
            self.bias_hh = None
        
        # Hidden state
        self.hidden = None
    
    def forward(self, x: Tensor, hidden: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through GRU cell.
        
        Args:
            x: Input tensor (batch_size x input_size)
            hidden: Optional previous hidden state h_{t-1}
        
        Returns:
            New hidden state h_t
        """
        # Ensure 2D input
        x_data = x.data if isinstance(x.data[0], list) else [x.data]
        x_2d = Tensor(x_data)
        batch_size = len(x_data)
        
        # Initialize hidden state if not provided
        if hidden is None:
            if self.hidden is None:
                h_prev = Tensor([[0.0] * self.hidden_size for _ in range(batch_size)])
            else:
                h_prev = self.hidden
        else:
            h_prev = hidden
        
        # Compute input projections: x @ W_ih^T
        W_ih_T = self._transpose(self.weight_ih)
        x_proj = ops.matmul(x_2d, W_ih_T)
        
        if self.bias and self.bias_ih is not None:
            bias_ih_T = self._transpose(self.bias_ih)
            x_proj = ops.add(x_proj, bias_ih_T)
        
        # Compute hidden projections: h @ W_hh^T
        W_hh_T = self._transpose(self.weight_hh)
        h_proj = ops.matmul(h_prev, W_hh_T)
        
        if self.bias and self.bias_hh is not None:
            bias_hh_T = self._transpose(self.bias_hh)
            h_proj = ops.add(h_proj, bias_hh_T)
        
        # Split projections into gates
        # Order: reset, update, new (candidate)
        h = self.hidden_size
        x_r, x_z, x_n = self._split_3(x_proj, h)
        h_r, h_z, h_n = self._split_3(h_proj, h)
        
        # Reset gate: r_t = sigmoid(x_r + h_r)
        r_t = ops.sigmoid(ops.add(x_r, h_r))
        
        # Update gate: z_t = sigmoid(x_z + h_z)
        z_t = ops.sigmoid(ops.add(x_z, h_z))
        
        # Candidate hidden: n_t = tanh(x_n + r_t * h_n)
        n_t = ops.tanh(ops.add(x_n, ops.mul(r_t, h_n)))
        
        # New hidden state: h_t = (1 - z_t) * n_t + z_t * h_{t-1}
        one_minus_z = ops.sub(Tensor([[1.0] * h for _ in range(batch_size)]), z_t)
        h_t = ops.add(ops.mul(one_minus_z, n_t), ops.mul(z_t, h_prev))
        
        # Store for next step
        self.hidden = h_t
        
        return h_t
    
    def _transpose(self, t: Tensor) -> Tensor:
        """Transpose a 2D tensor."""
        if not isinstance(t.data[0], list):
            data = [t.data]
        else:
            data = t.data
        
        transposed = [[data[j][i] for j in range(len(data))] 
                     for i in range(len(data[0]))]
        return Tensor(transposed, requires_grad=t.requires_grad)
    
    def _split_3(self, tensor: Tensor, h: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Split tensor into 3 chunks along last dimension."""
        data = tensor.data
        
        if isinstance(data[0], list):
            r_data = [[row[j] for j in range(0, h)] for row in data]
            z_data = [[row[j] for j in range(h, 2*h)] for row in data]
            n_data = [[row[j] for j in range(2*h, 3*h)] for row in data]
        else:
            r_data = [data[j] for j in range(0, h)]
            z_data = [data[j] for j in range(h, 2*h)]
            n_data = [data[j] for j in range(2*h, 3*h)]
        
        return (
            Tensor(r_data, requires_grad=tensor.requires_grad),
            Tensor(z_data, requires_grad=tensor.requires_grad),
            Tensor(n_data, requires_grad=tensor.requires_grad)
        )
    
    def reset_hidden(self) -> None:
        """Reset hidden state."""
        self.hidden = None
    
    def parameters(self) -> List[Tensor]:
        """Get all trainable parameters."""
        params = [self.weight_ih, self.weight_hh]
        if self.bias and self.bias_ih is not None:
            params.extend([self.bias_ih, self.bias_hh])
        return params
    
    def zero_grad(self) -> None:
        """Clear gradients for all parameters."""
        for p in self.parameters():
            p.zero_grad()


class GRU:
    """
    Multi-layer GRU module.
    
    Stacks multiple GRU layers for deeper sequence processing.
    
    Attributes:
        input_size: Size of input features
        hidden_size: Size of hidden state
        num_layers: Number of stacked GRU layers
    
    Examples:
        >>> gru = GRU(10, 20, num_layers=2)
        >>> x = Tensor([[[1.0] * 10] * 5])  # batch x seq x features
        >>> outputs, h_n = gru.forward(x)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = True
    ):
        """
        Initialize multi-layer GRU.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden state
            num_layers: Number of stacked GRU layers
            bias: Whether to include bias terms
            batch_first: If True, input shape is (batch, seq, features)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        
        # Create GRU cells for each layer
        self.cells: List[GRUCell] = []
        for i in range(num_layers):
            cell_input_size = input_size if i == 0 else hidden_size
            self.cells.append(GRUCell(cell_input_size, hidden_size, bias))
    
    def forward(
        self,
        x: Tensor,
        hidden: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through GRU.
        
        Args:
            x: Input tensor (batch x seq x features) if batch_first
            hidden: Optional initial h_0 for all layers
        
        Returns:
            Tuple of (outputs, h_n) where:
            - outputs: All hidden states for each timestep
            - h_n: Final hidden state for each layer
        """
        # Ensure proper shape
        data = x.data
        if not isinstance(data[0], list):
            data = [[data]]
        elif not isinstance(data[0][0], list):
            data = [data]
        
        batch_size = len(data)
        seq_len = len(data[0])
        
        # Initialize hidden states for all layers
        h_layers = [None] * self.num_layers
        
        # Process sequence
        outputs = []
        
        for t in range(seq_len):
            # Get input for this timestep
            x_t_data = [[data[b][t][f] for f in range(len(data[b][t]))] 
                       for b in range(batch_size)]
            x_t = Tensor(x_t_data)
            
            # Process through each layer
            layer_input = x_t
            for layer_idx, cell in enumerate(self.cells):
                h_prev = h_layers[layer_idx]
                h_t = cell.forward(layer_input, h_prev)
                h_layers[layer_idx] = h_t
                layer_input = h_t
            
            # Store output from last layer
            outputs.append(layer_input)
        
        # Stack outputs: (batch, seq, hidden)
        output_data = [
            [[float(outputs[t].data[b][h]) 
              for h in range(self.hidden_size)]
             for t in range(seq_len)]
            for b in range(batch_size)
        ]
        
        # Stack final states
        h_n_data = [[float(h_layers[l].data[b][h]) 
                    for h in range(self.hidden_size)]
                   for l in range(self.num_layers)
                   for b in range(batch_size)]
        
        return Tensor(output_data), Tensor(h_n_data)
    
    def reset_hidden(self) -> None:
        """Reset hidden states for all layers."""
        for cell in self.cells:
            cell.reset_hidden()
    
    def parameters(self) -> List[Tensor]:
        """Get all trainable parameters."""
        params = []
        for cell in self.cells:
            params.extend(cell.parameters())
        return params
    
    def zero_grad(self) -> None:
        """Clear gradients for all parameters."""
        for cell in self.cells:
            cell.zero_grad()
