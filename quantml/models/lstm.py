"""
Long Short-Term Memory (LSTM) implementation.

LSTM networks are a type of recurrent neural network that can learn
long-term dependencies through gating mechanisms.
"""

from typing import Optional, Tuple, List
import math
from quantml.tensor import Tensor
from quantml import ops


class LSTMCell:
    """
    A single LSTM cell.
    
    Implements the LSTM equations:
        f_t = sigmoid(x_t @ W_xf + h_{t-1} @ W_hf + b_f)  # forget gate
        i_t = sigmoid(x_t @ W_xi + h_{t-1} @ W_hi + b_i)  # input gate
        g_t = tanh(x_t @ W_xg + h_{t-1} @ W_hg + b_g)     # cell candidate
        o_t = sigmoid(x_t @ W_xo + h_{t-1} @ W_ho + b_o)  # output gate
        c_t = f_t * c_{t-1} + i_t * g_t                    # cell state
        h_t = o_t * tanh(c_t)                              # hidden state
    
    Attributes:
        input_size: Size of input features
        hidden_size: Size of hidden state
    
    Examples:
        >>> cell = LSTMCell(10, 20)
        >>> x = Tensor([[1.0] * 10])
        >>> h, c = cell.forward(x)
    """
    
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        """
        Initialize LSTM cell.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden state
            bias: Whether to include bias terms
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        
        # Initialize weights using Xavier/Glorot initialization
        # Combined weights for all 4 gates: [i, f, g, o]
        # Input-to-hidden weights: (4 * hidden_size, input_size)
        combined_size = 4 * hidden_size
        limit_ih = math.sqrt(6.0 / (input_size + hidden_size))
        
        weight_ih_data = [
            [(2.0 * limit_ih * ((i * input_size + j) % 1000) / 1000 - limit_ih)
             for j in range(input_size)]
            for i in range(combined_size)
        ]
        self.weight_ih = Tensor(weight_ih_data, requires_grad=True)
        
        # Hidden-to-hidden weights: (4 * hidden_size, hidden_size)
        limit_hh = math.sqrt(6.0 / (hidden_size + hidden_size))
        weight_hh_data = [
            [(2.0 * limit_hh * ((i * hidden_size + j) % 1000) / 1000 - limit_hh)
             for j in range(hidden_size)]
            for i in range(combined_size)
        ]
        self.weight_hh = Tensor(weight_hh_data, requires_grad=True)
        
        # Biases for all 4 gates
        if bias:
            # Initialize forget gate bias to 1.0 for better gradient flow
            bias_data = [[0.0] for _ in range(combined_size)]
            # Forget gate bias starts at index hidden_size
            for i in range(hidden_size, 2 * hidden_size):
                bias_data[i][0] = 1.0
            self.bias_ih = Tensor(bias_data, requires_grad=True)
            self.bias_hh = Tensor([[0.0] for _ in range(combined_size)], requires_grad=True)
        else:
            self.bias_ih = None
            self.bias_hh = None
        
        # Hidden and cell state
        self.hidden = None
        self.cell = None
    
    def forward(
        self, 
        x: Tensor, 
        hidden: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through LSTM cell.
        
        Args:
            x: Input tensor (batch_size x input_size)
            hidden: Optional tuple of (h_{t-1}, c_{t-1})
        
        Returns:
            Tuple of (h_t, c_t) - new hidden and cell states
        """
        # Ensure 2D input
        x_data = x.data if isinstance(x.data[0], list) else [x.data]
        x_2d = Tensor(x_data)
        batch_size = len(x_data)
        
        # Initialize hidden state if not provided
        if hidden is None:
            if self.hidden is None or self.cell is None:
                h_prev = Tensor([[0.0] * self.hidden_size for _ in range(batch_size)])
                c_prev = Tensor([[0.0] * self.hidden_size for _ in range(batch_size)])
            else:
                h_prev = self.hidden
                c_prev = self.cell
        else:
            h_prev, c_prev = hidden
        
        # Compute gates: x @ W_ih^T + h @ W_hh^T + b
        # W_ih is (4*hidden, input), W_hh is (4*hidden, hidden)
        W_ih_T = self._transpose(self.weight_ih)
        W_hh_T = self._transpose(self.weight_hh)
        
        gates = ops.matmul(x_2d, W_ih_T)
        gates = ops.add(gates, ops.matmul(h_prev, W_hh_T))
        
        if self.bias and self.bias_ih is not None and self.bias_hh is not None:
            # Reshape biases for broadcasting
            bias = ops.add(self.bias_ih, self.bias_hh)
            gates = ops.add(gates, self._transpose(bias))
        
        # Split gates: each is (batch_size x hidden_size)
        # Order: input, forget, cell candidate, output
        i_gate, f_gate, g_gate, o_gate = self._split_gates(gates)
        
        # Apply activations
        i_t = ops.sigmoid(i_gate)
        f_t = ops.sigmoid(f_gate)
        g_t = ops.tanh(g_gate)
        o_t = ops.sigmoid(o_gate)
        
        # Cell state update: c_t = f_t * c_{t-1} + i_t * g_t
        c_t = ops.add(ops.mul(f_t, c_prev), ops.mul(i_t, g_t))
        
        # Hidden state: h_t = o_t * tanh(c_t)
        h_t = ops.mul(o_t, ops.tanh(c_t))
        
        # Store for next step
        self.hidden = h_t
        self.cell = c_t
        
        return h_t, c_t
    
    def _transpose(self, t: Tensor) -> Tensor:
        """Transpose a 2D tensor."""
        if not isinstance(t.data[0], list):
            data = [t.data]
        else:
            data = t.data
        
        transposed = [[data[j][i] for j in range(len(data))] 
                     for i in range(len(data[0]))]
        return Tensor(transposed, requires_grad=t.requires_grad)
    
    def _split_gates(self, gates: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Split combined gates into individual gate tensors."""
        h = self.hidden_size
        data = gates.data
        
        if isinstance(data[0], list):
            i_data = [[row[j] for j in range(0, h)] for row in data]
            f_data = [[row[j] for j in range(h, 2*h)] for row in data]
            g_data = [[row[j] for j in range(2*h, 3*h)] for row in data]
            o_data = [[row[j] for j in range(3*h, 4*h)] for row in data]
        else:
            i_data = [data[j] for j in range(0, h)]
            f_data = [data[j] for j in range(h, 2*h)]
            g_data = [data[j] for j in range(2*h, 3*h)]
            o_data = [data[j] for j in range(3*h, 4*h)]
        
        return (
            Tensor(i_data, requires_grad=gates.requires_grad),
            Tensor(f_data, requires_grad=gates.requires_grad),
            Tensor(g_data, requires_grad=gates.requires_grad),
            Tensor(o_data, requires_grad=gates.requires_grad)
        )
    
    def reset_hidden(self) -> None:
        """Reset hidden and cell states."""
        self.hidden = None
        self.cell = None
    
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


class LSTM:
    """
    Multi-layer LSTM module.
    
    Stacks multiple LSTM layers for deeper sequence processing.
    
    Attributes:
        input_size: Size of input features
        hidden_size: Size of hidden state
        num_layers: Number of stacked LSTM layers
    
    Examples:
        >>> lstm = LSTM(10, 20, num_layers=2)
        >>> x = Tensor([[[1.0] * 10] * 5])  # batch x seq x features
        >>> outputs, (h_n, c_n) = lstm.forward(x)
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
        Initialize multi-layer LSTM.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden state
            num_layers: Number of stacked LSTM layers
            bias: Whether to include bias terms
            batch_first: If True, input shape is (batch, seq, features)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        
        # Create LSTM cells for each layer
        self.cells: List[LSTMCell] = []
        for i in range(num_layers):
            cell_input_size = input_size if i == 0 else hidden_size
            self.cells.append(LSTMCell(cell_input_size, hidden_size, bias))
    
    def forward(
        self,
        x: Tensor,
        hidden: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass through LSTM.
        
        Args:
            x: Input tensor (batch x seq x features) if batch_first
            hidden: Optional initial (h_0, c_0) for all layers
        
        Returns:
            Tuple of (outputs, (h_n, c_n)) where:
            - outputs: All hidden states for each timestep
            - h_n: Final hidden state for each layer
            - c_n: Final cell state for each layer
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
        if hidden is None:
            h_layers = [None] * self.num_layers
            c_layers = [None] * self.num_layers
        else:
            # Split initial states by layer
            h_0, c_0 = hidden
            h_layers = [None] * self.num_layers  # TODO: proper splitting
            c_layers = [None] * self.num_layers
        
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
                c_prev = c_layers[layer_idx]
                hidden_tuple = (h_prev, c_prev) if h_prev is not None else None
                
                h_t, c_t = cell.forward(layer_input, hidden_tuple)
                
                h_layers[layer_idx] = h_t
                c_layers[layer_idx] = c_t
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
        c_n_data = [[float(c_layers[l].data[b][h]) 
                    for h in range(self.hidden_size)]
                   for l in range(self.num_layers)
                   for b in range(batch_size)]
        
        return (
            Tensor(output_data),
            (Tensor(h_n_data), Tensor(c_n_data))
        )
    
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
