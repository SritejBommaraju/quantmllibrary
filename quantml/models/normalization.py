"""
Normalization layers.

Implementations of Batch Normalization and Layer Normalization.
"""

from typing import Optional, List, Union
import math
from quantml.tensor import Tensor
from quantml import ops


class BatchNorm1d:
    """
    Batch Normalization for 2D or 3D inputs.
    
    y = (x - mean) / sqrt(var + eps) * gamma + beta
    
    Attributes:
        num_features: Number of features in input
        eps: Small value for stability
        momentum: Momentum for running stats (default: 0.1)
        affine: Whether to learn gamma and beta
        track_running_stats: Whether to track running mean/var
    
    Examples:
        >>> bn = BatchNorm1d(10)
        >>> x = Tensor([[1.0] * 10])
        >>> y = bn.forward(x)
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True
    ):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.training = True
        
        # Learnable parameters
        if affine:
            self.gamma = Tensor([1.0] * num_features, requires_grad=True)
            self.beta = Tensor([0.0] * num_features, requires_grad=True)
        else:
            self.gamma = None
            self.beta = None
        
        # Running stats (not trainable)
        if track_running_stats:
            self.running_mean = Tensor([0.0] * num_features, requires_grad=False)
            self.running_var = Tensor([1.0] * num_features, requires_grad=False)
            self.num_batches_tracked = 0
        else:
            self.running_mean = None
            self.running_var = None
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size x num_features) or (batch x seq x features)
        
        Returns:
            Normalized tensor
        """
        # Handle 3D input (batch, seq, features) by flattening
        original_shape = None
        data = x.data
        if isinstance(data[0], list) and isinstance(data[0][0], list):
            # 3D input
            batch_size = len(data)
            seq_len = len(data[0])
            features = len(data[0][0])
            if features != self.num_features:
                raise ValueError(f"Expected {self.num_features} features, got {features}")
            
            # Flatten to (batch*seq, features)
            flat_data = [row for batch in data for row in batch]
            x_flat = Tensor(flat_data, requires_grad=x.requires_grad)
            original_shape = (batch_size, seq_len, features)
            
            # Recurse with flattened input
            out_flat = self._forward_2d(x_flat)
            
            # Reshape back directly using list comprehension
            flat_out_data = out_flat.data
            out_data = [
                [flat_out_data[b * seq_len + t] for t in range(seq_len)]
                for b in range(batch_size)
            ]
            
            # Note: We lose gradient history through reshaping in pure Python
            # unless we implement a proper Reshape op. For now, we return
            # a new tensor which breaks the graph for 3D inputs.
            # TODO: Implement Reshape/View op for full autograd support
            
            return Tensor(
                out_data, 
                requires_grad=out_flat.requires_grad,
                _prev={out_flat} if out_flat.requires_grad else set(),
                _op='reshape'
            )
        else:
            # 2D input
            return self._forward_2d(x)

    def _forward_2d(self, x: Tensor) -> Tensor:
        """Forward pass for 2D input."""
        # Calculate mean and var
        if self.training and self.track_running_stats:
            # Current batch stats
            batch_mean = ops.mean(x, axis=0)
            batch_var = ops.var(x, axis=0, unbiased=False)
            
            # Update running stats (no grad)
            n = x.data.shape[0] if hasattr(x.data, 'shape') else len(x.data)
            
            # Manual update to avoid graph creation
            if self.running_mean is not None:
                m = self.momentum
                # running_mean = (1 - m) * running_mean + m * batch_mean
                # Done manually on data
                rm_data = self.running_mean.data
                bm_data = batch_mean.data[0] if isinstance(batch_mean.data[0], list) else batch_mean.data
                
                new_rm = [
                    (1 - m) * float(rm_data[i]) + m * float(bm_data[i])
                    for i in range(self.num_features)
                ]
                self.running_mean._data_list = new_rm
                self.running_mean._np_array = None
            
            if self.running_var is not None:
                m = self.momentum
                # running_var = (1 - m) * running_var + m * batch_var * (n / (n-1))
                rv_data = self.running_var.data
                bv_data = batch_var.data[0] if isinstance(batch_var.data[0], list) else batch_var.data
                
                unbiased_factor = n / (n - 1) if n > 1 else 1.0
                new_rv = [
                    (1 - m) * float(rv_data[i]) + m * float(bv_data[i]) * unbiased_factor
                    for i in range(self.num_features)
                ]
                self.running_var._data_list = new_rv
                self.running_var._np_array = None
                
            self.num_batches_tracked += 1
            
            # Use batch stats for normalization
            mean = batch_mean
            var = batch_var
        else:
            # Use running stats
            if self.running_mean is not None:
                mean = self.running_mean
                var = self.running_var
            else:
                # No running stats, compute batch stats
                mean = ops.mean(x, axis=0)
                var = ops.var(x, axis=0, unbiased=False)
        
        # Normalize: (x - mean) / sqrt(var + eps)
        # Add eps
        var_plus_eps = ops.add(var, self.eps)
        std = ops.pow(var_plus_eps, 0.5)
        
        x_centered = ops.sub(x, mean)
        x_norm = ops.div(x_centered, std)
        
        # Scale and shift
        if self.affine and self.gamma is not None and self.beta is not None:
            # Expand gamma and beta for broadcasting
            # gamma is (features,), x_norm is (batch, features)
            # We construct a (features,) tensor that works with broadcasting
            out = ops.mul(x_norm, self.gamma)
            out = ops.add(out, self.beta)
            return out
        else:
            return x_norm
    
    def train(self, mode: bool = True) -> 'BatchNorm1d':
        """Set training mode."""
        self.training = mode
        return self
    
    def eval(self) -> 'BatchNorm1d':
        """Set evaluation mode."""
        return self.train(False)
    
    def parameters(self) -> List[Tensor]:
        """Get trainable parameters."""
        params = []
        if self.affine:
            if self.gamma is not None: params.append(self.gamma)
            if self.beta is not None: params.append(self.beta)
        return params
    
    def zero_grad(self) -> None:
        """Clear gradients."""
        for p in self.parameters():
            p.zero_grad()


class LayerNorm:
    """
    Layer Normalization.
    
    y = (x - mean) / sqrt(var + eps) * gamma + beta
    
    Applied over the last dimension.
    
    Attributes:
        normalized_shape: Input shape (int or list)
        eps: Small value for stability
        elementwise_affine: Whether to learn gamma and beta
    """
    
    def __init__(
        self,
        normalized_shape: Union[int, List[int]],
        eps: float = 1e-5,
        elementwise_affine: bool = True
    ):
        if isinstance(normalized_shape, int):
            self.normalized_shape = [normalized_shape]
        else:
            self.normalized_shape = list(normalized_shape)
            
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        # Total number of elements in normalized shape
        self.num_elements = 1
        for dim in self.normalized_shape:
            self.num_elements *= dim
            
        if elementwise_affine:
            self.gamma = Tensor([1.0] * self.num_elements, requires_grad=True)
            self.beta = Tensor([0.0] * self.num_elements, requires_grad=True)
        else:
            self.gamma = None
            self.beta = None
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        # Mean and var over last dim(s)
        # For simplicity, we assume normalized_shape corresponds to the last dimension(s)
        # and we currently only support 1D normalized_shape (last dim)
        
        axis = -1
        mean = ops.mean(x, axis=axis)
        var = ops.var(x, axis=axis, unbiased=False)
        
        # Add eps
        var_plus_eps = ops.add(var, self.eps)
        std = ops.pow(var_plus_eps, 0.5)
        
        # Normalize
        # We need to reshape mean/std to broadcast correctly if they were reduced
        # ops.sub and ops.div should handle broadcasting if implemented correctly
        x_centered = ops.sub(x, mean)
        x_norm = ops.div(x_centered, std)
        
        # Scale and shift
        if self.elementwise_affine:
            out = ops.mul(x_norm, self.gamma)
            out = ops.add(out, self.beta)
            return out
        else:
            return x_norm
            
    def parameters(self) -> List[Tensor]:
        """Get trainable parameters."""
        params = []
        if self.elementwise_affine:
            if self.gamma is not None: params.append(self.gamma)
            if self.beta is not None: params.append(self.beta)
        return params
            
    def zero_grad(self) -> None:
        """Clear gradients."""
        for p in self.parameters():
            p.zero_grad()
