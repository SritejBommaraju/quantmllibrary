"""
RMSProp optimizer implementation.

RMSProp (Root Mean Square Propagation) is an adaptive learning rate optimizer
that maintains a moving average of squared gradients.
"""

from typing import List, Optional, Dict, Any
from quantml.tensor import Tensor
from quantml import ops

# Try to import NumPy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


class RMSProp:
    """
    RMSProp optimizer.
    
    RMSProp maintains a moving average of squared gradients and divides
    the gradient by the root of this average.
    
    Attributes:
        lr: Learning rate
        alpha: Smoothing constant (decay factor)
        eps: Small value for numerical stability
        weight_decay: Weight decay coefficient
        momentum: Momentum factor (0 = no momentum)
        squared_avg: Moving average of squared gradients
    
    Examples:
        >>> optimizer = RMSProp(lr=0.001, alpha=0.99)
        >>> for param in model.parameters():
        >>>     optimizer.step(param)
    """
    
    def __init__(
        self,
        params: Optional[List[Tensor]] = None,
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0
    ):
        """
        Initialize RMSProp optimizer.
        
        Args:
            params: Optional list of parameters to optimize
            lr: Learning rate
            alpha: Smoothing constant (decay factor for squared gradient average)
            eps: Small value to prevent division by zero
            weight_decay: Weight decay (L2 regularization) coefficient
            momentum: Momentum factor (0.0 to disable)
        """
        self.params = params if params is not None else []
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.squared_avg: Dict[int, Any] = {}  # Moving average of squared gradients
        self.momentum_buffer: Dict[int, Any] = {}  # Momentum buffer
    
    def step(self, param: Optional[Tensor] = None):
        """
        Perform a single optimization step.
        
        Args:
            param: Optional single parameter to update
        """
        if param is not None:
            self._update_param(param)
        else:
            for p in self.params:
                self._update_param(p)
    
    def _update_param(self, param: Tensor):
        """Update a single parameter using RMSProp algorithm."""
        if not param.requires_grad:
            return
        
        if param.grad is None:
            return
        
        param_id = id(param)
        
        if HAS_NUMPY:
            try:
                # Get gradient and parameter as NumPy arrays
                grad = param.grad
                if isinstance(grad, np.ndarray):
                    grad_arr = grad
                else:
                    grad_arr = np.array(grad, dtype=np.float64)
                
                param_arr = param.numpy if param.numpy is not None else np.array(param.data, dtype=np.float64)
                
                # Apply weight decay
                if self.weight_decay > 0:
                    grad_arr = grad_arr + self.weight_decay * param_arr
                
                # Initialize squared average if needed
                if param_id not in self.squared_avg:
                    self.squared_avg[param_id] = np.zeros_like(param_arr, dtype=np.float64)
                
                # Update squared average: avg = alpha * avg + (1 - alpha) * grad^2
                sq_avg = self.squared_avg[param_id]
                sq_avg[:] = self.alpha * sq_avg + (1.0 - self.alpha) * (grad_arr ** 2)
                
                # Compute update: update = grad / (sqrt(avg) + eps)
                update = grad_arr / (np.sqrt(sq_avg) + self.eps)
                
                # Apply momentum if enabled
                if self.momentum > 0:
                    if param_id not in self.momentum_buffer:
                        self.momentum_buffer[param_id] = np.zeros_like(param_arr, dtype=np.float64)
                    
                    buf = self.momentum_buffer[param_id]
                    buf[:] = self.momentum * buf + update
                    update = buf
                
                # Update parameter: param = param - lr * update
                param_update = self.lr * update
                new_param_arr = param_arr - param_update
                param.data = new_param_arr
                
            except (ValueError, TypeError, AttributeError):
                self._update_param_fallback(param)
        else:
            self._update_param_fallback(param)
    
    def _update_param_fallback(self, param: Tensor):
        """Fallback update using Tensor operations."""
        if param.grad is None:
            return
        
        param_id = id(param)
        
        # Initialize squared average if needed
        if param_id not in self.squared_avg:
            if isinstance(param.data[0], list):
                self.squared_avg[param_id] = [[0.0] * len(row) for row in param.data]
            else:
                self.squared_avg[param_id] = [0.0] * len(param.data)
        
        grad = param.grad
        if self.weight_decay > 0:
            grad = ops.add(grad, ops.mul(param, self.weight_decay))
        
        # Update squared average
        grad_sq = ops.mul(grad, grad)
        sq_avg = Tensor(self.squared_avg[param_id])
        new_sq_avg = ops.add(
            ops.mul(sq_avg, self.alpha),
            ops.mul(grad_sq, 1.0 - self.alpha)
        )
        self.squared_avg[param_id] = new_sq_avg.data
        
        # Compute update
        sq_avg_sqrt = ops.pow(ops.add(new_sq_avg, self.eps), 0.5)
        update = ops.div(grad, sq_avg_sqrt)
        
        # Apply momentum
        if self.momentum > 0:
            if param_id not in self.momentum_buffer:
                if isinstance(param.data[0], list):
                    self.momentum_buffer[param_id] = [[0.0] * len(row) for row in param.data]
                else:
                    self.momentum_buffer[param_id] = [0.0] * len(param.data)
            
            buf = Tensor(self.momentum_buffer[param_id])
            new_buf = ops.add(ops.mul(buf, self.momentum), update)
            self.momentum_buffer[param_id] = new_buf.data
            update = new_buf
        
        # Update parameter
        param_update = ops.mul(update, self.lr)
        if param.requires_grad:
            param_detached = param.detach()
            param_detached.sub_(param_update)
            param.data = param_detached.data
        else:
            param.sub_(param_update)
    
    def zero_grad(self, param: Optional[Tensor] = None):
        """Clear gradients."""
        if param is not None:
            param.zero_grad()
        else:
            for p in self.params:
                p.zero_grad()
    
    def add_param_group(self, params: List[Tensor]):
        """Add a parameter group to optimize."""
        self.params.extend(params)

