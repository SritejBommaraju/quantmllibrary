"""
AdaGrad optimizer implementation.

AdaGrad (Adaptive Gradient) adapts learning rates by accumulating squared gradients.
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


class AdaGrad:
    """
    AdaGrad optimizer.
    
    AdaGrad adapts learning rates by accumulating squared gradients.
    Learning rates decrease for parameters with large gradients.
    
    Attributes:
        lr: Learning rate
        eps: Small value for numerical stability
        weight_decay: Weight decay coefficient
        accum: Accumulated squared gradients
    
    Examples:
        >>> optimizer = AdaGrad(lr=0.01)
        >>> for param in model.parameters():
        >>>     optimizer.step(param)
    """
    
    def __init__(
        self,
        params: Optional[List[Tensor]] = None,
        lr: float = 0.01,
        eps: float = 1e-10,
        weight_decay: float = 0.0
    ):
        """
        Initialize AdaGrad optimizer.
        
        Args:
            params: Optional list of parameters to optimize
            lr: Learning rate
            eps: Small value to prevent division by zero
            weight_decay: Weight decay (L2 regularization) coefficient
        """
        self.params = params if params is not None else []
        self.lr = lr
        self.eps = eps
        self.weight_decay = weight_decay
        self.accum: Dict[int, Any] = {}  # Accumulated squared gradients
    
    def step(self, param: Optional[Tensor] = None):
        """Perform a single optimization step."""
        if param is not None:
            self._update_param(param)
        else:
            for p in self.params:
                self._update_param(p)
    
    def _update_param(self, param: Tensor):
        """Update a single parameter using AdaGrad algorithm."""
        if not param.requires_grad:
            return
        
        if param.grad is None:
            return
        
        param_id = id(param)
        
        if HAS_NUMPY:
            try:
                grad = param.grad
                if isinstance(grad, np.ndarray):
                    grad_arr = grad
                else:
                    grad_arr = np.array(grad, dtype=np.float64)
                
                param_arr = param.numpy if param.numpy is not None else np.array(param.data, dtype=np.float64)
                
                if self.weight_decay > 0:
                    grad_arr = grad_arr + self.weight_decay * param_arr
                
                # Initialize accumulator if needed
                if param_id not in self.accum:
                    self.accum[param_id] = np.zeros_like(param_arr, dtype=np.float64)
                
                # Accumulate squared gradients
                accum = self.accum[param_id]
                accum[:] = accum + grad_arr ** 2
                
                # Update parameter: param = param - lr * grad / (sqrt(accum) + eps)
                update = grad_arr / (np.sqrt(accum) + self.eps)
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
        
        if param_id not in self.accum:
            if isinstance(param.data[0], list):
                self.accum[param_id] = [[0.0] * len(row) for row in param.data]
            else:
                self.accum[param_id] = [0.0] * len(param.data)
        
        grad = param.grad
        if self.weight_decay > 0:
            grad = ops.add(grad, ops.mul(param, self.weight_decay))
        
        # Accumulate squared gradients
        grad_sq = ops.mul(grad, grad)
        accum = Tensor(self.accum[param_id])
        new_accum = ops.add(accum, grad_sq)
        self.accum[param_id] = new_accum.data
        
        # Update parameter
        accum_sqrt = ops.pow(ops.add(new_accum, self.eps), 0.5)
        update = ops.div(grad, accum_sqrt)
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

