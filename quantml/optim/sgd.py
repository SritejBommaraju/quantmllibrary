"""
Stochastic Gradient Descent (SGD) optimizer.

This module provides the SGD optimizer with optional momentum and weight decay.
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


class SGD:
    """
    Stochastic Gradient Descent optimizer.
    
    Updates parameters using: param = param - lr * (grad + weight_decay * param)
    With momentum: v = momentum * v + grad, param = param - lr * v
    
    Attributes:
        lr: Learning rate
        momentum: Momentum factor (0 = no momentum)
        weight_decay: Weight decay (L2 regularization) factor
        velocity: Momentum velocity for each parameter
    
    Examples:
        >>> optimizer = SGD(lr=0.01, momentum=0.9)
        >>> for param in model.parameters():
        >>>     optimizer.step(param)
    """
    
    def __init__(
        self,
        params: Optional[List[Tensor]] = None,
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0
    ):
        """
        Initialize SGD optimizer.
        
        Args:
            params: Optional list of parameters to optimize
            lr: Learning rate
            momentum: Momentum factor (0.0 to disable)
            weight_decay: Weight decay coefficient
        """
        self.params = params if params is not None else []
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity: Dict[int, Any] = {}  # Store velocity as NumPy arrays
    
    def step(self, param: Optional[Tensor] = None):
        """
        Perform a single optimization step.
        
        If param is provided, updates that parameter.
        Otherwise, updates all parameters in self.params.
        
        Args:
            param: Optional single parameter to update
        """
        if param is not None:
            self._update_param(param)
        else:
            for p in self.params:
                self._update_param(p)
    
    def _update_param(self, param: Tensor):
        """Update a single parameter using direct NumPy operations."""
        if not param.requires_grad:
            return
        
        if param.grad is None:
            return
        
        param_id = id(param)
        
        # Get gradient as NumPy array if possible
        if HAS_NUMPY:
            try:
                # Get gradient as NumPy array
                grad = param.grad
                if isinstance(grad, np.ndarray):
                    grad_arr = grad
                else:
                    grad_arr = np.array(grad, dtype=np.float64)
                
                # Get parameter as NumPy array
                param_arr = param.numpy if param.numpy is not None else np.array(param.data, dtype=np.float64)
                
                # Apply weight decay
                if self.weight_decay > 0:
                    grad_arr = grad_arr + self.weight_decay * param_arr
                
                # Update velocity if momentum is used
                if self.momentum > 0:
                    if param_id not in self.velocity:
                        # Initialize velocity to zero
                        self.velocity[param_id] = np.zeros_like(param_arr, dtype=np.float64)
                    
                    # v = momentum * v + grad
                    vel = self.velocity[param_id]
                    vel[:] = self.momentum * vel + grad_arr
                    update = vel
                else:
                    update = grad_arr
                
                # Compute parameter update: param = param - lr * update
                param_update = self.lr * update
                new_param_arr = param_arr - param_update
                
                # Update parameter data directly
                param.data = new_param_arr
                
            except (ValueError, TypeError, AttributeError):
                # Fallback to Tensor operations
                self._update_param_fallback(param)
        else:
            # Fallback to Tensor operations
            self._update_param_fallback(param)
    
    def _update_param_fallback(self, param: Tensor):
        """Fallback update using Tensor operations."""
        if param.grad is None:
            return
        
        param_id = id(param)
        if param_id not in self.velocity:
            if isinstance(param.data[0], list):
                self.velocity[param_id] = [[0.0] * len(row) for row in param.data]
            else:
                self.velocity[param_id] = [0.0] * len(param.data)
        
        grad = param.grad
        if self.weight_decay > 0:
            grad = ops.add(grad, ops.mul(param, self.weight_decay))
        
        if self.momentum > 0:
            vel = self.velocity[param_id]
            vel_tensor = Tensor(vel)
            new_vel = ops.add(ops.mul(vel_tensor, self.momentum), grad)
            self.velocity[param_id] = new_vel.data
            update = new_vel
        else:
            update = grad
        
        if param.requires_grad:
            param_detached = param.detach()
            param_update = ops.mul(update, self.lr)
            param_detached.sub_(param_update)
            param.data = param_detached.data
        else:
            param_update = ops.mul(update, self.lr)
            param.sub_(param_update)
    
    def zero_grad(self, param: Optional[Tensor] = None):
        """
        Clear gradients.
        
        Args:
            param: Optional single parameter, otherwise clears all
        """
        if param is not None:
            param.zero_grad()
        else:
            for p in self.params:
                p.zero_grad()
    
    def add_param_group(self, params: List[Tensor]):
        """Add a parameter group to optimize."""
        self.params.extend(params)

