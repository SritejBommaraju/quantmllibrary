"""
RAdam (Rectified Adam) optimizer implementation.

RAdam rectifies the variance of the adaptive learning rate in Adam.
"""

from typing import List, Optional, Dict, Any
from quantml.tensor import Tensor
from quantml import ops
import math

# Try to import NumPy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


class RAdam:
    """
    RAdam (Rectified Adam) optimizer.
    
    Rectifies the variance of the adaptive learning rate in Adam,
    providing better convergence for quant training.
    
    Attributes:
        lr: Learning rate
        betas: Tuple of (beta1, beta2) for moment estimates
        eps: Small value for numerical stability
        weight_decay: Weight decay coefficient
        m: First moment estimates
        v: Second moment estimates
    
    Examples:
        >>> optimizer = RAdam(lr=0.001, betas=(0.9, 0.999))
        >>> for param in model.parameters():
        >>>     optimizer.step(param)
    """
    
    def __init__(
        self,
        params: Optional[List[Tensor]] = None,
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ):
        """
        Initialize RAdam optimizer.
        
        Args:
            params: Optional list of parameters to optimize
            lr: Learning rate
            betas: Tuple of (beta1, beta2) for exponential decay rates
            eps: Small value to prevent division by zero
            weight_decay: Weight decay coefficient
        """
        self.params = params if params is not None else []
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m: Dict[int, Any] = {}  # First moment
        self.v: Dict[int, Any] = {}  # Second moment
        self.step_count = 0
    
    def step(self, param: Optional[Tensor] = None):
        """Perform a single optimization step."""
        if param is not None:
            self._update_param(param)
        else:
            for p in self.params:
                self._update_param(p)
        self.step_count += 1
    
    def _update_param(self, param: Tensor):
        """Update a single parameter using RAdam algorithm."""
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
                
                if param_id not in self.m:
                    self.m[param_id] = np.zeros_like(param_arr, dtype=np.float64)
                    self.v[param_id] = np.zeros_like(param_arr, dtype=np.float64)
                
                if self.weight_decay > 0:
                    grad_arr = grad_arr + self.weight_decay * param_arr
                
                # Update moments
                m = self.m[param_id]
                v = self.v[param_id]
                m[:] = self.beta1 * m + (1.0 - self.beta1) * grad_arr
                v[:] = self.beta2 * v + (1.0 - self.beta2) * (grad_arr ** 2)
                
                # RAdam variance rectification
                beta2_t = self.beta2 ** self.step_count
                rho_inf = 2.0 / (1.0 - self.beta2) - 1.0
                rho_t = rho_inf - 2.0 * self.step_count * beta2_t / (1.0 - beta2_t)
                
                if rho_t > 4.0:
                    # Rectified update
                    r_t = math.sqrt((rho_t - 4.0) * (rho_t - 2.0) * rho_inf / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t))
                    m_hat = m / (1.0 - self.beta1 ** self.step_count)
                    v_hat = v / (1.0 - beta2_t)
                    update = r_t * m_hat / (np.sqrt(v_hat) + self.eps)
                else:
                    # Simple momentum update
                    m_hat = m / (1.0 - self.beta1 ** self.step_count)
                    update = m_hat
                
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
        
        if param_id not in self.m:
            if isinstance(param.data[0], list):
                self.m[param_id] = Tensor([[0.0] * len(row) for row in param.data])
                self.v[param_id] = Tensor([[0.0] * len(row) for row in param.data])
            else:
                self.m[param_id] = Tensor([0.0] * len(param.data))
                self.v[param_id] = Tensor([0.0] * len(param.data))
        
        grad = Tensor(param.grad)
        if self.weight_decay > 0:
            grad = ops.add(grad, ops.mul(param, self.weight_decay))
        
        m_prev = self.m[param_id]
        m_new = ops.add(ops.mul(m_prev, self.beta1), ops.mul(grad, 1.0 - self.beta1))
        self.m[param_id] = m_new
        
        v_prev = self.v[param_id]
        grad_sq = ops.mul(grad, grad)
        v_new = ops.add(ops.mul(v_prev, self.beta2), ops.mul(grad_sq, 1.0 - self.beta2))
        self.v[param_id] = v_new
        
        # Simplified RAdam (use regular Adam update in fallback)
        bias_correction1 = 1.0 - (self.beta1 ** self.step_count)
        bias_correction2 = 1.0 - (self.beta2 ** self.step_count)
        m_hat = ops.div(m_new, bias_correction1)
        v_hat = ops.div(v_new, bias_correction2)
        
        v_hat_sqrt = ops.pow(ops.add(v_hat, self.eps), 0.5)
        update = ops.div(m_hat, v_hat_sqrt)
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

