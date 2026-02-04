"""
AdaFactor optimizer implementation.

AdaFactor is a memory-efficient variant of Adam that uses factorized second moment estimates.
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


class AdaFactor:
    """
    AdaFactor optimizer - memory-efficient Adam variant.
    
    Uses factorized second moment estimates to reduce memory usage.
    Good for quant models with many parameters.
    
    Attributes:
        lr: Learning rate
        betas: Tuple of (beta1, beta2) for moment estimates
        eps: Small value for numerical stability
        weight_decay: Weight decay coefficient
        m: First moment estimates
        v: Factorized second moment estimates
    
    Examples:
        >>> optimizer = AdaFactor(lr=0.001, betas=(0.9, 0.999))
        >>> for param in model.parameters():
        >>>     optimizer.step(param)
    """
    
    def __init__(
        self,
        params: Optional[List[Tensor]] = None,
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-30,
        weight_decay: float = 0.0,
        factor_decay: float = 0.8
    ):
        """
        Initialize AdaFactor optimizer.
        
        Args:
            params: Optional list of parameters to optimize
            lr: Learning rate
            betas: Tuple of (beta1, beta2) for exponential decay rates
            eps: Small value to prevent division by zero
            weight_decay: Weight decay coefficient
            factor_decay: Decay factor for second moment factorization
        """
        self.params = params if params is not None else []
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.factor_decay = factor_decay
        self.m: Dict[int, Any] = {}  # First moment
        self.v_row: Dict[int, Any] = {}  # Row factors for second moment
        self.v_col: Dict[int, Any] = {}  # Column factors for second moment
        self.step_count = 0
    
    def step(self, param: Optional[Tensor] = None):
        """Perform a single optimization step."""
        self.step_count += 1
        if param is not None:
            self._update_param(param)
        else:
            for p in self.params:
                self._update_param(p)
    
    def _update_param(self, param: Tensor):
        """Update a single parameter using AdaFactor algorithm."""
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
                
                # Initialize moments if needed
                if param_id not in self.m:
                    self.m[param_id] = np.zeros_like(param_arr, dtype=np.float64)
                    # Factorize second moment for 2D arrays
                    if grad_arr.ndim == 2:
                        self.v_row[param_id] = np.zeros(grad_arr.shape[0], dtype=np.float64)
                        self.v_col[param_id] = np.zeros(grad_arr.shape[1], dtype=np.float64)
                    else:
                        self.v_row[param_id] = np.zeros_like(grad_arr, dtype=np.float64)
                        self.v_col[param_id] = None
                
                # Update first moment
                m = self.m[param_id]
                m[:] = self.beta1 * m + (1.0 - self.beta1) * grad_arr
                
                # Update factorized second moment
                if grad_arr.ndim == 2 and self.v_col[param_id] is not None:
                    # 2D case: factorize
                    v_row = self.v_row[param_id]
                    v_col = self.v_col[param_id]
                    grad_sq = grad_arr ** 2
                    v_row[:] = self.beta2 * v_row + (1.0 - self.beta2) * np.mean(grad_sq, axis=1)
                    v_col[:] = self.beta2 * v_col + (1.0 - self.beta2) * np.mean(grad_sq, axis=0)
                    v_hat = np.outer(v_row, v_col) / np.mean(v_row)
                else:
                    # 1D case: regular second moment
                    v_row = self.v_row[param_id]
                    v_row[:] = self.beta2 * v_row + (1.0 - self.beta2) * (grad_arr ** 2)
                    v_hat = v_row
                
                # Bias correction
                bias_correction1 = 1.0 - (self.beta1 ** self.step_count)
                m_hat = m / bias_correction1
                
                # Update parameter
                v_hat_sqrt = np.sqrt(v_hat) + self.eps
                update = m_hat / v_hat_sqrt
                param_update = self.lr * update
                new_param_arr = param_arr - param_update
                param.data = new_param_arr
                
            except (ValueError, TypeError, AttributeError):
                # Fallback to simplified Adam-like update
                self._update_param_fallback(param)
        else:
            self._update_param_fallback(param)
    
    def _update_param_fallback(self, param: Tensor):
        """Fallback update using Tensor operations (simplified)."""
        # Simplified fallback - use regular Adam-like update
        if param.grad is None:
            return
        
        param_id = id(param)
        
        if param_id not in self.m:
            if isinstance(param.data[0], list):
                self.m[param_id] = Tensor([[0.0] * len(row) for row in param.data])
                self.v_row[param_id] = Tensor([[0.0] * len(row) for row in param.data])
            else:
                self.m[param_id] = Tensor([0.0] * len(param.data))
                self.v_row[param_id] = Tensor([0.0] * len(param.data))
        
        grad = Tensor(param.grad)
        if self.weight_decay > 0:
            grad = ops.add(grad, ops.mul(param, self.weight_decay))
        
        m_prev = self.m[param_id]
        m_new = ops.add(ops.mul(m_prev, self.beta1), ops.mul(grad, 1.0 - self.beta1))
        self.m[param_id] = m_new
        
        v_prev = self.v_row[param_id]
        grad_sq = ops.mul(grad, grad)
        v_new = ops.add(ops.mul(v_prev, self.beta2), ops.mul(grad_sq, 1.0 - self.beta2))
        self.v_row[param_id] = v_new
        
        bias_correction1 = 1.0 - (self.beta1 ** self.step_count)
        m_hat = ops.div(m_new, bias_correction1)
        v_hat = v_new
        
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

