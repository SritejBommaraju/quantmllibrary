"""
Adam optimizer implementation.

Adam (Adaptive Moment Estimation) is an adaptive learning rate optimizer
that combines the benefits of AdaGrad and RMSProp.
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


class Adam:
    """
    Adam optimizer.
    
    Adam maintains per-parameter adaptive learning rates based on estimates
    of first and second moments of gradients.
    
    Attributes:
        lr: Learning rate
        betas: Tuple of (beta1, beta2) for moment estimates
        eps: Small value for numerical stability
        m: First moment estimates (momentum)
        v: Second moment estimates (variance)
        step_count: Number of steps taken
    
    Examples:
        >>> optimizer = Adam(lr=0.001, betas=(0.9, 0.999))
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
        Initialize Adam optimizer.
        
        Args:
            params: Optional list of parameters to optimize
            lr: Learning rate
            betas: Tuple of (beta1, beta2) for exponential decay rates
            eps: Small value to prevent division by zero
            weight_decay: Weight decay (L2 regularization) coefficient
        """
        self.params = params if params is not None else []
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Moment estimates (stored as NumPy arrays for efficiency)
        self.m: Dict[int, Any] = {}  # First moment
        self.v: Dict[int, Any] = {}  # Second moment
        self.step_count = 0
    
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
        
        self.step_count += 1
    
    def _update_param(self, param: Tensor):
        """Update a single parameter using Adam algorithm with direct NumPy operations."""
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
                
                # Initialize moments if needed
                if param_id not in self.m:
                    self.m[param_id] = np.zeros_like(param_arr, dtype=np.float64)
                    self.v[param_id] = np.zeros_like(param_arr, dtype=np.float64)
                
                # Apply weight decay
                if self.weight_decay > 0:
                    grad_arr = grad_arr + self.weight_decay * param_arr
                
                # Update biased first moment: m = beta1 * m + (1 - beta1) * grad
                m = self.m[param_id]
                m[:] = self.beta1 * m + (1.0 - self.beta1) * grad_arr
                
                # Update biased second moment: v = beta2 * v + (1 - beta2) * grad^2
                v = self.v[param_id]
                v[:] = self.beta2 * v + (1.0 - self.beta2) * (grad_arr ** 2)
                
                # Bias correction
                bias_correction1 = 1.0 - (self.beta1 ** self.step_count)
                bias_correction2 = 1.0 - (self.beta2 ** self.step_count)
                
                # Compute bias-corrected estimates
                m_hat = m / bias_correction1
                v_hat = v / bias_correction2
                
                # Update parameter: param = param - lr * m_hat / (sqrt(v_hat) + eps)
                v_hat_sqrt = np.sqrt(v_hat) + self.eps
                update = m_hat / v_hat_sqrt
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
        
        # Initialize moments if needed
        if param_id not in self.m:
            if isinstance(param.data[0], list):
                self.m[param_id] = Tensor([[0.0] * len(row) for row in param.data])
                self.v[param_id] = Tensor([[0.0] * len(row) for row in param.data])
            else:
                self.m[param_id] = Tensor([0.0] * len(param.data))
                self.v[param_id] = Tensor([0.0] * len(param.data))
        
        # Get gradient
        grad = Tensor(param.grad)
        
        # Apply weight decay
        if self.weight_decay > 0:
            grad = ops.add(grad, ops.mul(param, self.weight_decay))
        
        # Update biased first moment estimate: m = beta1 * m + (1 - beta1) * grad
        m_prev = self.m[param_id]
        m_new = ops.add(
            ops.mul(m_prev, self.beta1),
            ops.mul(grad, 1.0 - self.beta1)
        )
        self.m[param_id] = m_new
        
        # Update biased second moment estimate: v = beta2 * v + (1 - beta2) * grad^2
        v_prev = self.v[param_id]
        grad_sq = ops.mul(grad, grad)
        v_new = ops.add(
            ops.mul(v_prev, self.beta2),
            ops.mul(grad_sq, 1.0 - self.beta2)
        )
        self.v[param_id] = v_new
        
        # Bias correction
        bias_correction1 = 1.0 - (self.beta1 ** self.step_count)
        bias_correction2 = 1.0 - (self.beta2 ** self.step_count)
        
        # Compute bias-corrected estimates
        m_hat = ops.div(m_new, bias_correction1)
        v_hat = ops.div(v_new, bias_correction2)
        
        # Update parameter in-place: param = param - lr * m_hat / (sqrt(v_hat) + eps)
        v_hat_sqrt = ops.pow(ops.add(v_hat, self.eps), 0.5)
        update = ops.div(m_hat, v_hat_sqrt)
        param_update = ops.mul(update, self.lr)
        
        # Detach and update in-place
        if param.requires_grad:
            param_detached = param.detach()
            param_detached.sub_(param_update)
            param.data = param_detached.data
        else:
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
    
    def state_dict(self) -> dict:
        """Get optimizer state dictionary."""
        # Convert NumPy arrays to lists for serialization
        m_data = {}
        v_data = {}
        for k, v in self.m.items():
            if HAS_NUMPY and isinstance(v, np.ndarray):
                m_data[k] = v.tolist()
            elif isinstance(v, Tensor):
                m_data[k] = v.data
            else:
                m_data[k] = v
        
        for k, v in self.v.items():
            if HAS_NUMPY and isinstance(v, np.ndarray):
                v_data[k] = v.tolist()
            elif isinstance(v, Tensor):
                v_data[k] = v.data
            else:
                v_data[k] = v
        
        return {
            'step_count': self.step_count,
            'm': m_data,
            'v': v_data
        }
    
    def load_state_dict(self, state_dict: dict):
        """Load optimizer state from dictionary."""
        self.step_count = state_dict.get('step_count', 0)
        # Reconstruct moment arrays from data
        for k, m_data in state_dict.get('m', {}).items():
            if HAS_NUMPY:
                self.m[int(k)] = np.array(m_data, dtype=np.float64)
            else:
                self.m[int(k)] = Tensor(m_data)
        for k, v_data in state_dict.get('v', {}).items():
            if HAS_NUMPY:
                self.v[int(k)] = np.array(v_data, dtype=np.float64)
            else:
                self.v[int(k)] = Tensor(v_data)

