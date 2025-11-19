"""
QuantOptimizer - Custom optimizer for quant trading.

Adaptive learning rate based on market volatility and regime-aware parameter updates.
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


class QuantOptimizer:
    """
    QuantOptimizer - Custom optimizer for quant trading.
    
    Features:
    - Adaptive learning rate based on market volatility
    - Regime-aware parameter updates
    - Per-feature learning rate scaling
    
    Attributes:
        lr: Base learning rate
        volatility_window: Window size for volatility calculation
        regime_threshold: Threshold for regime detection
        feature_lrs: Per-feature learning rate multipliers
    
    Examples:
        >>> optimizer = QuantOptimizer(lr=0.001, volatility_window=20)
        >>> for param in model.parameters():
        >>>     optimizer.step(param, market_volatility=0.02)
    """
    
    def __init__(
        self,
        params: Optional[List[Tensor]] = None,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        volatility_window: int = 20,
        regime_threshold: float = 0.5
    ):
        """
        Initialize QuantOptimizer.
        
        Args:
            params: Optional list of parameters to optimize
            lr: Base learning rate
            beta1: First moment decay rate
            beta2: Second moment decay rate
            eps: Small value for numerical stability
            weight_decay: Weight decay coefficient
            volatility_window: Window size for volatility calculation
            regime_threshold: Threshold for regime detection
        """
        self.params = params if params is not None else []
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.volatility_window = volatility_window
        self.regime_threshold = regime_threshold
        
        self.m: Dict[int, Any] = {}  # First moment
        self.v: Dict[int, Any] = {}  # Second moment
        self.step_count = 0
        self.volatility_history: List[float] = []
        self.feature_lrs: Dict[int, Any] = {}  # Per-parameter learning rate multipliers
    
    def step(self, param: Optional[Tensor] = None, market_volatility: Optional[float] = None):
        """
        Perform a single optimization step.
        
        Args:
            param: Optional single parameter to update
            market_volatility: Current market volatility (for adaptive LR)
        """
        if param is not None:
            self._update_param(param, market_volatility)
        else:
            for p in self.params:
                self._update_param(p, market_volatility)
        self.step_count += 1
        
        # Update volatility history
        if market_volatility is not None:
            self.volatility_history.append(market_volatility)
            if len(self.volatility_history) > self.volatility_window:
                self.volatility_history.pop(0)
    
    def _update_param(self, param: Tensor, market_volatility: Optional[float] = None):
        """Update a single parameter with adaptive learning rate."""
        if not param.requires_grad:
            return
        
        if param.grad is None:
            return
        
        param_id = id(param)
        
        # Calculate adaptive learning rate based on volatility
        adaptive_lr = self.lr
        if market_volatility is not None and len(self.volatility_history) > 1:
            # Adjust LR based on volatility regime
            avg_vol = sum(self.volatility_history) / len(self.volatility_history)
            if market_volatility > avg_vol * (1.0 + self.regime_threshold):
                # High volatility regime - reduce LR
                adaptive_lr = self.lr * 0.5
            elif market_volatility < avg_vol * (1.0 - self.regime_threshold):
                # Low volatility regime - increase LR
                adaptive_lr = self.lr * 1.5
        
        # Apply per-parameter learning rate multiplier
        if param_id in self.feature_lrs:
            adaptive_lr = adaptive_lr * self.feature_lrs[param_id]
        
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
                
                # Update moments (Adam-like)
                m = self.m[param_id]
                v = self.v[param_id]
                m[:] = self.beta1 * m + (1.0 - self.beta1) * grad_arr
                v[:] = self.beta2 * v + (1.0 - self.beta2) * (grad_arr ** 2)
                
                # Bias correction
                bias_correction1 = 1.0 - (self.beta1 ** self.step_count)
                bias_correction2 = 1.0 - (self.beta2 ** self.step_count)
                m_hat = m / bias_correction1
                v_hat = v / bias_correction2
                
                # Update parameter with adaptive LR
                v_hat_sqrt = np.sqrt(v_hat) + self.eps
                update = m_hat / v_hat_sqrt
                param_update = adaptive_lr * update
                new_param_arr = param_arr - param_update
                param.data = new_param_arr
                
            except (ValueError, TypeError, AttributeError):
                self._update_param_fallback(param, adaptive_lr)
        else:
            self._update_param_fallback(param, adaptive_lr)
    
    def _update_param_fallback(self, param: Tensor, adaptive_lr: float):
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
        
        bias_correction1 = 1.0 - (self.beta1 ** self.step_count)
        bias_correction2 = 1.0 - (self.beta2 ** self.step_count)
        m_hat = ops.div(m_new, bias_correction1)
        v_hat = ops.div(v_new, bias_correction2)
        
        v_hat_sqrt = ops.pow(ops.add(v_hat, self.eps), 0.5)
        update = ops.div(m_hat, v_hat_sqrt)
        param_update = ops.mul(update, adaptive_lr)
        
        if param.requires_grad:
            param_detached = param.detach()
            param_detached.sub_(param_update)
            param.data = param_detached.data
        else:
            param.sub_(param_update)
    
    def set_feature_lr(self, param: Tensor, multiplier: float):
        """Set learning rate multiplier for a specific parameter."""
        param_id = id(param)
        self.feature_lrs[param_id] = multiplier
    
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

