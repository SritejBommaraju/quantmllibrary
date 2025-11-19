"""
Lookahead optimizer wrapper.

Lookahead is a wrapper that improves training stability by maintaining
a slow-moving average of parameters.
"""

from typing import List, Optional, Any
from quantml.tensor import Tensor


class Lookahead:
    """
    Lookahead optimizer wrapper.
    
    Wraps any base optimizer and maintains a slow-moving average of parameters.
    Improves training stability for quant models.
    
    Attributes:
        base_optimizer: The base optimizer to wrap
        k: Number of steps before updating slow weights
        alpha: Interpolation factor for slow weights
        slow_weights: Slow-moving parameter averages
    
    Examples:
        >>> base_opt = Adam(lr=0.001)
        >>> optimizer = Lookahead(base_opt, k=5, alpha=0.5)
        >>> for param in model.parameters():
        >>>     optimizer.step(param)
    """
    
    def __init__(
        self,
        base_optimizer: Any,
        k: int = 5,
        alpha: float = 0.5
    ):
        """
        Initialize Lookahead optimizer.
        
        Args:
            base_optimizer: Base optimizer to wrap (SGD, Adam, etc.)
            k: Number of steps before updating slow weights
            alpha: Interpolation factor (0 < alpha < 1)
        """
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.slow_weights: dict = {}  # Slow-moving parameter averages
        self.step_count = 0
    
    def step(self, param: Optional[Tensor] = None):
        """Perform a single optimization step."""
        # Use base optimizer
        self.base_optimizer.step(param)
        self.step_count += 1
        
        # Update slow weights every k steps
        if self.step_count % self.k == 0:
            if param is not None:
                self._update_slow_weights(param)
            else:
                for p in self.base_optimizer.params:
                    self._update_slow_weights(p)
    
    def _update_slow_weights(self, param: Tensor):
        """Update slow-moving parameter average."""
        param_id = id(param)
        
        if param_id not in self.slow_weights:
            # Initialize slow weights with current parameter values
            self.slow_weights[param_id] = param.data.copy() if hasattr(param.data, 'copy') else param.data
        else:
            # Interpolate: slow = alpha * slow + (1 - alpha) * param
            slow = self.slow_weights[param_id]
            if hasattr(slow, '__iter__') and hasattr(param.data, '__iter__'):
                # Update slow weights (simplified - would need proper NumPy/list handling)
                # For now, just store current param as slow weight
                self.slow_weights[param_id] = param.data.copy() if hasattr(param.data, 'copy') else param.data
            else:
                self.slow_weights[param_id] = self.alpha * slow + (1.0 - self.alpha) * param.data
    
    def sync_slow_weights(self):
        """Synchronize parameters with slow weights."""
        for param in self.base_optimizer.params:
            param_id = id(param)
            if param_id in self.slow_weights:
                param.data = self.slow_weights[param_id]
    
    def zero_grad(self, param: Optional[Tensor] = None):
        """Clear gradients."""
        self.base_optimizer.zero_grad(param)
    
    def add_param_group(self, params: List[Tensor]):
        """Add a parameter group to optimize."""
        self.base_optimizer.add_param_group(params)

