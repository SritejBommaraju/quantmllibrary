"""
Learning rate finder for optimal LR discovery.

Implements learning rate range test to find optimal learning rate.
"""

from typing import List, Optional, Callable, Any
from quantml.tensor import Tensor
import math


class LRFinder:
    """
    Learning rate finder using range test.
    
    Finds optimal learning rate by testing a range of values
    and identifying where loss decreases most rapidly.
    """
    
    def __init__(
        self,
        model: Any,
        optimizer: Any,
        loss_fn: Callable,
        min_lr: float = 1e-7,
        max_lr: float = 10.0,
        num_iterations: int = 100
    ):
        """
        Initialize LR finder.
        
        Args:
            model: Model to test
            optimizer: Optimizer
            loss_fn: Loss function
            min_lr: Minimum learning rate to test
            max_lr: Maximum learning rate to test
            num_iterations: Number of iterations to run
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_iterations = num_iterations
        self.lrs = []
        self.losses = []
    
    def range_test(self, x: Tensor, y: Tensor) -> tuple:
        """
        Run learning rate range test.
        
        Args:
            x: Input features
            y: Targets
        
        Returns:
            Tuple of (lrs, losses) lists
        """
        self.lrs = []
        self.losses = []
        
        # Calculate LR multiplier
        lr_mult = (self.max_lr / self.min_lr) ** (1.0 / self.num_iterations)
        
        original_lr = self.optimizer.lr
        
        for i in range(self.num_iterations):
            # Set learning rate
            current_lr = self.min_lr * (lr_mult ** i)
            self.optimizer.lr = current_lr
            self.lrs.append(current_lr)
            
            # Forward pass
            pred = self.model.forward(x)
            loss = self.loss_fn(pred, y)
            
            # Backward pass
            if loss.requires_grad:
                loss.backward()
                self.optimizer.step()
                self.model.zero_grad()
            
            # Record loss
            loss_val = self._get_value(loss)
            self.losses.append(loss_val)
        
        # Restore original LR
        self.optimizer.lr = original_lr
        
        return self.lrs, self.losses
    
    def suggest_lr(self) -> float:
        """
        Suggest optimal learning rate based on range test.
        
        Returns:
            Suggested learning rate
        """
        if not self.losses:
            return self.min_lr
        
        # Find steepest descent point
        # Look for point with maximum negative gradient
        best_idx = 0
        best_grad = float('-inf')
        
        for i in range(1, len(self.losses) - 1):
            # Calculate gradient (negative of loss change)
            grad = -(self.losses[i+1] - self.losses[i-1]) / (self.lrs[i+1] - self.lrs[i-1])
            if grad > best_grad and self.losses[i] < self.losses[0] * 0.5:
                best_grad = grad
                best_idx = i
        
        if best_idx > 0:
            return self.lrs[best_idx]
        return self.min_lr * 10.0  # Default suggestion
    
    def _get_value(self, tensor: Tensor) -> float:
        """Extract scalar value from tensor."""
        data = tensor.data
        if isinstance(data, list):
            if isinstance(data[0], list):
                return float(data[0][0])
            return float(data[0])
        return float(data)

