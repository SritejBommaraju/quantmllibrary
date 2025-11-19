"""
Online learning support for incremental model updates.

This module provides utilities for online learning scenarios where models
are updated incrementally as new data arrives, typical in HFT and quant trading.
"""

from typing import List, Optional, Callable, Any
from quantml.tensor import Tensor
from quantml.streaming import StreamingTensor


class OnlineOptimizer:
    """
    Base class for online optimizers that support incremental updates.
    
    Online optimizers update model parameters one sample at a time,
    which is essential for per-tick training in quantitative trading.
    """
    
    def __init__(self, learning_rate: float = 0.01):
        """
        Initialize online optimizer.
        
        Args:
            learning_rate: Learning rate for updates
        """
        self.learning_rate = learning_rate
        self.step_count = 0
    
    def step(self, grad: Tensor, param: Tensor):
        """
        Perform a single parameter update step.
        
        Args:
            grad: Gradient for the parameter
            param: Parameter tensor to update
        
        Note:
            This is a base implementation. Subclasses should override.
        """
        # Base implementation: simple gradient descent
        # In practice, this should update param.data in-place
        # For now, we'll just track the step
        self.step_count += 1
        # TODO: Implement actual parameter update
        # This requires in-place operations which we avoid in the base design
        # Subclasses can implement specific update rules
    
    def zero_grad(self):
        """Reset optimizer state."""
        self.step_count = 0


def incremental_update(
    model_params: List[Tensor],
    gradients: List[Tensor],
    learning_rate: float = 0.01
) -> List[Tensor]:
    """
    Perform incremental parameter update for online learning.
    
    This function updates model parameters using gradients computed from
    a single sample or small batch, suitable for per-tick training.
    
    Args:
        model_params: List of parameter tensors
        gradients: List of corresponding gradients
        learning_rate: Learning rate for the update
    
    Returns:
        Updated parameter tensors (new instances)
    
    Examples:
        >>> weights = [Tensor([[1.0, 2.0]], requires_grad=True)]
        >>> grads = [Tensor([[0.1, 0.2]])]
        >>> updated = incremental_update(weights, grads, lr=0.01)
    """
    if len(model_params) != len(gradients):
        raise ValueError("Number of parameters and gradients must match")
    
    updated_params = []
    for param, grad in zip(model_params, gradients):
        # Simple gradient descent: param_new = param - lr * grad
        from quantml import ops
        update = ops.mul(grad, learning_rate)
        new_param = ops.sub(param, update)
        updated_params.append(new_param)
    
    return updated_params


def per_tick_training_step(
    model: Any,
    x: Tensor,
    y: Tensor,
    loss_fn: Callable,
    optimizer: Optional[OnlineOptimizer] = None,
    learning_rate: float = 0.01
) -> float:
    """
    Perform a single training step on one tick of data.
    
    This is the core function for online learning in quant trading,
    where models are updated as each new tick arrives.
    
    Args:
        model: Model with forward() method and parameters
        x: Input tensor (single sample)
        y: Target tensor (single sample)
        loss_fn: Loss function (pred, target) -> loss
        optimizer: Optional online optimizer
        learning_rate: Learning rate if no optimizer provided
    
    Returns:
        Loss value for this step
    
    Examples:
        >>> from quantml.models import Linear
        >>> model = Linear(10, 1)
        >>> x = Tensor([[1.0] * 10])
        >>> y = Tensor([[0.5]])
        >>> loss = per_tick_training_step(model, x, y, lambda p, t: (p - t) ** 2)
    """
    # Forward pass
    pred = model.forward(x)
    
    # Compute loss
    loss = loss_fn(pred, y)
    
    # Backward pass
    if loss.requires_grad:
        loss.backward()
    
    # Update parameters
    if hasattr(model, 'parameters'):
        params = model.parameters()
        grads = [p.grad for p in params if p.grad is not None]
        
        if optimizer is not None:
            for param, grad in zip(params, grads):
                optimizer.step(grad, param)
        else:
            # Simple SGD update
            updated = incremental_update(params, grads, learning_rate)
            # In a real implementation, we'd update model parameters in-place
            # For now, this demonstrates the pattern
    
    # Get loss value
    if isinstance(loss.data, list):
        if isinstance(loss.data[0], list):
            loss_val = loss.data[0][0]
        else:
            loss_val = loss.data[0]
    else:
        loss_val = float(loss.data)
    
    return loss_val


class StreamingDataset:
    """
    Dataset wrapper for streaming data.
    
    This class provides an interface for training models on streaming
    market data, where new samples arrive continuously.
    """
    
    def __init__(self, x_stream: StreamingTensor, y_stream: Optional[StreamingTensor] = None):
        """
        Initialize streaming dataset.
        
        Args:
            x_stream: StreamingTensor for input features
            y_stream: Optional StreamingTensor for targets
        """
        self.x_stream = x_stream
        self.y_stream = y_stream
    
    def get_batch(self, size: int = 1) -> tuple:
        """
        Get a batch of recent samples.
        
        Args:
            size: Number of samples to return
        
        Returns:
            Tuple of (x_batch, y_batch) tensors
        """
        x_batch = self.x_stream.get_window(size)
        
        if self.y_stream is not None:
            y_batch = self.y_stream.get_window(size)
            return x_batch, y_batch
        else:
            return x_batch, None
    
    def append(self, x: float, y: Optional[float] = None):
        """
        Append a new sample to the dataset.
        
        Args:
            x: Input value
            y: Optional target value
        """
        self.x_stream.append(x)
        if y is not None and self.y_stream is not None:
            self.y_stream.append(y)

