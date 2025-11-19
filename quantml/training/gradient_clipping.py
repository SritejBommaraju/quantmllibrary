"""
Gradient clipping utilities for quant training.

Provides various gradient clipping strategies to stabilize training.
"""

from typing import Union, Optional
from quantml.tensor import Tensor

# Try to import NumPy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


def clip_grad_norm(parameters: list, max_norm: float, norm_type: float = 2.0) -> float:
    """
    Clip gradients by norm.
    
    Args:
        parameters: List of parameters with gradients
        max_norm: Maximum norm value
        norm_type: Type of norm (2.0 for L2, 1.0 for L1, etc.)
    
    Returns:
        Total norm before clipping
    """
    if HAS_NUMPY:
        total_norm = 0.0
        for param in parameters:
            if param.grad is not None:
                grad = param.grad
                if isinstance(grad, np.ndarray):
                    param_norm = np.linalg.norm(grad, ord=norm_type)
                else:
                    grad_arr = np.array(grad, dtype=np.float64)
                    param_norm = np.linalg.norm(grad_arr, ord=norm_type)
                total_norm += param_norm ** norm_type
        
        total_norm = total_norm ** (1.0 / norm_type)
        clip_coef = max_norm / (total_norm + 1e-6)
        
        if clip_coef < 1.0:
            for param in parameters:
                if param.grad is not None:
                    grad = param.grad
                    if isinstance(grad, np.ndarray):
                        param.grad = grad * clip_coef
                    else:
                        grad_arr = np.array(grad, dtype=np.float64)
                        param.grad = (grad_arr * clip_coef).tolist()
        
        return float(total_norm)
    else:
        # Fallback to list-based computation
        total_norm = 0.0
        for param in parameters:
            if param.grad is not None:
                grad = param.grad
                if isinstance(grad, list):
                    if isinstance(grad[0], list):
                        param_norm = sum(sum(x ** norm_type for x in row) for row in grad) ** (1.0 / norm_type)
                    else:
                        param_norm = sum(x ** norm_type for x in grad) ** (1.0 / norm_type)
                else:
                    param_norm = abs(grad) ** norm_type
                total_norm += param_norm ** norm_type
        
        total_norm = total_norm ** (1.0 / norm_type)
        clip_coef = max_norm / (total_norm + 1e-6)
        
        if clip_coef < 1.0:
            for param in parameters:
                if param.grad is not None:
                    grad = param.grad
                    if isinstance(grad, list):
                        if isinstance(grad[0], list):
                            param.grad = [[x * clip_coef for x in row] for row in grad]
                        else:
                            param.grad = [x * clip_coef for x in grad]
                    else:
                        param.grad = grad * clip_coef
        
        return float(total_norm)


def clip_grad_value(parameters: list, clip_value: float):
    """
    Clip gradients by value.
    
    Args:
        parameters: List of parameters with gradients
        clip_value: Maximum absolute value for gradients
    """
    if HAS_NUMPY:
        for param in parameters:
            if param.grad is not None:
                grad = param.grad
                if isinstance(grad, np.ndarray):
                    param.grad = np.clip(grad, -clip_value, clip_value)
                else:
                    grad_arr = np.array(grad, dtype=np.float64)
                    param.grad = np.clip(grad_arr, -clip_value, clip_value).tolist()
    else:
        for param in parameters:
            if param.grad is not None:
                grad = param.grad
                if isinstance(grad, list):
                    if isinstance(grad[0], list):
                        param.grad = [[max(-clip_value, min(clip_value, x)) for x in row] for row in grad]
                    else:
                        param.grad = [max(-clip_value, min(clip_value, x)) for x in grad]
                else:
                    param.grad = max(-clip_value, min(clip_value, grad))


class GradientNormClipper:
    """Gradient clipper that clips by norm."""
    
    def __init__(self, max_norm: float, norm_type: float = 2.0):
        """
        Initialize gradient norm clipper.
        
        Args:
            max_norm: Maximum norm value
            norm_type: Type of norm (2.0 for L2)
        """
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def __call__(self, parameters: list) -> float:
        """Clip gradients and return total norm."""
        return clip_grad_norm(parameters, self.max_norm, self.norm_type)


class GradientValueClipper:
    """Gradient clipper that clips by value."""
    
    def __init__(self, clip_value: float):
        """
        Initialize gradient value clipper.
        
        Args:
            clip_value: Maximum absolute value
        """
        self.clip_value = clip_value
    
    def __call__(self, parameters: list):
        """Clip gradients by value."""
        clip_grad_value(parameters, self.clip_value)


class AdaptiveClipper:
    """Adaptive gradient clipper based on gradient statistics."""
    
    def __init__(self, percentile: float = 95.0, factor: float = 2.0):
        """
        Initialize adaptive clipper.
        
        Args:
            percentile: Percentile to use for clipping threshold
            factor: Factor to multiply percentile by
        """
        self.percentile = percentile
        self.factor = factor
    
    def __call__(self, parameters: list):
        """Clip gradients adaptively."""
        if HAS_NUMPY:
            all_grads = []
            for param in parameters:
                if param.grad is not None:
                    grad = param.grad
                    if isinstance(grad, np.ndarray):
                        all_grads.extend(grad.flatten())
                    else:
                        grad_arr = np.array(grad, dtype=np.float64)
                        all_grads.extend(grad_arr.flatten())
            
            if all_grads:
                threshold = np.percentile(np.abs(all_grads), self.percentile) * self.factor
                clip_grad_value(parameters, threshold)
        else:
            # Fallback
            all_grads = []
            for param in parameters:
                if param.grad is not None:
                    grad = param.grad
                    if isinstance(grad, list):
                        if isinstance(grad[0], list):
                            all_grads.extend([x for row in grad for x in row])
                        else:
                            all_grads.extend(grad)
                    else:
                        all_grads.append(grad)
            
            if all_grads:
                abs_grads = [abs(g) for g in all_grads]
                abs_grads.sort()
                idx = int(len(abs_grads) * self.percentile / 100.0)
                threshold = abs_grads[min(idx, len(abs_grads) - 1)] * self.factor
                clip_grad_value(parameters, threshold)

