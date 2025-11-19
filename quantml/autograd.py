"""
Automatic differentiation engine for QuantML.

This module provides the core autograd functionality, including computation
graph construction, topological sorting, and gradient computation via
backpropagation.
"""

from typing import Set, List, Callable, Optional
from collections import deque
from quantml.tensor import Tensor


def build_topo(tensor: Tensor, visited: Optional[Set] = None, topo: Optional[List] = None) -> List[Tensor]:
    """
    Build topological ordering of computation graph.
    
    Performs a depth-first search to order tensors in reverse topological order
    (children before parents), which is needed for backpropagation.
    
    Args:
        tensor: Root tensor to start traversal from
        visited: Set of already visited tensors (internal)
        topo: List to accumulate topological order (internal)
    
    Returns:
        List of tensors in reverse topological order
    
    Examples:
        >>> x = Tensor([1.0], requires_grad=True)
        >>> y = Tensor([2.0], requires_grad=True)
        >>> z = x + y
        >>> topo = build_topo(z)
        >>> len(topo)  # z, x, y (or y, x)
    """
    if visited is None:
        visited = set()
    if topo is None:
        topo = []
    
    if tensor in visited:
        return topo
    
    visited.add(tensor)
    
    # Visit all parent tensors first
    for parent in tensor._prev:
        build_topo(parent, visited, topo)
    
    # Add current tensor after all its parents
    topo.append(tensor)
    
    return topo


def backward(tensor: Tensor, grad: Optional[any] = None):
    """
    Compute gradients for all tensors in the computation graph.
    
    This function builds the topological order and then calls backward()
    on each tensor in reverse order, propagating gradients through the graph.
    
    Args:
        tensor: The tensor to compute gradients for
        grad: Initial gradient (defaults to ones)
    
    Examples:
        >>> x = Tensor([1.0, 2.0], requires_grad=True)
        >>> y = Tensor([3.0, 4.0], requires_grad=True)
        >>> z = x * y
        >>> backward(z)
        >>> x.grad  # [3.0, 4.0]
        >>> y.grad  # [1.0, 2.0]
    """
    # Build topological order
    topo = build_topo(tensor)
    
    # Initialize gradient of root tensor
    if tensor.grad is None:
        if grad is None:
            # Create ones with same shape
            tensor.grad = _ones_like(tensor.data)
        else:
            tensor.grad = grad
    else:
        if grad is not None:
            tensor.grad = _add_grads(tensor.grad, grad)
    
    # Backpropagate through graph in reverse topological order
    for node in reversed(topo):
        if node.requires_grad and node._backward_fn is not None:
            # Ensure gradient exists
            if node.grad is None:
                node.grad = _ones_like(node.data)
            # Call backward function
            node._backward_fn(node.grad)


def _ones_like(data: any) -> any:
    """Create a tensor of ones with the same shape as data."""
    if isinstance(data, list):
        if len(data) == 0:
            return []
        if isinstance(data[0], list):
            return [[1.0] * len(row) for row in data]
        return [1.0] * len(data)
    return [[1.0]]


def _add_grads(grad1: any, grad2: any) -> any:
    """Add two gradients element-wise."""
    if isinstance(grad1, list) and isinstance(grad2, list):
        if len(grad1) == 0:
            return grad2
        if len(grad2) == 0:
            return grad1
        
        # Check if nested
        if isinstance(grad1[0], list) and isinstance(grad2[0], list):
            # 2D case
            result = []
            max_rows = max(len(grad1), len(grad2))
            for i in range(max_rows):
                row1 = grad1[i] if i < len(grad1) else [0.0] * len(grad1[0]) if grad1 else [0.0]
                row2 = grad2[i] if i < len(grad2) else [0.0] * len(grad2[0]) if grad2 else [0.0]
                max_cols = max(len(row1), len(row2))
                result.append([
                    (row1[j] if j < len(row1) else 0.0) + 
                    (row2[j] if j < len(row2) else 0.0)
                    for j in range(max_cols)
                ])
            return result
        else:
            # 1D case
            max_len = max(len(grad1), len(grad2))
            return [
                (grad1[i] if i < len(grad1) else 0.0) + 
                (grad2[i] if i < len(grad2) else 0.0)
                for i in range(max_len)
            ]
    elif isinstance(grad1, list):
        # grad1 is list, grad2 is scalar
        if isinstance(grad1[0], list):
            return [[g + grad2 for g in row] for row in grad1]
        return [g + grad2 for g in grad1]
    elif isinstance(grad2, list):
        # grad2 is list, grad1 is scalar
        if isinstance(grad2[0], list):
            return [[grad1 + g for g in row] for row in grad2]
        return [grad1 + g for g in grad2]
    else:
        # Both scalars
        return grad1 + grad2

