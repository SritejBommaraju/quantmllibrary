"""
Core operations for Tensor objects.

This module provides all the fundamental operations that can be performed
on tensors, including arithmetic, linear algebra, reductions, and activations.
All operations support automatic differentiation.
"""

from typing import Union, Optional, List, Any, Callable
from quantml.tensor import Tensor
from quantml.autograd import backward

# Try to import NumPy for optimized operations
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


def _to_tensor(x: Union[Tensor, float, int, List]) -> Tensor:
    """Convert input to Tensor if needed."""
    if isinstance(x, Tensor):
        return x
    return Tensor(x)


def _create_tensor_from_numpy(
    arr: Any,
    requires_grad: bool = False,
    _prev: Optional[set] = None,
    _op: Optional[str] = None,
    _backward_fn: Optional[Callable] = None
) -> Tensor:
    """
    Create a Tensor directly from a NumPy array, avoiding list conversion.
    
    This is an internal optimization to eliminate NumPy -> list -> NumPy conversions.
    """
    if HAS_NUMPY and isinstance(arr, np.ndarray):
        return Tensor(
            arr,  # Will be converted to NumPy in __init__
            requires_grad=requires_grad,
            _prev=_prev,
            _op=_op,
            _backward_fn=_backward_fn,
            _np_array=arr  # Direct pass to skip conversion
        )
    # Fallback to regular Tensor creation
    return Tensor(
        arr,
        requires_grad=requires_grad,
        _prev=_prev,
        _op=_op,
        _backward_fn=_backward_fn
    )


def _to_numpy(data: Any):
    """Convert data to NumPy array if NumPy is available."""
    if not HAS_NUMPY:
        return None
    if isinstance(data, np.ndarray):
        return data
    return np.array(data, dtype=np.float64)


def _from_numpy(arr) -> List:
    """Convert NumPy array to nested list."""
    if arr is None:
        return None
    if HAS_NUMPY and isinstance(arr, np.ndarray):
        return arr.tolist()
    return arr


def _broadcast_shape(shape1: tuple, shape2: tuple) -> tuple:
    """Compute broadcasted shape of two tensors."""
    # Simple broadcasting: if one is scalar, use the other's shape
    if shape1 == (1, 1) or shape1 == (1,):
        return shape2
    if shape2 == (1, 1) or shape2 == (1,):
        return shape1
    # For now, shapes must match or one must be scalar
    return shape1 if len(shape1) >= len(shape2) else shape2


def _broadcast_op(a: List, b: List, op: callable, op_type: Optional[str] = None) -> List:
    """Apply operation with broadcasting. Uses NumPy if available."""
    if HAS_NUMPY:
        try:
            a_arr = _to_numpy(a)
            b_arr = _to_numpy(b)
            if a_arr is not None and b_arr is not None:
                # Use NumPy broadcasting based on op_type
                if op_type == 'add':
                    result = (a_arr + b_arr).tolist()
                elif op_type == 'mul':
                    result = (a_arr * b_arr).tolist()
                elif op_type == 'sub':
                    result = (a_arr - b_arr).tolist()
                elif op_type == 'div':
                    result = (a_arr / b_arr).tolist()
                else:
                    # Fallback for custom ops
                    result = np.vectorize(op)(a_arr, b_arr).tolist()
                return result
        except (ValueError, TypeError):
            # Broadcasting failed, fall through to pure Python
            pass
    
    # Pure Python fallback
    if isinstance(a[0], list) and isinstance(b[0], list):
        # Both 2D
        if len(a) == len(b) and len(a[0]) == len(b[0]):
            return [[op(a[i][j], b[i][j]) for j in range(len(a[i]))] 
                    for i in range(len(a))]
        # Broadcast scalar
        if len(b) == 1 and len(b[0]) == 1:
            val = b[0][0]
            return [[op(a[i][j], val) for j in range(len(a[i]))] 
                    for i in range(len(a))]
        if len(a) == 1 and len(a[0]) == 1:
            val = a[0][0]
            return [[op(val, b[i][j]) for j in range(len(b[i]))] 
                    for i in range(len(b))]
    elif isinstance(a[0], list):
        # a is 2D, b is 1D or scalar
        if isinstance(b, list) and not isinstance(b[0], list):
            # b is 1D
            return [[op(a[i][j], b[j] if j < len(b) else b[0]) 
                    for j in range(len(a[i]))] for i in range(len(a))]
        else:
            # b is scalar
            val = float(b[0]) if isinstance(b, list) else float(b)
            return [[op(a[i][j], val) for j in range(len(a[i]))] 
                    for i in range(len(a))]
    elif isinstance(b[0], list):
        # b is 2D, a is 1D or scalar
        val = float(a[0]) if isinstance(a, list) else float(a)
        return [[op(val, b[i][j]) for j in range(len(b[i]))] 
                for i in range(len(b))]
    else:
        # Both 1D
        max_len = max(len(a), len(b))
        return [op(a[i] if i < len(a) else a[0], 
                  b[i] if i < len(b) else b[0]) 
                for i in range(max_len)]
    return a


def add(a: Union[Tensor, float, int], b: Union[Tensor, float, int]) -> Tensor:
    """
    Add two tensors element-wise.
    
    Args:
        a: First tensor or scalar
        b: Second tensor or scalar
    
    Returns:
        New tensor with result
    
    Examples:
        >>> x = Tensor([1.0, 2.0])
        >>> y = Tensor([3.0, 4.0])
        >>> z = add(x, y)  # [4.0, 6.0]
    """
    a = _to_tensor(a)
    b = _to_tensor(b)
    
    # Use NumPy arrays if available for better performance
    a_data = a.numpy if (HAS_NUMPY and a.numpy is not None) else a.data
    b_data = b.numpy if (HAS_NUMPY and b.numpy is not None) else b.data
    
    if HAS_NUMPY and isinstance(a_data, np.ndarray) and isinstance(b_data, np.ndarray):
        try:
            out_arr = a_data + b_data
            # Create tensor directly from NumPy array to avoid conversion
            out = _create_tensor_from_numpy(
                out_arr,
                requires_grad=a.requires_grad or b.requires_grad,
                _prev={a, b} if (a.requires_grad or b.requires_grad) else set(),
                _op='add'
            )
        except (ValueError, TypeError):
            out_data = _broadcast_op(a.data, b.data, lambda x, y: float(x) + float(y), 'add')
            out = Tensor(
                out_data,
                requires_grad=a.requires_grad or b.requires_grad,
                _prev={a, b} if (a.requires_grad or b.requires_grad) else set(),
                _op='add'
            )
    else:
        out_data = _broadcast_op(a_data, b_data, lambda x, y: float(x) + float(y), 'add')
        out = Tensor(
            out_data,
            requires_grad=a.requires_grad or b.requires_grad,
            _prev={a, b} if (a.requires_grad or b.requires_grad) else set(),
            _op='add'
        )
    
    def _backward(grad):
        if a.requires_grad:
            a.backward(grad)
        if b.requires_grad:
            b.backward(grad)
    
    if out.requires_grad:
        out._backward_fn = _backward
    
    return out


def sub(a: Union[Tensor, float, int], b: Union[Tensor, float, int]) -> Tensor:
    """Subtract two tensors element-wise."""
    return add(a, mul(b, -1.0))


def mul(a: Union[Tensor, float, int], b: Union[Tensor, float, int]) -> Tensor:
    """
    Multiply two tensors element-wise.
    
    Args:
        a: First tensor or scalar
        b: Second tensor or scalar
    
    Returns:
        New tensor with result
    """
    a = _to_tensor(a)
    b = _to_tensor(b)
    
    # Use NumPy arrays if available
    a_data = a.numpy if (HAS_NUMPY and a.numpy is not None) else a.data
    b_data = b.numpy if (HAS_NUMPY and b.numpy is not None) else b.data
    
    if HAS_NUMPY and isinstance(a_data, np.ndarray) and isinstance(b_data, np.ndarray):
        try:
            out_arr = a_data * b_data
            # Create tensor directly from NumPy array
            out = _create_tensor_from_numpy(
                out_arr,
                requires_grad=a.requires_grad or b.requires_grad,
                _prev={a, b} if (a.requires_grad or b.requires_grad) else set(),
                _op='mul'
            )
        except (ValueError, TypeError):
            out_data = _broadcast_op(a.data, b.data, lambda x, y: float(x) * float(y), 'mul')
            out = Tensor(
                out_data,
                requires_grad=a.requires_grad or b.requires_grad,
                _prev={a, b} if (a.requires_grad or b.requires_grad) else set(),
                _op='mul'
            )
    else:
        out_data = _broadcast_op(a_data, b_data, lambda x, y: float(x) * float(y), 'mul')
        out = Tensor(
            out_data,
            requires_grad=a.requires_grad or b.requires_grad,
            _prev={a, b} if (a.requires_grad or b.requires_grad) else set(),
            _op='mul'
        )
    
    def _backward(grad):
        if a.requires_grad:
            a_grad = _broadcast_op(grad, b.data, lambda g, b_val: float(g) * float(b_val), 'mul')
            a.backward(a_grad)
        if b.requires_grad:
            b_grad = _broadcast_op(grad, a.data, lambda g, a_val: float(g) * float(a_val), 'mul')
            b.backward(b_grad)
    
    if out.requires_grad:
        out._backward_fn = _backward
    
    return out


def div(a: Union[Tensor, float, int], b: Union[Tensor, float, int]) -> Tensor:
    """Divide two tensors element-wise."""
    return mul(a, pow(b, -1.0))


def pow(a: Union[Tensor, float, int], power: Union[float, int]) -> Tensor:
    """
    Raise tensor to a power.
    
    Args:
        a: Base tensor
        power: Exponent
    
    Returns:
        New tensor with result
    """
    a = _to_tensor(a)
    power = float(power)
    
    # Use NumPy if available
    if HAS_NUMPY:
        try:
            a_arr = _to_numpy(a.data)
            if a_arr is not None:
                out_arr = np.power(a_arr, power)
                # Create tensor directly from NumPy array
                out = _create_tensor_from_numpy(
                    out_arr,
                    requires_grad=a.requires_grad,
                    _prev={a} if a.requires_grad else set(),
                    _op='pow'
                )
                # Set backward function
                def _backward(grad):
                    if a.requires_grad:
                        # d/dx (x^n) = n * x^(n-1)
                        if HAS_NUMPY:
                            try:
                                grad_arr = np.array(grad, dtype=np.float64) if not isinstance(grad, np.ndarray) else grad
                                a_grad_arr = grad_arr * power * np.power(a_arr, power - 1)
                                a.backward(a_grad_arr)
                            except (ValueError, TypeError):
                                # Fallback
                                if isinstance(grad, list) and isinstance(a.data[0], list):
                                    a_grad = [[float(grad[i][j]) * power * (float(a.data[i][j]) ** (power - 1))
                                              for j in range(len(grad[i]))] for i in range(len(grad))]
                                elif isinstance(grad, list):
                                    a_grad = [float(grad[i]) * power * (float(a.data[i]) ** (power - 1))
                                             for i in range(len(grad))]
                                else:
                                    a_grad = float(grad) * power * (float(a.data[0][0]) ** (power - 1))
                                a.backward(a_grad)
                        else:
                            if isinstance(grad, list) and isinstance(a.data[0], list):
                                a_grad = [[float(grad[i][j]) * power * (float(a.data[i][j]) ** (power - 1))
                                          for j in range(len(grad[i]))] for i in range(len(grad))]
                            elif isinstance(grad, list):
                                a_grad = [float(grad[i]) * power * (float(a.data[i]) ** (power - 1))
                                         for i in range(len(grad))]
                            else:
                                a_grad = float(grad) * power * (float(a.data[0][0]) ** (power - 1))
                            a.backward(a_grad)
                
                if out.requires_grad:
                    out._backward_fn = _backward
                
                return out
            else:
                # Fallback
                def _pow_op(x):
                    return float(x) ** power
                if isinstance(a.data[0], list):
                    out_data = [[_pow_op(a.data[i][j]) for j in range(len(a.data[i]))] 
                                for i in range(len(a.data))]
                else:
                    out_data = [_pow_op(x) for x in a.data]
        except (ValueError, TypeError):
            # Fallback
            def _pow_op(x):
                return float(x) ** power
            if isinstance(a.data[0], list):
                out_data = [[_pow_op(a.data[i][j]) for j in range(len(a.data[i]))] 
                            for i in range(len(a.data))]
            else:
                out_data = [_pow_op(x) for x in a.data]
    else:
        # Pure Python
        def _pow_op(x):
            return float(x) ** power
        if isinstance(a.data[0], list):
            out_data = [[_pow_op(a.data[i][j]) for j in range(len(a.data[i]))] 
                        for i in range(len(a.data))]
        else:
            out_data = [_pow_op(x) for x in a.data]
    
    out = Tensor(
        out_data,
        requires_grad=a.requires_grad,
        _prev={a} if a.requires_grad else set(),
        _op='pow'
    )
    
    def _backward(grad):
        if a.requires_grad:
            # d/dx (x^n) = n * x^(n-1)
            if isinstance(grad, list) and isinstance(a.data[0], list):
                a_grad = [[float(grad[i][j]) * power * (float(a.data[i][j]) ** (power - 1))
                          for j in range(len(grad[i]))] for i in range(len(grad))]
            elif isinstance(grad, list):
                a_grad = [float(grad[i]) * power * (float(a.data[i]) ** (power - 1))
                         for i in range(len(grad))]
            else:
                a_grad = float(grad) * power * (float(a.data[0][0]) ** (power - 1))
            a.backward(a_grad)
    
    if out.requires_grad:
        out._backward_fn = _backward
    
    return out


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """
    Matrix multiplication of two tensors.
    
    Args:
        a: First matrix (m x n)
        b: Second matrix (n x p)
    
    Returns:
        Result matrix (m x p)
    
    Examples:
        >>> a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> b = Tensor([[5.0, 6.0], [7.0, 8.0]])
        >>> c = matmul(a, b)  # [[19.0, 22.0], [43.0, 50.0]]
    """
    a = _to_tensor(a)
    b = _to_tensor(b)
    
    # Ensure 2D
    a_data = a.data if isinstance(a.data[0], list) else [a.data]
    b_data = b.data if isinstance(b.data[0], list) else [b.data]
    
    # Matrix multiplication - use NumPy if available
    if HAS_NUMPY:
        try:
            # Use numpy arrays directly from tensors if available
            a_arr = a.numpy if (a.numpy is not None) else np.array(a_data, dtype=np.float64)
            b_arr = b.numpy if (b.numpy is not None) else np.array(b_data, dtype=np.float64)
            out_arr = np.dot(a_arr, b_arr)
            # Create tensor directly from NumPy array
            out = _create_tensor_from_numpy(
                out_arr,
                requires_grad=a.requires_grad or b.requires_grad,
                _prev={a, b} if (a.requires_grad or b.requires_grad) else set(),
                _op='matmul'
            )
        except (ValueError, TypeError):
            # Fallback to pure Python
            m, n = len(a_data), len(a_data[0])
            n2, p = len(b_data), len(b_data[0])
            if n != n2:
                raise ValueError(f"Matrix dimensions incompatible: {a.shape} x {b.shape}")
            out_data = [[sum(float(a_data[i][k]) * float(b_data[k][j]) 
                            for k in range(n)) 
                        for j in range(p)] 
                       for i in range(m)]
            out = Tensor(
                out_data,
                requires_grad=a.requires_grad or b.requires_grad,
                _prev={a, b} if (a.requires_grad or b.requires_grad) else set(),
                _op='matmul'
            )
    else:
        # Pure Python implementation
        m, n = len(a_data), len(a_data[0])
        n2, p = len(b_data), len(b_data[0])
        if n != n2:
            raise ValueError(f"Matrix dimensions incompatible: {a.shape} x {b.shape}")
        out_data = [[sum(float(a_data[i][k]) * float(b_data[k][j]) 
                        for k in range(n)) 
                    for j in range(p)] 
                   for i in range(m)]
        out = Tensor(
            out_data,
            requires_grad=a.requires_grad or b.requires_grad,
            _prev={a, b} if (a.requires_grad or b.requires_grad) else set(),
            _op='matmul'
        )
    
    def _backward(grad):
        if a.requires_grad:
            # dL/da = grad @ b.T
            if HAS_NUMPY:
                try:
                    grad_arr = np.array(grad, dtype=np.float64)
                    b_arr = np.array(b_data, dtype=np.float64)
                    a_grad_arr = np.dot(grad_arr, b_arr.T)
                    a_grad = a_grad_arr.tolist()
                except (ValueError, TypeError):
                    # Fallback
                    b_T = [[b_data[j][i] for j in range(len(b_data))] 
                           for i in range(len(b_data[0]))]
                    a_grad = [[sum(float(grad[i][k]) * float(b_T[k][j]) 
                                  for k in range(len(grad[0]))) 
                              for j in range(len(b_T[0]))] 
                             for i in range(len(grad))]
            else:
                b_T = [[b_data[j][i] for j in range(len(b_data))] 
                       for i in range(len(b_data[0]))]
                a_grad = [[sum(float(grad[i][k]) * float(b_T[k][j]) 
                              for k in range(len(grad[0]))) 
                          for j in range(len(b_T[0]))] 
                         for i in range(len(grad))]
            a.backward(a_grad)
        if b.requires_grad:
            # dL/db = a.T @ grad
            if HAS_NUMPY:
                try:
                    a_arr = np.array(a_data, dtype=np.float64)
                    grad_arr = np.array(grad, dtype=np.float64)
                    b_grad_arr = np.dot(a_arr.T, grad_arr)
                    b_grad = b_grad_arr.tolist()
                except (ValueError, TypeError):
                    # Fallback
                    a_T = [[a_data[j][i] for j in range(len(a_data))] 
                           for i in range(len(a_data[0]))]
                    b_grad = [[sum(float(a_T[i][k]) * float(grad[k][j]) 
                                  for k in range(len(a_T[0]))) 
                              for j in range(len(grad[0]))] 
                             for i in range(len(a_T))]
            else:
                a_T = [[a_data[j][i] for j in range(len(a_data))] 
                       for i in range(len(a_data[0]))]
                b_grad = [[sum(float(a_T[i][k]) * float(grad[k][j]) 
                              for k in range(len(a_T[0]))) 
                          for j in range(len(grad[0]))] 
                         for i in range(len(a_T))]
            b.backward(b_grad)
    
    if out.requires_grad:
        out._backward_fn = _backward
    
    return out


def dot(a: Tensor, b: Tensor) -> Tensor:
    """
    Dot product of two 1D tensors.
    
    Args:
        a: First vector
        b: Second vector
    
    Returns:
        Scalar result
    """
    a = _to_tensor(a)
    b = _to_tensor(b)
    
    # Flatten to 1D
    a_flat = a.data if not isinstance(a.data[0], list) else a.data[0]
    b_flat = b.data if not isinstance(b.data[0], list) else b.data[0]
    
    if len(a_flat) != len(b_flat):
        raise ValueError(f"Vectors must have same length: {len(a_flat)} vs {len(b_flat)}")
    
    # Use NumPy if available
    if HAS_NUMPY:
        try:
            a_arr = np.array(a_flat, dtype=np.float64)
            b_arr = np.array(b_flat, dtype=np.float64)
            result = float(np.dot(a_arr, b_arr))
        except (ValueError, TypeError):
            result = sum(float(a_flat[i]) * float(b_flat[i]) for i in range(len(a_flat)))
    else:
        result = sum(float(a_flat[i]) * float(b_flat[i]) for i in range(len(a_flat)))
    
    out = Tensor(
        [[result]],
        requires_grad=a.requires_grad or b.requires_grad,
        _prev={a, b} if (a.requires_grad or b.requires_grad) else set(),
        _op='dot'
    )
    
    def _backward(grad):
        grad_val = float(grad[0][0]) if isinstance(grad[0], list) else float(grad[0])
        if a.requires_grad:
            a.backward([grad_val * float(b_flat[i]) for i in range(len(b_flat))])
        if b.requires_grad:
            b.backward([grad_val * float(a_flat[i]) for i in range(len(a_flat))])
    
    if out.requires_grad:
        out._backward_fn = _backward
    
    return out


def sum(t: Tensor, axis: Optional[int] = None) -> Tensor:
    """
    Sum elements of tensor, optionally along an axis.
    
    Args:
        t: Input tensor
        axis: Axis to sum along (None for all elements)
    
    Returns:
        Sum result
    """
    t = _to_tensor(t)
    
    # Use NumPy if available
    if HAS_NUMPY:
        try:
            t_arr = t.numpy if (t.numpy is not None) else _to_numpy(t.data)
            if t_arr is not None:
                result = np.sum(t_arr, axis=axis, keepdims=True)
                # Create tensor directly from NumPy array
                out = _create_tensor_from_numpy(
                    result,
                    requires_grad=t.requires_grad,
                    _prev={t} if t.requires_grad else set(),
                    _op='sum'
                )
                # Set backward function
                def _backward(grad):
                    if t.requires_grad:
                        # Broadcast gradient back
                        if HAS_NUMPY:
                            try:
                                grad_arr = np.array(grad, dtype=np.float64) if not isinstance(grad, np.ndarray) else grad
                                if axis is None:
                                    # Broadcast to all elements
                                    t_grad = np.broadcast_to(grad_arr, t_arr.shape)
                                elif axis == 0:
                                    # Broadcast along axis 0
                                    t_grad = np.broadcast_to(grad_arr, t_arr.shape)
                                else:  # axis == 1
                                    t_grad = np.broadcast_to(grad_arr, t_arr.shape)
                                t.backward(t_grad)
                            except (ValueError, TypeError):
                                # Fallback to list operations
                                if axis is None:
                                    grad_val = float(grad[0][0]) if isinstance(grad[0], list) else float(grad[0])
                                    if isinstance(t.data[0], list):
                                        t_grad = [[grad_val for _ in row] for row in t.data]
                                    else:
                                        t_grad = [grad_val for _ in t.data]
                                elif axis == 0:
                                    grad_val = float(grad[0][0]) if isinstance(grad[0], list) else float(grad[0])
                                    if isinstance(t.data[0], list):
                                        t_grad = [[grad_val for _ in range(len(t.data))] 
                                                 for _ in range(len(t.data[0]))]
                                        t_grad = [[t_grad[j][i] for j in range(len(t_grad))] 
                                                 for i in range(len(t_grad[0]))]
                                    else:
                                        t_grad = [grad_val for _ in t.data]
                                else:  # axis == 1
                                    if isinstance(grad[0], list):
                                        t_grad = [[float(grad[i][0]) for _ in range(len(t.data[i]))] 
                                                 for i in range(len(t.data))]
                                    else:
                                        t_grad = [[float(grad[i]) for _ in range(len(t.data[i]))] 
                                                 for i in range(len(t.data))]
                                t.backward(t_grad)
                        else:
                            # Fallback (same as before)
                            if axis is None:
                                grad_val = float(grad[0][0]) if isinstance(grad[0], list) else float(grad[0])
                                if isinstance(t.data[0], list):
                                    t_grad = [[grad_val for _ in row] for row in t.data]
                                else:
                                    t_grad = [grad_val for _ in t.data]
                            elif axis == 0:
                                grad_val = float(grad[0][0]) if isinstance(grad[0], list) else float(grad[0])
                                if isinstance(t.data[0], list):
                                    t_grad = [[grad_val for _ in range(len(t.data))] 
                                             for _ in range(len(t.data[0]))]
                                    t_grad = [[t_grad[j][i] for j in range(len(t_grad))] 
                                             for i in range(len(t_grad[0]))]
                                else:
                                    t_grad = [grad_val for _ in t.data]
                            else:  # axis == 1
                                if isinstance(grad[0], list):
                                    t_grad = [[float(grad[i][0]) for _ in range(len(t.data[i]))] 
                                             for i in range(len(t.data))]
                                else:
                                    t_grad = [[float(grad[i]) for _ in range(len(t.data[i]))] 
                                             for i in range(len(t.data))]
                            t.backward(t_grad)
                
                if out.requires_grad:
                    out._backward_fn = _backward
                
                return out
        except (ValueError, TypeError):
            # Fallback to pure Python
            if axis is None:
                total = 0.0
                if isinstance(t.data[0], list):
                    for row in t.data:
                        total += sum(float(x) for x in row)
                else:
                    total = sum(float(x) for x in t.data)
                out_data = [[total]]
            elif axis == 0:
                if isinstance(t.data[0], list):
                    out_data = [[sum(float(t.data[i][j]) for i in range(len(t.data))) 
                                for j in range(len(t.data[0]))]]
                else:
                    out_data = [[sum(float(x) for x in t.data)]]
            elif axis == 1:
                if isinstance(t.data[0], list):
                    out_data = [[sum(float(row[j]) for j in range(len(row)))] 
                               for row in t.data]
                else:
                    out_data = t.data
            else:
                raise ValueError(f"Invalid axis: {axis}")
    else:
        # Pure Python implementation
        if axis is None:
            total = 0.0
            if isinstance(t.data[0], list):
                for row in t.data:
                    total += sum(float(x) for x in row)
            else:
                total = sum(float(x) for x in t.data)
            out_data = [[total]]
        elif axis == 0:
            if isinstance(t.data[0], list):
                out_data = [[sum(float(t.data[i][j]) for i in range(len(t.data))) 
                            for j in range(len(t.data[0]))]]
            else:
                out_data = [[sum(float(x) for x in t.data)]]
        elif axis == 1:
            if isinstance(t.data[0], list):
                out_data = [[sum(float(row[j]) for j in range(len(row)))] 
                           for row in t.data]
            else:
                out_data = t.data
        else:
            raise ValueError(f"Invalid axis: {axis}")
    
    out = Tensor(
        out_data,
        requires_grad=t.requires_grad,
        _prev={t} if t.requires_grad else set(),
        _op='sum'
    )
    
    def _backward(grad):
        if t.requires_grad:
            # Broadcast gradient back
            if axis is None:
                # Broadcast to all elements
                grad_val = float(grad[0][0]) if isinstance(grad[0], list) else float(grad[0])
                if isinstance(t.data[0], list):
                    t_grad = [[grad_val for _ in row] for row in t.data]
                else:
                    t_grad = [grad_val for _ in t.data]
            elif axis == 0:
                grad_val = float(grad[0][0]) if isinstance(grad[0], list) else float(grad[0])
                if isinstance(t.data[0], list):
                    t_grad = [[grad_val for _ in range(len(t.data))] 
                             for _ in range(len(t.data[0]))]
                    t_grad = [[t_grad[j][i] for j in range(len(t_grad))] 
                             for i in range(len(t_grad[0]))]
                else:
                    t_grad = [grad_val for _ in t.data]
            else:  # axis == 1
                if isinstance(grad[0], list):
                    t_grad = [[float(grad[i][0]) for _ in range(len(t.data[i]))] 
                             for i in range(len(t.data))]
                else:
                    t_grad = [[float(grad[i]) for _ in range(len(t.data[i]))] 
                             for i in range(len(t.data))]
            t.backward(t_grad)
    
    if out.requires_grad:
        out._backward_fn = _backward
    
    return out


def mean(t: Tensor, axis: Optional[int] = None) -> Tensor:
    """Compute mean of tensor elements."""
    t = _to_tensor(t)
    s = sum(t, axis=axis)
    if axis is None:
        count = 1.0
        if isinstance(t.data[0], list):
            count = len(t.data) * len(t.data[0])
        else:
            count = len(t.data)
    elif axis == 0:
        count = len(t.data) if isinstance(t.data[0], list) else 1.0
    else:
        count = len(t.data[0]) if isinstance(t.data[0], list) else len(t.data)
    
    return div(s, count)


def std(t: Tensor, axis: Optional[int] = None) -> Tensor:
    """Compute standard deviation of tensor elements."""
    t = _to_tensor(t)
    m = mean(t, axis=axis)
    # Expand mean to match shape for subtraction
    diff = sub(t, m)
    diff_sq = mul(diff, diff)
    var = mean(diff_sq, axis=axis)
    return pow(var, 0.5)


def relu(t: Tensor) -> Tensor:
    """
    ReLU activation function: max(0, x)
    
    Args:
        t: Input tensor
    
    Returns:
        Tensor with ReLU applied
    """
    t = _to_tensor(t)
    
    # Use NumPy if available
    if HAS_NUMPY:
        try:
            t_arr = _to_numpy(t.data)
            if t_arr is not None:
                out_arr = np.maximum(0, t_arr)
                out_data = out_arr.tolist()
            else:
                # Fallback
                def _relu_op(x):
                    return max(0.0, float(x))
                if isinstance(t.data[0], list):
                    out_data = [[_relu_op(t.data[i][j]) for j in range(len(t.data[i]))] 
                                for i in range(len(t.data))]
                else:
                    out_data = [_relu_op(x) for x in t.data]
        except (ValueError, TypeError):
            # Fallback
            def _relu_op(x):
                return max(0.0, float(x))
            if isinstance(t.data[0], list):
                out_data = [[_relu_op(t.data[i][j]) for j in range(len(t.data[i]))] 
                            for i in range(len(t.data))]
            else:
                out_data = [_relu_op(x) for x in t.data]
    else:
        # Pure Python
        def _relu_op(x):
            return max(0.0, float(x))
        if isinstance(t.data[0], list):
            out_data = [[_relu_op(t.data[i][j]) for j in range(len(t.data[i]))] 
                        for i in range(len(t.data))]
        else:
            out_data = [_relu_op(x) for x in t.data]
    
    out = Tensor(
        out_data,
        requires_grad=t.requires_grad,
        _prev={t} if t.requires_grad else set(),
        _op='relu'
    )
    
    def _backward(grad):
        if t.requires_grad:
            if isinstance(grad, list) and isinstance(t.data[0], list):
                t_grad = [[float(grad[i][j]) if float(t.data[i][j]) > 0 else 0.0
                          for j in range(len(grad[i]))] for i in range(len(grad))]
            elif isinstance(grad, list):
                t_grad = [float(grad[i]) if float(t.data[i]) > 0 else 0.0
                         for i in range(len(grad))]
            else:
                t_grad = float(grad) if float(t.data[0][0]) > 0 else 0.0
            t.backward(t_grad)
    
    if out.requires_grad:
        out._backward_fn = _backward
    
    return out


def sigmoid(t: Tensor) -> Tensor:
    """
    Sigmoid activation function: 1 / (1 + exp(-x))
    
    Args:
        t: Input tensor
    
    Returns:
        Tensor with sigmoid applied
    """
    t = _to_tensor(t)
    
    # Use NumPy if available
    if HAS_NUMPY:
        try:
            t_arr = _to_numpy(t.data)
            if t_arr is not None:
                out_arr = 1.0 / (1.0 + np.exp(-t_arr))
                out_data = out_arr.tolist()
            else:
                # Fallback
                import math
                def _sigmoid_op(x):
                    return 1.0 / (1.0 + math.exp(-float(x)))
                if isinstance(t.data[0], list):
                    out_data = [[_sigmoid_op(t.data[i][j]) for j in range(len(t.data[i]))] 
                                for i in range(len(t.data))]
                else:
                    out_data = [_sigmoid_op(x) for x in t.data]
        except (ValueError, TypeError):
            # Fallback
            import math
            def _sigmoid_op(x):
                return 1.0 / (1.0 + math.exp(-float(x)))
            if isinstance(t.data[0], list):
                out_data = [[_sigmoid_op(t.data[i][j]) for j in range(len(t.data[i]))] 
                            for i in range(len(t.data))]
            else:
                out_data = [_sigmoid_op(x) for x in t.data]
    else:
        # Pure Python
        import math
        def _sigmoid_op(x):
            return 1.0 / (1.0 + math.exp(-float(x)))
        if isinstance(t.data[0], list):
            out_data = [[_sigmoid_op(t.data[i][j]) for j in range(len(t.data[i]))] 
                        for i in range(len(t.data))]
        else:
            out_data = [_sigmoid_op(x) for x in t.data]
    
    out = Tensor(
        out_data,
        requires_grad=t.requires_grad,
        _prev={t} if t.requires_grad else set(),
        _op='sigmoid'
    )
    
    def _backward(grad):
        if t.requires_grad:
            # d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
            if isinstance(grad, list) and isinstance(t.data[0], list):
                t_grad = [[float(grad[i][j]) * float(out_data[i][j]) * (1.0 - float(out_data[i][j]))
                          for j in range(len(grad[i]))] for i in range(len(grad))]
            elif isinstance(grad, list):
                t_grad = [float(grad[i]) * float(out_data[i]) * (1.0 - float(out_data[i]))
                         for i in range(len(grad))]
            else:
                t_grad = float(grad) * float(out_data[0][0]) * (1.0 - float(out_data[0][0]))
            t.backward(t_grad)
    
    if out.requires_grad:
        out._backward_fn = _backward
    
    return out


def abs(t: Tensor) -> Tensor:
    """
    Absolute value of tensor elements.
    
    Args:
        t: Input tensor
    
    Returns:
        Tensor with absolute values
    """
    t = _to_tensor(t)
    
    # Use NumPy if available
    if HAS_NUMPY:
        try:
            t_arr = _to_numpy(t.data)
            if t_arr is not None:
                out_arr = np.abs(t_arr)
                out_data = out_arr.tolist()
            else:
                # Fallback
                if isinstance(t.data[0], list):
                    out_data = [[abs(float(t.data[i][j])) for j in range(len(t.data[i]))] 
                                for i in range(len(t.data))]
                else:
                    out_data = [abs(float(x)) for x in t.data]
        except (ValueError, TypeError):
            # Fallback
            if isinstance(t.data[0], list):
                out_data = [[abs(float(t.data[i][j])) for j in range(len(t.data[i]))] 
                            for i in range(len(t.data))]
            else:
                out_data = [abs(float(x)) for x in t.data]
    else:
        # Pure Python
        if isinstance(t.data[0], list):
            out_data = [[abs(float(t.data[i][j])) for j in range(len(t.data[i]))] 
                        for i in range(len(t.data))]
        else:
            out_data = [abs(float(x)) for x in t.data]
    
    out = Tensor(
        out_data,
        requires_grad=t.requires_grad,
        _prev={t} if t.requires_grad else set(),
        _op='abs'
    )
    
    def _backward(grad):
        if t.requires_grad:
            # d/dx |x| = sign(x) = x / |x|
            if isinstance(grad, list) and isinstance(t.data[0], list):
                t_grad = [[float(grad[i][j]) * (1.0 if float(t.data[i][j]) >= 0 else -1.0)
                          for j in range(len(grad[i]))] for i in range(len(grad))]
            elif isinstance(grad, list):
                t_grad = [float(grad[i]) * (1.0 if float(t.data[i]) >= 0 else -1.0)
                         for i in range(len(grad))]
            else:
                t_grad = float(grad) * (1.0 if float(t.data[0][0]) >= 0 else -1.0)
            t.backward(t_grad)
    
    if out.requires_grad:
        out._backward_fn = _backward
    
    return out


def maximum(a: Tensor, b: Union[Tensor, float, int]) -> Tensor:
    """
    Element-wise maximum of two tensors.
    
    Args:
        a: First tensor
        b: Second tensor or scalar
    
    Returns:
        Tensor with element-wise maximum
    """
    a = _to_tensor(a)
    b = _to_tensor(b)
    
    # Use NumPy if available
    a_data = a.numpy if (HAS_NUMPY and a.numpy is not None) else a.data
    b_data = b.numpy if (HAS_NUMPY and b.numpy is not None) else b.data
    
    if HAS_NUMPY and isinstance(a_data, np.ndarray) and isinstance(b_data, np.ndarray):
        try:
            out_arr = np.maximum(a_data, b_data)
            out_data = out_arr.tolist()
        except (ValueError, TypeError):
            out_data = _broadcast_op(a.data, b.data, lambda x, y: max(float(x), float(y)))
    else:
        out_data = _broadcast_op(a_data, b_data, lambda x, y: max(float(x), float(y)))
    
    out = Tensor(
        out_data,
        requires_grad=a.requires_grad or b.requires_grad,
        _prev={a, b} if (a.requires_grad or b.requires_grad) else set(),
        _op='maximum'
    )
    
    def _backward(grad):
        if a.requires_grad:
            # Gradient is grad where a >= b, else 0
            if isinstance(grad, list) and isinstance(a.data[0], list):
                a_grad = [[float(grad[i][j]) if float(a.data[i][j]) >= float(b.data[i][j] if isinstance(b.data[0], list) else b.data[j] if j < len(b.data) else b.data[0]) else 0.0
                          for j in range(len(grad[i]))] for i in range(len(grad))]
            else:
                a_grad = grad
            a.backward(a_grad)
        if b.requires_grad:
            # Gradient is grad where b > a, else 0
            if isinstance(grad, list) and isinstance(b.data[0], list):
                b_grad = [[float(grad[i][j]) if float(b.data[i][j]) > float(a.data[i][j] if isinstance(a.data[0], list) else a.data[j] if j < len(a.data) else a.data[0]) else 0.0
                          for j in range(len(grad[i]))] for i in range(len(grad))]
            else:
                b_grad = grad
            b.backward(b_grad)
    
    if out.requires_grad:
        out._backward_fn = _backward
    
    return out


def tanh(t: Tensor) -> Tensor:
    """
    Hyperbolic tangent activation function.
    
    Args:
        t: Input tensor
    
    Returns:
        Tensor with tanh applied
    """
    t = _to_tensor(t)
    
    # Use NumPy if available
    if HAS_NUMPY:
        try:
            t_arr = _to_numpy(t.data)
            if t_arr is not None:
                out_arr = np.tanh(t_arr)
                out_data = out_arr.tolist()
            else:
                # Fallback
                import math
                def _tanh_op(x):
                    return math.tanh(float(x))
                if isinstance(t.data[0], list):
                    out_data = [[_tanh_op(t.data[i][j]) for j in range(len(t.data[i]))] 
                                for i in range(len(t.data))]
                else:
                    out_data = [_tanh_op(x) for x in t.data]
        except (ValueError, TypeError):
            # Fallback
            import math
            def _tanh_op(x):
                return math.tanh(float(x))
            if isinstance(t.data[0], list):
                out_data = [[_tanh_op(t.data[i][j]) for j in range(len(t.data[i]))] 
                            for i in range(len(t.data))]
            else:
                out_data = [_tanh_op(x) for x in t.data]
    else:
        # Pure Python
        import math
        def _tanh_op(x):
            return math.tanh(float(x))
        if isinstance(t.data[0], list):
            out_data = [[_tanh_op(t.data[i][j]) for j in range(len(t.data[i]))] 
                        for i in range(len(t.data))]
        else:
            out_data = [_tanh_op(x) for x in t.data]
    
    out = Tensor(
        out_data,
        requires_grad=t.requires_grad,
        _prev={t} if t.requires_grad else set(),
        _op='tanh'
    )
    
    def _backward(grad):
        if t.requires_grad:
            # d/dx tanh(x) = 1 - tanh^2(x)
            if isinstance(grad, list) and isinstance(t.data[0], list):
                t_grad = [[float(grad[i][j]) * (1.0 - float(out_data[i][j]) ** 2)
                          for j in range(len(grad[i]))] for i in range(len(grad))]
            elif isinstance(grad, list):
                t_grad = [float(grad[i]) * (1.0 - float(out_data[i]) ** 2)
                         for i in range(len(grad))]
            else:
                t_grad = float(grad) * (1.0 - float(out_data[0][0]) ** 2)
            t.backward(t_grad)
    
    if out.requires_grad:
        out._backward_fn = _backward
    
    return out

