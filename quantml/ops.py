"""
Core operations for Tensor objects.

This module provides all the fundamental operations that can be performed
on tensors, including arithmetic, linear algebra, reductions, and activations.
All operations support automatic differentiation.
"""

from typing import Union, Optional, List, Any, Callable
from quantml.tensor import Tensor
from quantml.autograd import backward

# Save reference to builtin sum before it gets shadowed by ops.sum
_builtin_sum = sum

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
            out_data = [[_builtin_sum(float(a_data[i][k]) * float(b_data[k][j])
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
        out_data = [[_builtin_sum(float(a_data[i][k]) * float(b_data[k][j])
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
                    a_grad = [[_builtin_sum(float(grad[i][k]) * float(b_T[k][j])
                                  for k in range(len(grad[0])))
                              for j in range(len(b_T[0]))]
                             for i in range(len(grad))]
            else:
                b_T = [[b_data[j][i] for j in range(len(b_data))]
                       for i in range(len(b_data[0]))]
                a_grad = [[_builtin_sum(float(grad[i][k]) * float(b_T[k][j])
                              for k in range(len(grad[0])))
                          for j in range(len(b_T[0]))]
                         for i in range(len(grad))]
            a.backward(a_grad)
        if b.requires_grad:
            # dL/db = a.T @ grad (for 2D), or sum_b(a[b].T @ grad[b]) for 3D batched
            if HAS_NUMPY:
                try:
                    a_arr = np.array(a_data, dtype=np.float64)
                    grad_arr = np.array(grad, dtype=np.float64)
                    if a_arr.ndim == 3 and grad_arr.ndim == 3:
                        b_grad_arr = np.einsum('bsi,bsj->ij', a_arr, grad_arr)
                    else:
                        b_grad_arr = np.dot(a_arr.T, grad_arr)
                    b_grad = b_grad_arr.tolist()
                except (ValueError, TypeError):
                    # Fallback
                    if isinstance(a_data[0], list) and isinstance(a_data[0][0], list):
                        # 3D: sum over batches of a[b].T @ grad[b]
                        d_in = len(a_data[0][0])
                        d_out = len(grad[0][0]) if isinstance(grad[0][0], list) else len(grad[0])
                        b_grad = [[0.0] * d_out for _ in range(d_in)]
                        for batch_idx in range(len(a_data)):
                            a_b = a_data[batch_idx]
                            g_b = grad[batch_idx] if isinstance(grad[0], list) else grad
                            seq_len = len(a_b)
                            for i in range(d_in):
                                for j in range(d_out):
                                    b_grad[i][j] += _builtin_sum(
                                        float(a_b[s][i]) * float(g_b[s][j])
                                        for s in range(seq_len))
                    else:
                        a_T = [[a_data[j][i] for j in range(len(a_data))]
                               for i in range(len(a_data[0]))]
                        b_grad = [[_builtin_sum(float(a_T[i][k]) * float(grad[k][j])
                                      for k in range(len(a_T[0])))
                                  for j in range(len(grad[0]))]
                                 for i in range(len(a_T))]
            else:
                if isinstance(a_data[0], list) and isinstance(a_data[0][0], list):
                    d_in = len(a_data[0][0])
                    d_out = len(grad[0][0]) if isinstance(grad[0][0], list) else len(grad[0])
                    b_grad = [[0.0] * d_out for _ in range(d_in)]
                    for batch_idx in range(len(a_data)):
                        a_b = a_data[batch_idx]
                        g_b = grad[batch_idx] if isinstance(grad[0], list) else grad
                        seq_len = len(a_b)
                        for i in range(d_in):
                            for j in range(d_out):
                                b_grad[i][j] += _builtin_sum(
                                    float(a_b[s][i]) * float(g_b[s][j])
                                    for s in range(seq_len))
                else:
                    a_T = [[a_data[j][i] for j in range(len(a_data))]
                           for i in range(len(a_data[0]))]
                    b_grad = [[_builtin_sum(float(a_T[i][k]) * float(grad[k][j])
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
            result = _builtin_sum(float(a_flat[i]) * float(b_flat[i]) for i in range(len(a_flat)))
    else:
        result = _builtin_sum(float(a_flat[i]) * float(b_flat[i]) for i in range(len(a_flat)))
    
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



def transpose(t: Tensor) -> Tensor:
    """
    Transpose a 2D tensor.
    
    Args:
        t: Input tensor
    
    Returns:
        Transposed tensor
    """
    t = _to_tensor(t)
    
    # Use NumPy if available
    if HAS_NUMPY:
        try:
            t_arr = t.numpy if (t.numpy is not None) else _to_numpy(t.data)
            if t_arr is not None:
                if t_arr.ndim == 1:
                    out_arr = t_arr.reshape(1, -1).T
                elif t_arr.ndim == 2:
                    out_arr = t_arr.T
                else:
                    # Generic transpose for higher dim? default to reverse all dims or just swap last two
                    out_arr = np.transpose(t_arr) # Reverses dims by default
                
                out = _create_tensor_from_numpy(
                    out_arr,
                    requires_grad=t.requires_grad,
                    _prev={t} if t.requires_grad else set(),
                    _op='transpose'
                )
            else:
                out_data = _transpose_pure_python(t.data)
                out = Tensor(
                    out_data,
                    requires_grad=t.requires_grad,
                    _prev={t} if t.requires_grad else set(),
                    _op='transpose'
                )
        except (ValueError, TypeError):
            out_data = _transpose_pure_python(t.data)
            out = Tensor(
                out_data,
                requires_grad=t.requires_grad,
                _prev={t} if t.requires_grad else set(),
                _op='transpose'
            )
    else:
        out_data = _transpose_pure_python(t.data)
        out = Tensor(
            out_data,
            requires_grad=t.requires_grad,
            _prev={t} if t.requires_grad else set(),
            _op='transpose'
        )
    
    def _backward(grad):
        if t.requires_grad:
            # Gradient of transpose is transpose of gradient
            from quantml.ops import transpose
            t.backward(transpose(Tensor(grad) if not isinstance(grad, Tensor) else grad).data)
            
    if out.requires_grad:
        out._backward_fn = _backward
    
    return out


def _transpose_pure_python(data):
    if not isinstance(data, list):
        return [[data]]
    if not isinstance(data[0], list):
        return [[x] for x in data]
    return [[data[j][i] for j in range(len(data))] for i in range(len(data[0]))]


def select(t: Tensor, index: int, dim: int = 0) -> Tensor:
    """
    Select a slice from a tensor along a dimension, preserving the autograd graph.

    For a 3D tensor of shape (B, S, D) with dim=0, returns tensor[index]
    of shape (S, D).

    Args:
        t: Input tensor (must be 3D for dim=0)
        index: Index to select along the dimension
        dim: Dimension to select along (currently only 0 is supported)

    Returns:
        Tensor with one fewer dimension
    """
    t = _to_tensor(t)

    if dim != 0:
        raise ValueError(f"select currently only supports dim=0, got dim={dim}")

    if HAS_NUMPY:
        try:
            t_arr = t.numpy if (t.numpy is not None) else np.array(t.data, dtype=np.float64)
            if t_arr is not None:
                out_arr = t_arr[index].copy()
                out = _create_tensor_from_numpy(
                    out_arr,
                    requires_grad=t.requires_grad,
                    _prev={t} if t.requires_grad else set(),
                    _op='select'
                )
            else:
                out = Tensor(
                    t.data[index],
                    requires_grad=t.requires_grad,
                    _prev={t} if t.requires_grad else set(),
                    _op='select'
                )
        except (ValueError, TypeError, IndexError):
            out = Tensor(
                t.data[index],
                requires_grad=t.requires_grad,
                _prev={t} if t.requires_grad else set(),
                _op='select'
            )
    else:
        out = Tensor(
            t.data[index],
            requires_grad=t.requires_grad,
            _prev={t} if t.requires_grad else set(),
            _op='select'
        )

    _index = index
    _t = t

    def _backward(grad):
        if _t.requires_grad:
            if HAS_NUMPY:
                try:
                    grad_arr = np.array(grad, dtype=np.float64) if not isinstance(grad, np.ndarray) else grad
                    t_arr = _t.numpy if (_t.numpy is not None) else np.array(_t.data, dtype=np.float64)
                    t_grad = np.zeros_like(t_arr)
                    t_grad[_index] = grad_arr
                    _t.backward(t_grad)
                    return
                except (ValueError, TypeError):
                    pass
            # Pure Python fallback
            t_data = _t.data
            batch_size = len(t_data)
            t_grad = []
            for b in range(batch_size):
                if b == _index:
                    if isinstance(grad, list):
                        t_grad.append(grad)
                    else:
                        t_grad.append(np.array(grad, dtype=np.float64).tolist() if HAS_NUMPY else grad)
                else:
                    row = t_data[b]
                    if isinstance(row, list) and len(row) > 0 and isinstance(row[0], list):
                        t_grad.append([[0.0] * len(row[0]) for _ in range(len(row))])
                    elif isinstance(row, list):
                        t_grad.append([0.0] * len(row))
                    else:
                        t_grad.append(0.0)
            _t.backward(t_grad)

    if out.requires_grad:
        out._backward_fn = _backward

    return out


def stack(tensors: List[Tensor], dim: int = 0) -> Tensor:
    """
    Stack a list of tensors along a new dimension, preserving the autograd graph.

    For a list of N tensors each of shape (S, D), returns a tensor of
    shape (N, S, D) when dim=0.

    Args:
        tensors: List of tensors to stack (must all have the same shape)
        dim: Dimension along which to stack (currently only 0 is supported)

    Returns:
        Stacked tensor with one additional dimension
    """
    if dim != 0:
        raise ValueError(f"stack currently only supports dim=0, got dim={dim}")

    if len(tensors) == 0:
        raise ValueError("stack requires at least one tensor")

    tensors = [_to_tensor(t) for t in tensors]
    any_requires_grad = any(t.requires_grad for t in tensors)

    if HAS_NUMPY:
        try:
            arrays = []
            for t in tensors:
                arr = t.numpy if (t.numpy is not None) else np.array(t.data, dtype=np.float64)
                if arr is None:
                    raise TypeError("Cannot convert to numpy")
                arrays.append(arr)
            out_arr = np.stack(arrays, axis=0)
            out = _create_tensor_from_numpy(
                out_arr,
                requires_grad=any_requires_grad,
                _prev=set(tensors) if any_requires_grad else set(),
                _op='stack'
            )
        except (ValueError, TypeError):
            out_data = [t.data for t in tensors]
            out = Tensor(
                out_data,
                requires_grad=any_requires_grad,
                _prev=set(tensors) if any_requires_grad else set(),
                _op='stack'
            )
    else:
        out_data = [t.data for t in tensors]
        out = Tensor(
            out_data,
            requires_grad=any_requires_grad,
            _prev=set(tensors) if any_requires_grad else set(),
            _op='stack'
        )

    _tensors = tensors

    def _backward(grad):
        for i, t in enumerate(_tensors):
            if t.requires_grad:
                if HAS_NUMPY:
                    try:
                        grad_arr = np.array(grad, dtype=np.float64) if not isinstance(grad, np.ndarray) else grad
                        t.backward(grad_arr[i])
                        continue
                    except (ValueError, TypeError):
                        pass
                if isinstance(grad, list):
                    t.backward(grad[i])
                else:
                    t.backward(grad)

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
                        total += _builtin_sum(float(x) for x in row)
                else:
                    total = _builtin_sum(float(x) for x in t.data)
                out_data = [[total]]
            elif axis == 0:
                if isinstance(t.data[0], list):
                    out_data = [[_builtin_sum(float(t.data[i][j]) for i in range(len(t.data)))
                                for j in range(len(t.data[0]))]]
                else:
                    out_data = [[_builtin_sum(float(x) for x in t.data)]]
            elif axis == 1:
                if isinstance(t.data[0], list):
                    out_data = [[_builtin_sum(float(row[j]) for j in range(len(row)))]
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
                    total += _builtin_sum(float(x) for x in row)
            else:
                total = _builtin_sum(float(x) for x in t.data)
            out_data = [[total]]
        elif axis == 0:
            if isinstance(t.data[0], list):
                out_data = [[_builtin_sum(float(t.data[i][j]) for i in range(len(t.data)))
                            for j in range(len(t.data[0]))]]
            else:
                out_data = [[_builtin_sum(float(x) for x in t.data)]]
        elif axis == 1:
            if isinstance(t.data[0], list):
                out_data = [[_builtin_sum(float(row[j]) for j in range(len(row)))]
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


def var(t: Tensor, axis: Optional[int] = None, unbiased: bool = True) -> Tensor:
    """
    Compute variance of tensor elements.
    
    Args:
        t: Input tensor
        axis: Axis to compute variance over
        unbiased: If True, use Bessel's correction (N-1)
    
    Returns:
        Tensor with variance
    """
    t = _to_tensor(t)
    m = mean(t, axis=axis)
    
    # Expand mean to match shape for subtraction
    # If axis is 0 (cols), mean is 1xM, needs to be NxM
    # If axis is 1 (rows), mean is Nx1, needs to be NxM
    # Note: simple sub() might not broadcast correctly if shapes don't align perfectly
    # For now relying on sub's broadcasting or manual reshaping if needed
    
    diff = sub(t, m)
    diff_sq = mul(diff, diff)
    
    # Sum of squared differences
    s = sum(diff_sq, axis=axis)
    
    # Divide by N or N-1
    if axis is None:
        count = len(t.data) * len(t.data[0]) if isinstance(t.data[0], list) else len(t.data)
    elif axis == 0:
        count = len(t.data) if isinstance(t.data[0], list) else 1
    else:
        count = len(t.data[0]) if isinstance(t.data[0], list) else len(t.data)
    
    denom = count - 1 if unbiased and count > 1 else count
    
    return div(s, float(denom))


def std(t: Tensor, axis: Optional[int] = None, unbiased: bool = True) -> Tensor:
    """Compute standard deviation of tensor elements."""
    v = var(t, axis=axis, unbiased=unbiased)
    return pow(v, 0.5)


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
            if HAS_NUMPY:
                try:
                    # Convert to numpy if not already
                    grad_arr = np.array(grad, dtype=np.float64) if not isinstance(grad, np.ndarray) else grad
                    t_arr = _to_numpy(t.data)
                    # Gradient is passed where t > 0
                    t_grad_arr = np.where(t_arr > 0, grad_arr, 0.0)
                    t.backward(t_grad_arr)
                    return
                except (ValueError, TypeError):
                    pass
            
            # Pure Python fallback
            if isinstance(grad, list) and isinstance(t.data[0], list):
                t_grad = [[float(grad[i][j]) if float(t.data[i][j]) > 0 else 0.0
                          for j in range(len(grad[i]))] for i in range(len(grad))]
            elif isinstance(grad, list):
                t_grad = [float(grad[i]) if float(t.data[i]) > 0 else 0.0
                         for i in range(len(grad))]
            else:
                t_grad = float(grad) if float(t.data[0][0] if isinstance(t.data, list) else t.data) > 0 else 0.0
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
            if HAS_NUMPY:
                try:
                    grad_arr = np.array(grad, dtype=np.float64) if not isinstance(grad, np.ndarray) else grad
                    out_arr = _to_numpy(out_data)
                    t_grad_arr = grad_arr * out_arr * (1.0 - out_arr)
                    t.backward(t_grad_arr)
                    return
                except (ValueError, TypeError):
                    pass
            
            # Pure Python fallback
            if isinstance(grad, list) and isinstance(t.data[0], list):
                t_grad = [[float(grad[i][j]) * float(out_data[i][j]) * (1.0 - float(out_data[i][j]))
                          for j in range(len(grad[i]))] for i in range(len(grad))]
            elif isinstance(grad, list):
                t_grad = [float(grad[i]) * float(out_data[i]) * (1.0 - float(out_data[i]))
                         for i in range(len(grad))]
            else:
                s = float(out_data[0][0]) if isinstance(out_data, list) and isinstance(out_data[0], list) else float(out_data[0]) if isinstance(out_data, list) else float(out_data)
                t_grad = float(grad) * s * (1.0 - s)
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
            if HAS_NUMPY:
                try:
                    grad_arr = np.array(grad, dtype=np.float64) if not isinstance(grad, np.ndarray) else grad
                    out_arr = _to_numpy(out_data)
                    t_grad_arr = grad_arr * (1.0 - out_arr ** 2)
                    t.backward(t_grad_arr)
                    return
                except (ValueError, TypeError):
                    pass
            
            # Pure Python fallback
            if isinstance(grad, list) and isinstance(t.data[0], list):
                t_grad = [[float(grad[i][j]) * (1.0 - float(out_data[i][j]) ** 2)
                          for j in range(len(grad[i]))] for i in range(len(grad))]
            elif isinstance(grad, list):
                t_grad = [float(grad[i]) * (1.0 - float(out_data[i]) ** 2)
                         for i in range(len(grad))]
            else:
                s = float(out_data[0][0]) if isinstance(out_data, list) and isinstance(out_data[0], list) else float(out_data[0]) if isinstance(out_data, list) else float(out_data)
                t_grad = float(grad) * (1.0 - s ** 2)
            t.backward(t_grad)
    
    if out.requires_grad:
        out._backward_fn = _backward
    
    return out


def softmax(t: Tensor, axis: int = -1) -> Tensor:
    """
    Softmax activation function: exp(x) / sum(exp(x))
    
    Normalizes values to probabilities that sum to 1.
    
    Args:
        t: Input tensor
        axis: Axis along which to compute softmax (default: -1, last axis)
    
    Returns:
        Tensor with softmax applied
    
    Examples:
        >>> x = Tensor([[1.0, 2.0, 3.0]])
        >>> y = softmax(x)  # Probabilities summing to 1
    """
    t = _to_tensor(t)
    
    # Use NumPy if available
    if HAS_NUMPY:
        try:
            t_arr = _to_numpy(t.data)
            if t_arr is not None:
                # Numerically stable softmax
                t_max = np.max(t_arr, axis=axis, keepdims=True)
                exp_arr = np.exp(t_arr - t_max)
                out_arr = exp_arr / np.sum(exp_arr, axis=axis, keepdims=True)
                out_data = out_arr.tolist()
            else:
                # Fallback to pure Python
                out_data = _softmax_pure_python(t.data, axis)
        except (ValueError, TypeError):
            out_data = _softmax_pure_python(t.data, axis)
    else:
        out_data = _softmax_pure_python(t.data, axis)
    
    out = Tensor(
        out_data,
        requires_grad=t.requires_grad,
        _prev={t} if t.requires_grad else set(),
        _op='softmax'
    )
    
    def _backward(grad):
        if t.requires_grad:
            # Jacobian-vector product for softmax
            if HAS_NUMPY:
                try:
                    grad_arr = np.array(grad, dtype=np.float64) if not isinstance(grad, np.ndarray) else grad
                    s = _to_numpy(out_data)
                    
                    # optimized implementation: s * (g - sum(s*g))
                    # sum(s*g) along axis
                    sg_dot = np.sum(s * grad_arr, axis=axis, keepdims=True)
                    t_grad_arr = s * (grad_arr - sg_dot)
                    t.backward(t_grad_arr)
                    return
                except (ValueError, TypeError):
                    pass

            # Pure Python fallback
            if isinstance(grad, list) and isinstance(t.data[0], list):
                t_grad = []
                for i in range(len(grad)):
                    row_grad = []
                    s = out_data[i]  # softmax output
                    g = grad[i]  # upstream gradient
                    # Sum over all k: s[j] * (delta_jk - s[k]) * g[k]
                    dot_sg = _builtin_sum(float(s[k]) * float(g[k]) for k in range(len(s)))
                    for j in range(len(g)):
                        row_grad.append(float(s[j]) * (float(g[j]) - dot_sg))
                    t_grad.append(row_grad)
            elif isinstance(grad, list):
                s = out_data
                g = grad
                dot_sg = _builtin_sum(float(s[k]) * float(g[k]) for k in range(len(s)))
                t_grad = [float(s[j]) * (float(g[j]) - dot_sg) for j in range(len(g))]
            else:
                t_grad = grad
            t.backward(t_grad)
    
    if out.requires_grad:
        out._backward_fn = _backward
    
    return out


def _softmax_pure_python(data, axis=-1):
    """Pure Python softmax implementation."""
    import math
    
    if isinstance(data[0], list):
        # 2D case
        if axis == -1 or axis == 1:
            result = []
            for row in data:
                max_val = max(float(x) for x in row)
                exp_vals = [math.exp(float(x) - max_val) for x in row]
                sum_exp = sum(exp_vals)
                result.append([e / sum_exp for e in exp_vals])
            return result
        else:  # axis == 0
            # Transpose, apply, transpose back
            cols = len(data[0])
            rows = len(data)
            result = [[0.0] * cols for _ in range(rows)]
            for j in range(cols):
                col = [float(data[i][j]) for i in range(rows)]
                max_val = max(col)
                exp_vals = [math.exp(x - max_val) for x in col]
                sum_exp = sum(exp_vals)
                for i in range(rows):
                    result[i][j] = exp_vals[i] / sum_exp
            return result
    else:
        # 1D case
        max_val = max(float(x) for x in data)
        exp_vals = [math.exp(float(x) - max_val) for x in data]
        sum_exp = sum(exp_vals)
        return [e / sum_exp for e in exp_vals]


def leaky_relu(t: Tensor, negative_slope: float = 0.01) -> Tensor:
    """
    Leaky ReLU activation function: max(0, x) + negative_slope * min(0, x)
    
    Args:
        t: Input tensor
        negative_slope: Slope for negative values (default: 0.01)
    
    Returns:
        Tensor with Leaky ReLU applied
    
    Examples:
        >>> x = Tensor([[-1.0, 0.0, 1.0]])
        >>> y = leaky_relu(x, negative_slope=0.1)  # [[-0.1, 0.0, 1.0]]
    """
    t = _to_tensor(t)
    
    if HAS_NUMPY:
        try:
            t_arr = _to_numpy(t.data)
            if t_arr is not None:
                out_arr = np.where(t_arr > 0, t_arr, negative_slope * t_arr)
                out_data = out_arr.tolist()
            else:
                out_data = _leaky_relu_pure_python(t.data, negative_slope)
        except (ValueError, TypeError):
            out_data = _leaky_relu_pure_python(t.data, negative_slope)
    else:
        out_data = _leaky_relu_pure_python(t.data, negative_slope)
    
    out = Tensor(
        out_data,
        requires_grad=t.requires_grad,
        _prev={t} if t.requires_grad else set(),
        _op='leaky_relu'
    )
    
    def _backward(grad):
        if t.requires_grad:
            if HAS_NUMPY:
                try:
                    grad_arr = np.array(grad, dtype=np.float64) if not isinstance(grad, np.ndarray) else grad
                    t_arr = _to_numpy(t.data)
                    t_grad_arr = np.where(t_arr > 0, grad_arr, grad_arr * negative_slope)
                    t.backward(t_grad_arr)
                    return
                except (ValueError, TypeError):
                    pass
            
            # Pure Python fallback
            if isinstance(grad, list) and isinstance(t.data[0], list):
                t_grad = [[float(grad[i][j]) if float(t.data[i][j]) > 0 
                          else float(grad[i][j]) * negative_slope
                          for j in range(len(grad[i]))] for i in range(len(grad))]
            elif isinstance(grad, list):
                t_grad = [float(grad[i]) if float(t.data[i]) > 0 
                         else float(grad[i]) * negative_slope
                         for i in range(len(grad))]
            else:
                val = float(t.data[0][0]) if isinstance(t.data, list) and isinstance(t.data[0], list) else float(t.data[0]) if isinstance(t.data, list) else float(t.data)
                g_val = float(grad)
                t_grad = g_val if val > 0 else g_val * negative_slope
            t.backward(t_grad)
    
    if out.requires_grad:
        out._backward_fn = _backward
    
    return out


def _leaky_relu_pure_python(data, negative_slope):
    """Pure Python Leaky ReLU implementation."""
    if isinstance(data[0], list):
        return [[float(x) if float(x) > 0 else negative_slope * float(x) 
                for x in row] for row in data]
    else:
        return [float(x) if float(x) > 0 else negative_slope * float(x) 
               for x in data]


def gelu(t: Tensor) -> Tensor:
    """
    Gaussian Error Linear Unit (GELU) activation function.
    
    GELU(x) = x * Φ(x) where Φ is the CDF of the standard normal distribution.
    Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    
    Popular in Transformer models.
    
    Args:
        t: Input tensor
    
    Returns:
        Tensor with GELU applied
    
    Examples:
        >>> x = Tensor([[0.0, 1.0, 2.0]])
        >>> y = gelu(x)
    """
    t = _to_tensor(t)
    import math
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    
    if HAS_NUMPY:
        try:
            t_arr = _to_numpy(t.data)
            if t_arr is not None:
                inner = sqrt_2_over_pi * (t_arr + 0.044715 * t_arr ** 3)
                out_arr = 0.5 * t_arr * (1.0 + np.tanh(inner))
                out_data = out_arr.tolist()
            else:
                out_data = _gelu_pure_python(t.data, sqrt_2_over_pi)
        except (ValueError, TypeError):
            out_data = _gelu_pure_python(t.data, sqrt_2_over_pi)
    else:
        out_data = _gelu_pure_python(t.data, sqrt_2_over_pi)
    
    out = Tensor(
        out_data,
        requires_grad=t.requires_grad,
        _prev={t} if t.requires_grad else set(),
        _op='gelu'
    )
    
    def _backward(grad):
        if t.requires_grad:
            if HAS_NUMPY:
                try:
                    grad_arr = np.array(grad, dtype=np.float64) if not isinstance(grad, np.ndarray) else grad
                    x = _to_numpy(t.data)
                    inner = sqrt_2_over_pi * (x + 0.044715 * x ** 3)
                    tanh_inner = np.tanh(inner)
                    sech2_inner = 1.0 - tanh_inner ** 2
                    d_inner = sqrt_2_over_pi * (1.0 + 0.134145 * x ** 2)
                    d_gelu = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2_inner * d_inner
                    t_grad_arr = grad_arr * d_gelu
                    t.backward(t_grad_arr)
                    return
                except (ValueError, TypeError):
                    pass
            
            # Pure Python fallback
            # GELU derivative (using approximation)
            # d/dx GELU(x) = 0.5 * (1 + tanh(inner)) + 0.5 * x * sech^2(inner) * d_inner/dx
            if isinstance(grad, list) and isinstance(t.data[0], list):
                t_grad = []
                for i in range(len(grad)):
                    row_grad = []
                    for j in range(len(grad[i])):
                        x = float(t.data[i][j])
                        inner = sqrt_2_over_pi * (x + 0.044715 * x ** 3)
                        tanh_inner = math.tanh(inner)
                        sech2_inner = 1.0 - tanh_inner ** 2
                        d_inner = sqrt_2_over_pi * (1.0 + 0.134145 * x ** 2)
                        d_gelu = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2_inner * d_inner
                        row_grad.append(float(grad[i][j]) * d_gelu)
                    t_grad.append(row_grad)
            elif isinstance(grad, list):
                t_grad = []
                for i in range(len(grad)):
                    x = float(t.data[i])
                    inner = sqrt_2_over_pi * (x + 0.044715 * x ** 3)
                    tanh_inner = math.tanh(inner)
                    sech2_inner = 1.0 - tanh_inner ** 2
                    d_inner = sqrt_2_over_pi * (1.0 + 0.134145 * x ** 2)
                    d_gelu = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2_inner * d_inner
                    t_grad.append(float(grad[i]) * d_gelu)
            else:
                x = float(t.data[0][0] if isinstance(t.data, list) and isinstance(t.data[0], list) else t.data[0] if isinstance(t.data, list) else t.data)
                g_val = float(grad)
                inner = sqrt_2_over_pi * (x + 0.044715 * x ** 3)
                tanh_inner = math.tanh(inner)
                sech2_inner = 1.0 - tanh_inner ** 2
                d_inner = sqrt_2_over_pi * (1.0 + 0.134145 * x ** 2)
                d_gelu = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2_inner * d_inner
                t_grad = g_val * d_gelu
            t.backward(t_grad)
    
    if out.requires_grad:
        out._backward_fn = _backward
    
    return out


def _gelu_pure_python(data, sqrt_2_over_pi):
    """Pure Python GELU implementation."""
    import math
    
    def _gelu_element(x):
        x = float(x)
        inner = sqrt_2_over_pi * (x + 0.044715 * x ** 3)
        return 0.5 * x * (1.0 + math.tanh(inner))
    
    if isinstance(data[0], list):
        return [[_gelu_element(x) for x in row] for row in data]
    else:
        return [_gelu_element(x) for x in data]


def swish(t: Tensor, beta: float = 1.0) -> Tensor:
    """
    Swish activation function: x * sigmoid(beta * x)
    
    Also known as SiLU (Sigmoid Linear Unit) when beta=1.
    Self-gated activation that sometimes outperforms ReLU.
    
    Args:
        t: Input tensor
        beta: Scaling parameter (default: 1.0)
    
    Returns:
        Tensor with Swish applied
    
    Examples:
        >>> x = Tensor([[0.0, 1.0, 2.0]])
        >>> y = swish(x)
    """
    t = _to_tensor(t)
    import math
    
    if HAS_NUMPY:
        try:
            t_arr = _to_numpy(t.data)
            if t_arr is not None:
                sig = 1.0 / (1.0 + np.exp(-beta * t_arr))
                out_arr = t_arr * sig
                out_data = out_arr.tolist()
            else:
                out_data = _swish_pure_python(t.data, beta)
        except (ValueError, TypeError):
            out_data = _swish_pure_python(t.data, beta)
    else:
        out_data = _swish_pure_python(t.data, beta)
    
    out = Tensor(
        out_data,
        requires_grad=t.requires_grad,
        _prev={t} if t.requires_grad else set(),
        _op='swish'
    )
    
    def _backward(grad):
        if t.requires_grad:
            # d/dx swish(x) = swish(x) + sigmoid(beta*x) * (1 - swish(x))
            # = sigmoid(beta*x) * (1 + beta*x*(1 - sigmoid(beta*x)))
            if isinstance(grad, list) and isinstance(t.data[0], list):
                t_grad = []
                for i in range(len(grad)):
                    row_grad = []
                    for j in range(len(grad[i])):
                        x = float(t.data[i][j])
                        sig = 1.0 / (1.0 + math.exp(-beta * x))
                        sw = x * sig
                        d_swish = sw + sig * (1.0 - sw)
                        row_grad.append(float(grad[i][j]) * d_swish)
                    t_grad.append(row_grad)
            elif isinstance(grad, list):
                t_grad = []
                for i in range(len(grad)):
                    x = float(t.data[i])
                    sig = 1.0 / (1.0 + math.exp(-beta * x))
                    sw = x * sig
                    d_swish = sw + sig * (1.0 - sw)
                    t_grad.append(float(grad[i]) * d_swish)
            else:
                x = float(t.data[0][0])
                sig = 1.0 / (1.0 + math.exp(-beta * x))
                sw = x * sig
                d_swish = sw + sig * (1.0 - sw)
                t_grad = float(grad) * d_swish
            t.backward(t_grad)
    
    if out.requires_grad:
        out._backward_fn = _backward
    
    return out


def _swish_pure_python(data, beta):
    """Pure Python Swish implementation."""
    import math
    
    def _swish_element(x):
        x = float(x)
        sig = 1.0 / (1.0 + math.exp(-beta * x))
        return x * sig
    
    if isinstance(data[0], list):
        return [[_swish_element(x) for x in row] for row in data]
    else:
        return [_swish_element(x) for x in data]


