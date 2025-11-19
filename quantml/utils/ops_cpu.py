"""
CPU-optimized operations for QuantML.

This module provides low-level CPU-optimized implementations of operations.
It uses NumPy when available for better performance, with pure Python fallbacks.
"""

from typing import List, Union, Any
import math

# Try to import NumPy for optimized operations
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def add_cpu(a: Union[List, Any], b: Union[List, Any]) -> List:
    """
    CPU-optimized addition.
    
    Uses NumPy if available, otherwise falls back to pure Python.
    
    Args:
        a: First operand
        b: Second operand
    
    Returns:
        Result of addition
    """
    if HAS_NUMPY:
        a_arr = np.array(a)
        b_arr = np.array(b)
        result = (a_arr + b_arr).tolist()
        return result
    else:
        return _add_pure_python(a, b)


def mul_cpu(a: Union[List, Any], b: Union[List, Any]) -> List:
    """CPU-optimized multiplication."""
    if HAS_NUMPY:
        a_arr = np.array(a)
        b_arr = np.array(b)
        result = (a_arr * b_arr).tolist()
        return result
    else:
        return _mul_pure_python(a, b)


def matmul_cpu(a: Union[List, Any], b: Union[List, Any]) -> List:
    """CPU-optimized matrix multiplication."""
    if HAS_NUMPY:
        a_arr = np.array(a)
        b_arr = np.array(b)
        result = np.matmul(a_arr, b_arr).tolist()
        return result
    else:
        return _matmul_pure_python(a, b)


def sum_cpu(a: Union[List, Any], axis: int = None) -> Union[float, List]:
    """CPU-optimized sum."""
    if HAS_NUMPY:
        a_arr = np.array(a)
        result = np.sum(a_arr, axis=axis)
        if isinstance(result, np.ndarray):
            return result.tolist()
        return float(result)
    else:
        return _sum_pure_python(a, axis)


def mean_cpu(a: Union[List, Any], axis: int = None) -> Union[float, List]:
    """CPU-optimized mean."""
    if HAS_NUMPY:
        a_arr = np.array(a)
        result = np.mean(a_arr, axis=axis)
        if isinstance(result, np.ndarray):
            return result.tolist()
        return float(result)
    else:
        return _mean_pure_python(a, axis)


def std_cpu(a: Union[List, Any], axis: int = None) -> Union[float, List]:
    """CPU-optimized standard deviation."""
    if HAS_NUMPY:
        a_arr = np.array(a)
        result = np.std(a_arr, axis=axis)
        if isinstance(result, np.ndarray):
            return result.tolist()
        return float(result)
    else:
        return _std_pure_python(a, axis)


# Pure Python implementations (fallbacks)

def _add_pure_python(a: Any, b: Any) -> List:
    """Pure Python addition."""
    if isinstance(a, list) and isinstance(b, list):
        if isinstance(a[0], list) and isinstance(b[0], list):
            return [[a[i][j] + b[i][j] for j in range(len(a[i]))] 
                    for i in range(len(a))]
        else:
            return [a[i] + b[i] for i in range(len(a))]
    elif isinstance(a, list):
        if isinstance(a[0], list):
            return [[a[i][j] + b for j in range(len(a[i]))] 
                    for i in range(len(a))]
        return [a[i] + b for i in range(len(a))]
    elif isinstance(b, list):
        if isinstance(b[0], list):
            return [[a + b[i][j] for j in range(len(b[i]))] 
                    for i in range(len(b))]
        return [a + b[i] for i in range(len(b))]
    else:
        return [[a + b]]


def _mul_pure_python(a: Any, b: Any) -> List:
    """Pure Python multiplication."""
    if isinstance(a, list) and isinstance(b, list):
        if isinstance(a[0], list) and isinstance(b[0], list):
            return [[a[i][j] * b[i][j] for j in range(len(a[i]))] 
                    for i in range(len(a))]
        else:
            return [a[i] * b[i] for i in range(len(a))]
    elif isinstance(a, list):
        if isinstance(a[0], list):
            return [[a[i][j] * b for j in range(len(a[i]))] 
                    for i in range(len(a))]
        return [a[i] * b for i in range(len(a))]
    elif isinstance(b, list):
        if isinstance(b[0], list):
            return [[a * b[i][j] for j in range(len(b[i]))] 
                    for i in range(len(b))]
        return [a * b[i] for i in range(len(b))]
    else:
        return [[a * b]]


def _matmul_pure_python(a: List, b: List) -> List:
    """Pure Python matrix multiplication."""
    # Ensure 2D
    a_2d = a if isinstance(a[0], list) else [a]
    b_2d = b if isinstance(b[0], list) else [b]
    
    m, n = len(a_2d), len(a_2d[0])
    n2, p = len(b_2d), len(b_2d[0])
    
    if n != n2:
        raise ValueError(f"Incompatible dimensions: {m}x{n} x {n2}x{p}")
    
    result = [[sum(a_2d[i][k] * b_2d[k][j] for k in range(n)) 
              for j in range(p)] 
             for i in range(m)]
    return result


def _sum_pure_python(a: Any, axis: int = None) -> Union[float, List]:
    """Pure Python sum."""
    if axis is None:
        if isinstance(a, list):
            if isinstance(a[0], list):
                return sum(sum(row) for row in a)
            return sum(a)
        return float(a)
    elif axis == 0:
        if isinstance(a[0], list):
            return [sum(a[i][j] for i in range(len(a))) 
                   for j in range(len(a[0]))]
        return [sum(a)]
    elif axis == 1:
        if isinstance(a[0], list):
            return [sum(row) for row in a]
        return a
    else:
        raise ValueError(f"Invalid axis: {axis}")


def _mean_pure_python(a: Any, axis: int = None) -> Union[float, List]:
    """Pure Python mean."""
    s = _sum_pure_python(a, axis)
    if axis is None:
        count = 1.0
        if isinstance(a, list):
            if isinstance(a[0], list):
                count = len(a) * len(a[0])
            else:
                count = len(a)
        return s / count if count > 0 else 0.0
    elif axis == 0:
        count = len(a) if isinstance(a[0], list) else 1.0
        if isinstance(s, list):
            return [x / count for x in s]
        return s / count
    else:
        count = len(a[0]) if isinstance(a[0], list) else len(a)
        if isinstance(s, list):
            return [x / count for x in s]
        return s / count


def _std_pure_python(a: Any, axis: int = None) -> Union[float, List]:
    """Pure Python standard deviation."""
    m = _mean_pure_python(a, axis)
    
    # Compute variance
    if axis is None:
        if isinstance(a, list):
            if isinstance(a[0], list):
                diff_sq = sum((a[i][j] - m) ** 2 
                             for i in range(len(a)) 
                             for j in range(len(a[i])))
                count = len(a) * len(a[0])
            else:
                diff_sq = sum((a[i] - m) ** 2 for i in range(len(a)))
                count = len(a)
        else:
            diff_sq = (a - m) ** 2
            count = 1.0
        var = diff_sq / count if count > 0 else 0.0
        return math.sqrt(var)
    else:
        # For axis-specific std, use simplified approach
        # Full implementation would be more complex
        return m  # Placeholder - full implementation needed

