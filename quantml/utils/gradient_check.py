"""
Gradient checking utilities for verifying autograd correctness.

This module provides numerical gradient checking to verify that the
analytical gradients computed by the autograd engine are correct.
"""

from typing import Callable, List, Tuple, Optional, Union
from quantml.tensor import Tensor


def check_gradients(
    func: Callable[[Tensor], Tensor],
    inputs: Tensor,
    eps: float = 1e-5,
    rtol: float = 1e-3,
    atol: float = 1e-5
) -> Tuple[bool, List[dict]]:
    """
    Compare analytical gradients to numerical gradients.
    
    Uses finite differences to compute numerical gradients and compares
    them to the analytical gradients from backpropagation.
    
    Args:
        func: Function that takes a tensor and returns a scalar tensor
        inputs: Input tensor to compute gradients for
        eps: Small value for finite differences (default: 1e-5)
        rtol: Relative tolerance for comparison (default: 1e-3)
        atol: Absolute tolerance for comparison (default: 1e-5)
    
    Returns:
        Tuple of (passed, details) where:
        - passed: True if all gradients match within tolerance
        - details: List of dicts with 'index', 'numerical', 'analytical', 'diff'
    
    Examples:
        >>> def f(x):
        ...     return ops.sum(ops.mul(x, x))  # sum(x^2)
        >>> x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        >>> passed, details = check_gradients(f, x)
        >>> print(passed)  # True
        >>> # Analytical gradient should be 2*x = [2.0, 4.0, 6.0]
    """
    # Create a copy with requires_grad=True
    x = Tensor(inputs.data, requires_grad=True)
    
    # Forward pass
    y = func(x)
    
    # Backward pass to get analytical gradients
    y.backward()
    analytical_grad = x.grad
    
    # Get flattened data
    if isinstance(x.data[0], list):
        flat_data = [val for row in x.data for val in row]
        shape_2d = True
        rows = len(x.data)
        cols = len(x.data[0])
    else:
        flat_data = list(x.data)
        shape_2d = False
        rows = 1
        cols = len(x.data)
    
    # Flatten analytical gradient
    if analytical_grad is not None:
        if isinstance(analytical_grad, list):
            if analytical_grad and isinstance(analytical_grad[0], list):
                flat_analytical = [val for row in analytical_grad for val in row]
            else:
                flat_analytical = list(analytical_grad)
        else:
            # NumPy array
            flat_analytical = analytical_grad.flatten().tolist()
    else:
        flat_analytical = [0.0] * len(flat_data)
    
    # Compute numerical gradients using central differences
    numerical_grads = []
    details = []
    all_passed = True
    
    for i in range(len(flat_data)):
        # Perturb positively
        data_plus = flat_data.copy()
        data_plus[i] += eps
        
        # Reshape to original shape
        if shape_2d:
            reshaped_plus = [data_plus[r*cols:(r+1)*cols] for r in range(rows)]
        else:
            reshaped_plus = data_plus
        
        x_plus = Tensor(reshaped_plus, requires_grad=False)
        y_plus = func(x_plus)
        
        # Extract scalar value
        y_plus_val = _get_scalar(y_plus)
        
        # Perturb negatively
        data_minus = flat_data.copy()
        data_minus[i] -= eps
        
        if shape_2d:
            reshaped_minus = [data_minus[r*cols:(r+1)*cols] for r in range(rows)]
        else:
            reshaped_minus = data_minus
        
        x_minus = Tensor(reshaped_minus, requires_grad=False)
        y_minus = func(x_minus)
        y_minus_val = _get_scalar(y_minus)
        
        # Central difference
        numerical = (y_plus_val - y_minus_val) / (2 * eps)
        numerical_grads.append(numerical)
        
        # Get analytical gradient for this index
        analytical = float(flat_analytical[i]) if i < len(flat_analytical) else 0.0
        
        # Compare
        diff = abs(numerical - analytical)
        rel_diff = diff / (abs(analytical) + atol) if abs(analytical) > atol else diff
        
        passed = diff <= atol or rel_diff <= rtol
        if not passed:
            all_passed = False
        
        # Compute 2D index if applicable
        if shape_2d:
            idx_tuple = (i // cols, i % cols)
        else:
            idx_tuple = (i,)
        
        details.append({
            'index': idx_tuple,
            'numerical': numerical,
            'analytical': analytical,
            'diff': diff,
            'rel_diff': rel_diff,
            'passed': passed
        })
    
    return all_passed, details


def _get_scalar(t: Tensor) -> float:
    """Extract scalar value from tensor."""
    data = t.data
    if isinstance(data, list):
        if len(data) == 0:
            return 0.0
        if isinstance(data[0], list):
            return float(data[0][0])
        return float(data[0])
    # NumPy array or scalar
    try:
        return float(data.flat[0])
    except (AttributeError, TypeError):
        return float(data)


def gradient_check_layer(
    layer,
    input_tensor: Tensor,
    eps: float = 1e-5,
    rtol: float = 1e-3,
    atol: float = 1e-5
) -> Tuple[bool, dict]:
    """
    Check gradients for a layer's parameters.
    
    Verifies that the gradients w.r.t. the layer's weights and biases
    are correctly computed.
    
    Args:
        layer: A layer with forward() method and parameters() method
        input_tensor: Input to pass through the layer
        eps: Small value for finite differences
        rtol: Relative tolerance
        atol: Absolute tolerance
    
    Returns:
        Tuple of (passed, results) where results contains gradient check
        details for each parameter.
    
    Examples:
        >>> from quantml.models import Linear
        >>> from quantml import ops
        >>> layer = Linear(3, 2)
        >>> x = Tensor([[1.0, 2.0, 3.0]])
        >>> passed, results = gradient_check_layer(layer, x)
    """
    from quantml import ops
    
    results = {}
    all_passed = True
    
    # Get parameters
    params = layer.parameters()
    
    for param_idx, param in enumerate(params):
        param_name = f"param_{param_idx}"
        
        # Define function that uses this parameter
        def func_for_param(p):
            # Temporarily replace parameter data
            old_data = param.data
            param._data = p.data if hasattr(p, 'data') else p
            if hasattr(param, '_np_array'):
                param._np_array = None  # Clear cached numpy
            
            # Forward pass
            out = layer.forward(input_tensor)
            
            # Sum to get scalar
            scalar_out = ops.sum(out)
            
            # Restore
            param._data = old_data
            
            return scalar_out
        
        # Check gradients for this parameter
        passed, details = check_gradients(func_for_param, param, eps, rtol, atol)
        
        if not passed:
            all_passed = False
        
        results[param_name] = {
            'passed': passed,
            'details': details
        }
    
    return all_passed, results


def print_gradient_check_results(passed: bool, details: List[dict]) -> None:
    """
    Print formatted gradient check results.
    
    Args:
        passed: Overall pass/fail status
        details: List of detail dicts from check_gradients
    """
    print("=" * 60)
    print("GRADIENT CHECK RESULTS")
    print("=" * 60)
    print(f"Overall: {'PASSED ✓' if passed else 'FAILED ✗'}")
    print("-" * 60)
    print(f"{'Index':<15} {'Numerical':<15} {'Analytical':<15} {'Diff':<12} {'Status':<8}")
    print("-" * 60)
    
    for d in details:
        status = "✓" if d['passed'] else "✗"
        print(f"{str(d['index']):<15} {d['numerical']:<15.6f} {d['analytical']:<15.6f} {d['diff']:<12.2e} {status:<8}")
    
    print("=" * 60)


def quick_gradient_check(func: Callable, inputs: Tensor) -> bool:
    """
    Quick gradient check that returns True if gradients are correct.
    
    Args:
        func: Function to check
        inputs: Input tensor
    
    Returns:
        True if gradients pass, False otherwise
    """
    passed, _ = check_gradients(func, inputs)
    return passed
