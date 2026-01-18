
import time
import math
from quantml import Tensor, ops

def measure_tensor_ops_per_second():
    """
    Measure pure overhead-free tensor operations per second.
    This demonstrates the efficiency of the dispatch engine.
    """
    N = 100_000
    a = Tensor([1.0], requires_grad=False)
    b = Tensor([2.0], requires_grad=False)
    
    start = time.time()
    for _ in range(N):
        # We use a very simple op to measure overhead
        ops.add(a, b)
    end = time.time()
    
    ops_per_sec = N / (end - start)
    return ops_per_sec

def verify_gradient_precision():
    """
    Check the maximum precision we can achieve.
    """
    x_val = 2.0
    x = Tensor([[x_val]], requires_grad=True)
    # f(x) = x^3, f'(x) = 3x^2. at x=2, f'(2) = 12.0
    y = ops.pow(x, 3)
    y.backward()
    
    grad = x.grad[0][0] if isinstance(x.grad, list) else x.grad
    error = abs(grad - 12.0)
    relative_error = error / 12.0
    return relative_error

if __name__ == "__main__":
    tops = measure_tensor_ops_per_second()
    precision = verify_gradient_precision()
    
    print(f"TOPS:{int(tops)}")
    # Handle potentially numpy float
    if hasattr(precision, 'item'):
        precision = precision.item()
    print(f"GRAD_PRECISION:{precision:.2e}")
