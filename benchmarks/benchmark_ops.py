"""
Benchmark operations to measure performance improvements.

Compares NumPy-optimized operations vs pure Python fallback.
"""

import time
from quantml import Tensor

def benchmark_add(n=1000, size=100):
    """Benchmark addition operation."""
    print(f"\nBenchmarking add operation (n={n}, size={size})...")
    
    # Create test tensors
    a = Tensor([[1.0] * size] * size)
    b = Tensor([[2.0] * size] * size)
    
    # Warmup
    for _ in range(10):
        _ = a + b
    
    # Benchmark
    start = time.time()
    for _ in range(n):
        c = a + b
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.4f}s ({elapsed/n*1000:.4f}ms per op)")
    return elapsed


def benchmark_matmul(n=100, size=50):
    """Benchmark matrix multiplication."""
    print(f"\nBenchmarking matmul operation (n={n}, size={size})...")
    
    from quantml.ops import matmul
    
    a = Tensor([[1.0] * size] * size)
    b = Tensor([[2.0] * size] * size)
    
    # Warmup
    for _ in range(5):
        _ = matmul(a, b)
    
    # Benchmark
    start = time.time()
    for _ in range(n):
        c = matmul(a, b)
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.4f}s ({elapsed/n*1000:.4f}ms per op)")
    return elapsed


def benchmark_optimizer(n=100, size=100):
    """Benchmark optimizer step."""
    print(f"\nBenchmarking optimizer step (n={n}, size={size})...")
    
    from quantml.optim import SGD
    from quantml.models import Linear
    
    model = Linear(in_features=size, out_features=1, bias=True)
    optimizer = SGD(model.parameters(), lr=0.01)
    
    x = Tensor([[1.0] * size])
    y = Tensor([[1.0]])
    
    # Warmup
    for _ in range(5):
        pred = model.forward(x)
        loss = (pred - y) * (pred - y)
        loss.backward()
        optimizer.step()
        model.zero_grad()
    
    # Benchmark
    start = time.time()
    for _ in range(n):
        pred = model.forward(x)
        loss = (pred - y) * (pred - y)
        loss.backward()
        optimizer.step()
        model.zero_grad()
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.4f}s ({elapsed/n*1000:.4f}ms per step)")
    return elapsed


if __name__ == '__main__':
    print("=" * 60)
    print("QuantML Performance Benchmarks")
    print("=" * 60)
    
    benchmark_add(100, 50)
    benchmark_matmul(50, 30)
    benchmark_optimizer(50, 50)
    
    print("\n" + "=" * 60)
    print("Benchmarks completed!")
    print("=" * 60)

