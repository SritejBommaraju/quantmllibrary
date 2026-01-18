"""
Comprehensive benchmark comparing NumPy-optimized vs pure Python performance.

This script measures actual speedup numbers for resume/paper.
"""

import time
import sys
from quantml import Tensor
from quantml.models import Linear
from quantml.optim import SGD, Adam
from quantml.training import QuantTrainer
from quantml.training.losses import mse_loss

# Try to import NumPy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("NumPy not available - running pure Python benchmarks only")
    sys.exit(1)


def benchmark_tensor_operations():
    """Benchmark tensor operations with NumPy."""
    print("\n" + "=" * 70)
    print("Tensor Operations Benchmark")
    print("=" * 70)
    
    sizes = [100, 500, 1000]
    n_iterations = 1000
    
    for size in sizes:
        print(f"\nSize: {size}x{size}")
        
        # Create tensors
        a = Tensor([[1.0] * size] * size)
        b = Tensor([[2.0] * size] * size)
        
        # Warmup
        for _ in range(10):
            _ = a + b
        
        # Benchmark addition
        start = time.perf_counter()
        for _ in range(n_iterations):
            c = a + b
        elapsed = time.perf_counter() - start
        
        ops_per_sec = n_iterations / elapsed
        ms_per_op = (elapsed / n_iterations) * 1000
        
        print(f"  Addition: {elapsed:.4f}s total, {ms_per_op:.4f}ms per op, {ops_per_sec:.0f} ops/sec")
        
        # Benchmark multiplication
        from quantml.ops import mul
        start = time.perf_counter()
        for _ in range(n_iterations):
            c = mul(a, b)
        elapsed = time.perf_counter() - start
        
        ops_per_sec = n_iterations / elapsed
        ms_per_op = (elapsed / n_iterations) * 1000
        
        print(f"  Multiplication: {elapsed:.4f}s total, {ms_per_op:.4f}ms per op, {ops_per_sec:.0f} ops/sec")


def benchmark_matrix_multiplication():
    """Benchmark matrix multiplication."""
    print("\n" + "=" * 70)
    print("Matrix Multiplication Benchmark")
    print("=" * 70)
    
    sizes = [(50, 50), (100, 100), (200, 200)]
    n_iterations = 100
    
    for m, n in sizes:
        print(f"\nSize: {m}x{n} @ {n}x{n}")
        
        from quantml.ops import matmul
        
        a = Tensor([[1.0] * n] * m)
        b = Tensor([[2.0] * n] * n)
        
        # Warmup
        for _ in range(5):
            _ = matmul(a, b)
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(n_iterations):
            c = matmul(a, b)
        elapsed = time.perf_counter() - start
        
        ops_per_sec = n_iterations / elapsed
        ms_per_op = (elapsed / n_iterations) * 1000
        
        print(f"  MatMul: {elapsed:.4f}s total, {ms_per_op:.4f}ms per op, {ops_per_sec:.0f} ops/sec")


def benchmark_training_loop():
    """Benchmark full training loop."""
    print("\n" + "=" * 70)
    print("Training Loop Benchmark")
    print("=" * 70)
    
    configs = [
        (50, 32, 10),   # 50 features, batch 32, 10 epochs
        (100, 32, 10),  # 100 features, batch 32, 10 epochs
        (200, 32, 5),   # 200 features, batch 32, 5 epochs
    ]
    
    for n_features, batch_size, n_epochs in configs:
        print(f"\nConfig: {n_features} features, batch={batch_size}, epochs={n_epochs}")
        
        # Create model
        model = Linear(in_features=n_features, out_features=1, bias=True)
        optimizer = Adam(model.parameters(), lr=0.001)
        trainer = QuantTrainer(model, optimizer, mse_loss)
        
        # Generate data
        X = [Tensor([[1.0] * n_features]) for _ in range(batch_size)]
        y = [Tensor([[1.0]]) for _ in range(batch_size)]
        
        # Warmup
        for _ in range(2):
            for x, target in zip(X, y):
                trainer.train_step(x, target)
        
        # Benchmark
        start = time.perf_counter()
        for epoch in range(n_epochs):
            for x, target in zip(X, y):
                trainer.train_step(x, target)
        elapsed = time.perf_counter() - start
        
        total_steps = n_epochs * batch_size
        ms_per_step = (elapsed / total_steps) * 1000
        steps_per_sec = total_steps / elapsed
        
        print(f"  Total: {elapsed:.4f}s")
        print(f"  Per step: {ms_per_step:.4f}ms")
        print(f"  Throughput: {steps_per_sec:.0f} steps/sec")


def benchmark_inference():
    """Benchmark inference latency."""
    print("\n" + "=" * 70)
    print("Inference Latency Benchmark")
    print("=" * 70)
    
    configs = [
        (50, 1000),
        (100, 1000),
        (200, 500),
    ]
    
    for n_features, n_iterations in configs:
        print(f"\nConfig: {n_features} features, {n_iterations} iterations")
        
        model = Linear(in_features=n_features, out_features=1, bias=True)
        x = Tensor([[1.0] * n_features])
        
        # Warmup
        for _ in range(10):
            _ = model.forward(x)
        
        # Benchmark
        latencies = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            _ = model.forward(x)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
        
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]
        
        print(f"  Avg: {avg_latency:.4f}ms")
        print(f"  Min: {min_latency:.4f}ms")
        print(f"  Max: {max_latency:.4f}ms")
        print(f"  P95: {p95_latency:.4f}ms")
        print(f"  P99: {p99_latency:.4f}ms")


def benchmark_memory_usage():
    """Estimate memory usage."""
    print("\n" + "=" * 70)
    print("Memory Usage Estimate")
    print("=" * 70)
    
    try:
        import sys
        
        # Create model
        model = Linear(in_features=100, out_features=1, bias=True)
        
        # Estimate parameter count
        param_count = sum(len(p.data) if hasattr(p, 'data') else 0 for p in model.parameters())
        
        # Estimate memory (assuming float32 = 4 bytes)
        param_memory_mb = (param_count * 4) / (1024 * 1024)
        
        print(f"\nModel: Linear(100 -> 1)")
        print(f"  Parameters: ~{param_count}")
        print(f"  Estimated memory: ~{param_memory_mb:.4f} MB")
        
    except Exception as e:
        print(f"  Could not estimate memory: {e}")


def main():
    """Run all benchmarks."""
    print("=" * 70)
    print("QuantML Comprehensive Performance Benchmarks")
    print("=" * 70)
    print(f"NumPy available: {HAS_NUMPY}")
    print(f"Python version: {sys.version}")
    
    benchmark_tensor_operations()
    benchmark_matrix_multiplication()
    benchmark_training_loop()
    benchmark_inference()
    benchmark_memory_usage()
    
    print("\n" + "=" * 70)
    print("All benchmarks completed!")
    print("=" * 70)


if __name__ == '__main__':
    main()

