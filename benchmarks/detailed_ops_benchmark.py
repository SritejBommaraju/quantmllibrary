
import time
import numpy as np
from quantml import Tensor, ops

def benchmark_scalar_dispatch():
    """Measure pure framework overhead (ops/sec)."""
    N = 100_000
    a = Tensor([1.0], requires_grad=False)
    b = Tensor([2.0], requires_grad=False)
    
    start = time.time()
    for _ in range(N):
        ops.add(a, b)
    end = time.time()
    return N / (end - start)

def benchmark_vectorized_flops():
    """Measure element-wise throughput (FLOPS)."""
    size = 1024 * 1024  # 1M elements
    a = Tensor(np.random.randn(size).tolist(), requires_grad=False)
    b = Tensor(np.random.randn(size).tolist(), requires_grad=False)
    
    N = 100
    start = time.time()
    for _ in range(N):
        ops.add(a, b)
    end = time.time()
    
    # 1 add = 1 FLOP per element
    total_flops = size * N
    flops = total_flops / (end - start)
    return flops

def benchmark_matmul_flops():
    """Measure matrix multiplication throughput (FLOPS)."""
    size = 256
    # Create 256x256 matrices
    data = np.random.randn(size, size).tolist()
    a = Tensor(data, requires_grad=False)
    b = Tensor(data, requires_grad=False)
    
    N = 20
    start = time.time()
    for _ in range(N):
        ops.matmul(a, b)
    end = time.time()
    
    # Matmul FLOPS: 2 * n^3 (roughly)
    total_flops = 2 * (size ** 3) * N
    flops = total_flops / (end - start)
    return flops

def benchmark_activation_dispatch():
    """Measure scalar activation dispatch."""
    N = 100_000
    a = Tensor([0.5], requires_grad=False)
    
    start = time.time()
    for _ in range(N):
        ops.tanh(a)
    end = time.time()
    return N / (end - start)

if __name__ == "__main__":
    scalar_ops = benchmark_scalar_dispatch()
    act_ops = benchmark_activation_dispatch()
    vec_flops = benchmark_vectorized_flops()
    matmul_flops = benchmark_matmul_flops()
    
    print(f"SCALAR_DISPATCH_OPS:{int(scalar_ops)}")
    print(f"ACTIVATION_DISPATCH_OPS:{int(act_ops)}")
    print(f"VECTORIZED_MFLOPS:{vec_flops/1e6:.2f}")
    print(f"MATMUL_GFLOPS:{matmul_flops/1e9:.2f}")
