"""
Memory profiling benchmarks for QuantML.
"""

import time
import sys
import os
import psutil
from typing import Callable, List, Dict
from quantml import Tensor
from quantml import ops
from quantml.models import Linear, LSTM, MLP


class MemoryProfiler:
    """Simple memory profiler."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.baseline = self._get_memory()
    
    def _get_memory(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def measure(self, func: Callable, *args, **kwargs) -> Dict[str, float]:
        """Measure memory usage and execution time of a function."""
        # Force garbage collection
        import gc
        gc.collect()
        
        start_mem = self._get_memory()
        start_time = time.time()
        
        # Run function
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_mem = self._get_memory()
        
        # Calculate peak memory roughly
        peak_mem = end_mem  # Approximation
        
        return {
            'time_sec': end_time - start_time,
            'start_mem_mb': start_mem,
            'end_mem_mb': end_mem,
            'diff_mem_mb': end_mem - start_mem,
            'result': result
        }


def benchmark_tensor_ops(size: int = 1000):
    """Benchmark memory for basic tensor operations."""
    print(f"\nBenchmarking Tensor Ops (Size: {size}x{size})...")
    
    profiler = MemoryProfiler()
    
    def create_large_tensor():
        return Tensor([[1.0] * size for _ in range(size)])
    
    stats = profiler.measure(create_large_tensor)
    print(f"Creation: {stats['diff_mem_mb']:.2f} MB, {stats['time_sec']:.4f} sec")
    
    t1 = stats['result']
    t2 = Tensor([[2.0] * size for _ in range(size)])
    
    def add_op():
        return ops.add(t1, t2)
    
    stats = profiler.measure(add_op)
    print(f"Addition: {stats['diff_mem_mb']:.2f} MB, {stats['time_sec']:.4f} sec")
    
    def matmul_op():
        # Smaller usage for matmul to check overhead
        t_small = Tensor([[1.0] * 100 for _ in range(100)])
        return ops.matmul(t_small, t_small)
    
    stats = profiler.measure(matmul_op)
    print(f"MatMul (100x100): {stats['diff_mem_mb']:.2f} MB, {stats['time_sec']:.4f} sec")


def benchmark_model_training(steps: int = 10):
    """Benchmark memory during training simulation."""
    print(f"\nBenchmarking Model Training ({steps} steps)...")
    
    input_size = 100
    hidden_size = 200
    batch_size = 32
    
    # MLP
    print("Initializing MLP...")
    profiler = MemoryProfiler()
    
    def init_model():
        return MLP([input_size, hidden_size, hidden_size, 1])
    
    stats = profiler.measure(init_model)
    model = stats['result']
    print(f"Model Init: {stats['diff_mem_mb']:.2f} MB")
    
    # Data
    x = Tensor([[0.5] * input_size for _ in range(batch_size)])
    target = Tensor([[1.0] for _ in range(batch_size)])
    
    def train_loop():
        total_loss = 0
        for _ in range(steps):
            model.zero_grad()
            y = model.forward(x)
            loss = ops.mean(ops.mul(ops.sub(y, target), ops.sub(y, target)))
            loss.backward()
            total_loss += float(loss.data[0][0] if isinstance(loss.data[0], list) else loss.data[0])
        return total_loss
    
    stats = profiler.measure(train_loop)
    print(f"Training Loop: {stats['diff_mem_mb']:.2f} MB, {stats['time_sec']:.4f} sec")


if __name__ == "__main__":
    print("=" * 50)
    print("QuantML Memory Benchmarks")
    print("=" * 50)
    
    try:
        benchmark_tensor_ops()
        benchmark_model_training()
    except Exception as e:
        print(f"Benchmark failed: {e}")
    
    print("=" * 50)
