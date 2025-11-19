"""
Profiling and performance measurement utilities.

This module provides tools for measuring latency and performance of operations,
essential for optimizing quant trading systems.
"""

import time
import functools
from typing import Callable, Any, Optional
from collections import defaultdict


def timing(func: Callable) -> Callable:
    """
    Decorator to measure execution time of a function.
    
    The decorator prints the execution time and returns the result.
    
    Args:
        func: Function to time
    
    Returns:
        Wrapped function that measures execution time
    
    Examples:
        >>> @timing
        >>> def my_function():
        >>>     # ... do work ...
        >>>     return result
        >>> result = my_function()  # Prints: "my_function took 0.123 seconds"
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        print(f"{func.__name__} took {elapsed:.6f} seconds")
        return result
    return wrapper


def measure_latency(func: Callable, *args, **kwargs) -> tuple:
    """
    Measure the latency of a function call.
    
    Args:
        func: Function to measure
        *args: Positional arguments for function
        **kwargs: Keyword arguments for function
    
    Returns:
        Tuple of (result, latency_in_seconds)
    
    Examples:
        >>> result, latency = measure_latency(my_function, arg1, arg2)
        >>> print(f"Latency: {latency * 1000:.2f} ms")
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    latency = end - start
    return result, latency


def measure_latency_microseconds(func: Callable, *args, **kwargs) -> tuple:
    """
    Measure latency in microseconds (useful for HFT).
    
    Args:
        func: Function to measure
        *args: Positional arguments
        **kwargs: Keyword arguments
    
    Returns:
        Tuple of (result, latency_in_microseconds)
    """
    result, latency_seconds = measure_latency(func, *args, **kwargs)
    latency_us = latency_seconds * 1_000_000
    return result, latency_us


class PerformanceProfiler:
    """
    Performance profiler for tracking multiple function calls.
    
    Tracks statistics like mean, min, max latency for multiple calls.
    
    Attributes:
        stats: Dictionary mapping function names to statistics
    
    Examples:
        >>> profiler = PerformanceProfiler()
        >>> for _ in range(100):
        >>>     profiler.record('my_func', measure_latency(my_function))
        >>> print(profiler.get_stats('my_func'))
    """
    
    def __init__(self):
        """Initialize profiler."""
        self.stats = defaultdict(list)
    
    def record(self, name: str, latency: float):
        """
        Record a latency measurement.
        
        Args:
            name: Function or operation name
            latency: Latency in seconds
        """
        self.stats[name].append(latency)
    
    def get_stats(self, name: str) -> Optional[dict]:
        """
        Get statistics for a function.
        
        Args:
            name: Function name
        
        Returns:
            Dictionary with mean, min, max, count, or None if no data
        """
        if name not in self.stats or len(self.stats[name]) == 0:
            return None
        
        latencies = self.stats[name]
        return {
            'mean': sum(latencies) / len(latencies),
            'min': min(latencies),
            'max': max(latencies),
            'count': len(latencies),
            'total': sum(latencies)
        }
    
    def print_stats(self, name: Optional[str] = None):
        """
        Print statistics for one or all functions.
        
        Args:
            name: Optional function name, or None for all
        """
        if name is not None:
            stats = self.get_stats(name)
            if stats:
                print(f"{name}:")
                print(f"  Mean: {stats['mean']*1000:.3f} ms")
                print(f"  Min:  {stats['min']*1000:.3f} ms")
                print(f"  Max:  {stats['max']*1000:.3f} ms")
                print(f"  Count: {stats['count']}")
        else:
            for func_name in self.stats:
                self.print_stats(func_name)
    
    def clear(self):
        """Clear all recorded statistics."""
        self.stats.clear()


def benchmark(func: Callable, n_iterations: int = 100, *args, **kwargs) -> dict:
    """
    Benchmark a function over multiple iterations.
    
    Args:
        func: Function to benchmark
        n_iterations: Number of iterations
        *args: Positional arguments
        **kwargs: Keyword arguments
    
    Returns:
        Dictionary with benchmark statistics
    
    Examples:
        >>> stats = benchmark(my_function, n_iterations=1000, arg1, arg2)
        >>> print(f"Mean latency: {stats['mean']*1000:.2f} ms")
    """
    latencies = []
    for _ in range(n_iterations):
        _, latency = measure_latency(func, *args, **kwargs)
        latencies.append(latency)
    
    return {
        'mean': sum(latencies) / len(latencies),
        'min': min(latencies),
        'max': max(latencies),
        'median': sorted(latencies)[len(latencies) // 2],
        'p95': sorted(latencies)[int(len(latencies) * 0.95)],
        'p99': sorted(latencies)[int(len(latencies) * 0.99)],
        'count': len(latencies)
    }

