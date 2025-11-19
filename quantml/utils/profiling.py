"""
Profiling and performance measurement utilities.

This module provides tools for measuring latency and performance of operations,
essential for optimizing quant trading systems.
"""

import time
import functools
import sys
from typing import Callable, Any, Optional, Dict, List
from collections import defaultdict

# Try to import psutil for memory tracking
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None


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


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage.
    
    Returns:
        Dictionary with memory usage in MB
    """
    if HAS_PSUTIL:
        process = psutil.Process()
        mem_info = process.memory_info()
        return {
            'rss_mb': mem_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': mem_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent()
        }
    else:
        # Fallback using sys
        try:
            import resource
            mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            return {'rss_mb': mem_mb, 'vms_mb': 0.0, 'percent': 0.0}
        except (ImportError, AttributeError):
            return {'rss_mb': 0.0, 'vms_mb': 0.0, 'percent': 0.0}


class PipelineProfiler:
    """
    Profiler for entire pipeline (data loading, feature generation, training).
    
    Tracks timing and memory usage for each stage.
    """
    
    def __init__(self):
        """Initialize pipeline profiler."""
        self.stages = {}
        self.memory_snapshots = []
    
    def start_stage(self, stage_name: str):
        """Start timing a stage."""
        self.stages[stage_name] = {
            'start_time': time.perf_counter(),
            'start_memory': get_memory_usage()
        }
    
    def end_stage(self, stage_name: str):
        """End timing a stage."""
        if stage_name not in self.stages:
            return
        
        end_time = time.perf_counter()
        end_memory = get_memory_usage()
        
        start_info = self.stages[stage_name]
        duration = end_time - start_info['start_time']
        
        memory_delta = {
            'rss_mb': end_memory['rss_mb'] - start_info['start_memory']['rss_mb'],
            'vms_mb': end_memory['vms_mb'] - start_info['start_memory']['vms_mb']
        }
        
        self.stages[stage_name] = {
            'duration': duration,
            'memory_delta': memory_delta,
            'peak_memory': end_memory
        }
    
    def get_report(self) -> Dict[str, Any]:
        """Get profiling report."""
        total_time = sum(s.get('duration', 0) for s in self.stages.values())
        
        return {
            'stages': self.stages,
            'total_time': total_time,
            'bottleneck': self._identify_bottleneck()
        }
    
    def _identify_bottleneck(self) -> Optional[str]:
        """Identify the slowest stage."""
        if not self.stages:
            return None
        
        max_duration = 0
        bottleneck = None
        
        for stage_name, info in self.stages.items():
            duration = info.get('duration', 0)
            if duration > max_duration:
                max_duration = duration
                bottleneck = stage_name
        
        return bottleneck
    
    def print_report(self):
        """Print profiling report."""
        report = self.get_report()
        
        print("\n" + "=" * 70)
        print("Pipeline Profiling Report")
        print("=" * 70)
        
        for stage_name, info in report['stages'].items():
            duration = info.get('duration', 0)
            memory_delta = info.get('memory_delta', {})
            
            print(f"\n{stage_name}:")
            print(f"  Duration: {duration:.4f} seconds ({duration*1000:.2f} ms)")
            print(f"  Memory Delta: {memory_delta.get('rss_mb', 0):.2f} MB")
        
        print(f"\nTotal Time: {report['total_time']:.4f} seconds")
        if report['bottleneck']:
            print(f"Bottleneck: {report['bottleneck']} ({report['stages'][report['bottleneck']]['duration']:.4f}s)")
        print("=" * 70 + "\n")


def profile_training_loop(
    trainer,
    n_epochs: int,
    log_interval: int = 10
) -> Dict[str, Any]:
    """
    Profile a training loop.
    
    Args:
        trainer: Trainer instance
        n_epochs: Number of epochs
        log_interval: Log every N epochs
    
    Returns:
        Training profile dictionary
    """
    profiler = PipelineProfiler()
    epoch_times = []
    
    profiler.start_stage('total_training')
    
    for epoch in range(n_epochs):
        epoch_start = time.perf_counter()
        
        # Train epoch (assuming trainer has train_epoch method)
        if hasattr(trainer, 'train_epoch'):
            trainer.train_epoch()
        else:
            # Fallback: assume train_step is called externally
            pass
        
        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)
        
        if (epoch + 1) % log_interval == 0:
            avg_time = sum(epoch_times[-log_interval:]) / log_interval
            print(f"Epoch {epoch+1}/{n_epochs}: {avg_time:.4f}s per epoch")
    
    profiler.end_stage('total_training')
    
    return {
        'total_time': sum(epoch_times),
        'avg_epoch_time': sum(epoch_times) / len(epoch_times),
        'min_epoch_time': min(epoch_times),
        'max_epoch_time': max(epoch_times),
        'epoch_times': epoch_times,
        'memory': get_memory_usage()
    }

