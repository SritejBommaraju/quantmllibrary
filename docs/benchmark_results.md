# QuantML Benchmark Results

**Date**: November 2024  
**System**: Windows 10, Python 3.10.11, NumPy enabled  
**Hardware**: CPU-only (no GPU)

## Summary

QuantML achieves low-latency performance optimized for quantitative trading:

- **Inference Latency**: 0.02-0.07ms average (sub-millisecond)
- **Training Throughput**: 7,000-14,000 steps/second
- **Tensor Operations**: 250,000+ ops/sec for small tensors
- **Matrix Multiplication**: 2,000-14,000 ops/sec depending on size

## Detailed Results

### Tensor Operations

| Size | Operation | Time per Op | Throughput |
|------|-----------|-------------|------------|
| 100x100 | Addition | 0.0040ms | 252,621 ops/sec |
| 100x100 | Multiplication | 0.0035ms | 285,380 ops/sec |
| 500x500 | Addition | 0.5867ms | 1,705 ops/sec |
| 500x500 | Multiplication | 0.5745ms | 1,741 ops/sec |
| 1000x1000 | Addition | 2.3202ms | 431 ops/sec |
| 1000x1000 | Multiplication | 2.2198ms | 450 ops/sec |

### Matrix Multiplication

| Size | Time per Op | Throughput |
|------|-------------|------------|
| 50x50 @ 50x50 | 0.0070ms | 143,864 ops/sec |
| 100x100 @ 100x100 | 0.0502ms | 19,923 ops/sec |
| 200x200 @ 200x200 | 0.4432ms | 2,256 ops/sec |

### Training Loop Performance

| Features | Batch Size | Epochs | Time per Step | Throughput |
|----------|------------|--------|---------------|------------|
| 50 | 32 | 10 | 0.0719ms | 13,901 steps/sec |
| 100 | 32 | 10 | 0.1112ms | 8,994 steps/sec |
| 200 | 32 | 5 | 0.1392ms | 7,181 steps/sec |

### Inference Latency

| Features | Avg Latency | Min | Max | P95 | P99 |
|----------|-------------|-----|-----|-----|-----|
| 50 | 0.0199ms | 0.0144ms | 0.3483ms | 0.0455ms | 0.0586ms |
| 100 | 0.0365ms | 0.0259ms | 0.1922ms | 0.0784ms | 0.0910ms |
| 200 | 0.0662ms | 0.0475ms | 0.4724ms | 0.1520ms | 0.2243ms |

**Key Finding**: Average inference latency is **sub-millisecond** (0.02-0.07ms), making it suitable for real-time trading applications.

## Performance Characteristics

### Strengths

1. **Ultra-low inference latency**: Sub-millisecond inference makes it suitable for HFT/quant trading
2. **High training throughput**: 7k-14k steps/sec enables fast iteration
3. **Efficient tensor operations**: NumPy-optimized operations achieve high throughput
4. **CPU-optimized**: No GPU dependency, runs efficiently on CPU

### Optimization Impact

The library uses NumPy-optimized operations throughout, which provides:

- **Eliminated conversions**: Direct NumPy array operations avoid Python list overhead
- **Vectorized operations**: NumPy's optimized C implementations
- **Memory efficiency**: NumPy arrays are more memory-efficient than Python lists

## Comparison Context

These benchmarks demonstrate that QuantML achieves:

- **Low-latency inference** suitable for real-time trading (sub-ms)
- **High training throughput** for rapid model iteration
- **Efficient memory usage** through NumPy optimization

## Notes

- Benchmarks run on CPU-only (no GPU acceleration)
- Results may vary based on hardware and system load
- NumPy optimization provides significant performance benefits
- Library designed for quant trading use cases (small models, low latency)

## Running Benchmarks

To reproduce these results:

```bash
python benchmarks/benchmark_comparison.py
```

For basic benchmarks:

```bash
python benchmarks/benchmark_ops.py
python benchmarks/benchmark_training.py
```

