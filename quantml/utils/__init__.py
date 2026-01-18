"""
QuantML Utilities

This module provides utility functions for profiling, gradient checking,
model serialization, and CPU-optimized operations.
"""

from quantml.utils.profiling import (
    timing,
    measure_latency,
    measure_latency_microseconds,
    PerformanceProfiler,
    benchmark
)

from quantml.utils.gradient_check import (
    check_gradients,
    gradient_check_layer,
    print_gradient_check_results,
    quick_gradient_check
)

from quantml.utils.serialization import (
    save_model,
    load_model,
    save_checkpoint,
    load_checkpoint,
    get_model_state_dict,
    set_model_state_dict
)

__all__ = [
    # Profiling
    'timing',
    'measure_latency',
    'measure_latency_microseconds',
    'PerformanceProfiler',
    'benchmark',
    # Gradient checking
    'check_gradients',
    'gradient_check_layer',
    'print_gradient_check_results',
    'quick_gradient_check',
    # Serialization
    'save_model',
    'load_model',
    'save_checkpoint',
    'load_checkpoint',
    'get_model_state_dict',
    'set_model_state_dict',
]
