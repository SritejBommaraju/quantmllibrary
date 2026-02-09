"""
QuantML - Machine Learning Library for Quantitative Trading

A clean, minimal, hackable ML library optimized for:
- Streaming market data (tick-level or bar-level)
- Online/incremental learning
- Low-latency CPU-only inference
- Small models for HFT/quant research
- Time-series operations native to quant workflows
"""

__version__ = "0.1.1"

# Core components
from quantml.tensor import Tensor
from quantml import autograd
from quantml import ops
from quantml import functional as F
from quantml import time_series
from quantml import streaming
from quantml import online

# Models
from quantml.models import Linear, SimpleRNN, TCN

# Optimizers
from quantml.optim import SGD, Adam

# Utilities
from quantml.utils import profiling

# Training (optional import to avoid circular dependencies)
try:
    from quantml.training import (
        QuantTrainer,
        WalkForwardOptimizer,
        BacktestEngine,
        AlphaEvaluator,
        FeaturePipeline
    )
    HAS_TRAINING = True
except ImportError:
    HAS_TRAINING = False

__all__ = [
    # Core
    'Tensor',
    'autograd',
    'ops',
    'F',
    'time_series',
    'streaming',
    'online',
    # Models
    'Linear',
    'SimpleRNN',
    'TCN',
    # Optimizers
    'SGD',
    'Adam',
    # Utilities
    'profiling',
]

# Add training exports if available
if HAS_TRAINING:
    __all__.extend([
        'QuantTrainer',
        'WalkForwardOptimizer',
        'BacktestEngine',
        'AlphaEvaluator',
        'FeaturePipeline'
    ])

