# QuantML

A clean, minimal, hackable machine learning library optimized specifically for quantitative trading, streaming data, online learning, and low-latency CPU inference.

## Features

### Core Components
- **Tensor Operations**: Full-featured tensor class with automatic differentiation
- **Autograd Engine**: Dynamic computation graphs with backpropagation
- **NumPy-Optimized**: 2-5x faster operations with optional NumPy acceleration
- **Zero Dependencies**: Pure Python with optional NumPy for performance

### Quant-Specific Operations
- **Time-Series Ops**: EMA, WMA, rolling mean/std, volatility, z-score
- **Market Data**: VWAP, order flow imbalance, microprice
- **Returns**: Returns, log returns calculations
- **Streaming Support**: Ring buffer tensors for tick-level market data

### Neural Network Models
- **Linear**: Fully connected layer
- **SimpleRNN**: Recurrent neural network
- **TCN**: Temporal Convolutional Network

### Optimizers (8 Total)
- **SGD**: Stochastic Gradient Descent with momentum and weight decay
- **Adam**: Adaptive Moment Estimation
- **RMSProp**: Root Mean Square Propagation
- **AdaGrad**: Adaptive Gradient
- **AdaFactor**: Memory-efficient Adam variant
- **Lookahead**: Wrapper optimizer for training stability
- **RAdam**: Rectified Adam with variance rectification
- **QuantOptimizer**: Custom optimizer with volatility-aware learning rates

### Learning Rate Schedulers (6 Types)
- **StepLR**: Step decay by factor
- **CosineAnnealingLR**: Cosine annealing schedule
- **WarmupLR**: Linear/cosine warmup
- **ReduceLROnPlateau**: Reduce on metric plateau
- **CyclicLR**: Cyclic learning rates
- **OneCycleLR**: One cycle policy

### Training Utilities
- **QuantTrainer**: Training loop with early stopping, checkpointing, metrics tracking
- **Gradient Clipping**: Norm clipping, value clipping, adaptive clipping
- **Gradient Accumulation**: Effective larger batch sizes
- **Learning Rate Finder**: Automatic optimal LR discovery
- **Model Ensembling**: Weighted averaging, voting, stacking
- **Feature Importance**: Gradient-based and permutation-based importance
- **Cross-Validation**: TimeSeriesSplit, PurgedKFold for time-series data
- **Regularization**: Dropout layer

### Quant-Specific Loss Functions
- **Sharpe Loss**: Optimize for Sharpe ratio
- **Quantile Loss**: For quantile regression
- **Information Ratio Loss**: Optimize for information ratio
- **Huber Loss**: Robust to outliers
- **Asymmetric Loss**: Different penalties for over/under-prediction
- **Max Drawdown Loss**: Optimize to reduce drawdowns
- **Combined Quant Loss**: Multi-objective quant loss

### Financial Metrics
- **Performance**: Sharpe, Sortino, Calmar ratios, max drawdown
- **Alpha Metrics**: Information Coefficient (IC), Rank IC, hit rate
- **Risk Metrics**: VaR, CVaR, turnover
- **Backtesting**: Complete backtesting engine with transaction costs

### Feature Engineering
- **FeaturePipeline**: Reproducible feature creation
- **Lagged Features**: Multiple lag periods
- **Rolling Windows**: Mean, std, min, max over windows
- **Normalization**: Z-score, min-max, robust scaling

### Walk-Forward & Backtesting
- **WalkForwardOptimizer**: Time-series aware train/test splits
- **BacktestEngine**: Strategy backtesting with P&L tracking
- **AlphaEvaluator**: Comprehensive alpha signal evaluation

## Installation

```bash
pip install quantml
```

Or install from source:

```bash
git clone https://github.com/quantml/quantml.git
cd quantml
pip install -e .
```

## Quick Start

### Basic Tensor Operations

```python
from quantml import Tensor

# Create tensors
x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
y = Tensor([4.0, 5.0, 6.0], requires_grad=True)

# Operations
z = x + y
z.backward()

print(x.grad)  # [1.0, 1.0, 1.0]
```

### Quant Operations

```python
from quantml import Tensor
from quantml import time_series

# Price data
prices = Tensor([100.0, 101.0, 102.0, 103.0, 104.0])

# Exponential Moving Average
ema_20 = time_series.ema(prices, n=20)

# Rolling volatility
vol = time_series.volatility(prices, n=20)

# Returns
rets = time_series.returns(prices)

# VWAP
volumes = Tensor([100.0, 110.0, 105.0, 120.0, 115.0])
vwap = time_series.vwap(prices, volumes)
```

### Alpha Generation Example

```python
from quantml import Tensor
from quantml.models import Linear
from quantml.optim import QuantOptimizer, StepLR
from quantml.training import (
    QuantTrainer, FeaturePipeline, WalkForwardOptimizer, WindowType,
    AlphaEvaluator, BacktestEngine, GradientNormClipper
)
from quantml.training.losses import sharpe_loss

# 1. Create features
pipeline = FeaturePipeline()
pipeline.add_lagged_feature('price', lags=[1, 5, 10, 20])
pipeline.add_rolling_feature('price', window=20, func='mean')
features = pipeline.transform({'price': prices})

# 2. Create model
model = Linear(in_features=len(features[0]), out_features=1)

# 3. Setup optimizer with scheduler
optimizer = QuantOptimizer(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=100, gamma=0.9)

# 4. Train with gradient clipping
grad_clipper = GradientNormClipper(max_norm=1.0)
trainer = QuantTrainer(
    model=model,
    optimizer=optimizer,
    loss_fn=sharpe_loss,
    gradient_clipper=grad_clipper
)

# 5. Walk-forward training
wfo = WalkForwardOptimizer(
    window_type=WindowType.EXPANDING,
    train_size=500,
    test_size=100
)

for train_idx, test_idx in wfo.split(features, n_splits=5):
    # Train on train_idx, evaluate on test_idx
    # ... training code ...
    pass

# 6. Evaluate alpha
evaluator = AlphaEvaluator(predictions, actuals)
metrics = evaluator.evaluate()
print(f"IC: {metrics['ic']:.4f}, Rank IC: {metrics['rank_ic']:.4f}")

# 7. Backtest
backtest = BacktestEngine(initial_capital=100000.0)
results = backtest.run_with_predictions(predictions, prices)
print(f"Sharpe: {results['sharpe_ratio']:.4f}")
```

### Advanced Training Features

```python
from quantml.optim import RAdam, Lookahead, CosineAnnealingLR
from quantml.training import LRFinder, EnsembleModel, Dropout

# Learning rate finder
lr_finder = LRFinder(model, optimizer, loss_fn)
lrs, losses = lr_finder.range_test(x_sample, y_sample)
optimal_lr = lr_finder.suggest_lr()

# Model ensembling
ensemble = EnsembleModel(
    models=[model1, model2, model3],
    weights=[0.4, 0.3, 0.3],
    strategy='weighted_avg'
)

# Dropout regularization
dropout = Dropout(p=0.5)
x_dropped = dropout.forward(x)
```

## Architecture

```
quantml/
├── tensor.py              # Core Tensor class with NumPy optimization
├── autograd.py            # Automatic differentiation
├── ops.py                 # Operations (NumPy-optimized)
├── functional.py          # Functional API
├── time_series.py         # Quant-specific operations
├── streaming.py           # Streaming tensors with ring buffer
├── online.py              # Online learning utilities
├── models/                # Neural network models
│   ├── linear.py
│   ├── rnn.py
│   └── tcn.py
├── optim/                 # Optimizers and schedulers
│   ├── sgd.py
│   ├── adam.py
│   ├── rmsprop.py
│   ├── adagrad.py
│   ├── adafactor.py
│   ├── lookahead.py
│   ├── radam.py
│   ├── quant_optimizer.py
│   └── schedulers.py
├── training/              # Training utilities
│   ├── trainer.py
│   ├── losses.py
│   ├── metrics.py
│   ├── features.py
│   ├── walk_forward.py
│   ├── backtest.py
│   ├── alpha_eval.py
│   ├── gradient_clipping.py
│   ├── lr_finder.py
│   ├── ensemble.py
│   ├── feature_importance.py
│   ├── cv.py
│   ├── regularization.py
│   └── data_loader.py
└── utils/                 # Utilities
    ├── profiling.py
    └── ops_cpu.py
```

## Performance

With NumPy optimization:
- **Training Speed**: 2-5x faster (eliminated conversions)
- **Memory Usage**: 30-50% reduction (NumPy arrays, fewer Tensor objects)
- **Inference Latency**: 3-10x faster (optimized operations)
- **Optimizer Overhead**: 5-10x reduction (direct NumPy operations)

## Examples

See the `examples/` directory for complete examples:
- `alpha_training.py`: Complete alpha generation pipeline
- `walk_forward_training.py`: Walk-forward optimization example
- `backtest_strategy.py`: Strategy backtesting
- `online_regression.py`: Online learning with streaming data
- `streaming_training.py`: Per-tick model updates
- `futures_model.py`: Complete futures trading pipeline

## Design Philosophy

1. **Clean & Hackable**: Code is readable and easy to modify (micrograd-level clarity)
2. **Minimal Dependencies**: Pure Python with optional NumPy
3. **Quant-First**: Operations designed for trading workflows
4. **Streaming Native**: Built for tick-level data from the ground up
5. **Low Latency**: CPU-optimized for fast inference
6. **Performance**: NumPy-optimized operations throughout

## Comparison to PyTorch/JAX

QuantML is designed specifically for quant trading use cases:

- **Smaller footprint**: No CUDA, no complex backends
- **Streaming-first**: Native support for ring buffers and tick data
- **Quant ops**: Built-in EMA, VWAP, order flow, etc.
- **Online learning**: Per-tick updates out of the box
- **Simpler API**: Easier to understand and modify
- **Quant-specific**: Optimizers, losses, and metrics for trading
- **Walk-forward**: Built-in time-series cross-validation
- **Backtesting**: Integrated backtesting engine

## Key Use Cases

1. **Alpha Generation**: Train models to predict returns
2. **Signal Processing**: Real-time feature engineering on streaming data
3. **Online Learning**: Update models on every tick
4. **Strategy Backtesting**: Test trading strategies with realistic costs
5. **Feature Engineering**: Create reproducible quant features
6. **Model Ensembling**: Combine multiple models for robust predictions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details.

## Status

This is an alpha release. The API may change in future versions.
