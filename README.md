# QuantML

A clean, minimal, hackable machine learning library optimized specifically for quantitative trading, streaming data, online learning, and low-latency CPU inference.

## Table of Contents

- [Installation](#installation)
- [Dependencies](#dependencies)
- [Quick Start](#quick-start)
- [Directory Structure](#directory-structure)
- [Usage for Research](#usage-for-research)
- [Features](#features)
- [Architecture](#architecture)
- [Performance](#performance)
- [Examples](#examples)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Installation

### From PyPI (when published)
```bash
pip install quantmllibrary
```

### From Source

```bash
git clone https://github.com/SritejBommaraju/quantmllibrary.git
cd quantmllibrary
pip install -e .
```

### With Optional Dependencies

```bash
# With NumPy for performance
pip install quantmllibrary[numpy]

# Development dependencies
pip install -r requirements-dev.txt
```

### Using Conda

```bash
conda env create -f environment.yml
conda activate quantml
```

## Dependencies

### Core Dependencies

- **Python**: >=3.8, <3.13
- **NumPy**: >=1.20.0, <2.0.0 (optional, but recommended for performance)

### Optional Dependencies

- **pandas**: >=1.3.0 (for data loading and manipulation)
- **pyyaml**: >=6.0 (for YAML config support)
- **pyarrow**: (for Parquet feature caching)

### Development Dependencies

See `requirements-dev.txt` for testing, linting, and documentation tools.

**Note**: The library works without NumPy, but performance is significantly better (2-5x faster) with NumPy installed.

## Quick Start

### Complete Working Example

```python
from quantml import Tensor
from quantml.models import Linear
from quantml.optim import Adam
from quantml.training import QuantTrainer, FeaturePipeline
from quantml.training.losses import mse_loss
from quantml.training.features import normalize_features

# 1. Load your data (replace with your data source)
prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
volumes = [100.0, 110.0, 105.0, 120.0, 115.0, 125.0]

# 2. Create features
pipeline = FeaturePipeline()
pipeline.add_lagged_feature('price', lags=[1, 5, 10])
pipeline.add_rolling_feature('price', window=20, func='mean')
pipeline.add_time_series_feature('price', 'returns')

features = pipeline.transform({'price': prices})
features = normalize_features(features, method='zscore')

# 3. Create targets (forward returns)
targets = [(prices[i+1] - prices[i]) / prices[i] for i in range(len(prices)-1)]
features = features[:-1]  # Align

# 4. Train model
model = Linear(in_features=len(features[0]), out_features=1, bias=True)
optimizer = Adam(model.parameters(), lr=0.001)
trainer = QuantTrainer(model, optimizer, mse_loss)

# Train
for i in range(len(features)):
    x = Tensor([features[i]])
    y = Tensor([[targets[i]]])
    trainer.train_step(x, y)

# 5. Generate predictions
predictions = []
for feat in features:
    x = Tensor([feat])
    pred = model.forward(x)
    pred_val = pred.data[0][0] if isinstance(pred.data[0], list) else pred.data[0]
    predictions.append(pred_val)

print(f"Generated {len(predictions)} predictions")
```

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

## Directory Structure

```
quantmllibrary/
├── quantml/                    # Main library package
│   ├── __init__.py            # Package initialization
│   ├── tensor.py              # Core Tensor class
│   ├── autograd.py            # Automatic differentiation
│   ├── ops.py                 # Operations (NumPy-optimized)
│   ├── functional.py          # Functional API
│   ├── time_series.py         # Quant-specific operations
│   ├── streaming.py           # Streaming tensors
│   ├── online.py              # Online learning
│   ├── config/                # Configuration management
│   │   ├── __init__.py
│   │   └── config.py          # YAML/JSON config support
│   ├── data/                  # Data management
│   │   ├── __init__.py
│   │   ├── validators.py      # Data validation
│   │   ├── loaders.py         # Data loaders
│   │   ├── feature_store.py   # Feature caching
│   │   └── memory_optimizer.py # Memory optimization
│   ├── models/                # Neural network models
│   │   ├── linear.py
│   │   ├── rnn.py
│   │   └── tcn.py
│   ├── optim/                  # Optimizers and schedulers
│   │   ├── sgd.py
│   │   ├── adam.py
│   │   ├── rmsprop.py
│   │   ├── adagrad.py
│   │   ├── adafactor.py
│   │   ├── lookahead.py
│   │   ├── radam.py
│   │   ├── quant_optimizer.py
│   │   └── schedulers.py
│   ├── training/               # Training utilities
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   ├── metrics.py
│   │   ├── features.py
│   │   ├── walk_forward.py
│   │   ├── backtest.py
│   │   ├── alpha_eval.py
│   │   └── ...
│   ├── experiments/            # Experiment tracking
│   │   └── ...
│   └── utils/                  # Utilities
│       ├── logging.py
│       ├── reproducibility.py
│       └── profiling.py
├── examples/                   # Example scripts
│   ├── quick_alpha.py         # Quick alpha generation
│   ├── production_alpha.py   # Production pipeline
│   ├── alpha_training.py      # Alpha training example
│   └── ...
├── tests/                      # Test suite
│   ├── test_tensor.py
│   ├── test_ops.py
│   ├── test_models.py
│   └── integration/
├── configs/                   # Configuration files
│   ├── base.yaml
│   └── experiments/
├── docs/                       # Documentation
│   └── ...
├── benchmarks/                 # Performance benchmarks
├── requirements.txt            # Core dependencies
├── requirements-dev.txt        # Dev dependencies
├── environment.yml            # Conda environment
├── pyproject.toml             # Package configuration
└── README.md                  # This file
```

## Usage for Research

### Overnight Gap Prediction

```python
from quantml.config import load_config, ExperimentConfig
from quantml.data import load_csv_data, validate_price_data
from quantml.training import FeaturePipeline, QuantTrainer
from quantml.models import Linear
from quantml.optim import Adam

# Load configuration
config = load_config('configs/experiments/overnight_gap.yaml')

# Load and validate data
data = load_csv_data(
    config.data.data_path,
    price_column='close',
    volume_column='volume'
)

is_valid, errors = validate_price_data(data['prices'], data['volumes'])
if not is_valid:
    print(f"Data validation errors: {errors}")

# Create features for gap prediction
pipeline = FeaturePipeline()
pipeline.add_lagged_feature('price', lags=[1, 5, 10, 20])
pipeline.add_rolling_feature('price', window=20, func='mean')
pipeline.add_time_series_feature('price', 'volatility', n=20)

features = pipeline.transform({'price': data['prices']})

# Train model (see examples/alpha_training.py for full example)
```

### Multi-Instrument Support (ES, MES, NQ, MNQ)

```python
from quantml.config import load_config

# Load instrument-specific config
es_config = load_config('configs/instruments/ES.yaml')
mes_config = load_config('configs/instruments/MES.yaml')

# Run experiments for each instrument
for instrument, config in [('ES', es_config), ('MES', mes_config)]:
    # Load data for instrument
    # Create features
    # Train model
    # Evaluate
    pass
```

### Walk-Forward Optimization

```python
from quantml.training import WalkForwardOptimizer, WindowType

wfo = WalkForwardOptimizer(
    window_type=WindowType.EXPANDING,
    train_size=500,  # 500 days training
    test_size=100    # 100 days testing
)

for train_idx, test_idx in wfo.split(features, n_splits=5):
    # Train on train_idx (past data only)
    # Test on test_idx (future data)
    # No lookahead bias!
    pass
```

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
├── config/                # Configuration management
│   └── config.py          # YAML/JSON config support
├── data/                  # Data management
│   ├── validators.py      # Data validation
│   ├── loaders.py         # Data loaders
│   ├── feature_store.py   # Feature caching (Parquet)
│   └── memory_optimizer.py # Memory optimization
├── models/                # Neural network models
│   ├── linear.py
│   ├── rnn.py
│   └── tcn.py
├── optim/                 # Optimizers and schedulers
│   ├── sgd.py
│   ├── adam.py
│   └── ... (8 optimizers total)
├── training/              # Training utilities
│   ├── trainer.py
│   ├── losses.py
│   ├── metrics.py
│   ├── features.py
│   ├── walk_forward.py
│   ├── backtest.py
│   └── ... (15+ modules)
├── experiments/           # Experiment tracking
│   └── ...
└── utils/                 # Utilities
    ├── logging.py
    ├── reproducibility.py
    └── profiling.py
```

## Performance

**Measured Benchmarks** (see `docs/benchmark_results.md` for details):

- **Inference Latency**: 0.02-0.07ms average (sub-millisecond, suitable for HFT)
- **Training Throughput**: 7,000-14,000 steps/second
- **Tensor Operations**: 250,000+ ops/sec (100x100 tensors)
- **Matrix Multiplication**: 2,000-14,000 ops/sec (depending on size)

**Optimization Benefits**:
- **NumPy-optimized**: Direct array operations eliminate conversion overhead
- **Memory Efficient**: NumPy arrays reduce memory footprint vs Python lists
- **Low Latency**: Sub-millisecond inference enables real-time trading
- **High Throughput**: Optimized training loops for rapid iteration

## Examples

See the `examples/` directory for complete examples:

- **`quick_alpha.py`**: Generate alpha signals immediately
- **`alpha_training.py`**: Complete alpha generation pipeline
- **`production_alpha.py`**: Production-ready pipeline with walk-forward
- **`walk_forward_training.py`**: Walk-forward optimization example
- **`backtest_strategy.py`**: Strategy backtesting
- **`futures_model.py`**: Complete futures trading pipeline
- **`online_regression.py`**: Online learning with streaming data
- **`streaming_training.py`**: Per-tick model updates
- **`live_alpha_generator.py`**: Real-time alpha generation

## Documentation

- **`ALPHA_GUIDE.md`**: Complete guide to generating alpha
- **`examples/config_example.yaml`**: Configuration file example
- **API Documentation**: See docstrings in each module

## Design Philosophy

1. **Clean & Hackable**: Code is readable and easy to modify (micrograd-level clarity)
2. **Minimal Dependencies**: Pure Python with optional NumPy
3. **Quant-First**: Operations designed for trading workflows
4. **Streaming Native**: Built for tick-level data from the ground up
5. **Low Latency**: CPU-optimized for fast inference
6. **Performance**: NumPy-optimized operations throughout
7. **Reproducible**: Random seed management, experiment tracking, version control

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
7. **Futures Trading**: Overnight gap prediction, multi-instrument support
8. **Research**: Reproducible experiments for academic papers

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details.

## Status

This is an alpha release. The API may change in future versions.
