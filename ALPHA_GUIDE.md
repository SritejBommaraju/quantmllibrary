# Alpha Generation Guide

Complete guide to generating alpha using QuantML in real trading environments.

## Overview

This guide explains how to use QuantML to generate trading alpha (excess returns) using industry-standard methods from top quant firms like Jane Street and Optiver.

## Alpha Generation Methods

### 1. Momentum Alpha

**Concept**: Assets that have performed well recently tend to continue performing well (or poorly).

**Implementation**:
```python
from examples.alpha_factors import MomentumFactor

# Price momentum
momentum = MomentumFactor.price_momentum(prices, lookback=20)

# EMA momentum (crossover)
ema_mom = MomentumFactor.ema_momentum(prices, fast=10, slow=20)
```

**When to use**: Trending markets, strong directional moves

### 2. Mean Reversion Alpha

**Concept**: Prices that deviate from their average tend to revert back.

**Implementation**:
```python
from examples.alpha_factors import MeanReversionFactor

# Z-score (price deviation from rolling mean)
zscore = MeanReversionFactor.zscore_factor(prices, window=20)

# VWAP deviation
vwap_dev = MeanReversionFactor.vwap_deviation(prices, volumes, window=20)
```

**When to use**: Range-bound markets, overbought/oversold conditions

### 3. Microstructure Alpha

**Concept**: Order flow and market microstructure contain predictive information.

**Implementation**:
```python
from examples.alpha_factors import MicrostructureFactor

# Order flow imbalance
ofi = MicrostructureFactor.order_flow_imbalance(bids, asks, bid_sizes, ask_sizes)

# Microprice (weighted bid-ask midpoint)
microprice = MicrostructureFactor.microprice_factor(bids, asks, bid_sizes, ask_sizes)
```

**When to use**: High-frequency trading, tick-level data available

### 4. Volatility Alpha

**Concept**: Volatility patterns can predict future returns.

**Implementation**:
```python
from examples.alpha_factors import VolatilityFactor

# Realized volatility
vol = VolatilityFactor.realized_volatility(prices, window=20)

# Volatility of volatility
vol_of_vol = VolatilityFactor.vol_of_vol(prices, vol_window=20, vol_of_vol_window=10)
```

**When to use**: Volatility trading, risk-adjusted strategies

### 5. ML-Based Alpha

**Concept**: Machine learning models learn complex patterns from features.

**Implementation**:
```python
from quantml.models import Linear
from quantml.optim import QuantOptimizer
from quantml.training import QuantTrainer
from quantml.training.losses import sharpe_loss

# Train model
model = Linear(in_features=n_features, out_features=1)
optimizer = QuantOptimizer(model.parameters(), lr=0.001)
trainer = QuantTrainer(model, optimizer, sharpe_loss)

# Generate signals
signal = model.forward(features)
```

**When to use**: Complex patterns, multiple factors, non-linear relationships

## Quick Start

### Generate Alpha in 5 Minutes

```python
# 1. Load your data
prices = [100.0, 101.0, 102.0, ...]  # Your price data
volumes = [100.0, 110.0, 105.0, ...]  # Your volume data

# 2. Create features
from examples.alpha_factors import MomentumFactor, MeanReversionFactor, AlphaFactorCombiner

momentum = MomentumFactor.price_momentum(prices, lookback=20)
zscore = MeanReversionFactor.zscore_factor(prices, window=20)

# 3. Combine factors
combiner = AlphaFactorCombiner({
    'momentum': momentum,
    'zscore': zscore
}, weights={'momentum': 0.6, 'zscore': 0.4})

alpha_signals = combiner.combine()

# 4. Use signals
for i, signal in enumerate(alpha_signals):
    if signal > 0.01:
        # BUY signal
        pass
    elif signal < -0.01:
        # SELL signal
        pass
```

## Production Deployment

### Step 1: Data Integration

Connect to your data source:

```python
from examples.live_alpha_generator import DataAdapter, LiveAlphaGenerator

# Create adapter for your data source
def get_price(data_source, idx):
    # Your implementation
    return data_source.get_price(idx)

adapter = DataAdapter(your_data_source, get_price)

# Initialize generator
generator = LiveAlphaGenerator(model=trained_model, lookback_window=100)
```

### Step 2: Real-Time Signal Generation

```python
# On each new tick
generator.update(new_price, new_volume)
signal_data = generator.get_signal_with_metadata()

if signal_data['valid']:
    # Execute trade based on signal_data['signal']
    pass
```

### Step 3: Walk-Forward Training

```python
from quantml.training import WalkForwardOptimizer, WindowType

wfo = WalkForwardOptimizer(
    window_type=WindowType.EXPANDING,
    train_size=500,
    test_size=100
)

for train_idx, test_idx in wfo.split(features, n_splits=5):
    # Train on train_idx
    # Evaluate on test_idx
    pass
```

### Step 4: Alpha Evaluation

```python
from quantml.training import AlphaEvaluator

evaluator = AlphaEvaluator(predictions, actuals)
metrics = evaluator.evaluate()

# Key metrics:
# - IC (Information Coefficient): > 0.05 is good
# - Rank IC: > 0.05 is good
# - Hit Rate: > 0.5 is good
# - Turnover: Lower is better (less trading)
```

### Step 5: Risk Management

```python
from quantml.training import BacktestEngine

backtest = BacktestEngine(
    initial_capital=100000.0,
    commission=0.001,  # 0.1% commission
    slippage=0.0005     # 0.05% slippage
)

results = backtest.run_with_predictions(signals, prices)
# Check: Sharpe > 1.0, Max Drawdown < 20%
```

## Best Practices

### 1. Feature Engineering

- **Use domain knowledge**: Price, volume, order flow
- **Normalize features**: Z-score normalization prevents scale issues
- **Avoid lookahead bias**: Only use past data for features
- **Combine factors**: Multiple factors are more robust

### 2. Model Training

- **Walk-forward optimization**: Prevents overfitting to historical data
- **Use quant-specific losses**: Sharpe loss optimizes for risk-adjusted returns
- **Regular retraining**: Alpha decays over time, retrain monthly/quarterly
- **Ensemble models**: Combine multiple models for robustness

### 3. Signal Quality

- **IC threshold**: Only use signals with IC > 0.05
- **Signal strength**: Filter weak signals (|signal| < 0.01)
- **Turnover limits**: High turnover increases costs
- **Alpha decay monitoring**: Retrain when IC drops 30%+

### 4. Risk Management

- **Position sizing**: Scale positions by signal strength and volatility
- **Transaction costs**: Include commission and slippage in backtests
- **Drawdown limits**: Stop trading if drawdown exceeds threshold
- **Diversification**: Don't rely on single alpha source

## Real-World Integration

### Connecting to Data Sources

**Bloomberg API**:
```python
def get_price_bloomberg(security, field='PX_LAST'):
    # Your Bloomberg API code
    return bloomberg_api.get_field(security, field)
```

**Interactive Brokers**:
```python
def get_price_ib(contract):
    # Your IB API code
    return ib.reqMktData(contract)
```

**CSV Files**:
```python
import pandas as pd
df = pd.read_csv('prices.csv')
prices = df['close'].tolist()
```

### Deployment Architecture

```
┌─────────────┐
│ Data Source │ (Bloomberg, IB, CSV, etc.)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Data Adapter│ (Converts to QuantML format)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│Feature Eng. │ (Alpha factors, normalization)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Model     │ (Trained Linear/RNN/TCN)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Signal    │ (Alpha signal + metadata)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Risk Check  │ (IC, position limits, etc.)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Execution  │ (Order management system)
└─────────────┘
```

## Monitoring and Maintenance

### Key Metrics to Track

1. **Information Coefficient (IC)**: Should be > 0.05
2. **Sharpe Ratio**: Should be > 1.0
3. **Max Drawdown**: Should be < 20%
4. **Turnover**: Monitor for excessive trading
5. **Hit Rate**: Should be > 50%

### When to Retrain

- **Regular schedule**: Monthly or quarterly
- **Alpha decay**: When IC drops 30%+
- **Market regime change**: Volatility spikes, trend changes
- **Performance degradation**: Sharpe drops below threshold

### Alpha Decay Management

```python
# Monitor IC over time
if pipeline.check_alpha_decay(recent_ic):
    print("Alpha decay detected, retraining...")
    pipeline.train_models(new_features, new_targets)
```

## Example Workflows

### Intraday Trading

```python
# Update every minute
while market_open:
    generator.update(current_price, current_volume)
    signal = generator.generate_signal()
    
    if abs(signal) > threshold:
        execute_trade(signal)
    
    time.sleep(60)  # Wait 1 minute
```

### Daily Rebalancing

```python
# Train overnight
model = train_model(overnight_data)

# Generate signals at market open
signals = generate_signals(model, market_data)

# Execute trades
for signal in signals:
    if signal['valid']:
        execute_trade(signal)
```

### High-Frequency Trading

```python
# Per-tick updates
def on_tick(tick_data):
    generator.update(tick_data.price, tick_data.volume)
    signal = generator.generate_signal()
    
    if abs(signal) > threshold:
        execute_trade(signal)
```

## Troubleshooting

### Low IC (< 0.05)

- **Check features**: Are they predictive?
- **Add more factors**: Try microstructure or volatility factors
- **Longer training**: Train for more epochs
- **Different model**: Try RNN or TCN for time-series patterns

### High Turnover

- **Signal threshold**: Increase minimum signal strength
- **Smoothing**: Apply moving average to signals
- **Holding period**: Require minimum holding period

### Poor Backtest Performance

- **Transaction costs**: Ensure realistic costs in backtest
- **Lookahead bias**: Verify no future data leakage
- **Overfitting**: Use walk-forward optimization
- **Market conditions**: Strategy may not work in all regimes

## Resources

- **Examples**: See `examples/` directory
- **Alpha Factors**: `examples/alpha_factors.py`
- **Live Generator**: `examples/live_alpha_generator.py`
- **Production Pipeline**: `examples/production_alpha.py`

## Next Steps

1. Start with `examples/quick_alpha.py` to see immediate results
2. Customize features for your use case
3. Integrate with your data source
4. Deploy with proper risk management
5. Monitor and retrain regularly

Good luck generating alpha!

