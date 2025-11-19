# Feature Documentation

Complete documentation of all features available in QuantML, including formulas, interpretations, and usage examples.

## Table of Contents

- [Price-Based Features](#price-based-features)
- [Volume-Based Features](#volume-based-features)
- [Volatility Features](#volatility-features)
- [Time-Series Features](#time-series-features)
- [Alpha Factors](#alpha-factors)
- [Feature Engineering Pipeline](#feature-engineering-pipeline)

## Price-Based Features

### Lagged Price Features

**Description**: Historical price values at different time lags.

**Formula**: 
```
lag_k(t) = price(t - k)
```

**Usage**:
```python
pipeline.add_lagged_feature('price', lags=[1, 5, 10, 20])
```

**Interpretation**: Captures price momentum and mean reversion patterns. Short lags (1-5) capture immediate momentum, longer lags (10-20) capture trend persistence.

**Expected Distribution**: Similar to price distribution, typically log-normal.

### Rolling Mean

**Description**: Moving average over a rolling window.

**Formula**:
```
rolling_mean(t, n) = (1/n) * Σ price(t-i) for i=0 to n-1
```

**Usage**:
```python
pipeline.add_rolling_feature('price', window=20, func='mean')
```

**Interpretation**: Smooths price data, identifies trend direction. Price above rolling mean suggests uptrend.

**Expected Distribution**: Smoother than raw prices, lower variance.

### Rolling Standard Deviation

**Description**: Volatility measure over rolling window.

**Formula**:
```
rolling_std(t, n) = sqrt((1/n) * Σ (price(t-i) - mean)^2)
```

**Usage**:
```python
pipeline.add_rolling_feature('price', window=20, func='std')
```

**Interpretation**: Measures price volatility. High std indicates high volatility regime.

**Expected Distribution**: Positive values, typically right-skewed.

### Exponential Moving Average (EMA)

**Description**: Weighted moving average with exponential decay.

**Formula**:
```
EMA(t) = α * price(t) + (1 - α) * EMA(t-1)
where α = 2 / (n + 1)
```

**Usage**:
```python
from quantml import time_series
ema = time_series.ema(prices, n=20)
```

**Interpretation**: More responsive to recent prices than simple moving average. EMA crossover strategies are common.

**Expected Distribution**: Smoother than prices, follows trend.

## Volume-Based Features

### Volume-Weighted Average Price (VWAP)

**Description**: Average price weighted by volume.

**Formula**:
```
VWAP(t) = Σ (price(i) * volume(i)) / Σ volume(i) for i in window
```

**Usage**:
```python
from quantml import time_series
vwap = time_series.vwap(prices, volumes)
```

**Interpretation**: Institutional price level. Price above VWAP suggests buying pressure.

**Expected Distribution**: Similar to price but volume-weighted.

### VWAP Deviation

**Description**: Price deviation from VWAP.

**Formula**:
```
vwap_deviation(t) = (price(t) - VWAP(t)) / VWAP(t)
```

**Usage**:
```python
from examples.alpha_factors import MeanReversionFactor
dev = MeanReversionFactor.vwap_deviation(prices, volumes, window=20)
```

**Interpretation**: Mean reversion signal. Large positive deviation suggests overbought, negative suggests oversold.

**Expected Distribution**: Centered around 0, symmetric.

### Order Flow Imbalance (OFI)

**Description**: Imbalance between bid and ask volumes.

**Formula**:
```
OFI(t) = bid_volume(t) - ask_volume(t)
```

**Usage**:
```python
from quantml import time_series
ofi = time_series.orderflow_imbalance(bids, asks, bid_sizes, ask_sizes)
```

**Interpretation**: Positive OFI suggests buying pressure, negative suggests selling pressure.

**Expected Distribution**: Centered around 0, can be asymmetric.

## Volatility Features

### Realized Volatility

**Description**: Standard deviation of returns over rolling window.

**Formula**:
```
realized_vol(t, n) = std(returns(t-n:t)) * sqrt(252)  # Annualized
```

**Usage**:
```python
from quantml import time_series
vol = time_series.volatility(prices, n=20, annualize=True)
```

**Interpretation**: Historical volatility. High volatility suggests uncertainty, regime changes.

**Expected Distribution**: Positive, right-skewed, typically 0.1-0.5 for stocks.

### Volatility of Volatility

**Description**: Volatility of the volatility measure itself.

**Formula**:
```
vol_of_vol(t) = std(realized_vol(t-n:t))
```

**Usage**:
```python
from examples.alpha_factors import VolatilityFactor
vov = VolatilityFactor.vol_of_vol(prices, vol_window=20, vol_of_vol_window=10)
```

**Interpretation**: Measures stability of volatility. High vol-of-vol suggests regime uncertainty.

**Expected Distribution**: Positive, highly right-skewed.

### Z-Score

**Description**: Standardized price deviation from rolling mean.

**Formula**:
```
zscore(t, n) = (price(t) - rolling_mean(t, n)) / rolling_std(t, n)
```

**Usage**:
```python
from quantml import time_series
zscore = time_series.zscore(prices, n=20)
```

**Interpretation**: Mean reversion signal. |zscore| > 2 suggests extreme deviation, potential reversion.

**Expected Distribution**: Standard normal (mean=0, std=1) if prices are stationary.

## Time-Series Features

### Returns

**Description**: Percentage change in price.

**Formula**:
```
returns(t) = (price(t) - price(t-1)) / price(t-1)
```

**Usage**:
```python
from quantml import time_series
rets = time_series.returns(prices)
```

**Interpretation**: Price change magnitude and direction. Positive = up, negative = down.

**Expected Distribution**: Approximately normal, mean near 0, fat tails.

### Log Returns

**Description**: Natural logarithm of price ratio.

**Formula**:
```
log_returns(t) = ln(price(t) / price(t-1))
```

**Usage**:
```python
from quantml import time_series
log_rets = time_series.log_returns(prices)
```

**Interpretation**: Symmetric returns, better for statistical modeling.

**Expected Distribution**: More normal than simple returns.

## Alpha Factors

### Momentum Factor

**Description**: Price momentum over lookback period.

**Formula**:
```
momentum(t, n) = (price(t) - price(t-n)) / price(t-n)
```

**Usage**:
```python
from examples.alpha_factors import MomentumFactor
mom = MomentumFactor.price_momentum(prices, lookback=20)
```

**Interpretation**: Positive momentum suggests continuation, negative suggests reversal.

**Expected Distribution**: Centered around 0, can be asymmetric.

### EMA Momentum

**Description**: Difference between fast and slow EMA.

**Formula**:
```
ema_momentum(t) = (EMA_fast(t) - EMA_slow(t)) / EMA_slow(t)
```

**Usage**:
```python
from examples.alpha_factors import MomentumFactor
ema_mom = MomentumFactor.ema_momentum(prices, fast=10, slow=20)
```

**Interpretation**: Crossover signal. Positive when fast EMA > slow EMA (uptrend).

**Expected Distribution**: Centered around 0.

### Mean Reversion Factor

**Description**: Z-score based mean reversion signal.

**Formula**:
```
mean_reversion(t) = -zscore(t, n)  # Negative for reversion
```

**Usage**:
```python
from examples.alpha_factors import MeanReversionFactor
mr = MeanReversionFactor.zscore_factor(prices, window=20)
```

**Interpretation**: Negative z-score suggests price below mean, potential bounce.

**Expected Distribution**: Standard normal.

## Feature Engineering Pipeline

### Feature Pipeline

The `FeaturePipeline` class provides a reproducible way to create features:

```python
from quantml.training import FeaturePipeline

pipeline = FeaturePipeline()

# Add lagged features
pipeline.add_lagged_feature('price', lags=[1, 5, 10, 20])

# Add rolling features
pipeline.add_rolling_feature('price', window=20, func='mean')
pipeline.add_rolling_feature('price', window=20, func='std')

# Add time-series features
pipeline.add_time_series_feature('price', 'returns')
pipeline.add_time_series_feature('price', 'volatility', n=20)

# Transform data
features = pipeline.transform({'price': prices})
```

### Feature Normalization

Features should be normalized for model training:

```python
from quantml.training.features import normalize_features

# Z-score normalization (recommended)
features = normalize_features(features, method='zscore')

# Min-max normalization
features = normalize_features(features, method='minmax')

# Robust normalization (uses median and IQR)
features = normalize_features(features, method='robust')
```

### Feature Selection

For research, you may want to test different feature sets:

```python
# Momentum-focused features
momentum_features = ['lagged_price_1', 'lagged_price_5', 'ema_momentum']

# Mean reversion features
reversion_features = ['zscore', 'vwap_deviation', 'rolling_mean']

# Volatility features
volatility_features = ['realized_vol', 'vol_of_vol', 'rolling_std']
```

## Feature Correlations

Understanding feature correlations helps avoid multicollinearity:

- **Lagged prices**: Highly correlated with each other (0.8-0.95)
- **Rolling mean/std**: Correlated with lagged prices (0.6-0.8)
- **Returns**: Low correlation with levels (-0.1 to 0.1)
- **Volatility**: Correlated with absolute returns (0.3-0.5)
- **Momentum factors**: Correlated with recent returns (0.5-0.7)

**Recommendation**: Use feature selection or dimensionality reduction if using many correlated features.

## Feature Importance

Use QuantML's feature importance tools:

```python
from quantml.training import FeatureImportanceTracker

tracker = FeatureImportanceTracker()
importance = tracker.compute_gradient_importance(model, features, targets)
```

This helps identify which features contribute most to predictions.

## Best Practices

1. **Normalize features**: Always normalize before training
2. **Avoid lookahead**: Only use past data for features
3. **Handle missing data**: Use forward fill or interpolation
4. **Feature engineering**: Domain knowledge > more features
5. **Test feature sets**: Compare different feature combinations
6. **Monitor correlations**: Avoid highly correlated features
7. **Cache features**: Use feature store for large datasets

