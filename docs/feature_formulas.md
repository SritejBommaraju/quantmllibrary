# Feature Formulas Reference

Mathematical formulas for all features in QuantML.

## Price Features

### Lagged Price
```
lag_k(t) = price(t - k)
```

### Rolling Mean
```
μ(t, n) = (1/n) * Σ_{i=0}^{n-1} price(t - i)
```

### Rolling Standard Deviation
```
σ(t, n) = sqrt((1/n) * Σ_{i=0}^{n-1} (price(t-i) - μ(t, n))²)
```

### Exponential Moving Average
```
EMA(t, n) = α * price(t) + (1 - α) * EMA(t-1, n)
where α = 2 / (n + 1)
```

### Weighted Moving Average
```
WMA(t, n) = Σ_{i=0}^{n-1} w_i * price(t-i) / Σ w_i
where w_i = n - i
```

## Return Features

### Simple Returns
```
r(t) = (price(t) - price(t-1)) / price(t-1)
```

### Log Returns
```
r_log(t) = ln(price(t) / price(t-1))
```

### Cumulative Returns
```
R(t, n) = Π_{i=1}^{n} (1 + r(t-i)) - 1
```

## Volatility Features

### Realized Volatility (Annualized)
```
σ_realized(t, n) = std(r(t-n:t)) * sqrt(252)
```

### Parkinson Volatility (High-Low)
```
σ_parkinson(t) = sqrt((1/(4*ln(2))) * (ln(high(t)/low(t)))²)
```

### Garman-Klass Volatility
```
σ_gk(t) = sqrt(0.5 * (ln(high(t)/low(t)))² - (2*ln(2)-1) * (ln(close(t)/open(t)))²)
```

## Volume Features

### Volume-Weighted Average Price
```
VWAP(t, n) = Σ_{i=0}^{n-1} (price(t-i) * volume(t-i)) / Σ_{i=0}^{n-1} volume(t-i)
```

### VWAP Deviation
```
vwap_dev(t) = (price(t) - VWAP(t, n)) / VWAP(t, n)
```

### Volume Ratio
```
volume_ratio(t, n) = volume(t) / mean(volume(t-n:t))
```

## Order Flow Features

### Order Flow Imbalance
```
OFI(t) = bid_volume(t) - ask_volume(t)
```

### Microprice
```
microprice(t) = (bid(t) * ask_volume(t) + ask(t) * bid_volume(t)) / (bid_volume(t) + ask_volume(t))
```

### Bid-Ask Spread
```
spread(t) = ask(t) - bid(t)
```

### Mid-Price
```
mid(t) = (bid(t) + ask(t)) / 2
```

## Statistical Features

### Z-Score
```
zscore(t, n) = (price(t) - μ(t, n)) / σ(t, n)
```

### Skewness
```
skew(t, n) = (1/n) * Σ ((price(t-i) - μ) / σ)³
```

### Kurtosis
```
kurt(t, n) = (1/n) * Σ ((price(t-i) - μ) / σ)⁴ - 3
```

## Momentum Features

### Price Momentum
```
momentum(t, n) = (price(t) - price(t-n)) / price(t-n)
```

### Rate of Change (ROC)
```
ROC(t, n) = (price(t) - price(t-n)) / price(t-n) * 100
```

### Relative Strength Index (RSI)
```
RSI(t, n) = 100 - (100 / (1 + RS))
where RS = avg_gain / avg_loss over n periods
```

## Mean Reversion Features

### Price Deviation from Mean
```
deviation(t, n) = price(t) - μ(t, n)
```

### Percentile Rank
```
percentile_rank(t, n) = (count(price(t-i) < price(t)) / n) * 100
```

## Alpha Factor Formulas

### Momentum Alpha
```
α_momentum(t) = w1 * momentum(t, 5) + w2 * momentum(t, 20)
```

### Mean Reversion Alpha
```
α_reversion(t) = -w1 * zscore(t, 20) - w2 * vwap_dev(t)
```

### Combined Alpha
```
α_combined(t) = w_m * α_momentum(t) + w_r * α_reversion(t) + w_v * volatility_factor(t)
```

## Feature Normalization

### Z-Score Normalization
```
x_norm = (x - μ) / σ
```

### Min-Max Normalization
```
x_norm = (x - min) / (max - min)
```

### Robust Normalization
```
x_norm = (x - median) / IQR
where IQR = Q3 - Q1
```

## Feature Engineering Operations

### Feature Interaction
```
interaction(x1, x2) = x1 * x2
```

### Feature Ratio
```
ratio(x1, x2) = x1 / x2  (with zero handling)
```

### Feature Difference
```
diff(x1, x2) = x1 - x2
```

## Time-Based Features

### Day of Week
```
dow(t) ∈ {0, 1, 2, 3, 4}  # Monday=0, Friday=4
```

### Hour of Day
```
hour(t) ∈ {0, 1, ..., 23}
```

### Time Since Market Open
```
time_since_open(t) = minutes since 9:30 AM
```

## Regime Features

### Volatility Regime
```
regime_vol(t) = {
    0 if σ(t) < σ_25th_percentile  # Low vol
    1 if σ_25th ≤ σ(t) < σ_75th    # Normal vol
    2 if σ(t) ≥ σ_75th             # High vol
}
```

### Volume Regime
```
regime_vol(t) = {
    0 if volume(t) < volume_25th   # Low volume
    1 if volume_25th ≤ volume < volume_75th  # Normal
    2 if volume ≥ volume_75th      # High volume
}
```

