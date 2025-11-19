"""
Quick Alpha Generation - Generate alpha signals immediately

This script demonstrates how to generate alpha using the QuantML library.
Run this to see alpha signals generated from price data.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from quantml import Tensor
from quantml.models import Linear
from quantml.optim import Adam
from quantml.training import QuantTrainer, FeaturePipeline
from quantml.training.losses import sharpe_loss, mse_loss
from quantml.training.metrics import information_coefficient, sharpe_ratio
from quantml.training.alpha_eval import AlphaEvaluator
from quantml.training.backtest import BacktestEngine
from quantml.training.features import normalize_features
from examples.alpha_factors import MomentumFactor, MeanReversionFactor, AlphaFactorCombiner
import random


def generate_market_data(n=500):
    """Generate synthetic market data (replace with your data source)."""
    prices = []
    volumes = []
    base = 100.0
    
    for i in range(n):
        # Random walk with slight mean reversion
        drift = (100.0 - base) * 0.01
        noise = random.gauss(0, 0.5)
        base += drift + noise
        prices.append(base)
        volumes.append(100.0 + random.gauss(0, 10))
    
    return prices, volumes


def create_alpha_features(prices, volumes):
    """Create features using alpha factors and feature pipeline."""
    # Use alpha factors
    momentum = MomentumFactor.price_momentum(prices, lookback=20)
    ema_mom = MomentumFactor.ema_momentum(prices, fast=10, slow=20)
    zscore = MeanReversionFactor.zscore_factor(prices, window=20)
    vwap_dev = MeanReversionFactor.vwap_deviation(prices, volumes, window=20)
    
    # Clean NaN/inf values
    def clean_values(vals):
        return [0.0 if (not isinstance(x, (int, float)) or x != x or abs(x) > 1e10) else x for x in vals]
    
    momentum = clean_values(momentum)
    ema_mom = clean_values(ema_mom)
    zscore = clean_values(zscore)
    vwap_dev = clean_values(vwap_dev)
    
    # Combine factors
    combiner = AlphaFactorCombiner({
        'momentum': momentum,
        'ema_momentum': ema_mom,
        'zscore': zscore,
        'vwap_dev': vwap_dev
    }, weights={
        'momentum': 0.3,
        'ema_momentum': 0.3,
        'zscore': 0.2,
        'vwap_dev': 0.2
    })
    
    # Also use feature pipeline for additional features
    pipeline = FeaturePipeline()
    pipeline.add_lagged_feature('price', lags=[1, 5, 10])
    pipeline.add_rolling_feature('price', window=20, func='mean')
    pipeline.add_rolling_feature('price', window=20, func='std')
    pipeline.add_time_series_feature('price', 'returns')
    
    features = pipeline.transform({'price': prices})
    
    # Clean features before normalization
    for feat in features:
        for i, val in enumerate(feat):
            if not isinstance(val, (int, float)) or val != val or abs(val) > 1e10:
                feat[i] = 0.0
    
    features = normalize_features(features, method='zscore')
    
    # Add combined alpha factor to features
    combined_alpha = combiner.combine()
    combined_alpha = clean_values(combined_alpha)
    
    for i, feat in enumerate(features):
        if i < len(combined_alpha):
            feat.append(combined_alpha[i])
        else:
            feat.append(0.0)
    
    return features


def main():
    """Generate alpha signals."""
    print("=" * 70)
    print("Quick Alpha Generation")
    print("=" * 70)
    
    # Step 1: Get data
    print("\n[1/5] Loading market data...")
    prices, volumes = generate_market_data(500)
    print(f"Loaded {len(prices)} data points")
    
    # Step 2: Create features
    print("\n[2/5] Engineering alpha features...")
    features = create_alpha_features(prices, volumes)
    
    # Create targets (forward returns)
    targets = []
    for i in range(len(prices) - 1):
        ret = (prices[i + 1] - prices[i]) / prices[i] if prices[i] > 0 else 0.0
        targets.append(ret)
    
    features = features[:-1]  # Align
    
    print(f"Created {len(features)} samples with {len(features[0])} features")
    
    # Step 3: Train model
    print("\n[3/5] Training alpha model...")
    model = Linear(in_features=len(features[0]), out_features=1, bias=True)
    optimizer = Adam(model.parameters(), lr=0.001)
    # Use MSE loss for stability, can switch to sharpe_loss after model stabilizes
    trainer = QuantTrainer(model, optimizer, mse_loss)
    
    # Quick training
    train_size = int(len(features) * 0.7)
    X_train = features[:train_size]
    y_train = targets[:train_size]
    
    # Validate features
    for i, feat in enumerate(X_train):
        for j, val in enumerate(feat):
            if not isinstance(val, (int, float)) or val != val or abs(val) > 1e10:
                X_train[i][j] = 0.0
    
    for i, val in enumerate(y_train):
        if not isinstance(val, (int, float)) or val != val or abs(val) > 1e10:
            y_train[i] = 0.0
    
    for epoch in range(30):
        epoch_loss = 0.0
        count = 0
        for i in range(0, len(X_train), 10):
            try:
                x = Tensor([X_train[i]])
                y = Tensor([[y_train[i]]])
                loss = trainer.train_step(x, y)
                if isinstance(loss, (int, float)) and loss == loss:  # Check for NaN
                    epoch_loss += loss
                    count += 1
            except Exception:
                continue
        
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / count if count > 0 else 0.0
            print(f"  Epoch {epoch + 1}/30: Loss = {avg_loss:.6f}")
    
    # Step 4: Generate alpha signals
    print("\n[4/5] Generating alpha signals...")
    X_test = features[train_size:]
    y_test = targets[train_size:]
    
    predictions = []
    actuals = []
    
    for i in range(len(X_test)):
        x = Tensor([X_test[i]])
        pred = model.forward(x)
        pred_val = pred.data[0][0] if isinstance(pred.data[0], list) else pred.data[0]
        predictions.append(pred_val)
        actuals.append(y_test[i])
    
    # Step 5: Evaluate alpha
    print("\n[5/5] Evaluating alpha quality...")
    evaluator = AlphaEvaluator(predictions, actuals)
    metrics = evaluator.evaluate()
    
    print(f"\nAlpha Metrics:")
    print(f"  Information Coefficient (IC): {metrics['ic']:.4f}")
    print(f"  Rank IC: {metrics['rank_ic']:.4f}")
    print(f"  Hit Rate: {metrics['hit_rate']:.4f}")
    print(f"  Turnover: {metrics['turnover']:.4f}")
    
    # Backtest
    test_prices = prices[train_size+1:train_size+1+len(predictions)]
    backtest = BacktestEngine(initial_capital=100000.0)
    results = backtest.run_with_predictions(predictions, test_prices, targets=actuals)
    
    print(f"\nBacktest Results:")
    print(f"  Total Return: {results['total_return']*100:.2f}%")
    print(f"  Sharpe Ratio: {results['sharpe_ratio']:.4f}")
    print(f"  Max Drawdown: {results['max_drawdown']*100:.2f}%")
    
    # Show recent signals
    print(f"\nRecent Alpha Signals (last 10):")
    for i in range(max(0, len(predictions) - 10), len(predictions)):
        signal = predictions[i]
        if signal > 0.01:
            action = "BUY"
        elif signal < -0.01:
            action = "SELL"
        else:
            action = "HOLD"
        print(f"  Signal {i+1}: {signal:.4f} -> {action}")
    
    print("\n" + "=" * 70)
    print("Alpha generation complete!")
    print("=" * 70)
    print("\nTo use in production:")
    print("  1. Replace generate_market_data() with your data source")
    print("  2. Adjust feature engineering for your use case")
    print("  3. Train longer with walk-forward optimization")
    print("  4. Add risk management and position sizing")


if __name__ == "__main__":
    main()

