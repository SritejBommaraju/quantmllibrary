"""
Example: Alpha Training Pipeline

This example demonstrates a complete alpha generation and training pipeline:
1. Feature engineering
2. Model training with quant-specific loss
3. Alpha evaluation
4. Backtesting
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from quantml import Tensor
from quantml.models import Linear
from quantml.optim import Adam
from quantml.training import QuantTrainer
from quantml.training.losses import sharpe_loss, combined_quant_loss
from quantml.training.metrics import information_coefficient, sharpe_ratio
from quantml.training.alpha_eval import AlphaEvaluator
from quantml.training.backtest import BacktestEngine
from quantml.training.features import FeaturePipeline, normalize_features
from quantml import time_series


def generate_market_data(n_samples=1000):
    """Generate synthetic market data."""
    import random
    prices = []
    volumes = []
    base_price = 100.0
    
    for i in range(n_samples):
        # Random walk with drift
        change = random.gauss(0.0, 0.5)
        price = base_price + change
        prices.append(price)
        volumes.append(100.0 + random.gauss(0, 10))
        base_price = price
    
    return prices, volumes


def create_features(prices, volumes):
    """Create features using feature pipeline."""
    pipeline = FeaturePipeline()
    
    # Add lagged price features
    pipeline.add_lagged_feature('price', lags=[1, 5, 10, 20])
    
    # Add rolling features
    pipeline.add_rolling_feature('price', window=20, func='mean')
    pipeline.add_rolling_feature('price', window=20, func='std')
    
    # Add time-series features
    pipeline.add_time_series_feature('price', 'returns')
    pipeline.add_time_series_feature('price', 'volatility', n=20)
    
    # Transform
    data = {'price': prices, 'volume': volumes}
    features = pipeline.transform(data)
    
    # Normalize
    features = normalize_features(features, method='zscore')
    
    return features


def main():
    """Run alpha training example."""
    print("=== Alpha Training Pipeline Example ===\n")
    
    # Generate synthetic data
    print("Generating synthetic market data...")
    prices, volumes = generate_market_data(n_samples=1000)
    print(f"Generated {len(prices)} samples\n")
    
    # Create features
    print("Creating features...")
    features = create_features(prices, volumes)
    
    # Create targets (next period returns)
    targets = []
    for i in range(len(prices) - 1):
        ret = (prices[i + 1] - prices[i]) / prices[i] if prices[i] > 0 else 0.0
        targets.append(ret)
    
    # Align features and targets (skip last price, features already aligned)
    features = features[:-1]
    
    print(f"Created {len(features)} feature vectors with {len(features[0])} features each\n")
    
    # Split data
    train_size = int(len(features) * 0.7)
    X_train = features[:train_size]
    y_train = targets[:train_size]
    X_test = features[train_size:]
    y_test = targets[train_size:]
    
    print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples\n")
    
    # Create model
    print("Initializing model...")
    model = Linear(in_features=len(features[0]), out_features=1, bias=True)
    
    # Create optimizer
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # Create trainer with Sharpe loss
    print("Training model with Sharpe ratio loss...")
    trainer = QuantTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=sharpe_loss
    )
    
    # Train
    history = trainer.train(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        epochs=50,
        batch_size=32,
        early_stopping={'patience': 10, 'min_delta': 0.0001},
        verbose=True
    )
    
    print(f"\nFinal train loss: {history['train_loss'][-1]:.6f}")
    print(f"Final val loss: {history['val_loss'][-1]:.6f}\n")
    
    # Evaluate alpha
    print("Evaluating alpha signals...")
    predictions = []
    actuals = []
    
    for i in range(len(X_test)):
        x = Tensor([X_test[i]])
        pred = model.forward(x)
        pred_val = pred.data[0][0] if isinstance(pred.data[0], list) else pred.data[0]
        predictions.append(pred_val)
        actuals.append(y_test[i])
    
    evaluator = AlphaEvaluator(predictions, actuals)
    alpha_metrics = evaluator.evaluate()
    
    print(f"Information Coefficient: {alpha_metrics['ic']:.4f}")
    print(f"Rank IC: {alpha_metrics['rank_ic']:.4f}")
    print(f"Hit Rate: {alpha_metrics['hit_rate']:.4f}")
    print(f"Turnover: {alpha_metrics['turnover']:.4f}")
    print(f"Signal Half-Life: {alpha_metrics['half_life']} periods\n")
    
    # Backtest
    print("Running backtest...")
    test_prices = prices[train_size+1:train_size+1+len(predictions)]
    backtest_engine = BacktestEngine(initial_capital=100000.0)
    backtest_results = backtest_engine.run_with_predictions(
        predictions, test_prices, targets=actuals
    )
    
    print(f"Initial Capital: ${backtest_results['initial_capital']:,.2f}")
    print(f"Final Value: ${backtest_results['final_value']:,.2f}")
    print(f"Total Return: {backtest_results['total_return']*100:.2f}%")
    print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.4f}")
    print(f"Max Drawdown: {backtest_results['max_drawdown']*100:.2f}%")
    print(f"IC: {backtest_results.get('ic', 0.0):.4f}")
    print("\nExample completed!")


if __name__ == "__main__":
    main()

