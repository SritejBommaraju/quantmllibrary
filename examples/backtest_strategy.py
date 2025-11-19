"""
Example: Backtesting Strategy

This example demonstrates how to backtest a trading strategy using
the backtesting engine with position sizing and transaction costs.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from quantml import Tensor
from quantml.models import Linear
from quantml.training.backtest import BacktestEngine
from quantml.training.metrics import sharpe_ratio, max_drawdown
from quantml import time_series


def generate_price_data(n_samples=500):
    """Generate synthetic price data."""
    import random
    prices = []
    base_price = 100.0
    
    for i in range(n_samples):
        # Random walk with slight upward drift
        change = random.gauss(0.05, 1.0)
        price = base_price + change
        prices.append(price)
        base_price = price
    
    return prices


def generate_signals_from_model(prices, model):
    """Generate trading signals from a model."""
    signals = []
    
    # Create simple features (returns, volatility)
    price_tensor = Tensor([prices])
    returns = time_series.returns(price_tensor)
    volatility = time_series.volatility(price_tensor, n=20, annualize=False)
    
    ret_data = returns.data[0] if isinstance(returns.data[0], list) else returns.data
    vol_data = volatility.data[0] if isinstance(volatility.data[0], list) else volatility.data
    
    for i in range(20, len(prices)):  # Skip warm-up period
        # Feature: recent return and volatility
        feature = [ret_data[i] if i < len(ret_data) else 0.0,
                  vol_data[i] if i < len(vol_data) else 0.0]
        
        x = Tensor([feature])
        pred = model.forward(x)
        signal = pred.data[0][0] if isinstance(pred.data[0], list) else pred.data[0]
        
        # Normalize signal to [-1, 1]
        signals.append(max(-1.0, min(1.0, signal)))
    
    return signals


def main():
    """Run backtesting example."""
    print("=== Backtesting Strategy Example ===\n")
    
    # Generate price data
    print("Generating synthetic price data...")
    prices = generate_price_data(n_samples=500)
    print(f"Generated {len(prices)} price points\n")
    
    # Create a simple model (pretend it's trained)
    print("Creating model...")
    model = Linear(in_features=2, out_features=1, bias=True)
    
    # Generate signals
    print("Generating trading signals...")
    signals = generate_signals_from_model(prices, model)
    signal_prices = prices[20:]  # Align with signals
    
    print(f"Generated {len(signals)} signals\n")
    
    # Create backtest engine
    print("Running backtest...")
    engine = BacktestEngine(
        initial_capital=100000.0,
        commission=0.001,  # 0.1% commission
        slippage=0.0005    # 0.05% slippage
    )
    
    # Run backtest
    results = engine.run(signals, signal_prices)
    
    # Display results
    print("=== Backtest Results ===")
    print(f"Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"Final Value: ${results['final_value']:,.2f}")
    print(f"Total Return: {results['total_return']*100:.2f}%")
    print(f"Number of Trades: {results['n_trades']}")
    print(f"Win Rate: {results['win_rate']*100:.2f}%")
    print(f"\nPerformance Metrics:")
    print(f"  Sharpe Ratio: {results['sharpe_ratio']:.4f}")
    print(f"  Sortino Ratio: {results['sortino_ratio']:.4f}")
    print(f"  Calmar Ratio: {results['calmar_ratio']:.4f}")
    print(f"  Max Drawdown: {results['max_drawdown']*100:.2f}%")
    
    # Equity curve statistics
    equity = results['equity_curve']
    if len(equity) > 0:
        print(f"\nEquity Curve:")
        print(f"  Peak: ${max(equity):,.2f}")
        print(f"  Trough: ${min(equity):,.2f}")
        print(f"  Final: ${equity[-1]:,.2f}")
    
    print("\nExample completed!")


if __name__ == "__main__":
    main()

