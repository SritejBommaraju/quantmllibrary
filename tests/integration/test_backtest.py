"""
Integration tests for backtesting.
"""

import pytest
from quantml.training.backtest import BacktestEngine


def test_backtest_engine_init():
    """Test BacktestEngine initialization."""
    engine = BacktestEngine(initial_capital=100000.0)
    
    assert engine.initial_capital == 100000.0


def test_backtest_simple():
    """Test simple backtest."""
    engine = BacktestEngine(initial_capital=100000.0)
    
    signals = [0.1, -0.1, 0.2, -0.1, 0.1]
    prices = [100.0, 101.0, 102.0, 101.0, 102.0]
    
    results = engine.run_with_predictions(signals, prices)
    
    assert 'total_return' in results
    assert 'sharpe_ratio' in results
    assert 'max_drawdown' in results
    assert 'final_value' in results

