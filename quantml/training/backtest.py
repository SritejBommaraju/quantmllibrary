"""
Backtesting engine for strategy evaluation.

This module provides a backtesting framework for evaluating trading strategies
with position sizing, transaction costs, and performance metrics.
"""

from typing import List, Optional, Callable, Dict, Any
from quantml.training.metrics import (
    sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown
)

# Try to import NumPy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


class BacktestEngine:
    """
    Backtesting engine for strategy evaluation.
    
    This class simulates trading a strategy on historical data, tracking
    positions, P&L, and performance metrics.
    
    Attributes:
        initial_capital: Starting capital
        commission: Commission per trade (as fraction)
        slippage: Slippage per trade (as fraction)
        position_sizing: Position sizing function
    
    Examples:
        >>> engine = BacktestEngine(initial_capital=100000)
        >>> signals = [0.5, -0.3, 0.8, ...]  # Trading signals
        >>> prices = [100, 101, 102, ...]   # Price data
        >>> results = engine.run(signals, prices)
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.001,  # 0.1%
        slippage: float = 0.0005,   # 0.05%
        position_sizing: Optional[Callable] = None
    ):
        """
        Initialize backtesting engine.
        
        Args:
            initial_capital: Starting capital
            commission: Commission rate per trade
            slippage: Slippage rate per trade
            position_sizing: Function to determine position size from signal
                            Default: signal * capital (simple)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_sizing = position_sizing if position_sizing else self._default_position_sizing
    
    def _default_position_sizing(self, signal: float, capital: float, price: float) -> float:
        """Default position sizing: signal * capital / price."""
        return signal * capital / price if price > 0 else 0.0
    
    def run(
        self,
        signals: List[float],
        prices: List[float],
        volumes: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Run backtest on signals and prices.
        
        Args:
            signals: Trading signals (-1 to 1, or position sizes)
            prices: Price data
            volumes: Optional volume data (for VWAP execution)
        
        Returns:
            Dictionary with backtest results and metrics
        """
        if len(signals) != len(prices):
            raise ValueError("signals and prices must have same length")
        
        n = len(signals)
        capital = self.initial_capital
        position = 0.0  # Current position (number of shares)
        equity_curve = [capital]
        trades = []
        returns = []
        
        for i in range(n):
            signal = signals[i]
            price = prices[i]
            
            # Determine target position
            target_position_value = self.position_sizing(signal, capital, price)
            target_position = target_position_value / price if price > 0 else 0.0
            
            # Calculate trade
            trade_size = target_position - position
            
            if abs(trade_size) > 1e-6:  # Only trade if significant
                # Apply slippage to execution price
                execution_price = price * (1 + self.slippage * (1 if trade_size > 0 else -1))

                # Calculate costs (commission only - slippage already in execution_price)
                trade_value = abs(trade_size * execution_price)
                commission_cost = trade_value * self.commission
                total_cost = commission_cost

                # Update capital
                capital -= trade_size * execution_price + total_cost
                
                # Update position
                position = target_position
                
                trades.append({
                    'index': i,
                    'price': price,
                    'execution_price': execution_price,
                    'size': trade_size,
                    'cost': total_cost
                })
            
            # Update equity
            current_value = capital + position * price
            equity_curve.append(current_value)
            
            # Calculate return
            if i > 0:
                prev_value = equity_curve[-2]
                ret = (current_value - prev_value) / prev_value if prev_value > 0 else 0.0
                returns.append(ret)
        
        # Calculate metrics
        final_value = equity_curve[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Performance metrics
        sharpe = sharpe_ratio(returns) if returns else 0.0
        sortino = sortino_ratio(returns) if returns else 0.0
        calmar = calmar_ratio(returns) if returns else 0.0
        max_dd = max_drawdown(returns) if returns else 0.0
        
        # Calculate trade-level P&L
        completed_trades = []
        open_trade = None
        
        for i, trade in enumerate(trades):
            if open_trade is None:
                # Open new trade
                open_trade = {
                    'entry_index': trade['index'],
                    'entry_price': trade['execution_price'],
                    'entry_size': trade['size'],
                    'direction': 1 if trade['size'] > 0 else -1
                }
            else:
                # Check if trade closes position
                if (open_trade['direction'] > 0 and trade['size'] < 0) or \
                   (open_trade['direction'] < 0 and trade['size'] > 0):
                    # Close trade
                    exit_price = trade['execution_price']
                    exit_size = abs(trade['size'])
                    entry_size = abs(open_trade['entry_size'])
                    
                    # Calculate P&L
                    if open_trade['direction'] > 0:  # Long
                        pnl = (exit_price - open_trade['entry_price']) * min(entry_size, exit_size)
                    else:  # Short
                        pnl = (open_trade['entry_price'] - exit_price) * min(entry_size, exit_size)
                    
                    pnl -= trade['cost']  # Subtract exit cost
                    
                    completed_trades.append({
                        'entry_index': open_trade['entry_index'],
                        'exit_index': trade['index'],
                        'entry_price': open_trade['entry_price'],
                        'exit_price': exit_price,
                        'size': min(entry_size, exit_size),
                        'direction': open_trade['direction'],
                        'pnl': pnl,
                        'duration': trade['index'] - open_trade['entry_index'],
                        'return_pct': pnl / (open_trade['entry_price'] * min(entry_size, exit_size)) if open_trade['entry_price'] > 0 else 0.0
                    })
                    
                    # Update open trade if partial close
                    if exit_size < entry_size:
                        open_trade['entry_size'] = open_trade['entry_size'] - (exit_size if open_trade['direction'] > 0 else -exit_size)
                    else:
                        open_trade = None
        
        # Trade statistics
        n_trades = len(trades)
        n_completed_trades = len(completed_trades)
        winning_trades = [t for t in completed_trades if t['pnl'] > 0]
        losing_trades = [t for t in completed_trades if t['pnl'] <= 0]
        
        win_rate = len(winning_trades) / n_completed_trades if n_completed_trades > 0 else 0.0
        avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0.0
        avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0.0
        profit_factor = abs(sum(t['pnl'] for t in winning_trades) / sum(t['pnl'] for t in losing_trades)) if losing_trades and sum(t['pnl'] for t in losing_trades) != 0 else float('inf') if winning_trades else 0.0
        
        avg_duration = sum(t['duration'] for t in completed_trades) / n_completed_trades if n_completed_trades > 0 else 0.0
        
        return {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'equity_curve': equity_curve,
            'returns': returns,
            'trades': trades,
            'completed_trades': completed_trades,
            'n_trades': n_trades,
            'n_completed_trades': n_completed_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_duration': avg_duration,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_dd
        }
    
    def run_with_predictions(
        self,
        predictions: List[float],
        prices: List[float],
        targets: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Run backtest using model predictions as signals.
        
        Args:
            predictions: Model predictions (can be returns, prices, or signals)
            prices: Price data
            targets: Optional target returns (for comparison)
        
        Returns:
            Backtest results dictionary
        """
        # Convert predictions to signals (normalize to -1 to 1)
        if HAS_NUMPY:
            try:
                pred_arr = np.array(predictions)
                # Normalize to [-1, 1]
                pred_min, pred_max = np.min(pred_arr), np.max(pred_arr)
                if pred_max > pred_min:
                    signals = 2 * (pred_arr - pred_min) / (pred_max - pred_min) - 1
                else:
                    signals = pred_arr * 0  # All zeros
                signals = signals.tolist()
            except (ValueError, TypeError):
                signals = predictions
        else:
            # Pure Python normalization
            pred_min = min(predictions)
            pred_max = max(predictions)
            if pred_max > pred_min:
                signals = [2 * (p - pred_min) / (pred_max - pred_min) - 1 for p in predictions]
            else:
                signals = [0.0] * len(predictions)
        
        results = self.run(signals, prices)
        
        if targets is not None:
            # Add prediction accuracy metrics
            from quantml.training.metrics import information_coefficient, hit_rate
            results['ic'] = information_coefficient(predictions, targets)
            results['hit_rate'] = hit_rate(predictions, targets)
        
        return results

