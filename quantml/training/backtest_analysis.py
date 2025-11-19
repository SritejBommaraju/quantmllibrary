"""
Backtest analysis and visualization utilities.

Provides trade-level analysis, performance heatmaps, and regime-based breakdowns.
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict


def analyze_trades(completed_trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze completed trades for detailed statistics.
    
    Args:
        completed_trades: List of completed trade dictionaries
    
    Returns:
        Dictionary with trade analysis
    """
    if not completed_trades:
        return {
            'n_trades': 0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'avg_duration': 0.0
        }
    
    winning_trades = [t for t in completed_trades if t.get('pnl', 0) > 0]
    losing_trades = [t for t in completed_trades if t.get('pnl', 0) <= 0]
    
    win_rate = len(winning_trades) / len(completed_trades)
    avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0.0
    avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0.0
    
    total_win = sum(t['pnl'] for t in winning_trades)
    total_loss = abs(sum(t['pnl'] for t in losing_trades))
    profit_factor = total_win / total_loss if total_loss > 0 else float('inf') if total_win > 0 else 0.0
    
    avg_duration = sum(t.get('duration', 0) for t in completed_trades) / len(completed_trades)
    
    # Largest win/loss
    largest_win = max((t['pnl'] for t in winning_trades), default=0.0)
    largest_loss = min((t['pnl'] for t in losing_trades), default=0.0)
    
    return {
        'n_trades': len(completed_trades),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'avg_duration': avg_duration,
        'largest_win': largest_win,
        'largest_loss': largest_loss,
        'total_pnl': sum(t['pnl'] for t in completed_trades)
    }


def create_performance_heatmap(
    trades: List[Dict[str, Any]],
    prices: List[float],
    regimes: Optional[List[int]] = None,
    time_of_day: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Create performance heatmap by regime and time of day.
    
    Args:
        trades: List of trades
        prices: Price data
        regimes: Regime labels (0=low, 1=normal, 2=high)
        time_of_day: Hour of day (0-23)
    
    Returns:
        Dictionary with heatmap data
    """
    if not trades:
        return {}
    
    # Group trades by regime and time
    heatmap_data = defaultdict(lambda: {'pnl': 0.0, 'count': 0})
    
    for trade in trades:
        idx = trade.get('index', 0)
        
        regime = regimes[idx] if regimes and idx < len(regimes) else 1  # Default to normal
        hour = time_of_day[idx] if time_of_day and idx < len(time_of_day) else 12  # Default to noon
        
        key = f"regime_{regime}_hour_{hour}"
        heatmap_data[key]['pnl'] += trade.get('pnl', 0.0)
        heatmap_data[key]['count'] += 1
    
    # Convert to structured format
    result = {}
    for key, data in heatmap_data.items():
        result[key] = {
            'pnl': data['pnl'],
            'count': data['count'],
            'avg_pnl': data['pnl'] / data['count'] if data['count'] > 0 else 0.0
        }
    
    return result


def analyze_by_regime(
    completed_trades: List[Dict[str, Any]],
    regimes: List[int]
) -> Dict[int, Dict[str, Any]]:
    """
    Analyze performance by volatility/volume regime.
    
    Args:
        completed_trades: Completed trades
        regimes: Regime labels for each time period
    
    Returns:
        Dictionary mapping regime -> statistics
    """
    regime_trades = defaultdict(list)
    
    for trade in completed_trades:
        entry_idx = trade.get('entry_index', 0)
        if entry_idx < len(regimes):
            regime = regimes[entry_idx]
            regime_trades[regime].append(trade)
    
    results = {}
    for regime, trades in regime_trades.items():
        results[regime] = analyze_trades(trades)
        results[regime]['regime'] = regime
        results[regime]['n_trades'] = len(trades)
    
    return results


def create_trade_summary(completed_trades: List[Dict[str, Any]]) -> str:
    """
    Create human-readable trade summary.
    
    Args:
        completed_trades: Completed trades
    
    Returns:
        Formatted summary string
    """
    if not completed_trades:
        return "No completed trades."
    
    analysis = analyze_trades(completed_trades)
    
    summary = f"""
Trade Summary
=============
Total Trades: {analysis['n_trades']}
Win Rate: {analysis['win_rate']*100:.2f}%
Average Win: ${analysis['avg_win']:.2f}
Average Loss: ${analysis['avg_loss']:.2f}
Profit Factor: {analysis['profit_factor']:.2f}
Average Duration: {analysis['avg_duration']:.1f} periods
Largest Win: ${analysis['largest_win']:.2f}
Largest Loss: ${analysis['largest_loss']:.2f}
Total P&L: ${analysis['total_pnl']:.2f}
"""
    
    return summary

