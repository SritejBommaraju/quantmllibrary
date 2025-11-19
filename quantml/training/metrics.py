"""
Financial and prediction metrics for quant model evaluation.

This module provides metrics commonly used in quantitative trading,
including Sharpe ratio, Information Coefficient (IC), drawdown, and more.
"""

from typing import List, Optional, Union
import math

# Try to import NumPy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


def sharpe_ratio(returns: Union[List, any], risk_free_rate: float = 0.0, annualize: bool = True) -> float:
    """
    Calculate Sharpe ratio: (mean(returns) - risk_free_rate) / std(returns)
    
    Args:
        returns: List of returns
        risk_free_rate: Risk-free rate (default: 0.0)
        annualize: Whether to annualize (multiply by sqrt(252))
    
    Returns:
        Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0
    
    if HAS_NUMPY:
        try:
            ret_arr = np.array(returns, dtype=np.float64)
            mean_ret = np.mean(ret_arr)
            std_ret = np.std(ret_arr)
            if std_ret == 0:
                return 0.0
            sharpe = (mean_ret - risk_free_rate) / std_ret
            if annualize:
                sharpe *= math.sqrt(252)
            return float(sharpe)
        except (ValueError, TypeError):
            pass
    
    # Pure Python fallback
    mean_ret = sum(returns) / len(returns)
    variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
    std_ret = math.sqrt(variance) if variance > 0 else 0.0
    
    if std_ret == 0:
        return 0.0
    
    sharpe = (mean_ret - risk_free_rate) / std_ret
    if annualize:
        sharpe *= math.sqrt(252)
    return sharpe


def sortino_ratio(returns: Union[List, any], risk_free_rate: float = 0.0, annualize: bool = True) -> float:
    """
    Calculate Sortino ratio: (mean(returns) - risk_free_rate) / downside_std(returns)
    
    Args:
        returns: List of returns
        risk_free_rate: Risk-free rate
        annualize: Whether to annualize
    
    Returns:
        Sortino ratio
    """
    if len(returns) == 0:
        return 0.0
    
    if HAS_NUMPY:
        try:
            ret_arr = np.array(returns, dtype=np.float64)
            mean_ret = np.mean(ret_arr)
            # Downside deviation: only negative returns
            downside = ret_arr[ret_arr < 0]
            if len(downside) == 0:
                return float('inf') if mean_ret > risk_free_rate else 0.0
            downside_std = np.std(downside)
            if downside_std == 0:
                return 0.0
            sortino = (mean_ret - risk_free_rate) / downside_std
            if annualize:
                sortino *= math.sqrt(252)
            return float(sortino)
        except (ValueError, TypeError):
            pass
    
    # Pure Python fallback
    mean_ret = sum(returns) / len(returns)
    downside = [r for r in returns if r < 0]
    if len(downside) == 0:
        return float('inf') if mean_ret > risk_free_rate else 0.0
    
    downside_mean = sum(downside) / len(downside)
    downside_var = sum((d - downside_mean) ** 2 for d in downside) / len(downside)
    downside_std = math.sqrt(downside_var) if downside_var > 0 else 0.0
    
    if downside_std == 0:
        return 0.0
    
    sortino = (mean_ret - risk_free_rate) / downside_std
    if annualize:
        sortino *= math.sqrt(252)
    return sortino


def calmar_ratio(returns: Union[List, any], annualize: bool = True) -> float:
    """
    Calculate Calmar ratio: annual_return / max_drawdown
    
    Args:
        returns: List of returns
        annualize: Whether to annualize returns
    
    Returns:
        Calmar ratio
    """
    if len(returns) == 0:
        return 0.0
    
    annual_return = sum(returns) / len(returns)
    if annualize:
        annual_return *= 252
    
    max_dd = max_drawdown(returns)
    if max_dd == 0:
        return 0.0
    
    return abs(annual_return / max_dd)


def max_drawdown(returns: Union[List, any]) -> float:
    """
    Calculate maximum drawdown.
    
    Args:
        returns: List of returns
    
    Returns:
        Maximum drawdown (as positive value)
    """
    if len(returns) == 0:
        return 0.0
    
    if HAS_NUMPY:
        try:
            ret_arr = np.array(returns, dtype=np.float64)
            # Cumulative returns
            cum_ret = np.cumprod(1 + ret_arr)
            # Running maximum
            running_max = np.maximum.accumulate(cum_ret)
            # Drawdown
            drawdown = (cum_ret - running_max) / running_max
            return float(abs(np.min(drawdown)))
        except (ValueError, TypeError):
            pass
    
    # Pure Python fallback
    cum_ret = 1.0
    running_max = 1.0
    max_dd = 0.0
    
    for ret in returns:
        cum_ret *= (1 + ret)
        running_max = max(running_max, cum_ret)
        dd = (cum_ret - running_max) / running_max
        max_dd = min(max_dd, dd)
    
    return abs(max_dd)


def information_coefficient(predictions: Union[List, any], actuals: Union[List, any]) -> float:
    """
    Calculate Information Coefficient (IC): correlation between predictions and actuals.
    
    Args:
        predictions: Predicted values
        actuals: Actual/realized values
    
    Returns:
        IC (correlation coefficient)
    """
    if len(predictions) != len(actuals) or len(predictions) == 0:
        return 0.0
    
    if HAS_NUMPY:
        try:
            pred_arr = np.array(predictions, dtype=np.float64)
            actual_arr = np.array(actuals, dtype=np.float64)
            # Pearson correlation
            corr = np.corrcoef(pred_arr, actual_arr)[0, 1]
            return float(corr) if not np.isnan(corr) else 0.0
        except (ValueError, TypeError):
            pass
    
    # Pure Python fallback
    n = len(predictions)
    pred_mean = sum(predictions) / n
    actual_mean = sum(actuals) / n
    
    numerator = sum((predictions[i] - pred_mean) * (actuals[i] - actual_mean) for i in range(n))
    pred_var = sum((p - pred_mean) ** 2 for p in predictions)
    actual_var = sum((a - actual_mean) ** 2 for a in actuals)
    
    denominator = math.sqrt(pred_var * actual_var)
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def rank_ic(predictions: Union[List, any], actuals: Union[List, any]) -> float:
    """
    Calculate Rank IC: Spearman rank correlation.
    
    Args:
        predictions: Predicted values
        actuals: Actual values
    
    Returns:
        Rank IC
    """
    if len(predictions) != len(actuals) or len(predictions) == 0:
        return 0.0
    
    if HAS_NUMPY:
        try:
            from scipy.stats import spearmanr
            corr, _ = spearmanr(predictions, actuals)
            return float(corr) if not np.isnan(corr) else 0.0
        except ImportError:
            # Fallback to manual rank correlation
            pass
    
    # Manual rank correlation
    n = len(predictions)
    pred_ranks = _get_ranks(predictions)
    actual_ranks = _get_ranks(actuals)
    
    return information_coefficient(pred_ranks, actual_ranks)


def _get_ranks(values: List) -> List:
    """Get ranks of values."""
    sorted_vals = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0] * len(values)
    for rank, (idx, _) in enumerate(sorted_vals, 1):
        ranks[idx] = rank
    return ranks


def hit_rate(predictions: Union[List, any], actuals: Union[List, any]) -> float:
    """
    Calculate hit rate: percentage of correct directional predictions.
    
    Args:
        predictions: Predicted values
        actuals: Actual values
    
    Returns:
        Hit rate (0.0 to 1.0)
    """
    if len(predictions) != len(actuals) or len(predictions) == 0:
        return 0.0
    
    correct = 0
    for i in range(len(predictions)):
        pred_dir = 1 if predictions[i] > 0 else -1
        actual_dir = 1 if actuals[i] > 0 else -1
        if pred_dir == actual_dir:
            correct += 1
    
    return correct / len(predictions)


def var(returns: Union[List, any], confidence: float = 0.05) -> float:
    """
    Calculate Value at Risk (VaR).
    
    Args:
        returns: List of returns
        confidence: Confidence level (default: 0.05 for 95% VaR)
    
    Returns:
        VaR (negative value)
    """
    if len(returns) == 0:
        return 0.0
    
    if HAS_NUMPY:
        try:
            ret_arr = np.array(returns, dtype=np.float64)
            var_val = np.percentile(ret_arr, confidence * 100)
            return float(var_val)
        except (ValueError, TypeError):
            pass
    
    # Pure Python fallback
    sorted_returns = sorted(returns)
    idx = int(len(sorted_returns) * confidence)
    return sorted_returns[idx] if idx < len(sorted_returns) else sorted_returns[-1]


def cvar(returns: Union[List, any], confidence: float = 0.05) -> float:
    """
    Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.
    
    Args:
        returns: List of returns
        confidence: Confidence level
    
    Returns:
        CVaR (negative value)
    """
    if len(returns) == 0:
        return 0.0
    
    var_val = var(returns, confidence)
    
    if HAS_NUMPY:
        try:
            ret_arr = np.array(returns, dtype=np.float64)
            tail_losses = ret_arr[ret_arr <= var_val]
            if len(tail_losses) == 0:
                return var_val
            return float(np.mean(tail_losses))
        except (ValueError, TypeError):
            pass
    
    # Pure Python fallback
    tail_losses = [r for r in returns if r <= var_val]
    if len(tail_losses) == 0:
        return var_val
    return sum(tail_losses) / len(tail_losses)


def turnover(positions: Union[List, any]) -> float:
    """
    Calculate portfolio turnover.
    
    Args:
        positions: List of position values (can be weights)
    
    Returns:
        Average turnover
    """
    if len(positions) < 2:
        return 0.0
    
    if HAS_NUMPY:
        try:
            pos_arr = np.array(positions, dtype=np.float64)
            changes = np.abs(np.diff(pos_arr, axis=0))
            return float(np.mean(np.sum(changes, axis=1) if len(changes.shape) > 1 else changes))
        except (ValueError, TypeError):
            pass
    
    # Pure Python fallback
    total_turnover = 0.0
    for i in range(1, len(positions)):
        if isinstance(positions[i], list) and isinstance(positions[i-1], list):
            change = sum(abs(positions[i][j] - positions[i-1][j]) for j in range(len(positions[i])))
        else:
            change = abs(positions[i] - positions[i-1])
        total_turnover += change
    
    return total_turnover / (len(positions) - 1)

