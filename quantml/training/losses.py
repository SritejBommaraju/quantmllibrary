"""
Quant-specific loss functions for alpha generation.

This module provides loss functions optimized for quantitative trading,
including Sharpe ratio loss, quantile loss, and information ratio loss.
"""

from typing import Union, List, Optional
from quantml.tensor import Tensor
from quantml import ops

# Try to import NumPy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """
    Mean Squared Error loss.
    
    Args:
        pred: Predictions
        target: Targets
    
    Returns:
        MSE loss tensor
    """
    diff = ops.sub(pred, target)
    return ops.mean(ops.mul(diff, diff))


def mae_loss(pred: Tensor, target: Tensor) -> Tensor:
    """
    Mean Absolute Error loss.
    
    Args:
        pred: Predictions
        target: Targets
    
    Returns:
        MAE loss tensor
    """
    diff = ops.sub(pred, target)
    return ops.mean(ops.abs(diff))


def quantile_loss(pred: Tensor, target: Tensor, quantile: float = 0.5) -> Tensor:
    """
    Quantile loss (pinball loss) for robust regression.
    
    Useful for predicting percentiles and handling outliers.
    
    Args:
        pred: Predictions
        target: Targets
        quantile: Quantile level (0.0 to 1.0, default: 0.5 for median)
    
    Returns:
        Quantile loss tensor
    """
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0.0 and 1.0")
    
    diff = ops.sub(pred, target)
    
    # L_quantile = max(quantile * diff, (quantile - 1) * diff)
    pos_part = ops.mul(diff, quantile)
    neg_part = ops.mul(diff, quantile - 1.0)
    
    # Element-wise maximum
    loss = ops.maximum(pos_part, neg_part)
    return ops.mean(loss)


def sharpe_loss(pred: Tensor, target: Tensor, risk_free_rate: float = 0.0) -> Tensor:
    """
    Negative Sharpe ratio as loss (to maximize Sharpe ratio).
    
    This loss function directly optimizes for risk-adjusted returns.
    
    Args:
        pred: Predictions (returns)
        target: Targets (actual returns)
        risk_free_rate: Risk-free rate
    
    Returns:
        Negative Sharpe ratio (to minimize)
    """
    # Use predictions as portfolio returns
    returns = pred
    
    # Calculate mean and std
    mean_ret = ops.mean(returns)
    std_ret = ops.std(returns)
    
    # Sharpe = (mean - rf) / std
    # We want to maximize Sharpe, so minimize negative Sharpe
    numerator = ops.sub(mean_ret, risk_free_rate)
    sharpe = ops.div(numerator, ops.add(std_ret, 1e-8))  # Add small epsilon
    
    # Return negative to minimize (maximize Sharpe)
    return ops.mul(sharpe, -1.0)


def information_ratio_loss(pred: Tensor, target: Tensor) -> Tensor:
    """
    Negative Information Ratio as loss.
    
    Information Ratio = mean(alpha) / std(alpha), where alpha = pred - target
    
    Args:
        pred: Predictions
        target: Targets
    
    Returns:
        Negative Information Ratio
    """
    # Alpha = prediction error (we want to minimize this)
    alpha = ops.sub(pred, target)
    
    mean_alpha = ops.mean(alpha)
    std_alpha = ops.std(alpha)
    
    # IR = mean / std
    ir = ops.div(mean_alpha, ops.add(std_alpha, 1e-8))
    
    # Return negative to minimize (maximize IR)
    return ops.mul(ir, -1.0)


def huber_loss(pred: Tensor, target: Tensor, delta: float = 1.0) -> Tensor:
    """
    Huber loss: combines MSE and MAE, robust to outliers.
    
    Args:
        pred: Predictions
        target: Targets
        delta: Threshold parameter
    
    Returns:
        Huber loss tensor
    """
    diff = ops.sub(pred, target)
    abs_diff = ops.abs(diff)

    # L_huber = 0.5 * diff^2 if |diff| <= delta, else delta * (|diff| - 0.5*delta)
    # Use: squared + relu(|diff| - delta) * (linear - squared)
    # When |diff| <= delta: relu = 0, so result = squared = 0.5*diff^2  (correct)
    # When |diff| > delta: relu > 0, switches to linear region
    squared = ops.mul(ops.mul(diff, diff), 0.5)
    linear = ops.sub(ops.mul(abs_diff, delta), delta * delta * 0.5)

    # Indicator: relu(|diff| - delta) > 0 when |diff| > delta
    # We want: squared when |diff| <= delta, linear when |diff| > delta
    # Use: linear + relu(delta - |diff|) * (squared - linear)
    # When |diff| <= delta: relu(delta-|diff|) > 0, result = linear + (squared - linear) = squared
    # When |diff| > delta: relu(delta-|diff|) = 0, result = linear
    indicator = ops.relu(ops.sub(Tensor([[delta]]), abs_diff))
    # Clamp indicator to 0 or 1 by using min(indicator/eps, 1) approximation
    # Simpler: just pick between squared and linear based on threshold
    # loss = linear + relu(delta - |diff|) / (delta - |diff| + eps) * (squared - linear)
    # Actually simplest correct approach: loss = min(squared, linear + 0.5*delta^2)
    # But let's use the standard formulation:
    # huber = where(|diff| <= delta, 0.5*diff^2, delta*(|diff| - 0.5*delta))
    # Equivalent: huber = delta * |diff| - 0.5 * delta^2 + 0.5 * relu(delta - |diff|)^2
    # = delta * |diff| - 0.5*delta^2 + 0.5 * max(0, delta-|diff|)^2
    clamped = ops.relu(ops.sub(Tensor([[delta]]), abs_diff))
    loss = ops.add(linear, ops.mul(ops.mul(clamped, clamped), 0.5))

    return ops.mean(loss)


def asymmetric_loss(pred: Tensor, target: Tensor, asymmetry: float = 1.0) -> Tensor:
    """
    Asymmetric loss: penalizes over-prediction and under-prediction differently.
    
    Useful when false positives and false negatives have different costs.
    
    Args:
        pred: Predictions
        target: Targets
        asymmetry: Asymmetry factor (>1 penalizes over-prediction more)
    
    Returns:
        Asymmetric loss tensor
    """
    diff = ops.sub(pred, target)

    # L_asym = asymmetry * diff^2 if diff > 0, else diff^2
    diff_sq = ops.mul(diff, diff)

    # Split into positive and negative parts using relu
    # pos_sq = relu(diff)^2 = diff^2 when diff > 0, 0 otherwise
    # neg_sq = relu(-diff)^2 = diff^2 when diff < 0, 0 otherwise
    pos_diff = ops.relu(diff)
    neg_diff = ops.relu(ops.mul(diff, -1.0))
    pos_sq = ops.mul(pos_diff, pos_diff)
    neg_sq = ops.mul(neg_diff, neg_diff)

    # loss = asymmetry * pos_sq + neg_sq
    loss = ops.add(ops.mul(pos_sq, asymmetry), neg_sq)

    return ops.mean(loss)


def max_drawdown_loss(pred: Tensor, target: Tensor) -> Tensor:
    """
    Loss based on maximum drawdown of cumulative returns.
    
    This encourages predictions that lead to smoother equity curves.
    
    Args:
        pred: Predictions (returns)
        target: Targets (actual returns)
    
    Returns:
        Maximum drawdown (to minimize)
    """
    # Use predictions as returns
    returns = pred
    
    # For simplicity, we'll compute a proxy using variance
    # A full implementation would compute actual drawdown
    # This is a simplified version
    mean_ret = ops.mean(returns)
    std_ret = ops.std(returns)
    
    # Proxy: higher variance -> higher drawdown risk
    # We want to minimize this
    return std_ret


def combined_quant_loss(pred: Tensor, target: Tensor, 
                       mse_weight: float = 0.5,
                       sharpe_weight: float = 0.3,
                       drawdown_weight: float = 0.2) -> Tensor:
    """
    Combined loss function for quant trading.
    
    Combines MSE, Sharpe ratio, and drawdown considerations.
    
    Args:
        pred: Predictions
        target: Targets
        mse_weight: Weight for MSE component
        sharpe_weight: Weight for Sharpe component
        drawdown_weight: Weight for drawdown component
    
    Returns:
        Combined loss tensor
    """
    mse = mse_loss(pred, target)
    sharpe = sharpe_loss(pred, target)
    dd = max_drawdown_loss(pred, target)
    
    # Normalize components (simplified)
    combined = ops.add(
        ops.mul(mse, mse_weight),
        ops.add(
            ops.mul(sharpe, sharpe_weight),
            ops.mul(dd, drawdown_weight)
        )
    )
    
    return combined

