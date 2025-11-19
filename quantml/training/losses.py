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
    # We'll use a smooth approximation
    squared = ops.mul(ops.mul(diff, diff), 0.5)
    linear = ops.sub(ops.mul(abs_diff, delta), ops.mul(delta * delta, 0.5))
    
    # Use relu to switch between squared and linear
    # Simplified: use squared for small errors, linear for large
    loss = ops.add(squared, ops.mul(ops.relu(ops.sub(abs_diff, delta)), ops.sub(linear, squared)))
    
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
    
    # Weight over-predictions more
    over_pred = ops.mul(ops.relu(diff), ops.mul(diff_sq, asymmetry - 1.0))
    loss = ops.add(diff_sq, over_pred)
    
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

