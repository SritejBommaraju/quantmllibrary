"""
QuantML Training Utilities

This module provides training utilities for quant model development,
including metrics, loss functions, data loaders, and training loops.
"""

from quantml.training.metrics import (
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    max_drawdown,
    information_coefficient,
    rank_ic,
    hit_rate,
    var,
    cvar,
    turnover
)

from quantml.training.losses import (
    mse_loss,
    mae_loss,
    quantile_loss,
    sharpe_loss,
    information_ratio_loss,
    huber_loss,
    asymmetric_loss,
    max_drawdown_loss,
    combined_quant_loss
)

from quantml.training.walk_forward import (
    WalkForwardOptimizer,
    WindowType,
    walk_forward_validation
)

from quantml.training.backtest import BacktestEngine

from quantml.training.alpha_eval import (
    AlphaEvaluator,
    evaluate_alpha_signals
)

from quantml.training.features import (
    FeaturePipeline,
    create_lagged_features,
    normalize_features
)

from quantml.training.data_loader import (
    QuantDataLoader,
    TimeSeriesDataLoader
)

from quantml.training.trainer import QuantTrainer

from quantml.training.gradient_clipping import (
    GradientNormClipper,
    GradientValueClipper,
    AdaptiveClipper,
    clip_grad_norm,
    clip_grad_value
)

from quantml.training.lr_finder import LRFinder

from quantml.training.ensemble import EnsembleModel

from quantml.training.feature_importance import FeatureImportanceTracker

from quantml.training.cv import TimeSeriesSplit, PurgedKFold

from quantml.training.regularization import Dropout

__all__ = [
    # Metrics
    'sharpe_ratio',
    'sortino_ratio',
    'calmar_ratio',
    'max_drawdown',
    'information_coefficient',
    'rank_ic',
    'hit_rate',
    'var',
    'cvar',
    'turnover',
    # Losses
    'mse_loss',
    'mae_loss',
    'quantile_loss',
    'sharpe_loss',
    'information_ratio_loss',
    'huber_loss',
    'asymmetric_loss',
    'max_drawdown_loss',
    'combined_quant_loss',
    # Walk-forward
    'WalkForwardOptimizer',
    'WindowType',
    'walk_forward_validation',
    # Backtesting
    'BacktestEngine',
    # Alpha evaluation
    'AlphaEvaluator',
    'evaluate_alpha_signals',
    # Features
    'FeaturePipeline',
    'create_lagged_features',
    'normalize_features',
    # Data loaders
    'QuantDataLoader',
    'TimeSeriesDataLoader',
    # Trainer
    'QuantTrainer',
    # Gradient clipping
    'GradientNormClipper',
    'GradientValueClipper',
    'AdaptiveClipper',
    'clip_grad_norm',
    'clip_grad_value',
    # LR finder
    'LRFinder',
    # Ensembling
    'EnsembleModel',
    # Feature importance
    'FeatureImportanceTracker',
    # Cross-validation
    'TimeSeriesSplit',
    'PurgedKFold',
    # Regularization
    'Dropout'
]

