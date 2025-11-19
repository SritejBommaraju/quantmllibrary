"""
Alpha evaluation framework for signal quality assessment.

This module provides tools for evaluating alpha signals, including
Information Coefficient (IC), turnover analysis, decay analysis, and more.
"""

from typing import List, Optional, Dict, Any, Union
from quantml.training.metrics import information_coefficient, rank_ic, turnover

# Try to import NumPy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


class AlphaEvaluator:
    """
    Alpha signal quality evaluator.
    
    This class provides comprehensive evaluation of alpha signals,
    including IC analysis, turnover, decay, and factor exposure.
    
    Attributes:
        predictions: List of predictions/signals
        actuals: List of actual/realized values
        timestamps: Optional timestamps for time-based analysis
    
    Examples:
        >>> evaluator = AlphaEvaluator(predictions, actuals)
        >>> metrics = evaluator.evaluate()
        >>> print(f"IC: {metrics['ic']}, Rank IC: {metrics['rank_ic']}")
    """
    
    def __init__(
        self,
        predictions: List[float],
        actuals: List[float],
        timestamps: Optional[List] = None
    ):
        """
        Initialize alpha evaluator.
        
        Args:
            predictions: Predicted values/signals
            actuals: Actual/realized values
            timestamps: Optional timestamps for time-based analysis
        """
        if len(predictions) != len(actuals):
            raise ValueError("predictions and actuals must have same length")
        
        self.predictions = predictions
        self.actuals = actuals
        self.timestamps = timestamps if timestamps else list(range(len(predictions)))
        self.n = len(predictions)
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Perform comprehensive alpha evaluation.
        
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {}
        
        # Basic IC metrics
        metrics['ic'] = information_coefficient(self.predictions, self.actuals)
        metrics['rank_ic'] = rank_ic(self.predictions, self.actuals)
        
        # Turnover analysis
        metrics['turnover'] = turnover(self.predictions)
        
        # Decay analysis
        decay_metrics = self.decay_analysis()
        metrics.update(decay_metrics)
        
        # Hit rate
        from quantml.training.metrics import hit_rate
        metrics['hit_rate'] = hit_rate(self.predictions, self.actuals)
        
        # Signal statistics
        metrics['signal_mean'] = sum(self.predictions) / len(self.predictions)
        metrics['signal_std'] = self._std(self.predictions)
        
        return metrics
    
    def decay_analysis(self, max_lag: int = 5) -> Dict[str, Any]:
        """
        Analyze signal decay over time (how long signal remains predictive).
        
        Args:
            max_lag: Maximum lag to analyze
        
        Returns:
            Dictionary with decay metrics
        """
        decay_ics = []
        
        for lag in range(max_lag + 1):
            if lag == 0:
                # Current IC
                ic = information_coefficient(self.predictions, self.actuals)
            else:
                # IC at lag
                if lag < len(self.predictions):
                    pred_lag = self.predictions[:-lag] if lag > 0 else self.predictions
                    actual_lag = self.actuals[lag:]
                    if len(pred_lag) == len(actual_lag) and len(pred_lag) > 0:
                        ic = information_coefficient(pred_lag, actual_lag)
                    else:
                        ic = 0.0
                else:
                    ic = 0.0
            decay_ics.append(ic)
        
        # Calculate half-life (lag where IC drops to half)
        initial_ic = abs(decay_ics[0]) if decay_ics else 0.0
        half_ic = initial_ic / 2.0
        
        half_life = max_lag
        for i, ic in enumerate(decay_ics):
            if abs(ic) <= half_ic:
                half_life = i
                break
        
        return {
            'decay_ics': decay_ics,
            'half_life': half_life,
            'initial_ic': initial_ic,
            'decay_rate': (decay_ics[0] - decay_ics[-1]) / max_lag if max_lag > 0 else 0.0
        }
    
    def rolling_ic(self, window: int = 21) -> List[float]:
        """
        Calculate rolling Information Coefficient.
        
        Args:
            window: Rolling window size
        
        Returns:
            List of rolling IC values
        """
        rolling_ics = []
        
        for i in range(window, len(self.predictions)):
            pred_window = self.predictions[i-window:i]
            actual_window = self.actuals[i-window:i]
            ic = information_coefficient(pred_window, actual_window)
            rolling_ics.append(ic)
        
        return rolling_ics
    
    def factor_exposure(self, factors: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Calculate factor exposure of predictions.
        
        Args:
            factors: Dictionary of factor name -> factor values
        
        Returns:
            Dictionary of factor exposures (correlations)
        """
        exposures = {}
        
        for factor_name, factor_values in factors.items():
            if len(factor_values) != len(self.predictions):
                continue
            exposure = information_coefficient(self.predictions, factor_values)
            exposures[factor_name] = exposure
        
        return exposures
    
    def _std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) == 0:
            return 0.0
        mean_val = sum(values) / len(values)
        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
        return variance ** 0.5


def evaluate_alpha_signals(
    predictions: List[float],
    actuals: List[float],
    timestamps: Optional[List] = None
) -> Dict[str, Any]:
    """
    Convenience function for alpha evaluation.
    
    Args:
        predictions: Predicted values
        actuals: Actual values
        timestamps: Optional timestamps
    
    Returns:
        Evaluation metrics dictionary
    """
    evaluator = AlphaEvaluator(predictions, actuals, timestamps)
    return evaluator.evaluate()

