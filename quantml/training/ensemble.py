"""
Model ensembling utilities for quant models.

Provides utilities for combining multiple models.
"""

from typing import List, Optional, Callable, Any, Dict
from quantml.tensor import Tensor


class EnsembleModel:
    """
    Ensemble model that combines multiple models.
    
    Supports weighted averaging, voting, and stacking strategies.
    """
    
    def __init__(
        self,
        models: List[Any],
        weights: Optional[List[float]] = None,
        strategy: str = 'weighted_avg'
    ):
        """
        Initialize ensemble model.
        
        Args:
            models: List of models to ensemble
            weights: Optional weights for each model (defaults to equal weights)
            strategy: Ensemble strategy ('weighted_avg', 'voting', 'stacking')
        """
        self.models = models
        self.strategy = strategy
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through ensemble.
        
        Args:
            x: Input features
        
        Returns:
            Ensemble prediction
        """
        if self.strategy == 'weighted_avg':
            predictions = [model.forward(x) for model in self.models]
            # Weighted average
            from quantml import ops
            result = ops.mul(predictions[0], self.weights[0])
            for i in range(1, len(predictions)):
                result = ops.add(result, ops.mul(predictions[i], self.weights[i]))
            return result
        elif self.strategy == 'voting':
            # Simple voting (for classification - simplified for regression)
            predictions = [model.forward(x) for model in self.models]
            from quantml import ops
            result = ops.mul(predictions[0], self.weights[0])
            for i in range(1, len(predictions)):
                result = ops.add(result, ops.mul(predictions[i], self.weights[i]))
            return result
        else:  # stacking
            # Stacking would require a meta-learner (simplified here)
            predictions = [model.forward(x) for model in self.models]
            from quantml import ops
            result = ops.mul(predictions[0], self.weights[0])
            for i in range(1, len(predictions)):
                result = ops.add(result, ops.mul(predictions[i], self.weights[i]))
            return result
    
    def parameters(self):
        """Get all parameters from all models."""
        params = []
        for model in self.models:
            if hasattr(model, 'parameters'):
                params.extend(model.parameters())
        return params

