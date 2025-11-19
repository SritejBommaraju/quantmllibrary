"""
Feature importance tracking and analysis.

Provides utilities for tracking feature contributions to model predictions.
"""

from typing import List, Dict, Optional, Any, Callable
from quantml.tensor import Tensor

# Try to import NumPy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


class FeatureImportanceTracker:
    """
    Track feature importance for quant models.
    
    Tracks gradient-based and permutation-based feature importance.
    """
    
    def __init__(self):
        """Initialize feature importance tracker."""
        self.gradient_importance: Dict[int, List[float]] = {}
        self.permutation_importance: Dict[int, List[float]] = {}
    
    def compute_gradient_importance(self, model: Any, x: Tensor, y: Tensor) -> List[float]:
        """
        Compute gradient-based feature importance.
        
        Args:
            model: Model to analyze
            x: Input features
            y: Targets
        
        Returns:
            List of importance scores for each feature
        """
        # Forward pass
        pred = model.forward(x)
        loss = model.loss_fn(pred, y) if hasattr(model, 'loss_fn') else None
        
        if loss is None:
            from quantml.training.losses import mse_loss
            loss = mse_loss(pred, y)
        
        # Backward pass
        if loss.requires_grad:
            loss.backward()
        
        # Extract gradients from input
        # This is simplified - would need to track gradients through model
        importance = []
        if hasattr(x, 'grad') and x.grad is not None:
            grad = x.grad
            if HAS_NUMPY and isinstance(grad, np.ndarray):
                importance = np.abs(grad).mean(axis=0).tolist()
            elif isinstance(grad, list):
                if isinstance(grad[0], list):
                    # Average over samples
                    importance = [sum(abs(grad[i][j]) for i in range(len(grad))) / len(grad) 
                                 for j in range(len(grad[0]))]
                else:
                    importance = [abs(g) for g in grad]
        
        return importance
    
    def compute_permutation_importance(
        self, 
        model: Any, 
        x: Tensor, 
        y: Tensor, 
        metric_fn: Callable,
        n_permutations: int = 10
    ) -> List[float]:
        """
        Compute permutation-based feature importance.
        
        Args:
            model: Model to analyze
            x: Input features
            y: Targets
            metric_fn: Metric function to use
            n_permutations: Number of permutations per feature
        
        Returns:
            List of importance scores
        """
        # Baseline metric
        pred_baseline = model.forward(x)
        baseline_score = metric_fn(pred_baseline, y)
        
        importance = []
        x_data = x.data
        
        # For each feature
        num_features = len(x_data[0]) if isinstance(x_data[0], list) else len(x_data)
        
        for feat_idx in range(num_features):
            scores = []
            for _ in range(n_permutations):
                # Permute feature
                x_permuted = self._permute_feature(x_data, feat_idx)
                x_perm_tensor = Tensor(x_permuted)
                pred_perm = model.forward(x_perm_tensor)
                score = metric_fn(pred_perm, y)
                scores.append(score)
            
            # Importance is decrease in performance
            avg_score = sum(scores) / len(scores)
            importance.append(baseline_score - avg_score)
        
        return importance
    
    def _permute_feature(self, data: List, feat_idx: int) -> List:
        """Permute a single feature in the data."""
        if isinstance(data[0], list):
            # 2D case
            import random
            values = [row[feat_idx] for row in data]
            random.shuffle(values)
            return [[row[j] if j != feat_idx else values[i] 
                    for j in range(len(row))] 
                   for i, row in enumerate(data)]
        else:
            # 1D case
            import random
            values = data.copy()
            random.shuffle(values)
            return values

