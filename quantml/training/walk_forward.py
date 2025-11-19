"""
Walk-forward optimization for time-series model training.

Walk-forward optimization is essential for quant trading to prevent lookahead bias
and properly evaluate model performance on out-of-sample data.
"""

from typing import List, Tuple, Optional, Callable, Any
from enum import Enum


class WindowType(Enum):
    """Type of walk-forward window."""
    EXPANDING = "expanding"  # Training window grows over time
    ROLLING = "rolling"      # Training window has fixed size


class WalkForwardOptimizer:
    """
    Walk-forward optimization for time-series cross-validation.
    
    This class manages train/test splits for time-series data, ensuring
    no lookahead bias by only using past data for training.
    
    Attributes:
        window_type: Type of window (expanding or rolling)
        train_size: Initial training window size
        test_size: Test window size
        step_size: Step size for moving forward
        min_train_size: Minimum training window size
    
    Examples:
        >>> wfo = WalkForwardOptimizer(
        >>>     window_type=WindowType.EXPANDING,
        >>>     train_size=252,
        >>>     test_size=21
        >>> )
        >>> for train_idx, test_idx in wfo.split(data, n_splits=5):
        >>>     train_data = data[train_idx]
        >>>     test_data = data[test_idx]
    """
    
    def __init__(
        self,
        window_type: WindowType = WindowType.EXPANDING,
        train_size: int = 252,
        test_size: int = 21,
        step_size: Optional[int] = None,
        min_train_size: Optional[int] = None
    ):
        """
        Initialize walk-forward optimizer.
        
        Args:
            window_type: Type of window (expanding or rolling)
            train_size: Initial training window size (in samples)
            test_size: Test window size (in samples)
            step_size: Step size for moving forward (default: test_size)
            min_train_size: Minimum training window size (default: train_size)
        """
        self.window_type = window_type
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size if step_size is not None else test_size
        self.min_train_size = min_train_size if min_train_size is not None else train_size
    
    def split(self, data_length: int, n_splits: Optional[int] = None) -> List[Tuple[slice, slice]]:
        """
        Generate train/test splits.
        
        Args:
            data_length: Total length of data
            n_splits: Number of splits to generate (None for all possible)
        
        Returns:
            List of (train_slice, test_slice) tuples
        """
        splits = []
        start_idx = self.train_size
        
        split_count = 0
        while start_idx + self.test_size <= data_length:
            # Training window
            if self.window_type == WindowType.EXPANDING:
                train_start = 0
                train_end = start_idx
            else:  # ROLLING
                train_start = start_idx - self.train_size
                train_end = start_idx
            
            # Test window
            test_start = start_idx
            test_end = min(start_idx + self.test_size, data_length)
            
            # Ensure minimum training size
            if train_end - train_start >= self.min_train_size:
                train_slice = slice(train_start, train_end)
                test_slice = slice(test_start, test_end)
                splits.append((train_slice, test_slice))
                
                split_count += 1
                if n_splits is not None and split_count >= n_splits:
                    break
            
            # Move forward
            start_idx += self.step_size
        
        return splits
    
    def get_splits(self, data_length: int, n_splits: Optional[int] = None) -> List[Tuple[List[int], List[int]]]:
        """
        Get train/test indices as lists.
        
        Args:
            data_length: Total length of data
            n_splits: Number of splits to generate
        
        Returns:
            List of (train_indices, test_indices) tuples
        """
        splits = self.split(data_length, n_splits)
        return [
            (list(range(s[0].start, s[0].stop)), list(range(s[1].start, s[1].stop)))
            for s in splits
        ]


def walk_forward_validation(
    model: Any,
    X: List,
    y: List,
    train_fn: Callable,
    eval_fn: Callable,
    window_type: WindowType = WindowType.EXPANDING,
    train_size: int = 252,
    test_size: int = 21,
    n_splits: Optional[int] = None
) -> List[dict]:
    """
    Perform walk-forward validation on a model.
    
    Args:
        model: Model to train and evaluate
        X: Feature data
        y: Target data
        train_fn: Function to train model: train_fn(model, X_train, y_train) -> trained_model
        eval_fn: Function to evaluate: eval_fn(model, X_test, y_test) -> metrics_dict
        window_type: Type of window
        train_size: Initial training window size
        test_size: Test window size
        n_splits: Number of splits
    
    Returns:
        List of evaluation metrics for each split
    """
    if len(X) != len(y):
        raise ValueError("X and y must have same length")
    
    wfo = WalkForwardOptimizer(
        window_type=window_type,
        train_size=train_size,
        test_size=test_size
    )
    
    results = []
    splits = wfo.get_splits(len(X), n_splits)
    
    for train_idx, test_idx in splits:
        # Get train/test data
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_test = [y[i] for i in test_idx]
        
        # Train model
        trained_model = train_fn(model, X_train, y_train)
        
        # Evaluate
        metrics = eval_fn(trained_model, X_test, y_test)
        metrics['train_size'] = len(train_idx)
        metrics['test_size'] = len(test_idx)
        metrics['train_start'] = train_idx[0]
        metrics['train_end'] = train_idx[-1]
        metrics['test_start'] = test_idx[0]
        metrics['test_end'] = test_idx[-1]
        
        results.append(metrics)
    
    return results

