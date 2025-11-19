"""
Cross-validation utilities for time-series data.

Provides time-series aware cross-validation to avoid lookahead bias.
"""

from typing import List, Iterator, Tuple, Optional
import math


class TimeSeriesSplit:
    """
    Time-series aware cross-validation splitter.
    
    Splits data sequentially to avoid lookahead bias.
    """
    
    def __init__(self, n_splits: int = 5, test_size: Optional[int] = None):
        """
        Initialize TimeSeriesSplit.
        
        Args:
            n_splits: Number of splits
            test_size: Size of test set (if None, uses 1/n_splits)
        """
        self.n_splits = n_splits
        self.test_size = test_size
    
    def split(self, X: List, y: Optional[List] = None) -> Iterator[Tuple[List[int], List[int]]]:
        """
        Generate train/test splits.
        
        Args:
            X: Input data
            y: Optional target data
        
        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size
        
        for i in range(self.n_splits):
            test_start = (i + 1) * test_size
            test_end = min(test_start + test_size, n_samples)
            
            if test_start >= n_samples:
                break
            
            train_indices = list(range(test_start))
            test_indices = list(range(test_start, test_end))
            
            yield train_indices, test_indices


class PurgedKFold:
    """
    Purged K-fold for time-series.
    
    Removes overlapping periods to avoid leakage.
    """
    
    def __init__(self, n_splits: int = 5, purge_gap: int = 1):
        """
        Initialize PurgedKFold.
        
        Args:
            n_splits: Number of splits
            purge_gap: Number of samples to purge between train and test
        """
        self.n_splits = n_splits
        self.purge_gap = purge_gap
    
    def split(self, X: List, y: Optional[List] = None) -> Iterator[Tuple[List[int], List[int]]]:
        """
        Generate purged train/test splits.
        
        Args:
            X: Input data
            y: Optional target data
        
        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        fold_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = min((i + 1) * fold_size, n_samples)
            
            # Purge gap before test set
            train_end = max(0, test_start - self.purge_gap)
            train_indices = list(range(train_end))
            
            # Purge gap after test set
            train_start_after = min(n_samples, test_end + self.purge_gap)
            train_indices.extend(list(range(train_start_after, n_samples)))
            
            test_indices = list(range(test_start, test_end))
            
            yield train_indices, test_indices

