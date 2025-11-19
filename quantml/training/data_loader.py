"""
Efficient data loader for time-series aware batching.

This module provides data loaders optimized for quant training,
with support for time-series aware batching to prevent lookahead bias.
"""

from typing import List, Tuple, Optional, Iterator, Any
from quantml.tensor import Tensor

# Try to import NumPy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


class QuantDataLoader:
    """
    Data loader for quant model training with time-series awareness.
    
    This loader ensures no lookahead bias by only using past data
    for training batches.
    
    Attributes:
        X: Feature data
        y: Target data
        batch_size: Batch size
        shuffle: Whether to shuffle (should be False for time-series)
        drop_last: Whether to drop last incomplete batch
    
    Examples:
        >>> loader = QuantDataLoader(X, y, batch_size=32)
        >>> for batch_x, batch_y in loader:
        >>>     # Train on batch
    """
    
    def __init__(
        self,
        X: List,
        y: List,
        batch_size: int = 32,
        shuffle: bool = False,
        drop_last: bool = False
    ):
        """
        Initialize data loader.
        
        Args:
            X: Feature data
            y: Target data
            batch_size: Batch size
            shuffle: Whether to shuffle (False recommended for time-series)
            drop_last: Whether to drop last incomplete batch
        """
        if len(X) != len(y):
            raise ValueError("X and y must have same length")
        
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.n_samples = len(X)
    
    def __len__(self) -> int:
        """Get number of batches."""
        if self.drop_last:
            return self.n_samples // self.batch_size
        return (self.n_samples + self.batch_size - 1) // self.batch_size
    
    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        """Iterate over batches."""
        indices = list(range(self.n_samples))
        
        if self.shuffle:
            import random
            random.shuffle(indices)
        
        for i in range(0, self.n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            
            if self.drop_last and len(batch_indices) < self.batch_size:
                break
            
            batch_x = [self.X[idx] for idx in batch_indices]
            batch_y = [self.y[idx] for idx in batch_indices]
            
            # Convert to tensors
            x_tensor = Tensor(batch_x)
            y_tensor = Tensor(batch_y)
            
            yield x_tensor, y_tensor


class TimeSeriesDataLoader:
    """
    Time-series aware data loader with sequence support.
    
    This loader creates sequences for RNN/TCN models while ensuring
    no lookahead bias.
    
    Attributes:
        X: Feature data
        y: Target data
        sequence_length: Length of input sequences
        batch_size: Batch size
        stride: Stride for sequence creation
    
    Examples:
        >>> loader = TimeSeriesDataLoader(X, y, sequence_length=20, batch_size=16)
        >>> for seq_x, seq_y in loader:
        >>>     # Train on sequences
    """
    
    def __init__(
        self,
        X: List,
        y: List,
        sequence_length: int = 20,
        batch_size: int = 32,
        stride: int = 1
    ):
        """
        Initialize time-series data loader.
        
        Args:
            X: Feature data
            y: Target data
            sequence_length: Length of input sequences
            batch_size: Batch size
            stride: Stride for sequence creation
        """
        if len(X) != len(y):
            raise ValueError("X and y must have same length")
        
        self.X = X
        self.y = y
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.stride = stride
        
        # Create sequences
        self.sequences = self._create_sequences()
        self.n_sequences = len(self.sequences)
    
    def _create_sequences(self) -> List[Tuple[List, float]]:
        """Create sequences from data."""
        sequences = []
        
        for i in range(self.sequence_length, len(self.X), self.stride):
            seq_x = self.X[i - self.sequence_length:i]
            seq_y = self.y[i]
            sequences.append((seq_x, seq_y))
        
        return sequences
    
    def __len__(self) -> int:
        """Get number of batches."""
        return (self.n_sequences + self.batch_size - 1) // self.batch_size
    
    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        """Iterate over batches."""
        for i in range(0, self.n_sequences, self.batch_size):
            batch_sequences = self.sequences[i:i + self.batch_size]
            
            batch_x = [seq[0] for seq in batch_sequences]
            batch_y = [seq[1] for seq in batch_sequences]
            
            # Convert to tensors
            x_tensor = Tensor(batch_x)
            y_tensor = Tensor(batch_y)
            
            yield x_tensor, y_tensor

