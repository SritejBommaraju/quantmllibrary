"""
Streaming tensor support for tick-level data.

This module provides StreamingTensor, a ring buffer implementation optimized
for high-frequency trading data where new ticks arrive continuously.
"""

from typing import List, Optional, Union, Any
from collections import deque
from quantml.tensor import Tensor

# Try to import NumPy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


class StreamingTensor:
    """
    A tensor that maintains a fixed-size ring buffer for streaming data.
    
    This is optimized for tick-level market data where new values arrive
    continuously and only the most recent N values need to be kept in memory.
    
    Attributes:
        max_size: Maximum number of elements to store
        buffer: Ring buffer storing the data
        _tensor: Current Tensor representation of the buffer
    
    Examples:
        >>> st = StreamingTensor(max_size=100)
        >>> st.append(100.5)
        >>> st.append(101.0)
        >>> window = st.get_window(10)  # Get last 10 values
    """
    
    def __init__(self, max_size: int = 1000, initial_data: Optional[List] = None):
        """
        Initialize a StreamingTensor.
        
        Args:
            max_size: Maximum number of elements to store (ring buffer size)
            initial_data: Optional initial data to populate
        """
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        
        if initial_data is not None:
            for item in initial_data:
                self.buffer.append(float(item))
        
        self._tensor = None
        self._np_array = None  # Cache NumPy array
        self._update_tensor()
    
    def append(self, value: Union[float, int]):
        """
        Append a new value to the streaming tensor.
        
        If the buffer is full, the oldest value is automatically removed
        (ring buffer behavior).
        
        Args:
            value: New value to append
        """
        self.buffer.append(float(value))
        self._np_array = None  # Invalidate NumPy cache
        self._update_tensor()
    
    def extend(self, values: List[Union[float, int]]):
        """
        Append multiple values at once.
        
        Args:
            values: List of values to append
        """
        for value in values:
            self.buffer.append(float(value))
        self._np_array = None  # Invalidate NumPy cache
        self._update_tensor()
    
    def get_window(self, size: Optional[int] = None) -> Tensor:
        """
        Get a window of the most recent values as a Tensor.
        
        Uses NumPy views when available for better performance.
        
        Args:
            size: Number of recent values to return (default: all available)
        
        Returns:
            Tensor containing the window of values
        """
        if size is None:
            size = len(self.buffer)
        
        if size > len(self.buffer):
            size = len(self.buffer)
        
        if size == 0:
            return Tensor([])
        
        # Use NumPy if available for efficient slicing
        if HAS_NUMPY and self._np_array is not None and len(self._np_array) >= size:
            try:
                window_arr = self._np_array[-size:]
                return Tensor([window_arr.tolist()])
            except (ValueError, TypeError):
                pass
        
        # Fallback: Get last 'size' elements
        window_data = list(self.buffer)[-size:]
        return Tensor([window_data])
    
    def get_all(self) -> Tensor:
        """
        Get all current values as a Tensor.
        
        Returns:
            Tensor containing all values in the buffer
        """
        return self.get_window()
    
    def _update_tensor(self):
        """Update the internal tensor representation."""
        if len(self.buffer) == 0:
            self._tensor = Tensor([])
            self._np_array = None
        else:
            buffer_list = list(self.buffer)
            self._tensor = Tensor([buffer_list])
            
            # Update NumPy array cache
            if HAS_NUMPY:
                try:
                    self._np_array = np.array(buffer_list, dtype=np.float64)
                except (ValueError, TypeError):
                    self._np_array = None
            else:
                self._np_array = None
    
    def clear(self):
        """Clear all data from the buffer."""
        self.buffer.clear()
        self._tensor = Tensor([])
        self._np_array = None
    
    def __len__(self) -> int:
        """Return the number of elements currently stored."""
        return len(self.buffer)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"StreamingTensor(max_size={self.max_size}, current_size={len(self.buffer)})"
    
    @property
    def is_full(self) -> bool:
        """Check if the buffer is at maximum capacity."""
        return len(self.buffer) >= self.max_size
    
    @property
    def tensor(self) -> Tensor:
        """Get the current tensor representation."""
        return self._tensor
    
    def to_list(self) -> List[float]:
        """Convert to a Python list."""
        return list(self.buffer)

