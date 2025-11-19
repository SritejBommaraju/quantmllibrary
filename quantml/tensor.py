"""
Core Tensor class for QuantML library.

This module provides the Tensor class which is the fundamental building block
for all operations in the library. It supports automatic differentiation,
gradient tracking, and integration with the autograd engine.
"""

from typing import Optional, List, Union, Any, Callable
import math

# Try to import NumPy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


class Tensor:
    """
    A multi-dimensional array with automatic differentiation support.
    
    The Tensor class stores data and tracks computation graphs for automatic
    gradient computation. All operations return new tensors (no inplace operations).
    
    Attributes:
        data: The underlying data (list, nested lists, or numpy array)
        requires_grad: Whether to track gradients for this tensor
        grad: The gradient of this tensor (computed during backward pass)
        _prev: Set of parent tensors in the computation graph
        _op: String identifier for the operation that created this tensor
        _backward_fn: Function to call during backward pass
    
    Examples:
        >>> x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        >>> y = Tensor([4.0, 5.0, 6.0], requires_grad=True)
        >>> z = x + y
        >>> z.backward()
        >>> x.grad  # [1.0, 1.0, 1.0]
    """
    
    def __init__(
        self,
        data: Union[List, Any],
        requires_grad: bool = False,
        _prev: Optional[set] = None,
        _op: Optional[str] = None,
        _backward_fn: Optional[Callable] = None,
        _np_array: Optional[Any] = None
    ):
        """
        Initialize a Tensor.
        
        Args:
            data: The data to store (list, nested list, or numpy array)
            requires_grad: Whether to track gradients
            _prev: Parent tensors in computation graph (internal use)
            _op: Operation identifier (internal use)
            _backward_fn: Backward function (internal use)
            _np_array: Direct NumPy array (internal use, avoids conversion)
        """
        # Store as NumPy array if available, otherwise as list
        if _np_array is not None and HAS_NUMPY:
            # Direct NumPy array provided - skip conversion
            self._np_array = _np_array.astype(np.float64, copy=False)
            self._data_list = None  # Lazy conversion
        elif HAS_NUMPY:
            try:
                if isinstance(data, np.ndarray):
                    self._np_array = data.astype(np.float64, copy=False)
                    self._data_list = None  # Lazy conversion
                elif isinstance(data, (int, float)):
                    self._np_array = np.array([[float(data)]], dtype=np.float64)
                    self._data_list = None
                elif isinstance(data, list):
                    # Convert to NumPy array
                    self._np_array = np.array(data, dtype=np.float64)
                    self._data_list = None
                else:
                    self._np_array = np.array([[float(data)]], dtype=np.float64)
                    self._data_list = None
            except (ValueError, TypeError):
                # Fallback to list if NumPy conversion fails
                if isinstance(data, (int, float)):
                    self._data_list = [[float(data)]]
                elif isinstance(data, list):
                    self._data_list = self._ensure_nested_list(data)
                else:
                    self._data_list = [[float(data)]]
                self._np_array = None
        else:
            # No NumPy - use list storage
            if isinstance(data, (int, float)):
                self._data_list = [[float(data)]]
            elif isinstance(data, list):
                self._data_list = self._ensure_nested_list(data)
            else:
                self._data_list = [[float(data)]]
            self._np_array = None
        
        self.requires_grad = requires_grad
        self._grad_np = None  # NumPy array for gradient storage
        self._grad_list = None  # List gradient storage (fallback)
        self._prev = _prev if _prev is not None else set()
        self._op = _op
        self._backward_fn = _backward_fn
    
    @property
    def data(self) -> List:
        """Get data as nested list. Converts from NumPy if needed."""
        if self._np_array is not None:
            if self._data_list is None:
                self._data_list = self._np_array.tolist()
            return self._data_list
        return self._data_list
    
    @data.setter
    def data(self, value: Union[List, Any]):
        """Set data, converting to NumPy if available."""
        if HAS_NUMPY:
            try:
                if isinstance(value, np.ndarray):
                    self._np_array = value.astype(np.float64, copy=False)
                else:
                    self._np_array = np.array(value, dtype=np.float64)
                self._data_list = None  # Invalidate cache
            except (ValueError, TypeError):
                self._data_list = self._ensure_nested_list(value) if isinstance(value, list) else [[float(value)]]
                self._np_array = None
        else:
            self._data_list = self._ensure_nested_list(value) if isinstance(value, list) else [[float(value)]]
            self._np_array = None
    
    @property
    def numpy(self):
        """Get data as NumPy array. Returns None if NumPy not available."""
        if HAS_NUMPY and self._np_array is not None:
            return self._np_array
        elif HAS_NUMPY:
            try:
                self._np_array = np.array(self._data_list, dtype=np.float64)
                return self._np_array
            except (ValueError, TypeError):
                return None
        return None
    
    def to_numpy(self):
        """Convert to NumPy array."""
        return self.numpy
    
    @classmethod
    def from_numpy(cls, arr, requires_grad: bool = False):
        """Create Tensor from NumPy array."""
        if not HAS_NUMPY:
            raise ImportError("NumPy is required for from_numpy()")
        return cls(arr, requires_grad=requires_grad)
    
    def _ensure_nested_list(self, data: Any) -> List:
        """Ensure data is a nested list structure."""
        if not isinstance(data, list):
            return [[float(data)]]
        if len(data) == 0:
            return [[]]
        # Check if first element is a list (2D+)
        if isinstance(data[0], list):
            return data
        # 1D list - wrap it
        return [data]
    
    @property
    def shape(self) -> tuple:
        """Get the shape of the tensor. Uses NumPy if available."""
        if HAS_NUMPY and self._np_array is not None:
            return tuple(self._np_array.shape)
        # Fallback to list-based shape
        data = self.data
        if not isinstance(data, list):
            return (1,)
        if len(data) == 0:
            return (0,)
        if isinstance(data[0], list):
            return (len(data), len(data[0]))
        return (len(data),)
    
    def backward(self, grad: Optional[Union[List, Any, Any]] = None):
        """
        Compute gradients by backpropagation.
        
        Args:
            grad: Initial gradient (defaults to ones if not provided)
        """
        if not self.requires_grad:
            return
        
        # Initialize gradient if not set
        if self._grad_np is None and self._grad_list is None:
            if grad is None:
                # Default to ones with same shape
                if HAS_NUMPY and self._np_array is not None:
                    self._grad_np = np.ones_like(self._np_array, dtype=np.float64)
                    self._grad_list = None  # Lazy conversion
                else:
                    self._grad_list = self._ones_like(self.data)
                    self._grad_np = None
            else:
                # Convert grad to NumPy if possible
                if HAS_NUMPY:
                    try:
                        if isinstance(grad, np.ndarray):
                            self._grad_np = grad.astype(np.float64, copy=False)
                            self._grad_list = None
                        else:
                            grad_arr = np.array(grad, dtype=np.float64)
                            self._grad_np = grad_arr
                            self._grad_list = None
                    except (ValueError, TypeError):
                        self._grad_list = self._ensure_nested_list(grad)
                        self._grad_np = None
                else:
                    self._grad_list = self._ensure_nested_list(grad)
                    self._grad_np = None
        else:
            # Accumulate gradients
            if grad is not None:
                if HAS_NUMPY:
                    try:
                        grad_arr = np.array(grad, dtype=np.float64) if not isinstance(grad, np.ndarray) else grad
                        if self._grad_np is not None:
                            self._grad_np = self._grad_np + grad_arr
                            self._grad_list = None  # Invalidate list cache
                        else:
                            # Convert existing grad to NumPy and add
                            if self._grad_list is not None:
                                self._grad_np = np.array(self._grad_list, dtype=np.float64) + grad_arr
                                self._grad_list = None
                            else:
                                self._grad_np = grad_arr
                    except (ValueError, TypeError):
                        # Fallback to list operations
                        if self._grad_list is None and self._grad_np is not None:
                            self._grad_list = self._grad_np.tolist()
                            self._grad_np = None
                        if self._grad_list is None:
                            self._grad_list = self._ensure_nested_list(grad)
                        else:
                            self._grad_list = self._add(self._grad_list, self._ensure_nested_list(grad))
                else:
                    if self._grad_list is None:
                        self._grad_list = self._ensure_nested_list(grad)
                    else:
                        self._grad_list = self._add(self._grad_list, self._ensure_nested_list(grad))
        
        # Call backward function if it exists
        if self._backward_fn is not None:
            # Pass gradient in appropriate format
            if self._grad_np is not None:
                self._backward_fn(self._grad_np)
            else:
                self._backward_fn(self._grad_list)
    
    @property
    def grad(self):
        """Get gradient. Returns NumPy array if available, otherwise list."""
        if self._grad_np is not None:
            return self._grad_np
        return self._grad_list
    
    @grad.setter
    def grad(self, value):
        """Set gradient. Accepts NumPy array or list."""
        if value is None:
            self._grad_np = None
            self._grad_list = None
            return
        
        if HAS_NUMPY:
            try:
                if isinstance(value, np.ndarray):
                    self._grad_np = value.astype(np.float64, copy=False)
                    self._grad_list = None
                else:
                    self._grad_np = np.array(value, dtype=np.float64)
                    self._grad_list = None
            except (ValueError, TypeError):
                self._grad_list = self._ensure_nested_list(value) if isinstance(value, list) else [[float(value)]]
                self._grad_np = None
        else:
            self._grad_list = self._ensure_nested_list(value) if isinstance(value, list) else [[float(value)]]
            self._grad_np = None
    
    def zero_grad(self):
        """Clear the gradient."""
        self._grad_np = None
        self._grad_list = None
    
    def detach(self) -> 'Tensor':
        """Create a new tensor without gradient tracking."""
        return Tensor(self.data, requires_grad=False)
    
    def update_data(self, new_data: Union[List, Any]):
        """
        Update data in-place. Only works if requires_grad=False to prevent graph corruption.
        
        Args:
            new_data: New data to set
        """
        if self.requires_grad:
            raise RuntimeError("Cannot update data in-place for tensors with requires_grad=True")
        self.data = new_data
    
    def add_(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """
        In-place addition: self += other
        
        Args:
            other: Tensor or scalar to add
        
        Returns:
            self (modified in-place)
        """
        if self.requires_grad:
            raise RuntimeError("Cannot use in-place operations on tensors with requires_grad=True")
        
        from quantml.ops import add
        result = add(self, other)
        self.data = result.data
        return self
    
    def mul_(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """
        In-place multiplication: self *= other
        
        Args:
            other: Tensor or scalar to multiply
        
        Returns:
            self (modified in-place)
        """
        if self.requires_grad:
            raise RuntimeError("Cannot use in-place operations on tensors with requires_grad=True")
        
        from quantml.ops import mul
        result = mul(self, other)
        self.data = result.data
        return self
    
    def sub_(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """
        In-place subtraction: self -= other
        
        Args:
            other: Tensor or scalar to subtract
        
        Returns:
            self (modified in-place)
        """
        if self.requires_grad:
            raise RuntimeError("Cannot use in-place operations on tensors with requires_grad=True")
        
        from quantml.ops import sub
        result = sub(self, other)
        self.data = result.data
        return self
    
    def div_(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """
        In-place division: self /= other
        
        Args:
            other: Tensor or scalar to divide by
        
        Returns:
            self (modified in-place)
        """
        if self.requires_grad:
            raise RuntimeError("Cannot use in-place operations on tensors with requires_grad=True")
        
        from quantml.ops import div
        result = div(self, other)
        self.data = result.data
        return self
    
    def _ones_like(self, data: Any) -> List:
        """Create a tensor of ones with the same shape."""
        if isinstance(data, list):
            if len(data) == 0:
                return []
            if isinstance(data[0], list):
                return [[1.0] * len(row) for row in data]
            return [1.0] * len(data)
        return [[1.0]]
    
    def _add(self, a: List, b: List) -> List:
        """Element-wise addition of nested lists."""
        if isinstance(a[0], list) and isinstance(b[0], list):
            return [[a[i][j] + b[i][j] for j in range(len(a[i]))] 
                    for i in range(len(a))]
        elif isinstance(a[0], list):
            return [[a[i][j] + b[0] for j in range(len(a[i]))] 
                    for i in range(len(a))]
        elif isinstance(b[0], list):
            return [[a[0] + b[i][j] for j in range(len(b[i]))] 
                    for i in range(len(b))]
        else:
            return [a[i] + b[i] for i in range(len(a))]
    
    def __repr__(self) -> str:
        """String representation of the tensor."""
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"
    
    def __add__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """Add two tensors or tensor and scalar."""
        from quantml.ops import add
        return add(self, other)
    
    def __radd__(self, other: Union[float, int]) -> 'Tensor':
        """Right addition (scalar + tensor)."""
        from quantml.ops import add
        return add(other, self)
    
    def __sub__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """Subtract two tensors or tensor and scalar."""
        from quantml.ops import sub
        return sub(self, other)
    
    def __rsub__(self, other: Union[float, int]) -> 'Tensor':
        """Right subtraction (scalar - tensor)."""
        from quantml.ops import sub
        return sub(other, self)
    
    def __mul__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """Multiply two tensors or tensor and scalar."""
        from quantml.ops import mul
        return mul(self, other)
    
    def __rmul__(self, other: Union[float, int]) -> 'Tensor':
        """Right multiplication (scalar * tensor)."""
        from quantml.ops import mul
        return mul(other, self)
    
    def __truediv__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """Divide two tensors or tensor and scalar."""
        from quantml.ops import div
        return div(self, other)
    
    def __rtruediv__(self, other: Union[float, int]) -> 'Tensor':
        """Right division (scalar / tensor)."""
        from quantml.ops import div
        return div(other, self)
    
    def __neg__(self) -> 'Tensor':
        """Negate the tensor."""
        from quantml.ops import mul
        return mul(self, -1.0)
    
    def __pow__(self, power: Union[float, int]) -> 'Tensor':
        """Raise tensor to a power."""
        from quantml.ops import pow
        return pow(self, power)

