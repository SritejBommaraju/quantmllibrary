"""
Tests for Tensor class.
"""

import pytest
from quantml import Tensor
from quantml import ops


def test_tensor_creation():
    """Test tensor creation."""
    t = Tensor([1.0, 2.0, 3.0])
    assert t.data == [[1.0, 2.0, 3.0]] or t.data == [1.0, 2.0, 3.0]


def test_tensor_requires_grad():
    """Test requires_grad flag."""
    t1 = Tensor([1.0, 2.0], requires_grad=True)
    t2 = Tensor([1.0, 2.0], requires_grad=False)
    
    assert t1.requires_grad is True
    assert t2.requires_grad is False


def test_tensor_addition():
    """Test tensor addition."""
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    c = a + b
    
    # Check result
    c_data = c.data[0] if isinstance(c.data[0], list) else c.data
    assert c_data[0] == pytest.approx(5.0)
    assert c_data[1] == pytest.approx(7.0)
    assert c_data[2] == pytest.approx(9.0)


def test_tensor_multiplication():
    """Test tensor multiplication."""
    a = Tensor([2.0, 3.0])
    b = Tensor([4.0, 5.0])
    c = a * b
    
    c_data = c.data[0] if isinstance(c.data[0], list) else c.data
    assert c_data[0] == pytest.approx(8.0)
    assert c_data[1] == pytest.approx(15.0)


def test_tensor_backward():
    """Test backward pass."""
    x = Tensor([2.0], requires_grad=True)
    y = x * x  # y = x^2
    y.backward()
    
    # dy/dx = 2x = 4.0 at x=2.0
    grad = x.grad
    if isinstance(grad, list):
        grad_val = grad[0][0] if isinstance(grad[0], list) else grad[0]
    else:
        grad_val = grad[0] if hasattr(grad, '__getitem__') else grad
    
    assert grad_val == pytest.approx(4.0, abs=0.1)


def test_tensor_zero_grad():
    """Test zero_grad."""
    x = Tensor([1.0], requires_grad=True)
    y = x * 2.0
    y.backward()
    
    assert x.grad is not None
    
    x.zero_grad()
    assert x.grad is None


def test_tensor_scalar_ops():
    """Test scalar operations."""
    t = Tensor([2.0, 3.0])
    result = t * 2.0
    
    result_data = result.data[0] if isinstance(result.data[0], list) else result.data
    assert result_data[0] == pytest.approx(4.0)
    assert result_data[1] == pytest.approx(6.0)

