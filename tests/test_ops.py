"""
Tests for operations.
"""

import pytest
from quantml import Tensor
from quantml import ops


def test_add():
    """Test addition operation."""
    a = Tensor([1.0, 2.0])
    b = Tensor([3.0, 4.0])
    c = ops.add(a, b)
    
    c_data = c.data[0] if isinstance(c.data[0], list) else c.data
    assert c_data[0] == pytest.approx(4.0)
    assert c_data[1] == pytest.approx(6.0)


def test_mul():
    """Test multiplication operation."""
    a = Tensor([2.0, 3.0])
    b = Tensor([4.0, 5.0])
    c = ops.mul(a, b)
    
    c_data = c.data[0] if isinstance(c.data[0], list) else c.data
    assert c_data[0] == pytest.approx(8.0)
    assert c_data[1] == pytest.approx(15.0)


def test_matmul():
    """Test matrix multiplication."""
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])
    c = ops.matmul(a, b)
    
    # Result should be [[19, 22], [43, 50]]
    c_data = c.data
    assert c_data[0][0] == pytest.approx(19.0, abs=0.1)
    assert c_data[0][1] == pytest.approx(22.0, abs=0.1)


def test_sum():
    """Test sum operation."""
    t = Tensor([1.0, 2.0, 3.0, 4.0])
    s = ops.sum(t)
    
    s_data = s.data[0][0] if isinstance(s.data[0], list) else s.data[0]
    assert s_data == pytest.approx(10.0)


def test_relu():
    """Test ReLU activation."""
    t = Tensor([-1.0, 0.0, 1.0, 2.0])
    r = ops.relu(t)
    
    r_data = r.data[0] if isinstance(r.data[0], list) else r.data
    assert r_data[0] == pytest.approx(0.0)
    assert r_data[1] == pytest.approx(0.0)
    assert r_data[2] == pytest.approx(1.0)
    assert r_data[3] == pytest.approx(2.0)


def test_sigmoid():
    """Test sigmoid activation."""
    t = Tensor([0.0])
    s = ops.sigmoid(t)
    
    s_data = s.data[0][0] if isinstance(s.data[0], list) else s.data[0]
    assert s_data == pytest.approx(0.5, abs=0.1)

