"""
Tests for activation functions.
"""

import pytest
import math
from quantml import Tensor
from quantml import ops
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


class TestReLU:
    """Tests for ReLU activation."""
    
    def test_relu_positive(self):
        """Test ReLU on positive values."""
        x = Tensor([[1.0, 2.0, 3.0]])
        y = ops.relu(x)
        
        y_data = y.data[0] if isinstance(y.data[0], list) else y.data
        assert y_data[0] == pytest.approx(1.0)
        assert y_data[1] == pytest.approx(2.0)
        assert y_data[2] == pytest.approx(3.0)
    
    def test_relu_negative(self):
        """Test ReLU on negative values."""
        x = Tensor([[-1.0, -2.0, -3.0]])
        y = ops.relu(x)
        
        y_data = y.data[0] if isinstance(y.data[0], list) else y.data
        assert y_data[0] == pytest.approx(0.0)
        assert y_data[1] == pytest.approx(0.0)
        assert y_data[2] == pytest.approx(0.0)
    
    def test_relu_zero(self):
        """Test ReLU on zero."""
        x = Tensor([[0.0]])
        y = ops.relu(x)
        
        y_data = y.data[0] if isinstance(y.data[0], list) else y.data
        assert y_data[0] == pytest.approx(0.0)
    
    def test_relu_gradient(self):
        """Test ReLU gradient."""
        x = Tensor([[2.0, -1.0]], requires_grad=True)
        y = ops.relu(x)
        loss = ops.sum(y)
        loss.backward()
        
        grad = x.grad
        if HAS_NUMPY and isinstance(grad, np.ndarray):
            # Handle numpy case
            assert grad[0][0] == pytest.approx(1.0)
            assert grad[0][1] == pytest.approx(0.0)
        elif isinstance(grad, list) and isinstance(grad[0], list):
            assert grad[0][0] == pytest.approx(1.0)  # d/dx relu(2) = 1
            assert grad[0][1] == pytest.approx(0.0)  # d/dx relu(-1) = 0
        else:
            # Handle 1D list or other cases
            assert grad[0] == pytest.approx(1.0)
            assert grad[1] == pytest.approx(0.0)


class TestSigmoid:
    """Tests for sigmoid activation."""
    
    def test_sigmoid_zero(self):
        """Test sigmoid(0) = 0.5."""
        x = Tensor([[0.0]])
        y = ops.sigmoid(x)
        
        y_data = y.data[0] if isinstance(y.data[0], list) else y.data
        assert y_data[0] == pytest.approx(0.5)
    
    def test_sigmoid_large_positive(self):
        """Test sigmoid approaches 1 for large positive."""
        x = Tensor([[10.0]])
        y = ops.sigmoid(x)
        
        y_data = y.data[0] if isinstance(y.data[0], list) else y.data
        assert y_data[0] > 0.99
    
    def test_sigmoid_large_negative(self):
        """Test sigmoid approaches 0 for large negative."""
        x = Tensor([[-10.0]])
        y = ops.sigmoid(x)
        
        y_data = y.data[0] if isinstance(y.data[0], list) else y.data
        assert y_data[0] < 0.01
    
    def test_sigmoid_gradient(self):
        """Test sigmoid gradient: sigmoid(x) * (1 - sigmoid(x))."""
        x = Tensor([[0.0]], requires_grad=True)
        y = ops.sigmoid(x)
        y.backward()
        
        # At x=0, sigmoid=0.5, so gradient = 0.5 * 0.5 = 0.25
        grad = x.grad
        grad_val = grad[0][0] if isinstance(grad[0], list) else grad[0]
        assert grad_val == pytest.approx(0.25)


class TestTanh:
    """Tests for tanh activation."""
    
    def test_tanh_zero(self):
        """Test tanh(0) = 0."""
        x = Tensor([[0.0]])
        y = ops.tanh(x)
        
        y_data = y.data[0] if isinstance(y.data[0], list) else y.data
        assert y_data[0] == pytest.approx(0.0, abs=1e-10)
    
    def test_tanh_range(self):
        """Test tanh output in [-1, 1]."""
        x = Tensor([[-10.0, 0.0, 10.0]])
        y = ops.tanh(x)
        
        y_data = y.data[0] if isinstance(y.data[0], list) else y.data
        for val in y_data:
            assert -1.0 <= float(val) <= 1.0
    
    def test_tanh_gradient(self):
        """Test tanh gradient: 1 - tanh^2(x)."""
        x = Tensor([[0.0]], requires_grad=True)
        y = ops.tanh(x)
        y.backward()
        
        # At x=0, tanh=0, so gradient = 1 - 0 = 1
        grad = x.grad
        grad_val = grad[0][0] if isinstance(grad[0], list) else grad[0]
        assert grad_val == pytest.approx(1.0)


class TestSoftmax:
    """Tests for softmax activation."""
    
    def test_softmax_sums_to_one(self):
        """Test softmax output sums to 1."""
        x = Tensor([[1.0, 2.0, 3.0]])
        y = ops.softmax(x)
        
        y_data = y.data[0] if isinstance(y.data[0], list) else y.data
        total = sum(float(v) for v in y_data)
        assert total == pytest.approx(1.0)
    
    def test_softmax_positive(self):
        """Test softmax outputs are all positive."""
        x = Tensor([[-1.0, 0.0, 1.0]])
        y = ops.softmax(x)
        
        y_data = y.data[0] if isinstance(y.data[0], list) else y.data
        for val in y_data:
            assert float(val) > 0
    
    def test_softmax_ordering(self):
        """Test softmax preserves ordering."""
        x = Tensor([[1.0, 2.0, 3.0]])
        y = ops.softmax(x)
        
        y_data = y.data[0] if isinstance(y.data[0], list) else y.data
        assert float(y_data[0]) < float(y_data[1]) < float(y_data[2])


class TestLeakyReLU:
    """Tests for Leaky ReLU activation."""
    
    def test_leaky_relu_positive(self):
        """Test Leaky ReLU on positive values."""
        x = Tensor([[2.0]])
        y = ops.leaky_relu(x, negative_slope=0.01)
        
        y_data = y.data[0] if isinstance(y.data[0], list) else y.data
        assert y_data[0] == pytest.approx(2.0)
    
    def test_leaky_relu_negative(self):
        """Test Leaky ReLU on negative values."""
        x = Tensor([[-2.0]])
        y = ops.leaky_relu(x, negative_slope=0.1)
        
        y_data = y.data[0] if isinstance(y.data[0], list) else y.data
        assert y_data[0] == pytest.approx(-0.2)


class TestGELU:
    """Tests for GELU activation."""
    
    def test_gelu_zero(self):
        """Test GELU(0) ≈ 0."""
        x = Tensor([[0.0]])
        y = ops.gelu(x)
        
        y_data = y.data[0] if isinstance(y.data[0], list) else y.data
        assert y_data[0] == pytest.approx(0.0, abs=1e-6)
    
    def test_gelu_positive(self):
        """Test GELU on positive values."""
        x = Tensor([[2.0]])
        y = ops.gelu(x)
        
        y_data = y.data[0] if isinstance(y.data[0], list) else y.data
        # GELU(2) ≈ 1.954
        assert y_data[0] > 1.9


class TestSwish:
    """Tests for Swish activation."""
    
    def test_swish_zero(self):
        """Test Swish(0) = 0."""
        x = Tensor([[0.0]])
        y = ops.swish(x)
        
        y_data = y.data[0] if isinstance(y.data[0], list) else y.data
        assert y_data[0] == pytest.approx(0.0)
    
    def test_swish_positive(self):
        """Test Swish on positive values."""
        x = Tensor([[2.0]])
        y = ops.swish(x)
        
        y_data = y.data[0] if isinstance(y.data[0], list) else y.data
        # swish(2) = 2 * sigmoid(2) ≈ 2 * 0.88 ≈ 1.76
        assert y_data[0] > 1.7
