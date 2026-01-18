"""
Tests for gradient checking utility.
"""

import pytest
from quantml import Tensor
from quantml import ops
from quantml.utils.gradient_check import (
    check_gradients,
    quick_gradient_check,
    print_gradient_check_results
)


class TestCheckGradients:
    """Tests for gradient checking."""
    
    def test_check_gradients_add(self):
        """Test gradient check for addition."""
        def f(x):
            return ops.sum(x)
        
        x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        passed, details = check_gradients(f, x)
        
        assert passed, f"Gradient check failed: {details}"
    
    def test_check_gradients_mul(self):
        """Test gradient check for multiplication."""
        def f(x):
            return ops.sum(ops.mul(x, x))  # sum(x^2)
        
        x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        passed, details = check_gradients(f, x)
        
        # Analytical gradient: 2x = [2, 4, 6]
        assert passed, f"Gradient check failed: {details}"
    
    def test_check_gradients_relu(self):
        """Test gradient check for ReLU."""
        def f(x):
            return ops.sum(ops.relu(x))
        
        x = Tensor([[1.0, -1.0, 2.0]], requires_grad=True)
        passed, details = check_gradients(f, x)
        
        assert passed, f"Gradient check failed: {details}"
    
    def test_check_gradients_sigmoid(self):
        """Test gradient check for sigmoid."""
        def f(x):
            return ops.sum(ops.sigmoid(x))
        
        x = Tensor([[0.0, 1.0, -1.0]], requires_grad=True)
        passed, details = check_gradients(f, x)
        
        assert passed, f"Gradient check failed: {details}"
    
    def test_check_gradients_tanh(self):
        """Test gradient check for tanh."""
        def f(x):
            return ops.sum(ops.tanh(x))
        
        x = Tensor([[0.0, 0.5, -0.5]], requires_grad=True)
        passed, details = check_gradients(f, x)
        
        assert passed, f"Gradient check failed: {details}"
    
    def test_check_gradients_power(self):
        """Test gradient check for power."""
        def f(x):
            return ops.sum(ops.pow(x, 3))  # sum(x^3)
        
        x = Tensor([[1.0, 2.0]], requires_grad=True)
        passed, details = check_gradients(f, x)
        
        # Analytical gradient: 3x^2 = [3, 12]
        assert passed, f"Gradient check failed: {details}"


class TestQuickGradientCheck:
    """Tests for quick gradient check."""
    
    def test_quick_check_passes(self):
        """Test quick gradient check for passing case."""
        def f(x):
            return ops.sum(x)
        
        x = Tensor([[1.0, 2.0]], requires_grad=True)
        assert quick_gradient_check(f, x) == True
    
    def test_quick_check_composition(self):
        """Test quick gradient check for composed functions."""
        def f(x):
            y = ops.mul(x, x)  # x^2
            z = ops.relu(y)
            return ops.sum(z)
        
        x = Tensor([[1.0, 2.0, -1.0]], requires_grad=True)
        assert quick_gradient_check(f, x) == True


class TestPrintResults:
    """Tests for print functionality."""
    
    def test_print_results(self, capsys):
        """Test that print_gradient_check_results works."""
        details = [
            {'index': (0, 0), 'numerical': 1.0, 'analytical': 1.001, 
             'diff': 0.001, 'rel_diff': 0.001, 'passed': True}
        ]
        
        print_gradient_check_results(True, details)
        
        captured = capsys.readouterr()
        assert "PASSED" in captured.out


class TestNumericalStability:
    """Tests for numerical stability."""
    
    def test_small_values(self):
        """Test gradient check with small values."""
        def f(x):
            return ops.sum(ops.mul(x, x))
        
        x = Tensor([[0.001, 0.002, 0.003]], requires_grad=True)
        passed, details = check_gradients(f, x, eps=1e-6)
        
        assert passed, f"Gradient check failed for small values: {details}"
    
    def test_large_values(self):
        """Test gradient check with larger values."""
        def f(x):
            return ops.sum(x)
        
        x = Tensor([[100.0, 200.0, 300.0]], requires_grad=True)
        passed, details = check_gradients(f, x)
        
        assert passed, f"Gradient check failed for large values: {details}"
