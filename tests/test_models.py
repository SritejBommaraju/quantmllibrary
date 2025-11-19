"""
Tests for model classes.
"""

import pytest
from quantml import Tensor
from quantml.models import Linear, SimpleRNN


def test_linear_forward():
    """Test Linear layer forward pass."""
    model = Linear(in_features=3, out_features=2, bias=True)
    x = Tensor([[1.0, 2.0, 3.0]])
    y = model.forward(x)
    
    assert y is not None
    assert len(y.data) > 0


def test_linear_parameters():
    """Test Linear layer parameters."""
    model = Linear(in_features=5, out_features=1, bias=True)
    params = model.parameters()
    
    assert len(params) >= 2  # Weight and bias
    assert all(p.requires_grad for p in params)


def test_linear_zero_grad():
    """Test Linear layer zero_grad."""
    model = Linear(in_features=3, out_features=1)
    x = Tensor([[1.0, 2.0, 3.0]])
    y = model.forward(x)
    
    # Create a dummy loss and backward
    from quantml import ops
    loss = ops.sum(y)
    loss.backward()
    
    # Zero gradients
    model.zero_grad()
    
    # Check gradients are cleared
    for param in model.parameters():
        assert param.grad is None


def test_simple_rnn_forward():
    """Test SimpleRNN forward pass."""
    rnn = SimpleRNN(input_size=5, hidden_size=10)
    x = Tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    h = rnn.forward(x)
    
    assert h is not None
    assert len(h.data) > 0


def test_simple_rnn_parameters():
    """Test SimpleRNN parameters."""
    rnn = SimpleRNN(input_size=5, hidden_size=10)
    params = rnn.parameters()
    
    assert len(params) >= 2  # Weight_ih and weight_hh
    assert all(p.requires_grad for p in params)

