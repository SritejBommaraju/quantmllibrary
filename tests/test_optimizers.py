"""
Tests for optimizer implementations.
"""

import pytest
from quantml import Tensor
from quantml.models import Linear
from quantml.optim import Adam, SGD, RMSProp, AdaGrad, RAdam, AdaFactor


class TestAdam:
    """Tests for Adam optimizer."""
    
    def test_adam_init(self):
        """Test Adam initialization."""
        model = Linear(3, 1)
        optimizer = Adam(model.parameters(), lr=0.001)
        
        assert optimizer.lr == 0.001
        assert hasattr(optimizer, '_m')
        assert hasattr(optimizer, '_v')
    
    def test_adam_step(self):
        """Test Adam optimizer step."""
        model = Linear(3, 1)
        optimizer = Adam(model.parameters(), lr=0.01)
        
        # Get initial weights
        initial_weight = model.weight.data[0][0] if isinstance(model.weight.data[0], list) else model.weight.data[0]
        
        # Create input and target
        x = Tensor([[1.0, 2.0, 3.0]])
        y_pred = model.forward(x)
        
        # Compute loss (simplified MSE)
        target = Tensor([[1.0]])
        from quantml import ops
        loss = ops.mean(ops.mul(ops.sub(y_pred, target), ops.sub(y_pred, target)))
        
        # Backward and step
        loss.backward()
        optimizer.step()
        
        # Weights should have changed
        new_weight = model.weight.data[0][0] if isinstance(model.weight.data[0], list) else model.weight.data[0]
        assert new_weight != initial_weight
    
    def test_adam_momentum_update(self):
        """Test that Adam momentum buffers update correctly."""
        model = Linear(2, 1)
        optimizer = Adam(model.parameters(), lr=0.01)
        
        x = Tensor([[1.0, 1.0]])
        
        # Multiple steps
        for _ in range(3):
            model.zero_grad()
            y = model.forward(x)
            from quantml import ops
            loss = ops.sum(y)
            loss.backward()
            optimizer.step()
        
        # Momentum buffers should be non-zero
        assert len(optimizer._m) > 0


class TestSGD:
    """Tests for SGD optimizer."""
    
    def test_sgd_init(self):
        """Test SGD initialization."""
        model = Linear(3, 1)
        optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
        
        assert optimizer.lr == 0.01
        assert optimizer.momentum == 0.9
    
    def test_sgd_step(self):
        """Test SGD optimizer step."""
        model = Linear(3, 1)
        optimizer = SGD(model.parameters(), lr=0.1)
        
        initial_weight = model.weight.data[0][0] if isinstance(model.weight.data[0], list) else model.weight.data[0]
        
        x = Tensor([[1.0, 1.0, 1.0]])
        y = model.forward(x)
        from quantml import ops
        loss = ops.sum(y)
        loss.backward()
        optimizer.step()
        
        new_weight = model.weight.data[0][0] if isinstance(model.weight.data[0], list) else model.weight.data[0]
        assert new_weight != initial_weight
    
    def test_sgd_weight_decay(self):
        """Test SGD with weight decay."""
        model = Linear(2, 1)
        optimizer = SGD(model.parameters(), lr=0.1, weight_decay=0.01)
        
        x = Tensor([[1.0, 1.0]])
        y = model.forward(x)
        from quantml import ops
        loss = ops.sum(y)
        loss.backward()
        optimizer.step()
        
        # Should complete without error
        assert True


class TestRMSProp:
    """Tests for RMSProp optimizer."""
    
    def test_rmsprop_step(self):
        """Test RMSProp optimizer step."""
        model = Linear(2, 1)
        optimizer = RMSProp(model.parameters(), lr=0.01)
        
        initial_weight = model.weight.data[0][0] if isinstance(model.weight.data[0], list) else model.weight.data[0]
        
        x = Tensor([[1.0, 1.0]])
        y = model.forward(x)
        from quantml import ops
        loss = ops.sum(y)
        loss.backward()
        optimizer.step()
        
        new_weight = model.weight.data[0][0] if isinstance(model.weight.data[0], list) else model.weight.data[0]
        assert new_weight != initial_weight


class TestAdaGrad:
    """Tests for AdaGrad optimizer."""
    
    def test_adagrad_step(self):
        """Test AdaGrad optimizer step."""
        model = Linear(2, 1)
        optimizer = AdaGrad(model.parameters(), lr=0.1)
        
        x = Tensor([[1.0, 1.0]])
        y = model.forward(x)
        from quantml import ops
        loss = ops.sum(y)
        loss.backward()
        optimizer.step()
        
        # Should complete without error
        assert True


class TestRAdam:
    """Tests for RAdam optimizer."""
    
    def test_radam_step(self):
        """Test RAdam optimizer step."""
        model = Linear(2, 1)
        optimizer = RAdam(model.parameters(), lr=0.01)
        
        x = Tensor([[1.0, 1.0]])
        
        # RAdam needs a few steps to warm up
        for _ in range(5):
            model.zero_grad()
            y = model.forward(x)
            from quantml import ops
            loss = ops.sum(y)
            loss.backward()
            optimizer.step()
        
        # Should complete without error
        assert True


class TestZeroGrad:
    """Test gradient zeroing."""
    
    def test_zero_grad(self):
        """Test that zero_grad clears gradients."""
        model = Linear(2, 1)
        optimizer = Adam(model.parameters(), lr=0.01)
        
        x = Tensor([[1.0, 1.0]])
        y = model.forward(x)
        from quantml import ops
        loss = ops.sum(y)
        loss.backward()
        
        # Gradients should exist
        assert model.weight.grad is not None
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Gradients should be cleared
        assert model.weight.grad is None
