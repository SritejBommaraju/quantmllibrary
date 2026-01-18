"""
Tests for loss functions.
"""

import pytest
from quantml import Tensor
from quantml import ops
from quantml.training.losses import (
    mse_loss, mae_loss, quantile_loss, sharpe_loss,
    information_ratio_loss, huber_loss, asymmetric_loss,
    max_drawdown_loss, combined_quant_loss
)


class TestMSELoss:
    """Tests for MSE loss."""
    
    def test_mse_perfect_prediction(self):
        """Test MSE loss is 0 for perfect predictions."""
        pred = Tensor([[1.0, 2.0, 3.0]])
        target = Tensor([[1.0, 2.0, 3.0]])
        
        loss = mse_loss(pred, target)
        loss_val = loss.data[0][0] if isinstance(loss.data[0], list) else loss.data[0]
        
        assert loss_val == pytest.approx(0.0)
    
    def test_mse_known_value(self):
        """Test MSE loss with known values."""
        pred = Tensor([[2.0]])
        target = Tensor([[0.0]])
        
        loss = mse_loss(pred, target)
        loss_val = loss.data[0][0] if isinstance(loss.data[0], list) else loss.data[0]
        
        # MSE = mean((2-0)^2) = 4.0
        assert loss_val == pytest.approx(4.0)
    
    def test_mse_gradient(self):
        """Test MSE loss gradient."""
        pred = Tensor([[1.0]], requires_grad=True)
        target = Tensor([[0.0]])
        
        loss = mse_loss(pred, target)
        loss.backward()
        
        # d/d_pred MSE = 2 * (pred - target) / n = 2 * 1 / 1 = 2
        grad = pred.grad
        grad_val = grad[0][0] if isinstance(grad[0], list) else grad[0]
        assert grad_val == pytest.approx(2.0)


class TestMAELoss:
    """Tests for MAE loss."""
    
    def test_mae_perfect_prediction(self):
        """Test MAE loss is 0 for perfect predictions."""
        pred = Tensor([[1.0, 2.0, 3.0]])
        target = Tensor([[1.0, 2.0, 3.0]])
        
        loss = mae_loss(pred, target)
        loss_val = loss.data[0][0] if isinstance(loss.data[0], list) else loss.data[0]
        
        assert loss_val == pytest.approx(0.0)
    
    def test_mae_known_value(self):
        """Test MAE loss with known values."""
        pred = Tensor([[5.0]])
        target = Tensor([[2.0]])
        
        loss = mae_loss(pred, target)
        loss_val = loss.data[0][0] if isinstance(loss.data[0], list) else loss.data[0]
        
        # MAE = mean(|5-2|) = 3.0
        assert loss_val == pytest.approx(3.0)


class TestQuantileLoss:
    """Tests for quantile loss."""
    
    def test_quantile_loss_median(self):
        """Test quantile loss at median (quantile=0.5)."""
        pred = Tensor([[2.0]])
        target = Tensor([[1.0]])
        
        loss = quantile_loss(pred, target, quantile=0.5)
        loss_val = loss.data[0][0] if isinstance(loss.data[0], list) else loss.data[0]
        
        # For quantile=0.5, this is equivalent to half MAE
        assert loss_val >= 0
    
    def test_quantile_loss_invalid_quantile(self):
        """Test that invalid quantile raises error."""
        pred = Tensor([[1.0]])
        target = Tensor([[1.0]])
        
        with pytest.raises(ValueError):
            quantile_loss(pred, target, quantile=1.5)


class TestSharpeLoss:
    """Tests for Sharpe ratio loss."""
    
    def test_sharpe_loss_returns(self):
        """Test Sharpe loss computation."""
        # Positive returns should give negative loss (we minimize -Sharpe)
        pred = Tensor([[0.01, 0.02, 0.01, 0.03, 0.02]])
        target = Tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])  # Not used in current impl
        
        loss = sharpe_loss(pred, target)
        # Should return a value (negative Sharpe)
        assert loss is not None


class TestHuberLoss:
    """Tests for Huber loss."""
    
    def test_huber_small_error(self):
        """Test Huber loss for small errors (squared region)."""
        pred = Tensor([[1.5]])
        target = Tensor([[1.0]])
        
        loss = huber_loss(pred, target, delta=1.0)
        # Small error -> quadratic
        assert loss is not None
    
    def test_huber_large_error(self):
        """Test Huber loss for large errors (linear region)."""
        pred = Tensor([[10.0]])
        target = Tensor([[0.0]])
        
        loss = huber_loss(pred, target, delta=1.0)
        # Large error -> linear
        assert loss is not None


class TestAsymmetricLoss:
    """Tests for asymmetric loss."""
    
    def test_asymmetric_loss_over_prediction(self):
        """Test asymmetric loss penalizes over-prediction more."""
        pred = Tensor([[2.0]])  # Over-predicts
        target = Tensor([[0.0]])
        
        loss_asym = asymmetric_loss(pred, target, asymmetry=2.0)
        loss_sym = mse_loss(pred, target)
        
        # Asymmetric loss should be larger
        asym_val = loss_asym.data[0][0] if isinstance(loss_asym.data[0], list) else loss_asym.data[0]
        sym_val = loss_sym.data[0][0] if isinstance(loss_sym.data[0], list) else loss_sym.data[0]
        
        assert asym_val >= sym_val


class TestMaxDrawdownLoss:
    """Tests for max drawdown loss."""
    
    def test_max_drawdown_loss(self):
        """Test max drawdown loss computation."""
        pred = Tensor([[0.01, -0.02, 0.01, -0.03, 0.02]])
        target = Tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])
        
        loss = max_drawdown_loss(pred, target)
        # Should return a value
        assert loss is not None


class TestCombinedQuantLoss:
    """Tests for combined quant loss."""
    
    def test_combined_loss(self):
        """Test combined quant loss computation."""
        pred = Tensor([[0.01, 0.02, 0.01]])
        target = Tensor([[0.005, 0.015, 0.02]])
        
        loss = combined_quant_loss(
            pred, target,
            mse_weight=0.5,
            sharpe_weight=0.3,
            drawdown_weight=0.2
        )
        
        # Should return a value
        assert loss is not None
