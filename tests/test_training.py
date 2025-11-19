"""
Tests for training pipeline.
"""

import pytest
from quantml import Tensor
from quantml.models import Linear
from quantml.optim import Adam
from quantml.training import QuantTrainer
from quantml.training.losses import mse_loss
from quantml.training.walk_forward import WalkForwardOptimizer, WindowType


def test_quant_trainer_init():
    """Test QuantTrainer initialization."""
    model = Linear(in_features=3, out_features=1)
    optimizer = Adam(model.parameters(), lr=0.001)
    trainer = QuantTrainer(model, optimizer, mse_loss)
    
    assert trainer.model == model
    assert trainer.optimizer == optimizer


def test_quant_trainer_train_step():
    """Test training step."""
    model = Linear(in_features=3, out_features=1)
    optimizer = Adam(model.parameters(), lr=0.001)
    trainer = QuantTrainer(model, optimizer, mse_loss)
    
    x = Tensor([[1.0, 2.0, 3.0]])
    y = Tensor([[0.5]])
    
    loss = trainer.train_step(x, y)
    
    assert isinstance(loss, (int, float)) or loss is not None


def test_walk_forward_optimizer():
    """Test walk-forward optimizer."""
    wfo = WalkForwardOptimizer(
        window_type=WindowType.EXPANDING,
        train_size=10,
        test_size=5
    )
    
    data = list(range(50))
    splits = list(wfo.split(data, n_splits=3))
    
    assert len(splits) == 3
    for train_idx, test_idx in splits:
        assert len(train_idx) >= 10
        assert len(test_idx) == 5
        # Test indices should come after train indices
        assert min(test_idx) >= max(train_idx)

