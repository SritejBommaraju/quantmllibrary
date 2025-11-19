"""
Integration tests for end-to-end pipeline.
"""

import pytest
from quantml import Tensor
from quantml.models import Linear
from quantml.optim import Adam
from quantml.training import QuantTrainer, FeaturePipeline
from quantml.training.losses import mse_loss
from quantml.training.features import normalize_features


def test_end_to_end_pipeline():
    """Test complete pipeline from data to prediction."""
    # Generate sample data
    prices = [100.0 + i * 0.5 for i in range(100)]
    volumes = [100.0 + i for i in range(100)]
    
    # Create features
    pipeline = FeaturePipeline()
    pipeline.add_lagged_feature('price', lags=[1, 5])
    pipeline.add_rolling_feature('price', window=10, func='mean')
    
    features = pipeline.transform({'price': prices})
    features = normalize_features(features, method='zscore')
    
    # Create targets
    targets = []
    for i in range(len(prices) - 1):
        ret = (prices[i + 1] - prices[i]) / prices[i]
        targets.append(ret)
    
    features = features[:-1]  # Align
    
    # Create and train model
    model = Linear(in_features=len(features[0]), out_features=1, bias=True)
    optimizer = Adam(model.parameters(), lr=0.001)
    trainer = QuantTrainer(model, optimizer, mse_loss)
    
    # Train on a few samples
    for i in range(min(10, len(features))):
        x = Tensor([features[i]])
        y = Tensor([[targets[i]]])
        trainer.train_step(x, y)
    
    # Generate prediction
    x_test = Tensor([features[0]])
    pred = model.forward(x_test)
    
    assert pred is not None
    assert len(pred.data) > 0

