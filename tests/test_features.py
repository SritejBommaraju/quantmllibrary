"""
Tests for feature engineering.
"""

import pytest
from quantml.training.features import FeaturePipeline, normalize_features
from examples.alpha_factors import MomentumFactor, MeanReversionFactor


def test_feature_pipeline_lagged():
    """Test lagged feature creation."""
    pipeline = FeaturePipeline()
    pipeline.add_lagged_feature('price', lags=[1, 5])
    
    data = {'price': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]}
    features = pipeline.transform(data)
    
    assert len(features) == len(data['price'])
    # First few rows should have fill_value for lags
    assert features[0] is not None


def test_feature_pipeline_rolling():
    """Test rolling feature creation."""
    pipeline = FeaturePipeline()
    pipeline.add_rolling_feature('price', window=3, func='mean')
    
    data = {'price': [100.0, 101.0, 102.0, 103.0, 104.0]}
    features = pipeline.transform(data)
    
    assert len(features) == len(data['price'])


def test_normalize_features():
    """Test feature normalization."""
    features = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ]
    
    normalized = normalize_features(features, method='zscore')
    
    assert len(normalized) == len(features)
    assert len(normalized[0]) == len(features[0])


def test_momentum_factor():
    """Test momentum factor."""
    prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    momentum = MomentumFactor.price_momentum(prices, lookback=3)
    
    assert len(momentum) == len(prices)
    # First few should be 0 (not enough history)
    assert momentum[0] == 0.0


def test_mean_reversion_factor():
    """Test mean reversion factor."""
    prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    zscore = MeanReversionFactor.zscore_factor(prices, window=3)
    
    assert len(zscore) == len(prices)

