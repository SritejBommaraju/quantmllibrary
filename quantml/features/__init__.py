"""
QuantML Feature Registry

Plugin-based feature system for easy feature addition and configuration.
"""

from quantml.features.base import BaseFeature, Feature
from quantml.features.registry import FeatureRegistry, register_feature
from quantml.features.gap_features import OvernightGapFeature, GapSizeFeature
from quantml.features.volume_features import VolumeRegimeFeature, VolumeShockFeature
from quantml.features.volatility_features import VolatilityRegimeFeature, RealizedVolatilityFeature

__all__ = [
    'BaseFeature',
    'Feature',
    'FeatureRegistry',
    'register_feature',
    'OvernightGapFeature',
    'GapSizeFeature',
    'VolumeRegimeFeature',
    'VolumeShockFeature',
    'VolatilityRegimeFeature',
    'RealizedVolatilityFeature'
]

