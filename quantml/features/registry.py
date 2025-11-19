"""
Feature registry for plugin-based feature system.
"""

from typing import Dict, List, Optional, Callable
from quantml.features.base import BaseFeature, FeatureMetadata


class FeatureRegistry:
    """Registry for managing features."""
    
    _instance = None
    _features: Dict[str, BaseFeature] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def register(self, feature: BaseFeature, name: Optional[str] = None):
        """
        Register a feature.
        
        Args:
            feature: Feature instance
            name: Optional name override (default: feature.name)
        """
        feature_name = name or feature.name
        self._features[feature_name] = feature
    
    def get(self, name: str) -> Optional[BaseFeature]:
        """
        Get feature by name.
        
        Args:
            name: Feature name
        
        Returns:
            Feature instance or None
        """
        return self._features.get(name)
    
    def list_features(self) -> List[str]:
        """List all registered feature names."""
        return list(self._features.keys())
    
    def get_all(self) -> Dict[str, BaseFeature]:
        """Get all registered features."""
        return self._features.copy()
    
    def compute_features(
        self,
        feature_names: List[str],
        data: Dict[str, List[float]]
    ) -> Dict[str, List[float]]:
        """
        Compute multiple features.
        
        Args:
            feature_names: List of feature names to compute
            data: Input data dictionary
        
        Returns:
            Dictionary of feature name -> feature values
        """
        results = {}
        
        for name in feature_names:
            feature = self.get(name)
            if feature is None:
                raise ValueError(f"Feature not found: {name}")
            
            try:
                values = feature.compute(data)
                results[name] = values
            except Exception as e:
                raise RuntimeError(f"Error computing feature {name}: {e}")
        
        return results
    
    def get_metadata(self, name: str) -> Optional[FeatureMetadata]:
        """
        Get feature metadata.
        
        Args:
            name: Feature name
        
        Returns:
            FeatureMetadata or None
        """
        feature = self.get(name)
        return feature.get_metadata() if feature else None


# Global registry instance
_registry = FeatureRegistry()


def register_feature(feature: BaseFeature, name: Optional[str] = None):
    """
    Register a feature in the global registry.
    
    Args:
        feature: Feature instance
        name: Optional name override
    """
    _registry.register(feature, name)


def get_feature(name: str) -> Optional[BaseFeature]:
    """
    Get feature from global registry.
    
    Args:
        name: Feature name
    
    Returns:
        Feature instance or None
    """
    return _registry.get(name)


def compute_features(
    feature_names: List[str],
    data: Dict[str, List[float]]
) -> Dict[str, List[float]]:
    """
    Compute multiple features from registry.
    
    Args:
        feature_names: List of feature names
        data: Input data
    
    Returns:
        Dictionary of computed features
    """
    return _registry.compute_features(feature_names, data)

