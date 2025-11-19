"""
Base feature class for plugin-based feature system.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class FeatureMetadata:
    """Metadata for a feature."""
    name: str
    description: str
    formula: Optional[str] = None
    expected_range: Optional[tuple] = None
    unit: Optional[str] = None


class BaseFeature(ABC):
    """
    Base class for all features.
    
    All features should inherit from this class and implement the compute method.
    """
    
    def __init__(self, name: str, description: str = "", **kwargs):
        """
        Initialize feature.
        
        Args:
            name: Feature name
            description: Feature description
            **kwargs: Feature-specific parameters
        """
        self.name = name
        self.description = description
        self.params = kwargs
        self.metadata = FeatureMetadata(
            name=name,
            description=description
        )
    
    @abstractmethod
    def compute(self, data: Dict[str, List[float]]) -> List[float]:
        """
        Compute feature values.
        
        Args:
            data: Dictionary with required data (e.g., {'price': [...], 'volume': [...]})
        
        Returns:
            List of feature values
        """
        pass
    
    def get_metadata(self) -> FeatureMetadata:
        """Get feature metadata."""
        return self.metadata
    
    def validate_data(self, data: Dict[str, List[float]], required_keys: List[str]) -> bool:
        """
        Validate that required data keys are present.
        
        Args:
            data: Data dictionary
            required_keys: List of required keys
        
        Returns:
            True if all required keys present
        """
        return all(key in data for key in required_keys)


class Feature(BaseFeature):
    """
    Simple feature implementation.
    
    Use this for simple features that don't need complex logic.
    """
    
    def __init__(
        self,
        name: str,
        compute_fn: callable,
        description: str = "",
        **kwargs
    ):
        """
        Initialize feature with compute function.
        
        Args:
            name: Feature name
            compute_fn: Function to compute feature: compute_fn(data) -> List[float]
            description: Feature description
            **kwargs: Additional parameters
        """
        super().__init__(name, description, **kwargs)
        self.compute_fn = compute_fn
    
    def compute(self, data: Dict[str, List[float]]) -> List[float]:
        """Compute feature using provided function."""
        return self.compute_fn(data, **self.params)

