"""
Volume-based features.
"""

from typing import List, Dict
from quantml.features.base import BaseFeature, FeatureMetadata


class VolumeRegimeFeature(BaseFeature):
    """
    Volume regime feature (low/normal/high volume).
    """
    
    def __init__(self, low_threshold: float = 0.25, high_threshold: float = 0.75):
        """
        Initialize volume regime feature.
        
        Args:
            low_threshold: Percentile for low volume (default: 25th)
            high_threshold: Percentile for high volume (default: 75th)
        """
        super().__init__(
            name="volume_regime",
            description="Volume regime classification (0=low, 1=normal, 2=high)",
            low_threshold=low_threshold,
            high_threshold=high_threshold
        )
        self.metadata.formula = "regime = 0 if volume < p25, 1 if p25 <= volume < p75, 2 if volume >= p75"
        self.metadata.expected_range = (0, 2)
    
    def compute(self, data: Dict[str, List[float]]) -> List[float]:
        """Compute volume regimes."""
        if not self.validate_data(data, ['volume']):
            raise ValueError("VolumeRegimeFeature requires 'volume' in data")
        
        volumes = data['volume']
        
        if len(volumes) < 20:
            return [1.0] * len(volumes)  # Default to normal
        
        # Calculate percentiles
        sorted_vols = sorted(volumes)
        low_idx = int(len(sorted_vols) * self.params['low_threshold'])
        high_idx = int(len(sorted_vols) * self.params['high_threshold'])
        
        low_threshold = sorted_vols[low_idx]
        high_threshold = sorted_vols[high_idx]
        
        regimes = []
        for vol in volumes:
            if vol < low_threshold:
                regimes.append(0.0)
            elif vol >= high_threshold:
                regimes.append(2.0)
            else:
                regimes.append(1.0)
        
        return regimes


class VolumeShockFeature(BaseFeature):
    """
    Volume shock feature (unusual volume).
    """
    
    def __init__(self, window: int = 20, threshold: float = 2.0):
        """
        Initialize volume shock feature.
        
        Args:
            window: Rolling window for volume average
            threshold: Multiplier threshold for shock (default: 2x average)
        """
        super().__init__(
            name="volume_shock",
            description="Binary indicator for volume shock (1 if volume > threshold * avg)",
            window=window,
            threshold=threshold
        )
        self.metadata.formula = "shock = 1 if volume > threshold * rolling_mean(volume, window)"
        self.metadata.expected_range = (0, 1)
    
    def compute(self, data: Dict[str, List[float]]) -> List[float]:
        """Compute volume shocks."""
        if not self.validate_data(data, ['volume']):
            raise ValueError("VolumeShockFeature requires 'volume' in data")
        
        volumes = data['volume']
        window = self.params.get('window', 20)
        threshold = self.params.get('threshold', 2.0)
        
        shocks = [0.0] * min(window, len(volumes))  # Not enough data
        
        for i in range(window, len(volumes)):
            window_vols = volumes[i-window:i]
            avg_volume = sum(window_vols) / len(window_vols)
            
            shock = 1.0 if volumes[i] > threshold * avg_volume else 0.0
            shocks.append(shock)
        
        return shocks


class VolumeRatioFeature(BaseFeature):
    """
    Volume ratio feature (current volume / average volume).
    """
    
    def __init__(self, window: int = 20):
        """
        Initialize volume ratio feature.
        
        Args:
            window: Rolling window for average volume
        """
        super().__init__(
            name="volume_ratio",
            description="Ratio of current volume to rolling average",
            window=window
        )
        self.metadata.formula = "volume_ratio = volume(t) / mean(volume(t-window:t))"
        self.metadata.expected_range = (0.0, 10.0)
    
    def compute(self, data: Dict[str, List[float]]) -> List[float]:
        """Compute volume ratios."""
        if not self.validate_data(data, ['volume']):
            raise ValueError("VolumeRatioFeature requires 'volume' in data")
        
        volumes = data['volume']
        window = self.params.get('window', 20)
        
        ratios = [1.0] * min(window, len(volumes))
        
        for i in range(window, len(volumes)):
            window_vols = volumes[i-window:i]
            avg_volume = sum(window_vols) / len(window_vols)
            
            ratio = volumes[i] / avg_volume if avg_volume > 0 else 1.0
            ratios.append(ratio)
        
        return ratios

