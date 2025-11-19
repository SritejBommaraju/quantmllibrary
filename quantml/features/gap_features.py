"""
Overnight gap features for futures trading.
"""

from typing import List, Dict
from quantml.features.base import BaseFeature, FeatureMetadata


class OvernightGapFeature(BaseFeature):
    """
    Overnight gap feature.
    
    Computes the gap between previous day's close and current day's open.
    """
    
    def __init__(self, normalize: bool = True):
        """
        Initialize overnight gap feature.
        
        Args:
            normalize: Whether to normalize gap by previous close
        """
        super().__init__(
            name="overnight_gap",
            description="Gap between previous close and current open",
            normalize=normalize
        )
        self.metadata.formula = "gap = (open(t) - close(t-1)) / close(t-1) if normalize else open(t) - close(t-1)"
        self.metadata.expected_range = (-0.1, 0.1) if normalize else None
        self.metadata.unit = "fraction" if normalize else "price_units"
    
    def compute(self, data: Dict[str, List[float]]) -> List[float]:
        """Compute overnight gaps."""
        if not self.validate_data(data, ['open', 'close']):
            raise ValueError("OvernightGapFeature requires 'open' and 'close' in data")
        
        opens = data['open']
        closes = data['close']
        
        gaps = [0.0]  # First gap is 0 (no previous close)
        
        for i in range(1, len(opens)):
            if closes[i-1] > 0:
                if self.params.get('normalize', True):
                    gap = (opens[i] - closes[i-1]) / closes[i-1]
                else:
                    gap = opens[i] - closes[i-1]
            else:
                gap = 0.0
            gaps.append(gap)
        
        return gaps


class GapSizeFeature(BaseFeature):
    """
    Gap size feature (absolute value of gap).
    """
    
    def __init__(self, normalize: bool = True):
        """
        Initialize gap size feature.
        
        Args:
            normalize: Whether to normalize by previous close
        """
        super().__init__(
            name="gap_size",
            description="Absolute size of overnight gap",
            normalize=normalize
        )
        self.metadata.formula = "gap_size = |gap|"
        self.metadata.expected_range = (0.0, 0.1) if normalize else None
    
    def compute(self, data: Dict[str, List[float]]) -> List[float]):
        """Compute gap sizes."""
        gap_feature = OvernightGapFeature(normalize=self.params.get('normalize', True))
        gaps = gap_feature.compute(data)
        return [abs(g) for g in gaps]


class GapClosureFeature(BaseFeature):
    """
    Gap closure feature (whether gap closed during the day).
    """
    
    def __init__(self):
        """Initialize gap closure feature."""
        super().__init__(
            name="gap_closure",
            description="Binary indicator if overnight gap closed during day"
        )
        self.metadata.formula = "gap_closed = 1 if gap closed, 0 otherwise"
        self.metadata.expected_range = (0, 1)
    
    def compute(self, data: Dict[str, List[float]]) -> List[float]:
        """Compute gap closure indicators."""
        if not self.validate_data(data, ['open', 'close', 'high', 'low']):
            raise ValueError("GapClosureFeature requires 'open', 'close', 'high', 'low'")
        
        opens = data['open']
        closes = data['close']
        highs = data['high']
        lows = data['low']
        
        closures = [0.0]  # First day has no gap
        
        for i in range(1, len(opens)):
            prev_close = closes[i-1]
            gap = opens[i] - prev_close
            
            if abs(gap) < 0.0001:  # No gap
                closures.append(0.0)
            elif gap > 0:  # Gap up
                # Gap closed if low <= prev_close
                closed = 1.0 if lows[i] <= prev_close else 0.0
                closures.append(closed)
            else:  # Gap down
                # Gap closed if high >= prev_close
                closed = 1.0 if highs[i] >= prev_close else 0.0
                closures.append(closed)
        
        return closures

