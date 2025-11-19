"""
Volatility-based features.
"""

from typing import List, Dict
from quantml.features.base import BaseFeature, FeatureMetadata


class VolatilityRegimeFeature(BaseFeature):
    """
    Volatility regime feature (low/normal/high volatility).
    """
    
    def __init__(self, window: int = 20, low_threshold: float = 0.25, high_threshold: float = 0.75):
        """
        Initialize volatility regime feature.
        
        Args:
            window: Window for volatility calculation
            low_threshold: Percentile for low volatility
            high_threshold: Percentile for high volatility
        """
        super().__init__(
            name="volatility_regime",
            description="Volatility regime (0=low, 1=normal, 2=high)",
            window=window,
            low_threshold=low_threshold,
            high_threshold=high_threshold
        )
        self.metadata.formula = "regime based on realized volatility percentiles"
        self.metadata.expected_range = (0, 2)
    
    def compute(self, data: Dict[str, List[float]]) -> List[float]:
        """Compute volatility regimes."""
        if not self.validate_data(data, ['price']):
            raise ValueError("VolatilityRegimeFeature requires 'price' in data")
        
        prices = data['price']
        window = self.params.get('window', 20)
        
        # Calculate returns
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                ret = (prices[i] - prices[i-1]) / prices[i-1]
            else:
                ret = 0.0
            returns.append(ret)
        
        # Calculate rolling volatility
        volatilities = []
        for i in range(window, len(returns)):
            window_rets = returns[i-window:i]
            mean_ret = sum(window_rets) / len(window_rets)
            variance = sum((r - mean_ret) ** 2 for r in window_rets) / len(window_rets)
            vol = variance ** 0.5
            volatilities.append(vol)
        
        if len(volatilities) < 2:
            return [1.0] * len(prices)  # Default to normal
        
        # Calculate percentiles
        sorted_vols = sorted(volatilities)
        low_idx = int(len(sorted_vols) * self.params['low_threshold'])
        high_idx = int(len(sorted_vols) * self.params['high_threshold'])
        
        low_threshold = sorted_vols[low_idx]
        high_threshold = sorted_vols[high_idx]
        
        # Map to regimes
        regimes = [1.0] * window  # Not enough data
        
        for vol in volatilities:
            if vol < low_threshold:
                regimes.append(0.0)
            elif vol >= high_threshold:
                regimes.append(2.0)
            else:
                regimes.append(1.0)
        
        return regimes


class RealizedVolatilityFeature(BaseFeature):
    """
    Realized volatility feature.
    """
    
    def __init__(self, window: int = 20, annualize: bool = True):
        """
        Initialize realized volatility feature.
        
        Args:
            window: Rolling window
            annualize: Whether to annualize (multiply by sqrt(252))
        """
        super().__init__(
            name="realized_volatility",
            description="Realized volatility (std of returns)",
            window=window,
            annualize=annualize
        )
        self.metadata.formula = "vol = std(returns) * sqrt(252) if annualize"
        self.metadata.expected_range = (0.0, 1.0)
    
    def compute(self, data: Dict[str, List[float]]) -> List[float]:
        """Compute realized volatility."""
        if not self.validate_data(data, ['price']):
            raise ValueError("RealizedVolatilityFeature requires 'price' in data")
        
        prices = data['price']
        window = self.params.get('window', 20)
        annualize = self.params.get('annualize', True)
        
        # Calculate returns
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                ret = (prices[i] - prices[i-1]) / prices[i-1]
            else:
                ret = 0.0
            returns.append(ret)
        
        # Calculate rolling volatility
        volatilities = [0.0] * min(window, len(prices))
        
        for i in range(window, len(returns)):
            window_rets = returns[i-window:i]
            mean_ret = sum(window_rets) / len(window_rets)
            variance = sum((r - mean_ret) ** 2 for r in window_rets) / len(window_rets)
            vol = variance ** 0.5
            
            if annualize:
                import math
                vol = vol * math.sqrt(252)
            
            volatilities.append(vol)
        
        return volatilities

