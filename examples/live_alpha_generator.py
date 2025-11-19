"""
Live Alpha Generator - Real-time alpha signal generation

This module provides real-time alpha generation from streaming market data.
Works with any data source via adapter pattern.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import List, Optional, Callable, Dict, Any
from quantml import Tensor
from quantml.models import Linear
from quantml.streaming import StreamingTensor
from examples.alpha_factors import (
    MomentumFactor, MeanReversionFactor, 
    MicrostructureFactor, AlphaFactorCombiner
)


class DataAdapter:
    """Adapter for different data sources."""
    
    def __init__(self, data_source: Any, get_price_fn: Callable, get_volume_fn: Optional[Callable] = None):
        """
        Initialize data adapter.
        
        Args:
            data_source: Your data source (e.g., API client, file, database)
            get_price_fn: Function to get price: get_price_fn(data_source, index) -> float
            get_volume_fn: Optional function to get volume
        """
        self.data_source = data_source
        self.get_price = get_price_fn
        self.get_volume = get_volume_fn
    
    def get_prices(self, n: int) -> List[float]:
        """Get last N prices."""
        prices = []
        for i in range(n):
            price = self.get_price(self.data_source, i)
            if price is not None:
                prices.insert(0, price)  # Most recent first
        return prices
    
    def get_volumes(self, n: int) -> List[float]:
        """Get last N volumes."""
        if not self.get_volume:
            return [100.0] * n  # Default volume
        
        volumes = []
        for i in range(n):
            vol = self.get_volume(self.data_source, i)
            if vol is not None:
                volumes.insert(0, vol)
        return volumes


class LiveAlphaGenerator:
    """Real-time alpha signal generator."""
    
    def __init__(
        self,
        model: Linear,
        lookback_window: int = 100,
        feature_window: int = 20
    ):
        """
        Initialize live alpha generator.
        
        Args:
            model: Trained model for inference
            lookback_window: How much history to keep
            feature_window: Window for feature computation
        """
        self.model = model
        self.lookback_window = lookback_window
        self.feature_window = feature_window
        
        # Streaming buffers
        self.price_stream = StreamingTensor(max_size=lookback_window)
        self.volume_stream = StreamingTensor(max_size=lookback_window)
        
        # Cached features
        self.last_features = None
    
    def update(self, price: float, volume: Optional[float] = None):
        """Update with new market data."""
        self.price_stream.append(price)
        if volume is not None:
            self.volume_stream.append(volume)
        else:
            self.volume_stream.append(100.0)  # Default
    
    def generate_signal(self) -> float:
        """Generate alpha signal from current state."""
        # Get recent data
        prices = self.price_stream.get_window(self.feature_window)
        volumes = self.volume_stream.get_window(self.feature_window)
        
        if len(prices) < self.feature_window:
            return 0.0  # Not enough data
        
        # Compute alpha factors
        momentum = MomentumFactor.price_momentum(prices, lookback=min(10, len(prices)))
        zscore = MeanReversionFactor.zscore_factor(prices, window=min(10, len(prices)))
        
        # Get latest values
        mom_val = momentum[-1] if momentum else 0.0
        zscore_val = zscore[-1] if zscore else 0.0
        
        # Simple factor combination (can be replaced with model)
        signal = 0.5 * mom_val - 0.5 * zscore_val  # Momentum - mean reversion
        
        # Use model if available
        if self.model and len(prices) >= self.feature_window:
            try:
                # Create features
                features = self._create_features(prices, volumes)
                if features:
                    x = Tensor([features])
                    pred = self.model.forward(x)
                    pred_val = pred.data[0][0] if isinstance(pred.data[0], list) else pred.data[0]
                    signal = pred_val
            except Exception:
                pass  # Fall back to factor-based signal
        
        return signal
    
    def _create_features(self, prices: List[float], volumes: List[float]) -> List[float]:
        """Create feature vector from prices and volumes."""
        if len(prices) < 5:
            return None
        
        features = []
        
        # Price features
        features.append(prices[-1])  # Current price
        features.append((prices[-1] - prices[-2]) / prices[-2] if prices[-2] > 0 else 0.0)  # Return
        features.append((prices[-1] - prices[-5]) / prices[-5] if prices[-5] > 0 else 0.0)  # 5-period return
        
        # Rolling stats
        recent_prices = prices[-10:] if len(prices) >= 10 else prices
        features.append(sum(recent_prices) / len(recent_prices))  # Mean
        mean = sum(recent_prices) / len(recent_prices)
        variance = sum((p - mean) ** 2 for p in recent_prices) / len(recent_prices)
        features.append(variance ** 0.5)  # Std
        
        # Volume
        if volumes:
            features.append(volumes[-1] if volumes else 100.0)
            recent_vols = volumes[-10:] if len(volumes) >= 10 else volumes
            features.append(sum(recent_vols) / len(recent_vols))
        else:
            features.extend([100.0, 100.0])
        
        return features
    
    def get_signal_with_metadata(self) -> Dict[str, Any]:
        """Get signal with metadata."""
        signal = self.generate_signal()
        
        prices = self.price_stream.get_window(10)
        volumes = self.volume_stream.get_window(10)
        
        return {
            'signal': signal,
            'price': prices[-1] if prices else 0.0,
            'volume': volumes[-1] if volumes else 0.0,
            'data_points': len(prices),
            'action': 'BUY' if signal > 0.01 else ('SELL' if signal < -0.01 else 'HOLD')
        }


def create_simple_adapter(prices: List[float], volumes: Optional[List[float]] = None):
    """Create a simple in-memory data adapter for testing."""
    class SimpleSource:
        def __init__(self, p, v):
            self.prices = p
            self.volumes = v or [100.0] * len(p)
    
    source = SimpleSource(prices, volumes or [100.0] * len(prices))
    
    def get_price(src, idx):
        if 0 <= idx < len(src.prices):
            return src.prices[-(idx+1)]  # Reverse index
        return None
    
    def get_volume(src, idx):
        if 0 <= idx < len(src.volumes):
            return src.volumes[-(idx+1)]
        return None
    
    return DataAdapter(source, get_price, get_volume)


# Example usage
if __name__ == "__main__":
    print("Live Alpha Generator Example")
    print("=" * 70)
    
    # Create generator (without model for simplicity)
    generator = LiveAlphaGenerator(model=None, lookback_window=100, feature_window=20)
    
    # Simulate streaming data
    import random
    base_price = 100.0
    
    print("\nSimulating live market data...")
    for i in range(50):
        # New tick
        change = random.gauss(0, 0.3)
        base_price += change
        volume = 100.0 + random.gauss(0, 10)
        
        # Update generator
        generator.update(base_price, volume)
        
        # Generate signal
        if i >= 20:  # Wait for enough data
            signal_data = generator.get_signal_with_metadata()
            print(f"Tick {i+1}: Price={signal_data['price']:.2f}, "
                  f"Signal={signal_data['signal']:.4f}, "
                  f"Action={signal_data['action']}")
    
    print("\n" + "=" * 70)
    print("To use with real data:")
    print("  1. Create DataAdapter with your data source")
    print("  2. Initialize LiveAlphaGenerator with trained model")
    print("  3. Call update() on each new tick")
    print("  4. Call generate_signal() to get alpha signal")

