"""
Alpha Factor Library

Pre-built alpha factors commonly used in quantitative trading:
- Momentum factors
- Mean reversion factors
- Microstructure factors
- Volatility factors
"""

from typing import List, Optional, Dict
from quantml import Tensor
from quantml import time_series


class MomentumFactor:
    """Momentum-based alpha factors."""
    
    @staticmethod
    def price_momentum(prices: List[float], lookback: int = 20) -> List[float]:
        """Simple price momentum: (price_t - price_{t-n}) / price_{t-n}"""
        if len(prices) < lookback:
            return [0.0] * len(prices)
        
        factors = [0.0] * lookback
        for i in range(lookback, len(prices)):
            if prices[i - lookback] > 0:
                mom = (prices[i] - prices[i - lookback]) / prices[i - lookback]
            else:
                mom = 0.0
            factors.append(mom)
        return factors
    
    @staticmethod
    def ema_momentum(prices: List[float], fast: int = 10, slow: int = 20) -> List[float]:
        """EMA crossover momentum."""
        price_tensor = Tensor(prices)
        ema_fast = time_series.ema(price_tensor, n=fast)
        ema_slow = time_series.ema(price_tensor, n=slow)
        
        ema_fast_data = ema_fast.data[0] if isinstance(ema_fast.data[0], list) else ema_fast.data
        ema_slow_data = ema_slow.data[0] if isinstance(ema_slow.data[0], list) else ema_slow.data
        
        factors = []
        for i in range(len(prices)):
            if i < slow:
                factors.append(0.0)
            else:
                fast_val = ema_fast_data[i] if isinstance(ema_fast_data, list) else ema_fast_data
                slow_val = ema_slow_data[i] if isinstance(ema_slow_data, list) else ema_slow_data
                if slow_val > 0:
                    factors.append((fast_val - slow_val) / slow_val)
                else:
                    factors.append(0.0)
        return factors


class MeanReversionFactor:
    """Mean reversion alpha factors."""
    
    @staticmethod
    def zscore_factor(prices: List[float], window: int = 20) -> List[float]:
        """Z-score of price relative to rolling mean/std."""
        price_tensor = Tensor(prices)
        zscores = time_series.zscore(price_tensor, n=window)
        zscore_data = zscores.data[0] if isinstance(zscores.data[0], list) else zscores.data
        return zscore_data if isinstance(zscore_data, list) else [zscore_data]
    
    @staticmethod
    def vwap_deviation(prices: List[float], volumes: List[float], window: int = 20) -> List[float]:
        """Price deviation from VWAP."""
        if len(prices) != len(volumes) or len(prices) < window:
            return [0.0] * len(prices)
        
        price_tensor = Tensor(prices)
        volume_tensor = Tensor(volumes)
        vwap = time_series.vwap(price_tensor, volume_tensor)
        
        vwap_data = vwap.data[0] if isinstance(vwap.data[0], list) else vwap.data
        
        factors = []
        for i in range(len(prices)):
            if i < window:
                factors.append(0.0)
            else:
                vwap_val = vwap_data[i] if isinstance(vwap_data, list) else vwap_data
                if vwap_val > 0:
                    factors.append((prices[i] - vwap_val) / vwap_val)
                else:
                    factors.append(0.0)
        return factors


class MicrostructureFactor:
    """Microstructure-based alpha factors."""
    
    @staticmethod
    def order_flow_imbalance(bids: List[float], asks: List[float], 
                            bid_sizes: Optional[List[float]] = None,
                            ask_sizes: Optional[List[float]] = None) -> List[float]:
        """Order flow imbalance factor."""
        if len(bids) != len(asks):
            return [0.0] * max(len(bids), len(asks))
        
        factors = []
        for i in range(len(bids)):
            if bid_sizes and ask_sizes:
                ofi = time_series.orderflow_imbalance(
                    Tensor([bids[i]]), Tensor([asks[i]]),
                    Tensor([bid_sizes[i]]), Tensor([ask_sizes[i]])
                )
            else:
                ofi = time_series.orderflow_imbalance(
                    Tensor([bids[i]]), Tensor([asks[i]])
                )
            ofi_val = ofi.data[0][0] if isinstance(ofi.data[0], list) else ofi.data[0]
            factors.append(ofi_val)
        return factors
    
    @staticmethod
    def microprice_factor(bids: List[float], asks: List[float],
                         bid_sizes: Optional[List[float]] = None,
                         ask_sizes: Optional[List[float]] = None) -> List[float]:
        """Microprice factor (weighted bid-ask midpoint)."""
        if len(bids) != len(asks):
            return [0.0] * max(len(bids), len(asks))
        
        factors = []
        for i in range(len(bids)):
            if bid_sizes and ask_sizes:
                mp = time_series.microprice(
                    Tensor([bids[i]]), Tensor([asks[i]]),
                    Tensor([bid_sizes[i]]), Tensor([ask_sizes[i]])
                )
            else:
                mp = time_series.microprice(
                    Tensor([bids[i]]), Tensor([asks[i]])
                )
            mp_val = mp.data[0][0] if isinstance(mp.data[0], list) else mp.data[0]
            mid = (bids[i] + asks[i]) / 2.0 if asks[i] > 0 else bids[i]
            if mid > 0:
                factors.append((mp_val - mid) / mid)
            else:
                factors.append(0.0)
        return factors


class VolatilityFactor:
    """Volatility-based alpha factors."""
    
    @staticmethod
    def realized_volatility(prices: List[float], window: int = 20) -> List[float]:
        """Realized volatility factor."""
        price_tensor = Tensor(prices)
        vol = time_series.volatility(price_tensor, n=window, annualize=False)
        vol_data = vol.data[0] if isinstance(vol.data[0], list) else vol.data
        return vol_data if isinstance(vol_data, list) else [vol_data]
    
    @staticmethod
    def vol_of_vol(prices: List[float], vol_window: int = 20, vol_of_vol_window: int = 10) -> List[float]:
        """Volatility of volatility factor."""
        vol = VolatilityFactor.realized_volatility(prices, vol_window)
        vol_tensor = Tensor(vol)
        vol_of_vol = time_series.rolling_std(vol_tensor, n=vol_of_vol_window)
        vov_data = vol_of_vol.data[0] if isinstance(vol_of_vol.data[0], list) else vol_of_vol.data
        return vov_data if isinstance(vov_data, list) else [vov_data]


class AlphaFactorCombiner:
    """Combine multiple alpha factors into a single signal."""
    
    def __init__(self, factors: Dict[str, List[float]], weights: Optional[Dict[str, float]] = None):
        """
        Initialize factor combiner.
        
        Args:
            factors: Dictionary of factor name -> factor values
            weights: Optional weights for each factor (default: equal weights)
        """
        self.factors = factors
        self.weights = weights if weights else {name: 1.0 / len(factors) for name in factors.keys()}
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}
    
    def combine(self) -> List[float]:
        """Combine factors into single alpha signal."""
        if not self.factors:
            return []
        
        # Get length from first factor
        n = len(list(self.factors.values())[0])
        combined = [0.0] * n
        
        for i in range(n):
            signal = 0.0
            for factor_name, factor_values in self.factors.items():
                if i < len(factor_values):
                    weight = self.weights.get(factor_name, 0.0)
                    signal += weight * factor_values[i]
            combined[i] = signal
        
        return combined
    
    @staticmethod
    def cross_sectional_rank(factors: List[float]) -> List[float]:
        """Rank factors cross-sectionally (0 to 1 scale)."""
        if not factors:
            return []
        
        # Sort and get ranks
        sorted_indices = sorted(range(len(factors)), key=lambda i: factors[i])
        ranks = [0.0] * len(factors)
        
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = rank / max(len(factors) - 1, 1)  # Normalize to [0, 1]
        
        return ranks

