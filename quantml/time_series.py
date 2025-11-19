"""
Quantitative trading specific time-series operations.

This module provides operations commonly used in quantitative trading,
including moving averages, volatility measures, order flow indicators,
and price transformations.
"""

from typing import Union, Optional, List
from quantml.tensor import Tensor
from quantml import ops

# Try to import NumPy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

# Helper to convert to tensor
def _to_tensor(x):
    """Convert input to Tensor if needed."""
    if isinstance(x, Tensor):
        return x
    return Tensor(x)


def ema(t: Tensor, n: int, alpha: Optional[float] = None) -> Tensor:
    """
    Exponential Moving Average (EMA).
    
    EMA is calculated as: EMA_t = alpha * price_t + (1 - alpha) * EMA_{t-1}
    where alpha = 2 / (n + 1) by default.
    
    Args:
        t: Input tensor (price series)
        n: Period for EMA
        alpha: Smoothing factor (default: 2/(n+1))
    
    Returns:
        Tensor with EMA values
    
    Examples:
        >>> prices = Tensor([100.0, 101.0, 102.0, 103.0, 104.0])
        >>> ema_5 = ema(prices, n=5)
    """
    if alpha is None:
        alpha = 2.0 / (n + 1.0)
    
    t = _to_tensor(t)
    data = t.data if not isinstance(t.data[0], list) else t.data[0]
    
    if len(data) < n:
        # Not enough data, return original
        return t
    
    # Use NumPy if available for vectorized computation
    if HAS_NUMPY:
        try:
            data_arr = np.array(data, dtype=np.float64)
            ema_arr = np.zeros_like(data_arr)
            ema_arr[0] = data_arr[0]
            
            # Vectorized EMA calculation
            for i in range(1, len(data_arr)):
                ema_arr[i] = alpha * data_arr[i] + (1.0 - alpha) * ema_arr[i-1]
            
            return Tensor([ema_arr.tolist()], requires_grad=t.requires_grad)
        except (ValueError, TypeError):
            pass
    
    # Pure Python fallback
    ema_values = []
    ema_prev = float(data[0])
    ema_values.append(ema_prev)
    
    for i in range(1, len(data)):
        ema_curr = alpha * float(data[i]) + (1.0 - alpha) * ema_prev
        ema_values.append(ema_curr)
        ema_prev = ema_curr
    
    return Tensor([ema_values], requires_grad=t.requires_grad)


def wma(t: Tensor, n: int) -> Tensor:
    """
    Weighted Moving Average (WMA).
    
    WMA gives more weight to recent values. Weights are: n, n-1, ..., 1
    
    Args:
        t: Input tensor
        n: Period for WMA
    
    Returns:
        Tensor with WMA values
    """
    t = _to_tensor(t)
    data = t.data if not isinstance(t.data[0], list) else t.data[0]
    
    if len(data) < n:
        return t
    
    wma_values = []
    total_weight = sum(range(1, n + 1))
    
    for i in range(n - 1, len(data)):
        weighted_sum = sum(float(data[i - j]) * (n - j) for j in range(n))
        wma_values.append(weighted_sum / total_weight)
    
    # Pad beginning with first value
    for _ in range(n - 1):
        wma_values.insert(0, float(data[0]))
    
    return Tensor([wma_values], requires_grad=t.requires_grad)


def rolling_mean(t: Tensor, n: int) -> Tensor:
    """
    Rolling mean over a window of size n.
    
    Args:
        t: Input tensor
        n: Window size
    
    Returns:
        Tensor with rolling mean values
    """
    t = _to_tensor(t)
    data = t.data if not isinstance(t.data[0], list) else t.data[0]
    
    if len(data) < n:
        return t
    
    # Use NumPy if available
    if HAS_NUMPY:
        try:
            data_arr = np.array(data, dtype=np.float64)
            rolling_means = np.convolve(data_arr, np.ones(n)/n, mode='same')
            # Pad beginning
            rolling_means[:n-1] = data_arr[0]
            return Tensor([rolling_means.tolist()], requires_grad=t.requires_grad)
        except (ValueError, TypeError):
            pass
    
    # Pure Python fallback
    rolling_means = []
    for i in range(n - 1):
        rolling_means.append(float(data[0]))
    
    for i in range(n - 1, len(data)):
        window_sum = sum(float(data[j]) for j in range(i - n + 1, i + 1))
        rolling_means.append(window_sum / n)
    
    return Tensor([rolling_means], requires_grad=t.requires_grad)


def rolling_std(t: Tensor, n: int) -> Tensor:
    """
    Rolling standard deviation over a window of size n.
    
    Args:
        t: Input tensor
        n: Window size
    
    Returns:
        Tensor with rolling std values
    """
    t = _to_tensor(t)
    data = t.data if not isinstance(t.data[0], list) else t.data[0]
    
    if len(data) < n:
        return Tensor([[0.0] * len(data)], requires_grad=t.requires_grad)
    
    # Use NumPy if available
    if HAS_NUMPY:
        try:
            data_arr = np.array(data, dtype=np.float64)
            rolling_stds = np.zeros_like(data_arr)
            
            for i in range(n - 1, len(data_arr)):
                window = data_arr[i - n + 1:i + 1]
                rolling_stds[i] = np.std(window)
            
            return Tensor([rolling_stds.tolist()], requires_grad=t.requires_grad)
        except (ValueError, TypeError):
            pass
    
    # Pure Python fallback
    rolling_stds = []
    for i in range(n - 1):
        rolling_stds.append(0.0)
    
    for i in range(n - 1, len(data)):
        window = [float(data[j]) for j in range(i - n + 1, i + 1)]
        mean_val = sum(window) / n
        variance = sum((x - mean_val) ** 2 for x in window) / n
        rolling_stds.append(variance ** 0.5)
    
    return Tensor([rolling_stds], requires_grad=t.requires_grad)


def zscore(t: Tensor, n: int) -> Tensor:
    """
    Z-score normalization: (x - mean) / std over rolling window.
    
    Args:
        t: Input tensor
        n: Window size for mean and std calculation
    
    Returns:
        Tensor with z-scores
    """
    t = _to_tensor(t)
    mean_vals = rolling_mean(t, n)
    std_vals = rolling_std(t, n)
    
    # Avoid division by zero
    std_safe = ops.add(std_vals, Tensor([[1e-8]]))
    return ops.div(ops.sub(t, mean_vals), std_safe)


def volatility(t: Tensor, n: int, annualize: bool = True) -> Tensor:
    """
    Rolling volatility (annualized standard deviation of returns).
    
    Args:
        t: Price tensor
        n: Window size
        annualize: Whether to annualize (multiply by sqrt(252))
    
    Returns:
        Tensor with volatility values
    """
    returns_t = returns(t)
    vol = rolling_std(returns_t, n)
    
    if annualize:
        import math
        vol = ops.mul(vol, math.sqrt(252.0))
    
    return vol


def returns(t: Tensor) -> Tensor:
    """
    Calculate returns: (p_t - p_{t-1}) / p_{t-1}
    
    Args:
        t: Price tensor
    
    Returns:
        Tensor with returns
    """
    t = _to_tensor(t)
    data = t.data if not isinstance(t.data[0], list) else t.data[0]
    
    if len(data) < 2:
        return Tensor([[0.0] * len(data)], requires_grad=t.requires_grad)
    
    rets = [0.0]  # First return is 0
    for i in range(1, len(data)):
        if float(data[i - 1]) != 0:
            ret = (float(data[i]) - float(data[i - 1])) / float(data[i - 1])
        else:
            ret = 0.0
        rets.append(ret)
    
    return Tensor([rets], requires_grad=t.requires_grad)


def log_returns(t: Tensor) -> Tensor:
    """
    Calculate log returns: log(p_t / p_{t-1})
    
    Args:
        t: Price tensor
    
    Returns:
        Tensor with log returns
    """
    import math
    t = _to_tensor(t)
    data = t.data if not isinstance(t.data[0], list) else t.data[0]
    
    if len(data) < 2:
        return Tensor([[0.0] * len(data)], requires_grad=t.requires_grad)
    
    log_rets = [0.0]
    for i in range(1, len(data)):
        if float(data[i - 1]) > 0 and float(data[i]) > 0:
            log_ret = math.log(float(data[i]) / float(data[i - 1]))
        else:
            log_ret = 0.0
        log_rets.append(log_ret)
    
    return Tensor([log_rets], requires_grad=t.requires_grad)


def vwap(price: Tensor, volume: Tensor) -> Tensor:
    """
    Volume-Weighted Average Price (VWAP).
    
    VWAP = sum(price * volume) / sum(volume)
    
    Args:
        price: Price tensor
        volume: Volume tensor
    
    Returns:
        Tensor with VWAP values
    """
    price = _to_tensor(price)
    volume = _to_tensor(volume)
    
    price_data = price.data if not isinstance(price.data[0], list) else price.data[0]
    volume_data = volume.data if not isinstance(volume.data[0], list) else volume.data[0]
    
    if len(price_data) != len(volume_data):
        raise ValueError("Price and volume must have same length")
    
    # Use NumPy if available
    if HAS_NUMPY:
        try:
            price_arr = np.array(price_data, dtype=np.float64)
            volume_arr = np.array(volume_data, dtype=np.float64)
            
            # Cumulative sums
            cum_price_vol = np.cumsum(price_arr * volume_arr)
            cum_vol = np.cumsum(volume_arr)
            
            # VWAP = cumulative price*volume / cumulative volume
            vwap_arr = np.where(cum_vol > 0, cum_price_vol / cum_vol, price_arr)
            
            return Tensor([vwap_arr.tolist()], requires_grad=price.requires_grad or volume.requires_grad)
        except (ValueError, TypeError):
            pass
    
    # Pure Python fallback
    vwap_values = []
    cum_price_vol = 0.0
    cum_vol = 0.0
    
    for i in range(len(price_data)):
        p = float(price_data[i])
        v = float(volume_data[i])
        cum_price_vol += p * v
        cum_vol += v
        
        if cum_vol > 0:
            vwap_values.append(cum_price_vol / cum_vol)
        else:
            vwap_values.append(p)
    
    return Tensor([vwap_values], requires_grad=price.requires_grad or volume.requires_grad)


def orderflow_imbalance(bid: Tensor, ask: Tensor) -> Tensor:
    """
    Order Flow Imbalance (OFI).
    
    OFI = (bid_size - ask_size) / (bid_size + ask_size)
    
    Args:
        bid: Bid size tensor
        ask: Ask size tensor
    
    Returns:
        Tensor with OFI values (ranges from -1 to 1)
    """
    bid = _to_tensor(bid)
    ask = _to_tensor(ask)
    
    bid_data = bid.data if not isinstance(bid.data[0], list) else bid.data[0]
    ask_data = ask.data if not isinstance(ask.data[0], list) else ask.data[0]
    
    if len(bid_data) != len(ask_data):
        raise ValueError("Bid and ask must have same length")
    
    ofi_values = []
    for i in range(len(bid_data)):
        b = float(bid_data[i])
        a = float(ask_data[i])
        total = b + a
        if total > 0:
            ofi = (b - a) / total
        else:
            ofi = 0.0
        ofi_values.append(ofi)
    
    return Tensor([ofi_values], requires_grad=bid.requires_grad or ask.requires_grad)


def microprice(bid: Tensor, ask: Tensor, bid_size: Optional[Tensor] = None, 
               ask_size: Optional[Tensor] = None) -> Tensor:
    """
    Microprice: weighted average of bid and ask prices.
    
    If sizes are provided: microprice = (bid * ask_size + ask * bid_size) / (bid_size + ask_size)
    Otherwise: microprice = (bid + ask) / 2 (mid price)
    
    Args:
        bid: Bid price tensor
        ask: Ask price tensor
        bid_size: Bid size tensor (optional)
        ask_size: Ask size tensor (optional)
    
    Returns:
        Tensor with microprice values
    """
    bid = _to_tensor(bid)
    ask = _to_tensor(ask)
    
    bid_data = bid.data if not isinstance(bid.data[0], list) else bid.data[0]
    ask_data = ask.data if not isinstance(ask.data[0], list) else ask.data[0]
    
    if len(bid_data) != len(ask_data):
        raise ValueError("Bid and ask must have same length")
    
    if bid_size is None or ask_size is None:
        # Simple mid price
        microprice_values = [(float(bid_data[i]) + float(ask_data[i])) / 2.0 
                            for i in range(len(bid_data))]
    else:
        bid_size = _to_tensor(bid_size)
        ask_size = _to_tensor(ask_size)
        
        bid_size_data = bid_size.data if not isinstance(bid_size.data[0], list) else bid_size.data[0]
        ask_size_data = ask_size.data if not isinstance(ask_size.data[0], list) else ask_size.data[0]
        
        microprice_values = []
        for i in range(len(bid_data)):
            b = float(bid_data[i])
            a = float(ask_data[i])
            bs = float(bid_size_data[i])
            asz = float(ask_size_data[i])
            total_size = bs + asz
            
            if total_size > 0:
                mp = (b * asz + a * bs) / total_size
            else:
                mp = (b + a) / 2.0
            microprice_values.append(mp)
    
    return Tensor([microprice_values], requires_grad=bid.requires_grad or ask.requires_grad)

