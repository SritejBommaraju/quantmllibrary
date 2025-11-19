"""
Feature engineering pipeline for quant models.

This module provides utilities for creating features from raw market data,
including lagged features, rolling windows, cross-sectional features, and normalization.
"""

from typing import List, Optional, Union, Callable, Dict, Any
from quantml.tensor import Tensor
from quantml import time_series

# Try to import NumPy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


class FeaturePipeline:
    """
    Feature engineering pipeline for reproducible feature creation.
    
    This class provides a framework for creating features from raw data,
    with support for lagged features, rolling windows, normalization, and more.
    
    Attributes:
        features: List of feature definitions
        normalizers: Dictionary of normalizers for each feature
    
    Examples:
        >>> pipeline = FeaturePipeline()
        >>> pipeline.add_lagged_feature('price', lags=[1, 5, 10])
        >>> pipeline.add_rolling_feature('price', window=20, func='mean')
        >>> features = pipeline.transform(prices)
    """
    
    def __init__(self):
        """Initialize feature pipeline."""
        self.features = []
        self.normalizers = {}
    
    def add_lagged_feature(
        self,
        name: str,
        lags: List[int],
        fill_value: float = 0.0
    ):
        """
        Add lagged features.
        
        Args:
            name: Feature name
            lags: List of lag values (e.g., [1, 5, 10] for 1, 5, 10 period lags)
            fill_value: Value to use for missing lags (at beginning)
        """
        self.features.append({
            'type': 'lagged',
            'name': name,
            'lags': lags,
            'fill_value': fill_value
        })
    
    def add_rolling_feature(
        self,
        name: str,
        window: int,
        func: str = 'mean',
        min_periods: Optional[int] = None
    ):
        """
        Add rolling window feature.
        
        Args:
            name: Feature name
            window: Window size
            func: Function to apply ('mean', 'std', 'min', 'max', 'sum')
            min_periods: Minimum periods required (default: window)
        """
        if min_periods is None:
            min_periods = window
        
        self.features.append({
            'type': 'rolling',
            'name': name,
            'window': window,
            'func': func,
            'min_periods': min_periods
        })
    
    def add_time_series_feature(
        self,
        name: str,
        func: str,
        **kwargs
    ):
        """
        Add time-series feature (EMA, returns, etc.).
        
        Args:
            name: Feature name
            func: Function name ('ema', 'returns', 'volatility', etc.)
            **kwargs: Arguments for the function
        """
        self.features.append({
            'type': 'time_series',
            'name': name,
            'func': func,
            'kwargs': kwargs
        })
    
    def add_normalization(
        self,
        feature_name: str,
        method: str = 'zscore',
        window: Optional[int] = None
    ):
        """
        Add normalization to a feature.
        
        Args:
            feature_name: Name of feature to normalize
            method: Normalization method ('zscore', 'minmax', 'robust')
            window: Rolling window for normalization (None for global)
        """
        if feature_name not in self.normalizers:
            self.normalizers[feature_name] = []
        self.normalizers[feature_name].append({
            'method': method,
            'window': window
        })
    
    def transform(self, data: Dict[str, List[float]]) -> List[List[float]]:
        """
        Transform raw data into features.
        
        Args:
            data: Dictionary of feature name -> values
        
        Returns:
            List of feature vectors (one per time step)
        """
        n = len(list(data.values())[0]) if data else 0
        if n == 0:
            return []
        
        feature_matrix = []
        
        for i in range(n):
            feature_vector = []
            
            for feat_def in self.features:
                feat_name = feat_def['name']
                if feat_name not in data:
                    continue
                
                values = data[feat_name]
                
                if feat_def['type'] == 'lagged':
                    for lag in feat_def['lags']:
                        if i >= lag:
                            feature_vector.append(values[i - lag])
                        else:
                            feature_vector.append(feat_def['fill_value'])
                
                elif feat_def['type'] == 'rolling':
                    window = feat_def['window']
                    func = feat_def['func']
                    min_periods = feat_def['min_periods']
                    
                    start_idx = max(0, i - window + 1)
                    window_data = values[start_idx:i+1]
                    
                    if len(window_data) >= min_periods:
                        if func == 'mean':
                            feat_val = sum(window_data) / len(window_data)
                        elif func == 'std':
                            mean_val = sum(window_data) / len(window_data)
                            variance = sum((x - mean_val) ** 2 for x in window_data) / len(window_data)
                            feat_val = variance ** 0.5
                        elif func == 'min':
                            feat_val = min(window_data)
                        elif func == 'max':
                            feat_val = max(window_data)
                        elif func == 'sum':
                            feat_val = sum(window_data)
                        else:
                            feat_val = 0.0
                    else:
                        feat_val = 0.0
                    
                    feature_vector.append(feat_val)
                
                elif feat_def['type'] == 'time_series':
                    func_name = feat_def['func']
                    kwargs = feat_def['kwargs']
                    
                    # Convert to tensor for time_series operations
                    tensor_data = Tensor([values[:i+1]])
                    
                    if func_name == 'ema':
                        n_periods = kwargs.get('n', 20)
                        ema_vals = time_series.ema(tensor_data, n=n_periods)
                        if isinstance(ema_vals.data[0], list) and len(ema_vals.data[0]) > 0:
                            feature_vector.append(ema_vals.data[0][-1])
                        else:
                            feature_vector.append(0.0)
                    
                    elif func_name == 'returns':
                        rets = time_series.returns(tensor_data)
                        if isinstance(rets.data[0], list) and len(rets.data[0]) > 0:
                            feature_vector.append(rets.data[0][-1])
                        else:
                            feature_vector.append(0.0)
                    
                    elif func_name == 'volatility':
                        n_periods = kwargs.get('n', 20)
                        vol = time_series.volatility(tensor_data, n=n_periods)
                        if isinstance(vol.data[0], list) and len(vol.data[0]) > 0:
                            feature_vector.append(vol.data[0][-1])
                        else:
                            feature_vector.append(0.0)
                    
                    else:
                        feature_vector.append(0.0)
            
            feature_matrix.append(feature_vector)
        
        # Apply normalization
        if self.normalizers:
            feature_matrix = self._apply_normalization(feature_matrix, data)
        
        return feature_matrix
    
    def _apply_normalization(
        self,
        feature_matrix: List[List[float]],
        original_data: Dict[str, List[float]]
    ) -> List[List[float]]:
        """Apply normalization to features."""
        # Simplified normalization - in practice would track feature indices
        # For now, apply z-score normalization globally
        if HAS_NUMPY:
            try:
                matrix = np.array(feature_matrix, dtype=np.float64)
                mean_vals = np.mean(matrix, axis=0)
                std_vals = np.std(matrix, axis=0)
                std_vals = np.where(std_vals == 0, 1.0, std_vals)
                normalized = (matrix - mean_vals) / std_vals
                return normalized.tolist()
            except (ValueError, TypeError):
                pass
        
        # Pure Python fallback
        if len(feature_matrix) == 0:
            return feature_matrix
        
        n_features = len(feature_matrix[0])
        means = [sum(row[i] for row in feature_matrix) / len(feature_matrix) 
                for i in range(n_features)]
        stds = []
        for i in range(n_features):
            variance = sum((row[i] - means[i]) ** 2 for row in feature_matrix) / len(feature_matrix)
            stds.append(variance ** 0.5 if variance > 0 else 1.0)
        
        normalized = []
        for row in feature_matrix:
            normalized.append([
                (row[i] - means[i]) / stds[i] if stds[i] > 0 else 0.0
                for i in range(n_features)
            ])
        
        return normalized


def create_lagged_features(data: List[float], lags: List[int]) -> List[List[float]]:
    """
    Create lagged features from a time series.
    
    Args:
        data: Time series data
        lags: List of lag values
    
    Returns:
        List of feature vectors with lagged values
    """
    features = []
    for i in range(len(data)):
        feature_vec = []
        for lag in lags:
            if i >= lag:
                feature_vec.append(data[i - lag])
            else:
                feature_vec.append(0.0)
        features.append(feature_vec)
    return features


def normalize_features(
    features: List[List[float]],
    method: str = 'zscore'
) -> List[List[float]]:
    """
    Normalize feature matrix.
    
    Args:
        features: Feature matrix
        method: Normalization method ('zscore', 'minmax')
    
    Returns:
        Normalized feature matrix
    """
    if len(features) == 0:
        return features
    
    if HAS_NUMPY:
        try:
            matrix = np.array(features, dtype=np.float64)
            if method == 'zscore':
                mean_vals = np.mean(matrix, axis=0)
                std_vals = np.std(matrix, axis=0)
                std_vals = np.where(std_vals == 0, 1.0, std_vals)
                normalized = (matrix - mean_vals) / std_vals
            elif method == 'minmax':
                min_vals = np.min(matrix, axis=0)
                max_vals = np.max(matrix, axis=0)
                ranges = max_vals - min_vals
                ranges = np.where(ranges == 0, 1.0, ranges)
                normalized = (matrix - min_vals) / ranges
            else:
                normalized = matrix
            return normalized.tolist()
        except (ValueError, TypeError):
            pass
    
    # Pure Python fallback
    n_features = len(features[0])
    
    if method == 'zscore':
        means = [sum(row[i] for row in features) / len(features) for i in range(n_features)]
        stds = []
        for i in range(n_features):
            variance = sum((row[i] - means[i]) ** 2 for row in features) / len(features)
            stds.append(variance ** 0.5 if variance > 0 else 1.0)
        
        normalized = [
            [(row[i] - means[i]) / stds[i] if stds[i] > 0 else 0.0 for i in range(n_features)]
            for row in features
        ]
    elif method == 'minmax':
        mins = [min(row[i] for row in features) for i in range(n_features)]
        maxs = [max(row[i] for row in features) for i in range(n_features)]
        ranges = [maxs[i] - mins[i] if maxs[i] > mins[i] else 1.0 for i in range(n_features)]
        
        normalized = [
            [(row[i] - mins[i]) / ranges[i] if ranges[i] > 0 else 0.0 for i in range(n_features)]
            for row in features
        ]
    else:
        normalized = features
    
    return normalized

