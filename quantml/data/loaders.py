"""
Standardized data loaders for market data.
"""

from typing import List, Optional, Dict, Any, Callable
import os

# Try to import pandas
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None


def load_csv_data(
    filepath: str,
    price_column: str = "close",
    volume_column: str = "volume",
    timestamp_column: Optional[str] = "timestamp",
    date_format: Optional[str] = None
) -> Dict[str, List]:
    """
    Load market data from CSV file.
    
    Args:
        filepath: Path to CSV file
        price_column: Name of price column
        volume_column: Name of volume column
        timestamp_column: Name of timestamp column (optional)
        date_format: Date format string (optional)
    
    Returns:
        Dictionary with 'prices', 'volumes', and optionally 'timestamps'
    """
    if not HAS_PANDAS:
        raise ImportError("pandas required for CSV loading. Install with: pip install pandas")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Validate columns exist
    if price_column not in df.columns:
        raise ValueError(f"Price column '{price_column}' not found in CSV")
    
    if volume_column not in df.columns:
        raise ValueError(f"Volume column '{volume_column}' not found in CSV")
    
    # Extract data
    prices = df[price_column].tolist()
    volumes = df[volume_column].tolist()
    
    result = {
        'prices': prices,
        'volumes': volumes
    }
    
    # Add timestamps if available
    if timestamp_column and timestamp_column in df.columns:
        if date_format:
            timestamps = pd.to_datetime(df[timestamp_column], format=date_format).tolist()
        else:
            timestamps = pd.to_datetime(df[timestamp_column]).tolist()
        result['timestamps'] = timestamps
    
    return result


def load_dataframe(
    df: Any,
    price_column: str = "close",
    volume_column: str = "volume",
    timestamp_column: Optional[str] = "timestamp"
) -> Dict[str, List]:
    """
    Load market data from pandas DataFrame.
    
    Args:
        df: Pandas DataFrame
        price_column: Name of price column
        volume_column: Name of volume column
        timestamp_column: Name of timestamp column (optional)
    
    Returns:
        Dictionary with 'prices', 'volumes', and optionally 'timestamps'
    """
    if not HAS_PANDAS:
        raise ImportError("pandas required. Install with: pip install pandas")
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    
    # Validate columns
    if price_column not in df.columns:
        raise ValueError(f"Price column '{price_column}' not found")
    
    if volume_column not in df.columns:
        raise ValueError(f"Volume column '{volume_column}' not found")
    
    # Extract data
    prices = df[price_column].tolist()
    volumes = df[volume_column].tolist()
    
    result = {
        'prices': prices,
        'volumes': volumes
    }
    
    # Add timestamps if available
    if timestamp_column and timestamp_column in df.columns:
        timestamps = pd.to_datetime(df[timestamp_column]).tolist()
        result['timestamps'] = timestamps
    
    return result


class DataLoader:
    """Generic data loader with validation."""
    
    def __init__(
        self,
        load_fn: Callable,
        validate: bool = True,
        handle_missing: str = "forward_fill"
    ):
        """
        Initialize data loader.
        
        Args:
            load_fn: Function to load data
            validate: Whether to validate loaded data
            handle_missing: How to handle missing values (forward_fill, drop, interpolate)
        """
        self.load_fn = load_fn
        self.validate = validate
        self.handle_missing = handle_missing
    
    def load(
        self,
        *args,
        **kwargs
    ) -> Dict[str, List]:
        """
        Load data using the configured function.
        
        Args:
            *args: Arguments for load function
            **kwargs: Keyword arguments for load function
        
        Returns:
            Dictionary with loaded data
        """
        data = self.load_fn(*args, **kwargs)
        
        if self.validate:
            from quantml.data.validators import validate_price_data
            is_valid, errors = validate_price_data(
                data.get('prices', []),
                data.get('volumes'),
                data.get('timestamps')
            )
            if not is_valid:
                raise ValueError(f"Data validation failed: {errors}")
        
        # Handle missing values
        if self.handle_missing != "drop":
            data = self._handle_missing(data)
        
        return data
    
    def _handle_missing(self, data: Dict[str, List]) -> Dict[str, List]:
        """Handle missing values in data."""
        if self.handle_missing == "forward_fill":
            return self._forward_fill(data)
        elif self.handle_missing == "interpolate":
            return self._interpolate(data)
        else:
            return data
    
    def _forward_fill(self, data: Dict[str, List]) -> Dict[str, List]:
        """Forward fill missing values."""
        import math
        
        prices = data.get('prices', [])
        volumes = data.get('volumes', [])
        
        # Forward fill prices
        last_valid_price = None
        for i, price in enumerate(prices):
            if price is None or (isinstance(price, float) and math.isnan(price)):
                if last_valid_price is not None:
                    prices[i] = last_valid_price
            else:
                last_valid_price = price
        
        # Forward fill volumes
        if volumes:
            last_valid_volume = None
            for i, volume in enumerate(volumes):
                if volume is None or (isinstance(volume, float) and math.isnan(volume)):
                    if last_valid_volume is not None:
                        volumes[i] = last_valid_volume
                else:
                    last_valid_volume = volume
        
        data['prices'] = prices
        if volumes:
            data['volumes'] = volumes
        
        return data
    
    def _interpolate(self, data: Dict[str, List]) -> Dict[str, List]:
        """Interpolate missing values."""
        if not HAS_PANDAS:
            # Fallback to forward fill
            return self._forward_fill(data)
        
        import math
        
        prices = data.get('prices', [])
        
        # Convert to Series and interpolate
        prices_series = pd.Series(prices)
        prices_series = prices_series.interpolate(method='linear')
        data['prices'] = prices_series.tolist()
        
        if 'volumes' in data:
            volumes_series = pd.Series(data['volumes'])
            volumes_series = volumes_series.interpolate(method='linear')
            data['volumes'] = volumes_series.tolist()
        
        return data

