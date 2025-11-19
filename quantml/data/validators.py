"""
Data validation and integrity checks for market data.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import math

# Try to import pandas
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None


@dataclass
class DataQualityReport:
    """Data quality report."""
    total_rows: int
    missing_timestamps: int
    duplicate_timestamps: int
    zero_volume_count: int
    nan_price_count: int
    nan_volume_count: int
    negative_price_count: int
    negative_volume_count: int
    gaps_detected: List[Tuple[datetime, datetime]]
    quality_score: float  # 0.0 to 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_rows': self.total_rows,
            'missing_timestamps': self.missing_timestamps,
            'duplicate_timestamps': self.duplicate_timestamps,
            'zero_volume_count': self.zero_volume_count,
            'nan_price_count': self.nan_price_count,
            'nan_volume_count': self.nan_volume_count,
            'negative_price_count': self.negative_price_count,
            'negative_volume_count': self.negative_volume_count,
            'gaps_detected': len(self.gaps_detected),
            'quality_score': self.quality_score
        }


def validate_timestamps(
    timestamps: List,
    expected_frequency: Optional[str] = None,
    allow_gaps: bool = True
) -> Tuple[bool, List[Tuple[Any, Any]]]:
    """
    Validate timestamp sequence.
    
    Args:
        timestamps: List of timestamps
        expected_frequency: Expected frequency (e.g., '1min', '5min', '1H')
        allow_gaps: Whether gaps are allowed (e.g., weekends, holidays)
    
    Returns:
        Tuple of (is_valid, list of gaps)
    """
    if len(timestamps) < 2:
        return True, []
    
    gaps = []
    
    # Convert to datetime if needed
    if HAS_PANDAS:
        try:
            timestamps = pd.to_datetime(timestamps)
        except Exception:
            pass
    
    # Check for duplicates
    seen = set()
    for ts in timestamps:
        if ts in seen:
            return False, [(ts, ts)]  # Duplicate
        seen.add(ts)
    
    # Sort timestamps
    sorted_ts = sorted(timestamps)
    
    # Detect gaps if frequency specified
    if expected_frequency and not allow_gaps:
        # Parse frequency
        if expected_frequency.endswith('min'):
            minutes = int(expected_frequency[:-3])
            delta = timedelta(minutes=minutes)
        elif expected_frequency.endswith('H'):
            hours = int(expected_frequency[:-1])
            delta = timedelta(hours=hours)
        elif expected_frequency == '1D':
            delta = timedelta(days=1)
        else:
            delta = None
        
        if delta:
            for i in range(len(sorted_ts) - 1):
                expected_next = sorted_ts[i] + delta
                if sorted_ts[i + 1] != expected_next:
                    gaps.append((sorted_ts[i], sorted_ts[i + 1]))
    
    return len(gaps) == 0, gaps


def check_duplicates(data: List, key: Optional[str] = None) -> List[int]:
    """
    Check for duplicate values.
    
    Args:
        data: List of values or list of dictionaries
        key: Key to check if data is list of dicts
    
    Returns:
        List of indices with duplicates
    """
    if key:
        values = [item[key] for item in data if isinstance(item, dict)]
    else:
        values = data
    
    seen = {}
    duplicates = []
    
    for i, val in enumerate(values):
        if val in seen:
            duplicates.append(i)
            if seen[val] not in duplicates:
                duplicates.append(seen[val])
        else:
            seen[val] = i
    
    return duplicates


def check_missing_values(
    prices: List[float],
    volumes: Optional[List[float]] = None
) -> Dict[str, int]:
    """
    Check for missing values (NaN, None, zero).
    
    Args:
        prices: List of prices
        volumes: Optional list of volumes
    
    Returns:
        Dictionary with counts of missing values
    """
    results = {
        'nan_prices': 0,
        'zero_prices': 0,
        'negative_prices': 0,
        'nan_volumes': 0,
        'zero_volumes': 0,
        'negative_volumes': 0
    }
    
    for price in prices:
        if price is None or (isinstance(price, float) and math.isnan(price)):
            results['nan_prices'] += 1
        elif price == 0:
            results['zero_prices'] += 1
        elif price < 0:
            results['negative_prices'] += 1
    
    if volumes:
        for volume in volumes:
            if volume is None or (isinstance(volume, float) and math.isnan(volume)):
                results['nan_volumes'] += 1
            elif volume == 0:
                results['zero_volumes'] += 1
            elif volume < 0:
                results['negative_volumes'] += 1
    
    return results


def validate_price_data(
    prices: List[float],
    volumes: Optional[List[float]] = None,
    timestamps: Optional[List] = None,
    min_price: float = 0.01,
    max_price: float = 1e6
) -> Tuple[bool, List[str]]:
    """
    Validate price data for common issues.
    
    Args:
        prices: List of prices
        volumes: Optional list of volumes
        timestamps: Optional list of timestamps
        min_price: Minimum valid price
        max_price: Maximum valid price
    
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    
    if not prices:
        errors.append("Empty price data")
        return False, errors
    
    # Check prices
    for i, price in enumerate(prices):
        if price is None or (isinstance(price, float) and math.isnan(price)):
            errors.append(f"NaN price at index {i}")
        elif price <= 0:
            errors.append(f"Non-positive price at index {i}: {price}")
        elif price < min_price:
            errors.append(f"Price too low at index {i}: {price}")
        elif price > max_price:
            errors.append(f"Price too high at index {i}: {price}")
    
    # Check volumes if provided
    if volumes:
        if len(volumes) != len(prices):
            errors.append(f"Volume length ({len(volumes)}) != price length ({len(prices)})")
        else:
            for i, volume in enumerate(volumes):
                if volume is None or (isinstance(volume, float) and math.isnan(volume)):
                    errors.append(f"NaN volume at index {i}")
                elif volume < 0:
                    errors.append(f"Negative volume at index {i}: {volume}")
    
    # Check timestamps if provided
    if timestamps:
        if len(timestamps) != len(prices):
            errors.append(f"Timestamp length ({len(timestamps)}) != price length ({len(prices)})")
        else:
            is_valid, gaps = validate_timestamps(timestamps)
            if not is_valid:
                errors.append(f"Invalid timestamps: {len(gaps)} gaps detected")
    
    return len(errors) == 0, errors


def generate_data_quality_report(
    prices: List[float],
    volumes: Optional[List[float]] = None,
    timestamps: Optional[List] = None,
    expected_frequency: Optional[str] = None
) -> DataQualityReport:
    """
    Generate comprehensive data quality report.
    
    Args:
        prices: List of prices
        volumes: Optional list of volumes
        timestamps: Optional list of timestamps
        expected_frequency: Expected data frequency
    
    Returns:
        DataQualityReport object
    """
    total_rows = len(prices)
    
    # Check missing values
    missing = check_missing_values(prices, volumes)
    
    # Check timestamps
    missing_ts = 0
    duplicate_ts = 0
    gaps = []
    
    if timestamps:
        is_valid, gaps = validate_timestamps(timestamps, expected_frequency)
        if not is_valid:
            # Count issues
            seen = set()
            for ts in timestamps:
                if ts is None:
                    missing_ts += 1
                elif ts in seen:
                    duplicate_ts += 1
                else:
                    seen.add(ts)
    
    # Calculate quality score
    issues = (
        missing['nan_prices'] +
        missing['zero_prices'] +
        missing['negative_prices'] +
        missing['nan_volumes'] +
        missing['zero_volumes'] +
        missing['negative_volumes'] +
        missing_ts +
        duplicate_ts +
        len(gaps)
    )
    
    quality_score = max(0.0, 1.0 - (issues / max(total_rows, 1)))
    
    return DataQualityReport(
        total_rows=total_rows,
        missing_timestamps=missing_ts,
        duplicate_timestamps=duplicate_ts,
        zero_volume_count=missing['zero_volumes'],
        nan_price_count=missing['nan_prices'],
        nan_volume_count=missing['nan_volumes'],
        negative_price_count=missing['negative_prices'],
        negative_volume_count=missing['negative_volumes'],
        gaps_detected=gaps,
        quality_score=quality_score
    )


def validate_futures_data(
    prices: List[float],
    opens: Optional[List[float]] = None,
    closes: Optional[List[float]] = None,
    volumes: Optional[List[float]] = None,
    timestamps: Optional[List] = None,
    instrument: str = "ES"
) -> Tuple[bool, List[str]]:
    """
    Validate futures-specific data issues.
    
    Args:
        prices: Price data
        opens: Opening prices (for gap detection)
        closes: Closing prices (for gap detection)
        volumes: Volume data
        timestamps: Timestamp data
        instrument: Instrument symbol
    
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    
    # Basic price validation
    is_valid, price_errors = validate_price_data(prices, volumes, timestamps)
    if not is_valid:
        errors.extend(price_errors)
    
    # Check for holiday gaps
    if opens and closes and len(opens) > 1 and len(closes) > 1:
        for i in range(1, min(len(opens), len(closes))):
            if closes[i-1] > 0:
                gap = abs((opens[i] - closes[i-1]) / closes[i-1])
                # Large gaps might indicate holiday or data issue
                if gap > 0.1:  # 10% gap
                    if timestamps and i < len(timestamps) and i-1 < len(timestamps):
                        days_diff = (timestamps[i] - timestamps[i-1]).days
                        if days_diff <= 1:
                            errors.append(f"Large gap at index {i} ({gap*100:.2f}%) without holiday")
    
    # Check for roll dates (volume drops)
    if volumes and len(volumes) > 1:
        for i in range(1, len(volumes)):
            if volumes[i] < volumes[i-1] * 0.2:  # 80% drop
                errors.append(f"Potential contract roll at index {i} (volume drop)")
    
    return len(errors) == 0, errors


def validate_roll_dates(
    roll_indices: List[int],
    data_length: int
) -> Tuple[bool, List[str]]:
    """
    Validate contract roll dates.
    
    Args:
        roll_indices: List of roll indices
        data_length: Total data length
    
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    
    for idx in roll_indices:
        if idx < 0 or idx >= data_length:
            errors.append(f"Invalid roll index: {idx} (data length: {data_length})")
    
    # Check for too frequent rolls
    if len(roll_indices) > 1:
        for i in range(1, len(roll_indices)):
            if roll_indices[i] - roll_indices[i-1] < 10:
                errors.append(f"Rolls too frequent: {roll_indices[i-1]} -> {roll_indices[i]}")
    
    return len(errors) == 0, errors

