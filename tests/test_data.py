"""
Tests for data validation and loading.
"""

import pytest
from quantml.data.validators import (
    validate_price_data,
    check_missing_values,
    generate_data_quality_report
)


def test_validate_price_data_valid():
    """Test validation with valid data."""
    prices = [100.0, 101.0, 102.0, 103.0]
    volumes = [100.0, 110.0, 105.0, 120.0]
    
    is_valid, errors = validate_price_data(prices, volumes)
    
    assert is_valid is True
    assert len(errors) == 0


def test_validate_price_data_nan():
    """Test validation with NaN values."""
    import math
    prices = [100.0, math.nan, 102.0]
    volumes = [100.0, 110.0, 105.0]
    
    is_valid, errors = validate_price_data(prices, volumes)
    
    assert is_valid is False
    assert len(errors) > 0


def test_validate_price_data_negative():
    """Test validation with negative prices."""
    prices = [100.0, -1.0, 102.0]
    volumes = [100.0, 110.0, 105.0]
    
    is_valid, errors = validate_price_data(prices, volumes)
    
    assert is_valid is False
    assert any("negative" in err.lower() or "non-positive" in err.lower() for err in errors)


def test_check_missing_values():
    """Test missing value detection."""
    import math
    prices = [100.0, math.nan, 0.0, 102.0]
    volumes = [100.0, 110.0, 0.0, 105.0]
    
    missing = check_missing_values(prices, volumes)
    
    assert missing['nan_prices'] > 0
    assert missing['zero_prices'] > 0
    assert missing['zero_volumes'] > 0


def test_generate_data_quality_report():
    """Test data quality report generation."""
    prices = [100.0, 101.0, 102.0, 103.0, 104.0]
    volumes = [100.0, 110.0, 105.0, 120.0, 115.0]
    
    report = generate_data_quality_report(prices, volumes)
    
    assert report.total_rows == 5
    assert 0.0 <= report.quality_score <= 1.0
    assert report.nan_price_count == 0
    assert report.nan_volume_count == 0

