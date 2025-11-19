"""
Tests for futures-specific data handling.
"""

import pytest
from quantml.data.futures import FuturesDataHandler
from quantml.data.validators import validate_futures_data, validate_roll_dates
from datetime import datetime


def test_contract_roll_detection():
    """Test contract roll detection."""
    handler = FuturesDataHandler("ES", roll_method="volume")
    
    prices = [100.0] * 100
    volumes = [1000.0] * 50 + [100.0] * 50  # Volume drop at index 50
    
    roll_indices = handler.detect_contract_rolls(prices, volumes)
    
    assert len(roll_indices) > 0, "Should detect at least one roll"
    assert 50 in roll_indices or any(abs(idx - 50) < 5 for idx in roll_indices)


def test_overnight_gap_detection():
    """Test overnight gap detection."""
    handler = FuturesDataHandler("ES")
    
    closes = [100.0, 101.0, 102.0, 103.0]
    opens = [100.0, 102.0, 101.0, 105.0]  # Gaps at indices 1, 2, 3
    
    gaps = handler.detect_overnight_gaps(closes, opens)
    
    assert len(gaps) == 3, "Should detect 3 gaps"
    assert gaps[0]['index'] == 1
    assert abs(gaps[0]['gap'] - 0.01) < 0.001  # 1% gap


def test_futures_data_validation():
    """Test futures-specific data validation."""
    prices = [100.0, 101.0, 102.0, 103.0]
    opens = [100.0, 102.0, 101.0, 103.0]
    closes = [100.0, 101.0, 102.0, 103.0]
    volumes = [1000.0, 1100.0, 1050.0, 1200.0]
    
    is_valid, errors = validate_futures_data(
        prices,
        opens=opens,
        closes=closes,
        volumes=volumes,
        instrument="ES"
    )
    
    # Should be valid (small gaps)
    assert is_valid or len(errors) == 0


def test_roll_date_validation():
    """Test roll date validation."""
    data_length = 100
    
    # Valid rolls
    valid_rolls = [20, 50, 80]
    is_valid, errors = validate_roll_dates(valid_rolls, data_length)
    assert is_valid, "Valid rolls should pass"
    
    # Invalid rolls (out of bounds)
    invalid_rolls = [20, 50, 150]
    is_valid, errors = validate_roll_dates(invalid_rolls, data_length)
    assert not is_valid, "Out of bounds rolls should fail"
    
    # Too frequent rolls
    frequent_rolls = [20, 25, 30]
    is_valid, errors = validate_roll_dates(frequent_rolls, data_length)
    assert not is_valid, "Too frequent rolls should fail"


def test_session_filtering():
    """Test session-based filtering."""
    handler = FuturesDataHandler("ES")
    
    data = {
        'price': [100.0, 101.0, 102.0, 103.0],
        'volume': [1000.0, 1100.0, 1050.0, 1200.0]
    }
    
    # Without timestamps, should return data as-is
    filtered = handler.filter_session(data, session_type="RTH")
    assert len(filtered['price']) == len(data['price'])


def test_missing_data_handling():
    """Test missing data handling."""
    handler = FuturesDataHandler("ES")
    
    data = {
        'price': [100.0, None, 102.0, 0.0, 104.0],
        'volume': [1000.0, 1100.0, None, 1200.0, 1300.0]
    }
    
    cleaned = handler.handle_missing_data(data, method="forward_fill")
    
    # Should handle missing values
    assert all(p is not None and p > 0 for p in cleaned['price'])

