"""
Pytest configuration and fixtures.
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from quantml import Tensor
from quantml.utils.reproducibility import set_random_seed


@pytest.fixture(autouse=True)
def set_seed():
    """Set random seed for all tests."""
    set_random_seed(42)


@pytest.fixture
def sample_prices():
    """Sample price data for testing."""
    return [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.0, 103.0, 102.0, 101.0]


@pytest.fixture
def sample_volumes():
    """Sample volume data for testing."""
    return [100.0, 110.0, 105.0, 120.0, 115.0, 125.0, 130.0, 120.0, 110.0, 100.0]


@pytest.fixture
def sample_tensor():
    """Sample tensor for testing."""
    return Tensor([1.0, 2.0, 3.0], requires_grad=True)

