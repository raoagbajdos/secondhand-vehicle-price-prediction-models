"""
Test configuration and fixtures.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

@pytest.fixture
def sample_car_data():
    """Sample car data for testing."""
    return pd.DataFrame({
        'brand': ['mercedes', 'ford', 'toyota'],
        'title': ['Mercedes C-Class 2020', 'Ford Focus 2019', 'Toyota Corolla 2018'],
        'price': [25000, 12000, 15000],
        'mileage': [30000, 45000, 60000],
        'year': [2020, 2019, 2018],
        'source': ['autotrader', 'autotrader', 'autotrader'],
        'scraped_at': ['2024-01-01T10:00:00', '2024-01-01T10:01:00', '2024-01-01T10:02:00']
    })

@pytest.fixture
def temp_directory():
    """Temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def mock_settings(temp_directory):
    """Mock settings with temporary directories."""
    mock_settings = MagicMock()
    mock_settings.raw_data_dir = temp_directory / "raw"
    mock_settings.processed_data_dir = temp_directory / "processed"
    mock_settings.models_dir = temp_directory / "models"
    mock_settings.supported_brands = ['mercedes', 'ford', 'toyota', 'audi', 'bmw']
    mock_settings.random_seed = 42
    mock_settings.test_size = 0.2
    
    # Create directories
    mock_settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
    mock_settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
    mock_settings.models_dir.mkdir(parents=True, exist_ok=True)
    
    return mock_settings
