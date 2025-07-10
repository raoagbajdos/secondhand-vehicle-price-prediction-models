"""
Second-Hand Car Price Prediction Package

A comprehensive ML pipeline for predicting second-hand car prices using web-scraped data.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .config import settings
from .utils.logger import get_logger

logger = get_logger(__name__)

# Supported car brands
SUPPORTED_BRANDS = [
    "mercedes",
    "ford", 
    "fiat",
    "toyota",
    "tesla",
    "audi",
    "bmw",
    "honda"
]

# Model types
MODEL_TYPES = [
    "classification",
    "regression", 
    "valuation"
]
