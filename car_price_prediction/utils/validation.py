"""
Validation utilities for input parameters.
"""

from typing import List

from .. import SUPPORTED_BRANDS, MODEL_TYPES


def validate_brand(brand: str) -> bool:
    """
    Validate if a brand is supported.
    
    Args:
        brand: Car brand name
        
    Returns:
        True if brand is supported
        
    Raises:
        ValueError: If brand is not supported
    """
    if brand.lower() not in SUPPORTED_BRANDS:
        raise ValueError(
            f"Brand '{brand}' not supported. "
            f"Supported brands: {', '.join(SUPPORTED_BRANDS)}"
        )
    return True


def validate_model_type(model_type: str) -> bool:
    """
    Validate if a model type is supported.
    
    Args:
        model_type: Model type name
        
    Returns:
        True if model type is supported
        
    Raises:
        ValueError: If model type is not supported
    """
    if model_type.lower() not in MODEL_TYPES:
        raise ValueError(
            f"Model type '{model_type}' not supported. "
            f"Supported types: {', '.join(MODEL_TYPES)}"
        )
    return True


def validate_brands(brands: List[str]) -> bool:
    """
    Validate a list of brands.
    
    Args:
        brands: List of brand names
        
    Returns:
        True if all brands are supported
    """
    for brand in brands:
        validate_brand(brand)
    return True
