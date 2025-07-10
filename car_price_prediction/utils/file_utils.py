"""
File utility functions for model saving/loading and file management.
"""

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..config import settings


def get_dated_filename(base_name: str, extension: str = ".pkl") -> str:
    """
    Generate a dated filename.
    
    Args:
        base_name: Base name for the file
        extension: File extension
        
    Returns:
        Dated filename string
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}{extension}"


def save_model(
    model: Any, 
    model_type: str, 
    brand: str, 
    model_name: str,
    metadata: Optional[dict] = None
) -> Path:
    """
    Save a trained model with metadata.
    
    Args:
        model: Trained model object
        model_type: Type of model (classification, regression, valuation)
        brand: Car brand
        model_name: Name of the model algorithm
        metadata: Optional metadata dictionary
        
    Returns:
        Path to saved model file
    """
    # Create model directory structure
    model_dir = settings.models_dir / model_type / brand
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate dated filename
    base_name = f"{brand}_{model_type}_{model_name}"
    filename = get_dated_filename(base_name)
    model_path = model_dir / filename
    
    # Prepare data to save
    save_data = {
        "model": model,
        "metadata": metadata or {},
        "model_type": model_type,
        "brand": brand,
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "version": "0.1.0"
    }
    
    # Save model
    with open(model_path, "wb") as f:
        pickle.dump(save_data, f)
    
    return model_path


def load_model(model_path: Path) -> dict:
    """
    Load a saved model.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Dictionary containing model and metadata
    """
    with open(model_path, "rb") as f:
        return pickle.load(f)


def get_latest_model(model_type: str, brand: str, model_name: str) -> Optional[Path]:
    """
    Get the latest model file for given parameters.
    
    Args:
        model_type: Type of model
        brand: Car brand  
        model_name: Model algorithm name
        
    Returns:
        Path to latest model file or None if not found
    """
    model_dir = settings.models_dir / model_type / brand
    if not model_dir.exists():
        return None
    
    pattern = f"{brand}_{model_type}_{model_name}_*.pkl"
    model_files = list(model_dir.glob(pattern))
    
    if not model_files:
        return None
    
    # Sort by modification time and return latest
    return max(model_files, key=lambda x: x.stat().st_mtime)
