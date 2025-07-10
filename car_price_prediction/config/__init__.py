"""
Configuration settings for the car price prediction project.
"""

import os
from pathlib import Path
from typing import List, Optional

try:
    from pydantic import BaseSettings, validator
except ImportError:
    # Fallback for when pydantic is not installed
    class BaseSettings:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    def validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


class Settings(BaseSettings):
    """Application settings."""
    
    # Project paths
    project_root: Path = Path(__file__).parent.parent.parent
    data_dir: Path = project_root / "data"
    output_dir: Path = project_root / "outputs"
    config_dir: Path = project_root / "configs"
    
    # Data directories
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"
    external_data_dir: Path = data_dir / "external"
    
    # Output directories
    models_dir: Path = output_dir / "models"
    logs_dir: Path = output_dir / "logs"
    plots_dir: Path = output_dir / "plots"
    
    # Supported brands
    supported_brands: List[str] = [
        "mercedes", "ford", "fiat", "toyota", 
        "tesla", "audi", "bmw", "honda"
    ]
    
    # Model settings
    random_seed: int = 42
    test_size: float = 0.2
    validation_size: float = 0.1
    
    # Scraping settings
    scraping_delay: float = 1.0
    max_pages_per_brand: int = 100
    request_timeout: int = 30
    
    # Model training settings
    n_trials: int = 100  # for Optuna optimization
    cv_folds: int = 5
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}"
    
    # MLflow
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment_name: str = "car_price_prediction"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create all necessary directories
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Path) and attr_name.endswith('_dir'):
                attr.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
