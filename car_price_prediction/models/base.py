"""
Base classes for machine learning models.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)

from ...config import settings
from ...utils import get_logger, save_model

logger = get_logger(__name__)


class BaseModel(ABC):
    """Base class for all machine learning models."""
    
    def __init__(self, brand: str, model_name: str):
        self.brand = brand
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.training_history = {}
        
    @abstractmethod
    def create_model(self, **kwargs) -> Any:
        """Create the model instance."""
        pass
    
    @abstractmethod
    def train(self, x_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> Dict:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, x_test: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def evaluate(self, x_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate the model."""
        pass
    
    def save(self, metadata: Optional[Dict] = None) -> Path:
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_path = save_model(
            model=self.model,
            model_type=self.__class__.__module__.split('.')[-2],  # Get model type from module
            brand=self.brand,
            model_name=self.model_name,
            metadata={
                **(metadata or {}),
                **self.training_history,
                'model_class': self.__class__.__name__
            }
        )
        
        logger.info(f"Model saved to {model_path}")
        return model_path


class BaseClassificationModel(BaseModel):
    """Base class for classification models."""
    
    def create_price_categories(self, prices: pd.Series) -> pd.Series:
        """
        Create price categories from continuous prices.
        
        Args:
            prices: Series of car prices
            
        Returns:
            Series of price categories
        """
        # Define price ranges (can be customized per brand)
        if self.brand in ['mercedes', 'audi', 'bmw', 'tesla']:
            # Luxury brands - higher thresholds
            bins = [0, 15000, 35000, 60000, np.inf]
            labels = ['budget', 'mid_range', 'premium', 'luxury']
        else:
            # Mass market brands - lower thresholds  
            bins = [0, 8000, 20000, 40000, np.inf]
            labels = ['budget', 'mid_range', 'premium', 'luxury']
        
        return pd.cut(prices, bins=bins, labels=labels, include_lowest=True)
    
    def evaluate(self, x_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate classification model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Convert continuous prices to categories
        y_test_cat = self.create_price_categories(y_test)
        
        # Make predictions
        y_pred = self.predict(x_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_cat, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'classification_report': classification_report(
                y_test_cat, y_pred, output_dict=True
            ),
            'confusion_matrix': confusion_matrix(y_test_cat, y_pred).tolist()
        }
        
        logger.info(f"Classification accuracy: {accuracy:.4f}")
        return metrics


class BaseRegressionModel(BaseModel):
    """Base class for regression models."""
    
    def evaluate(self, x_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate regression model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        y_pred = self.predict(x_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate percentage error
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
        
        logger.info(f"Regression R²: {r2:.4f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
        return metrics


class BaseValuationModel(BaseRegressionModel):
    """Base class for valuation models with uncertainty estimation."""
    
    @abstractmethod
    def predict_with_uncertainty(self, x_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty estimates."""
        pass
    
    def evaluate(self, x_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate valuation model with uncertainty metrics."""
        # Get base regression metrics
        base_metrics = super().evaluate(x_test, y_test)
        
        # Add uncertainty metrics
        try:
            y_pred, y_std = self.predict_with_uncertainty(x_test)
            
            # Calculate prediction intervals (95% confidence)
            y_lower = y_pred - 1.96 * y_std
            y_upper = y_pred + 1.96 * y_std
            
            # Coverage probability (what percentage of true values fall within prediction intervals)
            coverage = np.mean((y_test >= y_lower) & (y_test <= y_upper))
            
            # Average prediction interval width
            avg_interval_width = np.mean(y_upper - y_lower)
            
            uncertainty_metrics = {
                'coverage_probability': coverage,
                'avg_interval_width': avg_interval_width,
                'avg_uncertainty': np.mean(y_std)
            }
            
            base_metrics.update(uncertainty_metrics)
            logger.info(f"Coverage probability: {coverage:.4f}")
            
        except Exception as e:
            logger.warning(f"Could not calculate uncertainty metrics: {e}")
        
        return base_metrics
