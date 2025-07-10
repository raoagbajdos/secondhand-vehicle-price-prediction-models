"""
Random Forest regression model for car price prediction.
"""

from typing import Dict

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from ..base import BaseRegressionModel
from ...utils import get_logger

logger = get_logger(__name__)


class RandomForestRegressionModel(BaseRegressionModel):
    """Random Forest model for price regression."""
    
    def __init__(self, brand: str, **kwargs):
        super().__init__(brand, "random_forest")
        self.hyperparameters = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            **kwargs
        }
    
    def create_model(self, **kwargs) -> RandomForestRegressor:
        """Create Random Forest regressor."""
        params = {**self.hyperparameters, **kwargs}
        return RandomForestRegressor(**params)
    
    def train(self, x_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> Dict:
        """Train the Random Forest model."""
        logger.info(f"Training Random Forest regressor for {self.brand}")
        
        # Create and train model
        self.model = self.create_model(**kwargs)
        self.model.fit(x_train, y_train)
        
        # Calculate training metrics
        train_score = self.model.score(x_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, x_train, y_train, cv=5, scoring='r2')
        
        self.training_history = {
            'train_r2': train_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': dict(zip(
                x_train.columns, 
                self.model.feature_importances_
            ))
        }
        
        self.is_trained = True
        
        logger.info(f"Training completed. CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        return self.training_history
    
    def predict(self, x_test: pd.DataFrame):
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(x_test)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        return self.training_history.get('feature_importance', {})
