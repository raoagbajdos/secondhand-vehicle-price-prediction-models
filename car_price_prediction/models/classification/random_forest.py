"""
Random Forest classification model for car price prediction.
"""

from typing import Dict

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from ..base import BaseClassificationModel
from ...utils import get_logger

logger = get_logger(__name__)


class RandomForestClassificationModel(BaseClassificationModel):
    """Random Forest model for price category classification."""
    
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
    
    def create_model(self, **kwargs) -> RandomForestClassifier:
        """Create Random Forest classifier."""
        params = {**self.hyperparameters, **kwargs}
        return RandomForestClassifier(**params)
    
    def train(self, x_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> Dict:
        """Train the Random Forest model."""
        logger.info(f"Training Random Forest classifier for {self.brand}")
        
        # Convert continuous prices to categories
        y_train_cat = self.create_price_categories(y_train)
        
        # Create and train model
        self.model = self.create_model(**kwargs)
        self.model.fit(x_train, y_train_cat)
        
        # Calculate training metrics
        train_score = self.model.score(x_train, y_train_cat)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, x_train, y_train_cat, cv=5)
        
        self.training_history = {
            'train_accuracy': train_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': dict(zip(
                x_train.columns, 
                self.model.feature_importances_
            ))
        }
        
        self.is_trained = True
        
        logger.info(f"Training completed. CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        return self.training_history
    
    def predict(self, x_test: pd.DataFrame):
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(x_test)
    
    def predict_proba(self, x_test: pd.DataFrame):
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict_proba(x_test)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        return self.training_history.get('feature_importance', {})
