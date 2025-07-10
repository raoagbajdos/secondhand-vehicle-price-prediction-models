"""
Ensemble valuation model with uncertainty estimation.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

from ..base import BaseValuationModel
from ...utils import get_logger

logger = get_logger(__name__)


class EnsembleValuationModel(BaseValuationModel):
    """Ensemble model for car valuation with uncertainty estimation."""
    
    def __init__(self, brand: str, **kwargs):
        super().__init__(brand, "ensemble_valuation")
        self.base_models = {}
        self.model_weights = {}
        self.hyperparameters = {
            'random_state': 42,
            **kwargs
        }
    
    def create_model(self, **kwargs) -> Dict:
        """Create ensemble of base models."""
        random_state = self.hyperparameters.get('random_state', 42)
        
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=random_state
            ),
            'linear_regression': LinearRegression()
        }
        
        return models
    
    def train(self, x_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> Dict:
        """Train the ensemble model."""
        logger.info(f"Training ensemble valuation model for {self.brand}")
        
        # Create base models
        self.base_models = self.create_model(**kwargs)
        
        # Train each base model and calculate weights based on CV performance
        cv_scores = {}
        
        for name, model in self.base_models.items():
            logger.info(f"Training {name}")
            model.fit(x_train, y_train)
            
            # Calculate cross-validation score
            scores = cross_val_score(model, x_train, y_train, cv=5, scoring='r2')
            cv_scores[name] = scores.mean()
            
            logger.info(f"{name} CV R²: {scores.mean():.4f} ± {scores.std():.4f}")
        
        # Calculate weights based on performance (softmax of CV scores)
        scores_array = np.array(list(cv_scores.values()))
        exp_scores = np.exp(scores_array - np.max(scores_array))  # Numerical stability
        weights = exp_scores / np.sum(exp_scores)
        
        self.model_weights = dict(zip(cv_scores.keys(), weights))
        
        # Calculate ensemble performance
        ensemble_pred = self._ensemble_predict(x_train)
        ensemble_score = self._calculate_r2(y_train, ensemble_pred)
        
        self.training_history = {
            'base_model_scores': cv_scores,
            'model_weights': self.model_weights,
            'ensemble_train_r2': ensemble_score
        }
        
        self.is_trained = True
        
        logger.info(f"Ensemble training completed. Train R²: {ensemble_score:.4f}")
        logger.info(f"Model weights: {self.model_weights}")
        
        return self.training_history
    
    def predict(self, x_test: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self._ensemble_predict(x_test)
    
    def predict_with_uncertainty(self, x_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty estimates."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Get predictions from all base models
        predictions = {}
        for name, model in self.base_models.items():
            predictions[name] = model.predict(x_test)
        
        # Calculate weighted ensemble prediction
        ensemble_pred = self._ensemble_predict(x_test)
        
        # Calculate uncertainty as weighted standard deviation of base model predictions
        pred_matrix = np.array(list(predictions.values())).T  # Shape: (n_samples, n_models)
        weights_array = np.array(list(self.model_weights.values()))
        
        # Weighted variance calculation
        weighted_mean = ensemble_pred
        weighted_var = np.average(
            (pred_matrix - weighted_mean.reshape(-1, 1)) ** 2,
            weights=weights_array,
            axis=1
        )
        uncertainty = np.sqrt(weighted_var)
        
        return ensemble_pred, uncertainty
    
    def _ensemble_predict(self, x: pd.DataFrame) -> np.ndarray:
        """Make weighted ensemble prediction."""
        predictions = []
        weights = []
        
        for name, model in self.base_models.items():
            pred = model.predict(x)
            weight = self.model_weights[name]
            
            predictions.append(pred * weight)
            weights.append(weight)
        
        # Weighted average
        ensemble_pred = np.sum(predictions, axis=0)
        return ensemble_pred
    
    def _calculate_r2(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate R² score."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def get_model_contributions(self, x_test: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get individual model contributions to ensemble prediction."""
        if not self.is_trained:
            raise ValueError("Model must be trained to get contributions")
        
        contributions = {}
        for name, model in self.base_models.items():
            pred = model.predict(x_test)
            weight = self.model_weights[name]
            contributions[name] = pred * weight
        
        return contributions
