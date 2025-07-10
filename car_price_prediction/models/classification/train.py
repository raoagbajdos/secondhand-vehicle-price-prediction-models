"""
Training script for classification models.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import hydra
from omegaconf import DictConfig

from ...config import settings
from ...data import DataLoader
from ...utils import get_logger, validate_brand, validate_brands
from . import RandomForestClassificationModel

logger = get_logger(__name__)

# Available classification models
CLASSIFICATION_MODELS = {
    'random_forest': RandomForestClassificationModel,
    # Add more models as implemented
}


def train_classification_model(
    brand: str,
    model_name: str = 'random_forest',
    hyperparameters: Optional[Dict] = None,
    save_model: bool = True
) -> Dict:
    """
    Train a classification model for a specific brand.
    
    Args:
        brand: Car brand to train on
        model_name: Name of the model algorithm
        hyperparameters: Model hyperparameters
        save_model: Whether to save the trained model
        
    Returns:
        Training results dictionary
    """
    validate_brand(brand)
    
    if model_name not in CLASSIFICATION_MODELS:
        raise ValueError(f"Model '{model_name}' not available. Choose from: {list(CLASSIFICATION_MODELS.keys())}")
    
    logger.info(f"Starting classification training for {brand} using {model_name}")
    
    # Load data
    data_loader = DataLoader()
    x_train, x_test, y_train, y_test = data_loader.load_for_brand_training(brand)
    
    logger.info(f"Loaded data - Train: {x_train.shape}, Test: {x_test.shape}")
    
    # Initialize model
    model_class = CLASSIFICATION_MODELS[model_name]
    model = model_class(brand, **(hyperparameters or {}))
    
    # Train model
    training_results = model.train(x_train, y_train)
    
    # Evaluate model
    evaluation_results = model.evaluate(x_test, y_test)
    
    # Combine results
    results = {
        'brand': brand,
        'model_name': model_name,
        'training_results': training_results,
        'evaluation_results': evaluation_results,
        'data_info': {
            'train_samples': len(x_train),
            'test_samples': len(x_test),
            'features': list(x_train.columns)
        }
    }
    
    # Save model if requested
    if save_model:
        model_path = model.save(metadata=results)
        results['model_path'] = str(model_path)
    
    logger.info(f"Classification training completed for {brand}")
    logger.info(f"Accuracy: {evaluation_results.get('accuracy', 'N/A'):.4f}")
    
    return results


def train_multiple_brands(
    brands: List[str],
    model_name: str = 'random_forest',
    hyperparameters: Optional[Dict] = None
) -> Dict[str, Dict]:
    """
    Train classification models for multiple brands.
    
    Args:
        brands: List of car brands
        model_name: Name of the model algorithm
        hyperparameters: Model hyperparameters
        
    Returns:
        Dictionary mapping brands to their training results
    """
    validate_brands(brands)
    
    results = {}
    
    for brand in brands:
        try:
            brand_results = train_classification_model(
                brand=brand,
                model_name=model_name,
                hyperparameters=hyperparameters
            )
            results[brand] = brand_results
            
        except Exception as e:
            logger.error(f"Failed to train {model_name} for {brand}: {e}")
            results[brand] = {'error': str(e)}
    
    return results


@hydra.main(version_base=None, config_path="../../../configs", config_name="classification")
def train_with_config(cfg: DictConfig) -> None:
    """
    Train classification models using Hydra configuration.
    
    Args:
        cfg: Hydra configuration
    """
    brands = cfg.get('brands', settings.supported_brands)
    model_name = cfg.get('model_name', 'random_forest')
    hyperparameters = cfg.get('hyperparameters', {})
    
    if isinstance(brands, str):
        brands = [brands]
    
    logger.info(f"Training {model_name} classification models for brands: {brands}")
    
    results = train_multiple_brands(
        brands=brands,
        model_name=model_name,
        hyperparameters=hyperparameters
    )
    
    # Log summary
    successful = sum(1 for r in results.values() if 'error' not in r)
    total = len(results)
    
    logger.info(f"Training completed: {successful}/{total} models trained successfully")
    
    for brand, result in results.items():
        if 'error' not in result:
            accuracy = result['evaluation_results'].get('accuracy', 0)
            logger.info(f"{brand}: Accuracy = {accuracy:.4f}")
        else:
            logger.error(f"{brand}: Failed - {result['error']}")


def main():
    """CLI entry point for classification training."""
    parser = argparse.ArgumentParser(description="Train classification models")
    parser.add_argument(
        "--brands",
        type=str,
        default=",".join(settings.supported_brands),
        help="Comma-separated list of brands to train"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="random_forest",
        choices=list(CLASSIFICATION_MODELS.keys()),
        help="Model algorithm to use"
    )
    parser.add_argument(
        "--config-name",
        type=str,
        help="Hydra config name to use (overrides other arguments)"
    )
    
    args = parser.parse_args()
    
    if args.config_name:
        # Use Hydra configuration
        train_with_config()
    else:
        # Use command line arguments
        brands = [b.strip() for b in args.brands.split(",")]
        
        results = train_multiple_brands(
            brands=brands,
            model_name=args.model
        )
        
        # Print summary
        for brand, result in results.items():
            if 'error' not in result:
                accuracy = result['evaluation_results'].get('accuracy', 0)
                print(f"{brand}: Accuracy = {accuracy:.4f}")
            else:
                print(f"{brand}: Failed - {result['error']}")


if __name__ == "__main__":
    main()
