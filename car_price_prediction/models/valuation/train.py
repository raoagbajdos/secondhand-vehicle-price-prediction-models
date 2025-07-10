"""
Training script for valuation models.
"""

import argparse
from typing import Dict, List, Optional

from ...config import settings
from ...data import DataLoader
from ...utils import get_logger, validate_brand, validate_brands
from .ensemble_valuation import EnsembleValuationModel

logger = get_logger(__name__)

# Available valuation models
VALUATION_MODELS = {
    'ensemble': EnsembleValuationModel,
    # Add more models as implemented
}


def train_valuation_model(
    brand: str,
    model_name: str = 'ensemble',
    hyperparameters: Optional[Dict] = None,
    save_model: bool = True
) -> Dict:
    """
    Train a valuation model for a specific brand.
    
    Args:
        brand: Car brand to train on
        model_name: Name of the model algorithm
        hyperparameters: Model hyperparameters
        save_model: Whether to save the trained model
        
    Returns:
        Training results dictionary
    """
    validate_brand(brand)
    
    if model_name not in VALUATION_MODELS:
        raise ValueError(f"Model '{model_name}' not available. Choose from: {list(VALUATION_MODELS.keys())}")
    
    logger.info(f"Starting valuation training for {brand} using {model_name}")
    
    # Load data
    data_loader = DataLoader()
    x_train, x_test, y_train, y_test = data_loader.load_for_brand_training(brand)
    
    logger.info(f"Loaded data - Train: {x_train.shape}, Test: {x_test.shape}")
    
    # Initialize model
    model_class = VALUATION_MODELS[model_name]
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
    
    logger.info(f"Valuation training completed for {brand}")
    logger.info(f"R²: {evaluation_results.get('r2', 'N/A'):.4f}")
    logger.info(f"Coverage: {evaluation_results.get('coverage_probability', 'N/A'):.4f}")
    
    return results


def train_multiple_brands(
    brands: List[str],
    model_name: str = 'ensemble',
    hyperparameters: Optional[Dict] = None
) -> Dict[str, Dict]:
    """
    Train valuation models for multiple brands.
    
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
            brand_results = train_valuation_model(
                brand=brand,
                model_name=model_name,
                hyperparameters=hyperparameters
            )
            results[brand] = brand_results
            
        except Exception as e:
            logger.error(f"Failed to train {model_name} for {brand}: {e}")
            results[brand] = {'error': str(e)}
    
    return results


def main():
    """CLI entry point for valuation training."""
    parser = argparse.ArgumentParser(description="Train valuation models")
    parser.add_argument(
        "--brands",
        type=str,
        default=",".join(settings.supported_brands),
        help="Comma-separated list of brands to train"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ensemble",
        choices=list(VALUATION_MODELS.keys()),
        help="Model algorithm to use"
    )
    
    args = parser.parse_args()
    
    brands = [b.strip() for b in args.brands.split(",")]
    
    results = train_multiple_brands(
        brands=brands,
        model_name=args.model
    )
    
    # Print summary
    for brand, result in results.items():
        if 'error' not in result:
            r2 = result['evaluation_results'].get('r2', 0)
            coverage = result['evaluation_results'].get('coverage_probability', 0)
            print(f"{brand}: R² = {r2:.4f}, Coverage = {coverage:.4f}")
        else:
            print(f"{brand}: Failed - {result['error']}")


if __name__ == "__main__":
    main()
