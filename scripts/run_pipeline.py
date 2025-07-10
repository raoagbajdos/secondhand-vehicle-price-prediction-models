#!/usr/bin/env python3
"""
Example script to run the complete pipeline.
"""

import logging
from pathlib import Path

from car_price_prediction.data import scrape_brand_data, preprocess_data
from car_price_prediction.models.classification.train import train_multiple_brands as train_classification
from car_price_prediction.models.regression.train import train_multiple_brands as train_regression
from car_price_prediction.models.valuation.train import train_multiple_brands as train_valuation
from car_price_prediction.config import settings
from car_price_prediction.utils import get_logger

logger = get_logger(__name__)

def main():
    """Run the complete pipeline."""
    brands = ['mercedes', 'ford', 'toyota']  # Example subset
    
    logger.info("Starting complete pipeline")
    
    # Step 1: Scrape data
    logger.info("Step 1: Scraping data")
    try:
        scrape_results = scrape_brand_data(brands, max_pages=5)  # Limited for example
        logger.info(f"Scraped data for {len(scrape_results)} brands")
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        return
    
    # Step 2: Preprocess data
    logger.info("Step 2: Preprocessing data")
    try:
        for brand in brands:
            preprocess_data(brand)
        logger.info("Data preprocessing completed")
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return
    
    # Step 3: Train models
    logger.info("Step 3: Training models")
    
    # Train classification models
    logger.info("Training classification models")
    classification_results = train_classification(brands)
    
    # Train regression models
    logger.info("Training regression models")
    regression_results = train_regression(brands)
    
    # Train valuation models
    logger.info("Training valuation models")
    valuation_results = train_valuation(brands)
    
    # Summary
    logger.info("Pipeline completed!")
    logger.info("Results summary:")
    
    for brand in brands:
        logger.info(f"\n{brand.upper()}:")
        
        if brand in classification_results and 'error' not in classification_results[brand]:
            acc = classification_results[brand]['evaluation_results'].get('accuracy', 0)
            logger.info(f"  Classification accuracy: {acc:.4f}")
        
        if brand in regression_results and 'error' not in regression_results[brand]:
            r2 = regression_results[brand]['evaluation_results'].get('r2', 0)
            logger.info(f"  Regression R²: {r2:.4f}")
        
        if brand in valuation_results and 'error' not in valuation_results[brand]:
            r2 = valuation_results[brand]['evaluation_results'].get('r2', 0)
            coverage = valuation_results[brand]['evaluation_results'].get('coverage_probability', 0)
            logger.info(f"  Valuation R²: {r2:.4f}, Coverage: {coverage:.4f}")

if __name__ == "__main__":
    main()
