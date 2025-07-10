# Second-Hand Car Price Prediction - Usage Guide

## Project Overview

This is a comprehensive Python mono repo for training machine learning models on second-hand car price data. The project follows best practices with modern Python tooling and provides three types of models:

- **Classification Models**: Categorize cars into price ranges
- **Regression Models**: Predict exact car prices  
- **Valuation Models**: Provide price estimates with confidence intervals

## Supported Car Brands

- Mercedes
- Ford  
- Fiat
- Toyota
- Tesla
- Audi
- BMW
- Honda

## Project Structure

The project is organized as a mono repo with clear separation of concerns:

```
car_price_prediction/
├── car_price_prediction/      # Main package
│   ├── config/               # Configuration management
│   ├── data/                 # Data handling (scraping, preprocessing, loading)
│   ├── models/               # ML models organized by type
│   │   ├── classification/   # Price category classification
│   │   ├── regression/       # Price prediction
│   │   └── valuation/        # Price valuation with uncertainty
│   ├── features/             # Feature engineering
│   ├── utils/                # Utility functions
│   └── visualization/        # Plotting and visualization
├── configs/                  # Hydra configuration files
├── data/                     # Data storage (git-ignored)
├── outputs/                  # Model outputs and logs (git-ignored)
├── tests/                    # Test suite
└── scripts/                  # Utility scripts
```

## Key Features

### 1. Modern Python Tooling
- **uv**: Fast Python package installer and resolver
- **pyproject.toml**: Modern Python project configuration
- **Type hints**: Full type annotation support
- **Pre-commit hooks**: Code quality enforcement

### 2. Data Pipeline
- **Web scraping**: Automated data collection from car marketplaces
- **Data preprocessing**: Robust cleaning and feature engineering
- **Data validation**: Input validation and error handling

### 3. Machine Learning
- **Multiple model types**: Classification, regression, and valuation
- **Model versioning**: Dated model files with metadata
- **Hyperparameter optimization**: Optuna integration (optional)
- **Cross-validation**: Robust model evaluation

### 4. Configuration Management
- **Hydra integration**: Flexible configuration management
- **Environment variables**: Secure configuration via .env files
- **YAML configs**: Easy-to-modify configuration files

### 5. Logging and Monitoring
- **Structured logging**: Loguru-based logging with rotation
- **MLflow integration**: Experiment tracking (optional)
- **Error handling**: Comprehensive error management

## Model Types Explained

### Classification Models
Transform continuous prices into categories (budget, mid-range, premium, luxury) based on brand-specific thresholds.

**Use cases:**
- Market segmentation
- Inventory categorization
- Quick price range assessment

### Regression Models  
Predict exact car prices using various algorithms (Random Forest, XGBoost, etc.).

**Use cases:**
- Price prediction for individual cars
- Market value estimation
- Pricing strategy development

### Valuation Models
Provide price estimates with confidence intervals using ensemble methods.

**Use cases:**
- Risk assessment in car trading
- Insurance valuations
- Investment decisions with uncertainty quantification

## Command Line Interface

The project provides convenient CLI commands:

```bash
# Data collection
scrape-data --brands mercedes,ford --max-pages 10

# Data preprocessing  
preprocess-data --brand mercedes

# Model training
train-classification --brands mercedes,ford
train-regression --brands mercedes,ford  
train-valuation --brands mercedes,ford
```

## Configuration

The project uses Hydra for configuration management. Config files are stored in `configs/`:

- `classification.yaml`: Classification model settings
- `regression.yaml`: Regression model settings  
- `valuation.yaml`: Valuation model settings

Environment variables can be set in `.env` file.

## Output Structure

All model outputs are organized with timestamps:

```
outputs/
├── models/
│   ├── classification/
│   │   └── mercedes/
│   │       └── mercedes_classification_random_forest_20240710_143022.pkl
│   ├── regression/
│   └── valuation/
├── logs/
└── plots/
```

## Getting Started

1. **Setup**: Run `python setup.py` for automated setup
2. **Configure**: Edit `.env` with your settings
3. **Collect Data**: Use the scraping tools or provide your own data
4. **Train Models**: Use the CLI commands or Python API
5. **Evaluate**: Review outputs and logs for model performance

## Best Practices Implemented

- **Separation of concerns**: Clear module organization
- **Error handling**: Comprehensive exception management
- **Documentation**: Extensive docstrings and type hints
- **Testing**: Test structure provided
- **Version control**: Proper gitignore and structure
- **Dependency management**: Modern Python packaging
- **Code quality**: Linting and formatting tools integrated

This project serves as a template for production-ready ML projects with proper software engineering practices.
