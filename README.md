# Second-Hand Car Price Prediction

A comprehensive machine learning pipeline for predicting second-hand car prices using web-scraped data from multiple car brands.

## Features

- **Multi-brand Support**: Mercedes, Ford, Fiat, Toyota, Tesla, Audi, BMW, and Honda
- **Multiple Model Types**: Classification, Regression, and Valuation models
- **Web Scraping Pipeline**: Automated data collection from various car marketplace websites
- **Best Practices**: Modern Python tooling with uv, pyproject.toml, and structured logging
- **Model Versioning**: Dated model outputs with MLflow tracking
- **Hyperparameter Optimization**: Optuna integration for model tuning

## Project Structure

```
car_price_prediction/
├── pyproject.toml              # Project configuration and dependencies
├── README.md                   # This file
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore rules
├── .pre-commit-config.yaml    # Pre-commit hooks configuration
├── car_price_prediction/      # Main package
│   ├── __init__.py
│   ├── config/                # Configuration files
│   ├── data/                  # Data handling modules
│   ├── models/                # Model implementations
│   │   ├── classification/    # Classification models
│   │   ├── regression/        # Regression models
│   │   └── valuation/         # Valuation models
│   ├── features/              # Feature engineering
│   ├── utils/                 # Utility functions
│   └── visualization/         # Plotting and visualization
├── configs/                   # Hydra configuration files
├── data/                      # Data storage
│   ├── raw/                   # Raw scraped data
│   ├── processed/             # Processed data
│   └── external/              # External data sources
├── outputs/                   # Model outputs and artifacts
│   ├── models/                # Trained models (.pkl files)
│   ├── logs/                  # Training logs
│   └── plots/                 # Generated plots
├── tests/                     # Test suite
└── scripts/                   # Utility scripts
```

## Quick Start

### 1. Setup Project

```bash
# Clone/download the project
cd second-hand-car-price-prediction

# Run setup script
python setup.py
```

### 2. Manual Setup (Alternative)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

### 3. Setup Environment

```bash
cp .env.example .env
# Edit .env with your configuration
```

### 4. Run Complete Pipeline

```bash
# Option 1: Use the example script
python scripts/run_pipeline.py

# Option 2: Run individual steps
scrape-data --brands mercedes,ford,toyota --max-pages 5
preprocess-data --brand mercedes
train-regression --brands mercedes
train-classification --brands mercedes
train-valuation --brands mercedes
```

## Model Types

### Classification Models
- **Purpose**: Classify cars into price categories (budget, mid-range, luxury)
- **Algorithms**: Random Forest, XGBoost, LightGBM
- **Output**: Price category predictions

### Regression Models
- **Purpose**: Predict exact car prices
- **Algorithms**: Linear Regression, Random Forest, Gradient Boosting
- **Output**: Continuous price predictions

### Valuation Models
- **Purpose**: Comprehensive car valuation with uncertainty estimation
- **Algorithms**: Ensemble methods, Bayesian models
- **Output**: Price estimates with confidence intervals

## Development

### Code Quality

```bash
# Format code
black car_price_prediction/
isort car_price_prediction/

# Lint code
flake8 car_price_prediction/
mypy car_price_prediction/

# Run tests
pytest
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

## License

MIT License - see LICENSE file for details.
