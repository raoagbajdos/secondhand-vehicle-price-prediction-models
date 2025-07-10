# GitHub Actions Pipeline Documentation

## Overview

This project includes a comprehensive GitHub Actions pipeline that automatically trains machine learning models on a monthly schedule. The pipeline ensures your car price prediction models are continuously updated with fresh data.

## Main Pipeline: Monthly Model Training

**File**: `.github/workflows/monthly-training.yml`

### Schedule
- **Frequency**: Monthly (1st of every month)
- **Time**: 2:00 AM UTC  
- **Cron Expression**: `0 2 1 * *`

### Pipeline Jobs

1. **Data Collection & Preprocessing** (`setup-and-scrape`)
   - Scrapes fresh car data from automotive websites
   - Preprocesses data for all supported brands
   - Stores processed data as artifacts (90 days retention)

2. **Classification Model Training** (`train-classification`)
   - Trains price categorization models
   - Uses Random Forest algorithm
   - Parallel execution for different model types
   - 365-day artifact retention

3. **Regression Model Training** (`train-regression`)
   - Trains exact price prediction models
   - Uses Random Forest algorithm
   - Parallel execution for different model types
   - 365-day artifact retention

4. **Valuation Model Training** (`train-valuation`)
   - Trains ensemble models with uncertainty estimation
   - Uses multiple algorithms (Random Forest, Gradient Boosting, Linear Regression)
   - 365-day artifact retention

5. **Artifact Collection** (`collect-artifacts`)
   - Consolidates all trained models
   - Creates training report with metadata
   - 730-day retention for consolidated artifacts
   - Generates timestamped model packages

6. **Notification** (`notify-completion`)
   - Creates workflow summary
   - Reports training status for each brand
   - Provides next steps guidance

### Artifacts Generated

#### Model Files
All model files are automatically timestamped with the format:
```
{brand}_{model_type}_{algorithm}_{YYYYMMDD_HHMMSS}.pkl
```

Examples:
- `mercedes_classification_random_forest_20240710_143022.pkl`
- `ford_regression_random_forest_20240710_143025.pkl`
- `toyota_valuation_ensemble_20240710_143030.pkl`

#### Artifact Structure
```
monthly-models-YYYYMMDD_HHMMSS/
├── classification/
│   ├── mercedes/
│   │   └── mercedes_classification_random_forest_YYYYMMDD_HHMMSS.pkl
│   ├── ford/
│   └── ...
├── regression/
│   ├── mercedes/
│   │   └── mercedes_regression_random_forest_YYYYMMDD_HHMMSS.pkl
│   └── ...
├── valuation/
│   └── ...
└── model_inventory.md
```

#### Retention Policy
- **Models**: 365-730 days
- **Logs**: 30-90 days  
- **Data**: 90 days
- **Consolidated packages**: 730 days (2 years)

## Manual Triggering

The pipeline can be manually triggered via GitHub Actions UI with options:

### Parameters
- **brands**: Comma-separated list of brands (default: all supported brands)
- **max_pages**: Maximum pages to scrape per brand (default: 20)

### Example Manual Trigger
1. Go to Actions tab in GitHub
2. Select "Monthly Model Training Pipeline"
3. Click "Run workflow"
4. Optionally specify:
   - `brands`: `mercedes,ford,toyota`
   - `max_pages`: `10`

## Supported Brands

The pipeline automatically trains models for these car brands:
- Mercedes
- Ford
- Fiat
- Toyota
- Tesla
- Audi
- BMW
- Honda

## Environment Configuration

The pipeline uses these environment variables:
```bash
SCRAPING_DELAY=2.0
REQUEST_TIMEOUT=60
LOG_LEVEL=INFO
```

## Monitoring and Debugging

### Checking Pipeline Status
1. Go to Actions tab in your GitHub repository
2. Look for "Monthly Model Training Pipeline" runs
3. Click on any run to see detailed logs

### Downloading Models
1. Navigate to the completed workflow run
2. Scroll to "Artifacts" section
3. Download the desired model packages:
   - `monthly-models-YYYYMMDD_HHMMSS` (all models)
   - `classification-models-*` (classification only)
   - `regression-models-*` (regression only)
   - `valuation-models-*` (valuation only)

### Common Issues

#### Data Scraping Failures
- Check if target websites have changed structure
- Verify scraping delays are sufficient
- Review error logs in the setup-and-scrape job

#### Model Training Failures
- Check data quality in preprocessing step
- Verify sufficient data exists for the brand
- Review training logs for specific errors

#### Artifact Upload Issues
- Ensure models directory exists and contains .pkl files
- Check file permissions and sizes
- Verify GitHub Actions storage limits

## Performance Optimization

### Parallel Execution
The pipeline uses matrix strategies for parallel training:
```yaml
strategy:
  fail-fast: false
  matrix:
    model: [random_forest]
```

### Resource Management
- **Timeouts**: Set per job to prevent hanging
- **Memory**: Uses ubuntu-latest with sufficient resources
- **Concurrency**: Parallel model training reduces total runtime

### Scaling Considerations
- Add more model types to matrix for broader coverage
- Increase max_pages for more comprehensive data
- Add more brands to supported brands list

## Future Enhancements

### Planned Features
1. **Model Comparison**: Automatic benchmarking between monthly versions
2. **Data Quality Metrics**: Automated data validation reports
3. **Performance Alerts**: Notifications for significant accuracy drops
4. **A/B Testing**: Deploy and compare different model versions
5. **Cloud Storage**: Integration with cloud storage for larger model artifacts

### Adding New Models
To add a new model type:
1. Implement the model class in appropriate module
2. Add to the matrix strategy in the workflow
3. Update model registry in training scripts

### Customization
The pipeline is designed to be easily customizable:
- Modify cron schedule in the workflow file
- Adjust retention days for different artifact types
- Add new notification channels
- Integrate with external monitoring systems

## Best Practices

### Reliability
- All steps use `continue-on-error: true` to ensure pipeline completion
- Multiple job dependencies prevent cascading failures
- Comprehensive error logging for debugging

### Security
- No sensitive data in code
- Uses GitHub's secure environment variables
- Minimal required permissions

### Maintainability
- Clear job names and descriptions
- Modular pipeline design
- Comprehensive artifact organization
- Detailed logging and summaries

This pipeline ensures your car price prediction models are always up-to-date and ready for production use!
