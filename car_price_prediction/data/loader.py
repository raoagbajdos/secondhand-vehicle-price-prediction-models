"""
Data loading utilities for processed car data.
"""

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import settings
from ..utils import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Data loader for processed car data."""
    
    def __init__(self):
        self.feature_columns = []
        self.target_column = 'price'
    
    def load_processed_data(self, brand: str) -> pd.DataFrame:
        """
        Load processed data for a specific brand.
        
        Args:
            brand: Car brand name
            
        Returns:
            Processed DataFrame
        """
        # Find the latest processed data file for the brand
        pattern = f"{brand}_processed_data_*.csv"
        data_files = list(settings.processed_data_dir.glob(pattern))
        
        if not data_files:
            raise FileNotFoundError(f"No processed data found for brand: {brand}")
        
        # Get the most recent file
        latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
        
        df = pd.read_csv(latest_file)
        logger.info(f"Loaded {len(df)} records for {brand} from {latest_file.name}")
        
        return df
    
    def load_all_processed_data(self) -> pd.DataFrame:
        """
        Load all processed data.
        
        Returns:
            Combined DataFrame of all processed data
        """
        data_files = list(settings.processed_data_dir.glob("*_processed_data_*.csv"))
        
        if not data_files:
            raise FileNotFoundError("No processed data files found")
        
        # Group files by brand and get latest for each
        brand_files = {}
        for file_path in data_files:
            brand = file_path.name.split('_')[0]
            if brand not in brand_files or file_path.stat().st_mtime > brand_files[brand].stat().st_mtime:
                brand_files[brand] = file_path
        
        # Load and combine all brand data
        dfs = []
        for brand, file_path in brand_files.items():
            try:
                df = pd.read_csv(file_path)
                dfs.append(df)
                logger.info(f"Loaded {len(df)} records for {brand}")
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
        
        if not dfs:
            raise ValueError("No valid processed data files could be loaded")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined processed dataset shape: {combined_df.shape}")
        
        return combined_df
    
    def prepare_train_test_split(
        self, 
        df: pd.DataFrame,
        target_column: str = 'price',
        test_size: float = None,
        random_state: int = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare train/test split.
        
        Args:
            df: Processed DataFrame
            target_column: Name of target column
            test_size: Proportion of test set
            random_state: Random seed
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        test_size = test_size or settings.test_size
        random_state = random_state or settings.random_seed
        
        # Separate features and target
        feature_columns = [col for col in df.columns if col != target_column]
        X = df[feature_columns]
        y = df[target_column]
        
        # Store feature columns
        self.feature_columns = feature_columns
        self.target_column = target_column
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=None  # Can't stratify continuous target
        )
        
        logger.info(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def prepare_train_val_test_split(
        self,
        df: pd.DataFrame,
        target_column: str = 'price',
        test_size: float = None,
        val_size: float = None,
        random_state: int = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Prepare train/validation/test split.
        
        Args:
            df: Processed DataFrame
            target_column: Name of target column
            test_size: Proportion of test set
            val_size: Proportion of validation set
            random_state: Random seed
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        test_size = test_size or settings.test_size
        val_size = val_size or settings.validation_size
        random_state = random_state or settings.random_seed
        
        # First split: train+val vs test
        X_train_val, X_test, y_train_val, y_test = self.prepare_train_test_split(
            df, target_column, test_size, random_state
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)  # Adjust validation size
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_ratio,
            random_state=random_state
        )
        
        logger.info(
            f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def load_for_brand_training(
        self, 
        brand: str,
        include_other_brands: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Load data specifically for training a brand-specific model.
        
        Args:
            brand: Target brand
            include_other_brands: Whether to include other brands as features
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        if include_other_brands:
            # Load all data
            df = self.load_all_processed_data()
            # Filter to include target brand and others as features
            # This could be useful for learning brand-specific patterns
        else:
            # Load only target brand data
            df = self.load_processed_data(brand)
        
        return self.prepare_train_test_split(df)


def load_processed_data(brand: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to load processed data.
    
    Args:
        brand: Specific brand to load, or None for all brands
        
    Returns:
        Processed DataFrame
    """
    loader = DataLoader()
    
    if brand:
        return loader.load_processed_data(brand)
    else:
        return loader.load_all_processed_data()


def prepare_training_data(
    brand: Optional[str] = None,
    include_other_brands: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Convenience function to prepare training data.
    
    Args:
        brand: Specific brand, or None for all brands
        include_other_brands: Whether to include other brands
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    loader = DataLoader()
    
    if brand:
        return loader.load_for_brand_training(brand, include_other_brands)
    else:
        df = loader.load_all_processed_data()
        return loader.prepare_train_test_split(df)
