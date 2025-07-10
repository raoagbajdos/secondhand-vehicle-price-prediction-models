"""
Data preprocessing functionality for car price prediction.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ..config import settings
from ..utils import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """Data preprocessing pipeline for car data."""
    
    def __init__(self):
        self.label_encoders = {}
        self.scalers = {}
        self.feature_columns = []
        self.target_column = 'price'
    
    def load_raw_data(self, brand: Optional[str] = None) -> pd.DataFrame:
        """
        Load raw scraped data.
        
        Args:
            brand: Specific brand to load, or None for all brands
            
        Returns:
            Combined DataFrame of raw data
        """
        data_files = []
        
        if brand:
            pattern = f"{brand}_scraped_data_*.csv"
            data_files = list(settings.raw_data_dir.glob(pattern))
        else:
            data_files = list(settings.raw_data_dir.glob("*_scraped_data_*.csv"))
        
        if not data_files:
            raise FileNotFoundError("No raw data files found")
        
        # Load and combine all data files
        dfs = []
        for file_path in data_files:
            try:
                df = pd.read_csv(file_path)
                dfs.append(df)
                logger.info(f"Loaded {len(df)} records from {file_path.name}")
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
        
        if not dfs:
            raise ValueError("No valid data files could be loaded")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined dataset shape: {combined_df.shape}")
        
        return combined_df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare raw data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Remove duplicates
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        logger.info(f"Removed {initial_rows - len(df_clean)} duplicate rows")
        
        # Clean price column
        df_clean = self._clean_price_column(df_clean)
        
        # Clean mileage column
        df_clean = self._clean_mileage_column(df_clean)
        
        # Extract features from title
        df_clean = self._extract_title_features(df_clean)
        
        # Clean year column
        df_clean = self._clean_year_column(df_clean)
        
        # Remove rows with missing critical information
        critical_columns = ['price', 'brand']
        df_clean = df_clean.dropna(subset=critical_columns)
        
        # Filter unrealistic values
        df_clean = self._filter_unrealistic_values(df_clean)
        
        logger.info(f"Cleaned dataset shape: {df_clean.shape}")
        return df_clean
    
    def _clean_price_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the price column."""
        df = df.copy()
        
        # Convert price to numeric, handling various formats
        if 'price' in df.columns:
            # Handle string prices
            df['price'] = df['price'].astype(str)
            df['price'] = df['price'].str.replace(r'[£$,]', '', regex=True)
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
        
        return df
    
    def _clean_mileage_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the mileage column."""
        df = df.copy()
        
        if 'mileage' in df.columns:
            # Convert mileage to numeric
            df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')
            
            # Fill missing mileage with median by brand and year
            for brand in df['brand'].unique():
                brand_mask = df['brand'] == brand
                if 'year' in df.columns:
                    for year in df[brand_mask]['year'].unique():
                        if pd.isna(year):
                            continue
                        mask = brand_mask & (df['year'] == year)
                        median_mileage = df.loc[mask, 'mileage'].median()
                        if not pd.isna(median_mileage):
                            df.loc[mask & df['mileage'].isna(), 'mileage'] = median_mileage
        
        return df
    
    def _extract_title_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from car title."""
        df = df.copy()
        
        if 'title' not in df.columns:
            return df
        
        # Extract fuel type
        fuel_patterns = {
            'petrol': r'\b(petrol|gasoline)\b',
            'diesel': r'\bdiesel\b',
            'hybrid': r'\bhybrid\b',
            'electric': r'\b(electric|ev)\b'
        }
        
        df['fuel_type'] = 'unknown'
        for fuel_type, pattern in fuel_patterns.items():
            mask = df['title'].str.contains(pattern, case=False, na=False)
            df.loc[mask, 'fuel_type'] = fuel_type
        
        # Extract transmission
        df['transmission'] = 'unknown'
        automatic_pattern = r'\b(automatic|auto)\b'
        manual_pattern = r'\bmanual\b'
        
        auto_mask = df['title'].str.contains(automatic_pattern, case=False, na=False)
        manual_mask = df['title'].str.contains(manual_pattern, case=False, na=False)
        
        df.loc[auto_mask, 'transmission'] = 'automatic'
        df.loc[manual_mask, 'transmission'] = 'manual'
        
        # Extract engine size (basic pattern)
        engine_pattern = r'(\d+\.\d+)L?'
        df['engine_size'] = df['title'].str.extract(engine_pattern, expand=False)
        df['engine_size'] = pd.to_numeric(df['engine_size'], errors='coerce')
        
        return df
    
    def _clean_year_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the year column."""
        df = df.copy()
        
        if 'year' in df.columns:
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            
            # Filter reasonable years (e.g., 1990-2024)
            current_year = pd.Timestamp.now().year
            df = df[(df['year'] >= 1990) & (df['year'] <= current_year + 1)]
            
            # Calculate age
            df['age'] = current_year - df['year']
        
        return df
    
    def _filter_unrealistic_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out unrealistic values."""
        df = df.copy()
        
        # Filter price range (£500 to £200,000)
        if 'price' in df.columns:
            df = df[(df['price'] >= 500) & (df['price'] <= 200000)]
        
        # Filter mileage range (0 to 300,000)
        if 'mileage' in df.columns:
            df = df[(df['mileage'] >= 0) & (df['mileage'] <= 300000)]
        
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        df_features = df.copy()
        
        # Price per mile (for cars with mileage > 0)
        if 'price' in df_features.columns and 'mileage' in df_features.columns:
            df_features['price_per_mile'] = np.where(
                df_features['mileage'] > 0,
                df_features['price'] / df_features['mileage'],
                np.nan
            )
        
        # Age categories
        if 'age' in df_features.columns:
            df_features['age_category'] = pd.cut(
                df_features['age'],
                bins=[0, 3, 7, 15, np.inf],
                labels=['new', 'young', 'mature', 'old']
            )
        
        # Mileage categories
        if 'mileage' in df_features.columns:
            df_features['mileage_category'] = pd.cut(
                df_features['mileage'],
                bins=[0, 20000, 60000, 100000, np.inf],
                labels=['low', 'medium', 'high', 'very_high']
            )
        
        # Brand luxury indicator
        luxury_brands = ['mercedes', 'audi', 'bmw', 'tesla']
        df_features['is_luxury'] = df_features['brand'].isin(luxury_brands)
        
        return df_features
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            df: DataFrame with features
            fit: Whether to fit encoders or use existing ones
            
        Returns:
            DataFrame with encoded features
        """
        df_encoded = df.copy()
        
        categorical_columns = [
            'brand', 'fuel_type', 'transmission', 'age_category', 'mileage_category'
        ]
        
        for col in categorical_columns:
            if col in df_encoded.columns:
                if fit:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                    df_encoded[f'{col}_encoded'] = self.label_encoders[col].fit_transform(
                        df_encoded[col].fillna('unknown')
                    )
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        known_categories = self.label_encoders[col].classes_
                        df_encoded[col] = df_encoded[col].fillna('unknown')
                        
                        # Replace unseen categories with 'unknown'
                        mask = ~df_encoded[col].isin(known_categories)
                        df_encoded.loc[mask, col] = 'unknown'
                        
                        df_encoded[f'{col}_encoded'] = self.label_encoders[col].transform(
                            df_encoded[col]
                        )
        
        return df_encoded
    
    def prepare_training_data(
        self, 
        df: pd.DataFrame, 
        target_column: str = 'price'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare final training data.
        
        Args:
            df: Processed DataFrame
            target_column: Name of target column
            
        Returns:
            Features DataFrame and target Series
        """
        # Define feature columns
        feature_columns = [
            'mileage', 'age', 'engine_size', 'price_per_mile',
            'brand_encoded', 'fuel_type_encoded', 'transmission_encoded',
            'age_category_encoded', 'mileage_category_encoded', 'is_luxury'
        ]
        
        # Filter to only include available columns
        available_features = [col for col in feature_columns if col in df.columns]
        
        X = df[available_features].copy()
        y = df[target_column].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Store feature columns
        self.feature_columns = available_features
        
        return X, y
    
    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            X: Features DataFrame
            fit: Whether to fit scaler or use existing one
            
        Returns:
            Scaled features DataFrame
        """
        X_scaled = X.copy()
        
        numerical_columns = X.select_dtypes(include=[np.number]).columns
        
        if fit:
            self.scalers['standard'] = StandardScaler()
            X_scaled[numerical_columns] = self.scalers['standard'].fit_transform(
                X_scaled[numerical_columns]
            )
        else:
            if 'standard' in self.scalers:
                X_scaled[numerical_columns] = self.scalers['standard'].transform(
                    X_scaled[numerical_columns]
                )
        
        return X_scaled
    
    def save_processed_data(self, X: pd.DataFrame, y: pd.Series, brand: str) -> Path:
        """Save processed data to file."""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Combine features and target
        processed_df = X.copy()
        processed_df[self.target_column] = y
        
        # Save to processed data directory
        filename = f"{brand}_processed_data_{timestamp}.csv"
        output_path = settings.processed_data_dir / filename
        
        processed_df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
        
        return output_path


def preprocess_data(brand: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Convenience function to preprocess data for a brand.
    
    Args:
        brand: Brand to preprocess, or None for all brands
        
    Returns:
        Features and target data
    """
    preprocessor = DataPreprocessor()
    
    # Load and clean data
    raw_data = preprocessor.load_raw_data(brand)
    clean_data = preprocessor.clean_data(raw_data)
    
    # Create features
    featured_data = preprocessor.create_features(clean_data)
    
    # Encode categorical features
    encoded_data = preprocessor.encode_categorical_features(featured_data)
    
    # Prepare training data
    X, y = preprocessor.prepare_training_data(encoded_data)
    
    # Scale features
    X_scaled = preprocessor.scale_features(X)
    
    return X_scaled, y


def main():
    """CLI entry point for preprocessing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess car data")
    parser.add_argument(
        "--brand", 
        type=str, 
        help="Brand to preprocess (leave empty for all brands)"
    )
    
    args = parser.parse_args()
    
    try:
        X, y = preprocess_data(args.brand)
        logger.info(f"Preprocessing completed. Features shape: {X.shape}, Target shape: {y.shape}")
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise


if __name__ == "__main__":
    main()
