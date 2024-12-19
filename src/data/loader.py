"""Data loading and preprocessing module."""
from pathlib import Path
from typing import List, Optional, Set, Tuple
import pandas as pd
from src.config.config import DataConfig

class DataLoader:
    """Handles data loading, validation, and preprocessing."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        
    def load_data(self, data_files: Optional[List[Path]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and preprocess data from multiple files.
        
        Args:
            data_files: Optional list of data files. If None, loads all Excel files from data_dir.
            
        Returns:
            Tuple of (training_data, test_data) DataFrames.
        """
        if data_files is None:
            data_files = sorted(list(self.config.data_dir.glob('*.xls*')))
            
        if not data_files:
            raise ValueError(f"No data files found in {self.config.data_dir}")
            
        dfs = []
        for file in data_files:
            df = pd.read_excel(file)
            if self._validate_data(df):
                dfs.append(self._preprocess_data(df))
                
        if not dfs:
            raise ValueError("No valid data files found")
            
        # Combine all data and sort chronologically
        df = pd.concat(dfs, ignore_index=True)
        df = df.sort_values('Date')
        
        # Split into training and test sets based on cutoff year
        train_data = df[df['Date'].dt.year < self.config.train_cutoff_year]
        test_data = df[df['Date'].dt.year >= self.config.train_cutoff_year]
        
        if len(train_data) < self.config.min_training_samples:
            raise ValueError(f"Insufficient training samples: {len(train_data)}")
            
        return train_data, test_data
    
    def _validate_data(self, df: pd.DataFrame) -> bool:
        """Validate that DataFrame contains required columns and data quality.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if DataFrame is valid, False otherwise
        """
        # Check required columns
        missing_cols = self.config.required_columns - set(df.columns)
        if missing_cols:
            print(f"Warning: Missing required columns: {missing_cols}")
            return False
        
        # Check for duplicate matches
        df['Match'] = df['Date'].astype(str) + '_' + df['HomeTeam'] + '_' + df['AwayTeam']
        duplicates = df['Match'].duplicated()
        if duplicates.any():
            print(f"Warning: Found {duplicates.sum()} duplicate matches")
            return False
        
        # Check for invalid odds (<=1.0)
        odds_columns = [col for col in df.columns if 'B365' in col]
        invalid_odds = (df[odds_columns] <= 1.0).any().any()
        if invalid_odds:
            print("Warning: Found invalid odds (<=1.0)")
            return False
        
        return True
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess raw data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        # Convert date column to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Sort by date to ensure chronological order
        df = df.sort_values('Date')
        
        # Add season column
        df['Season'] = df['Date'].dt.year.where(df['Date'].dt.month < 7, 
                                               df['Date'].dt.year - 1)
        
        # Fill missing values with appropriate defaults
        df = self._fill_missing_values(df)
        
        # Add derived columns
        df = self._add_derived_columns(df)
        
        return df
    
    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with appropriate defaults.
        
        Args:
            df: DataFrame with missing values
            
        Returns:
            DataFrame with filled values
        """
        # Fill missing odds with high values to prevent betting
        odds_columns = [col for col in df.columns if 'B365' in col]
        df[odds_columns] = df[odds_columns].fillna(100.0)
        
        # Fill missing stats with 0
        stats_columns = ['FTHG', 'FTAG', 'HTHG', 'HTAG', 'HC', 'AC', 'HY', 'AY']
        df[stats_columns] = df[stats_columns].fillna(0)
        
        return df
    
    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived columns useful for feature engineering.
        
        Args:
            df: DataFrame without derived columns
            
        Returns:
            DataFrame with derived columns
        """
        # Add total goals
        df['TotalGoals'] = df['FTHG'] + df['FTAG']
        df['HTTotalGoals'] = df['HTHG'] + df['HTAG']
        
        # Add total corners and cards
        df['TotalCorners'] = df['HC'] + df['AC']
        df['TotalCards'] = df['HY'] + df['AY']
        
        # Add goal difference
        df['GoalDiff'] = df['FTHG'] - df['FTAG']
        df['HTGoalDiff'] = df['HTHG'] - df['HTAG']
        
        return df 