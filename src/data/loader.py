import pandas as pd
import numpy as np
import logging
import os
import glob
from typing import Tuple, List

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.data_dir = os.path.join(self.project_root, self.config.data_dir)

    def load_data(self, test_mode: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Main entry point to load and split data."""
        seasons = self._get_seasons(test_mode)
        data = self._load_seasons(seasons)
        return self._split_data(data, test_mode)

    def _get_seasons(self, test_mode: bool) -> List[str]:
        """Get list of seasons to load based on mode."""
        seasons = sorted([d for d in os.listdir(self.data_dir) if d.count('-') == 1])
        if not seasons:
            raise ValueError(f"No season folders found in {self.data_dir}")
        
        if test_mode:
            num_test_seasons = max(2, int(len(seasons) * self.config.test_size))
            return seasons[-num_test_seasons:]
        return seasons

    def _load_seasons(self, seasons: List[str]) -> pd.DataFrame:
        """Load and combine data from specified seasons."""
        all_data = []
        
        for season in seasons:
            season_path = os.path.join(self.data_dir, season)
            data_files = sorted(glob.glob(os.path.join(season_path, '*.csv')))
            
            for file in data_files:
                try:
                    df = pd.read_csv(file)
                    if len(df) == 0:
                        continue
                        
                    league = os.path.basename(file).split('.')[0]
                    df['League'] = league
                    df['Season'] = season
                    
                    logger.info(f"Loaded {len(df)} rows from {league} - Season {season}")
                    all_data.append(df)
                    
                except Exception as e:
                    logger.warning(f"Error reading {file}: {str(e)}")
                    
        if not all_data:
            raise ValueError("No valid data files could be loaded")
            
        data = pd.concat(all_data, ignore_index=True)
        return self._preprocess_data(data)

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and filter the data."""
        # Convert date and handle missing values
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
        data[data.select_dtypes(include=[np.number]).columns] = data.select_dtypes(include=[np.number]).fillna(0)
        
        # Drop rows with missing critical data
        critical_columns = ['HomeTeam', 'AwayTeam', 'FTR', 'B365H', 'B365D', 'B365A']
        data = data.dropna(subset=critical_columns)
        
        # Filter odds within configured range
        odds_columns = ['B365H', 'B365D', 'B365A']
        for col in odds_columns:
            data = data[
                (data[col] >= self.config.min_odds) & 
                (data[col] <= self.config.max_odds)
            ]
        
        return data

    def _split_data(self, data: pd.DataFrame, test_mode: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and test sets."""
        if not test_mode:
            return data, pd.DataFrame(columns=data.columns)
            
        data = data.sort_values('Date')
        unique_seasons = sorted(data['Season'].unique())
        num_test_seasons = max(1, int(len(unique_seasons) * self.config.test_size))
        
        test_seasons = unique_seasons[-num_test_seasons:]
        train_seasons = unique_seasons[:-num_test_seasons]
        
        logger.info(f"Train seasons: {train_seasons}")
        logger.info(f"Test seasons: {test_seasons}")
        
        train_data = data[data['Season'].isin(train_seasons)].copy()
        test_data = data[data['Season'].isin(test_seasons)].copy()
        
        return train_data, test_data