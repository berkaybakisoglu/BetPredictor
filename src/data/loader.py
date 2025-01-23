import pandas as pd
import numpy as np
import logging
import os
import glob

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def load_data(self, test_mode=False):
        """Load data with efficient handling of test mode."""
        if test_mode:
            # First, get list of seasons without loading data
            data_dir = os.path.join(self.project_root, self.config.data_dir)
            season_folders = sorted([d for d in os.listdir(data_dir) if d.count('-') == 1])
            
            if not season_folders:
                raise ValueError("No season folders found in {0}".format(data_dir))
            
            # Calculate how many seasons we need
            num_test_seasons = max(2, int(len(season_folders) * self.config.test_size))
            selected_seasons = season_folders[-num_test_seasons:]  # Take only the most recent seasons needed
            
            logger.info(f"Test mode: Loading only {len(selected_seasons)} seasons: {selected_seasons}")
            data = self.load_seasons(selected_seasons)
        else:
            data = self.load_all_data()
        
        return self.split_data(data, test_mode)

    def load_all_data(self):
        """Load all data from the data directory without splitting."""
        data_dir = os.path.join(self.project_root, self.config.data_dir)
        logger.info("Loading data from {0}".format(data_dir))
        
        # Look for season folders
        season_folders = sorted([d for d in os.listdir(data_dir) if d.count('-') == 1])  # Folders like "2009-2010"
        if not season_folders:
            raise ValueError("No season folders found in {0}".format(data_dir))
        
        logger.info("Found {0} season folders: {1}".format(len(season_folders), season_folders))
        
        all_data = []
        for season in season_folders:
            season_path = os.path.join(data_dir, season)
            data_files = sorted(glob.glob(os.path.join(season_path, '*.csv')))
            
            for file in data_files:
                try:
                    logger.info("Reading {0}".format(file))
                    
                    df = pd.read_csv(file)
                    if len(df) == 0:
                        continue
                    
                    filename = os.path.basename(file)
                    league = filename.split('.')[0]  # Get league code from filename (e.g., 'E0' from 'E0.csv')
                    
                    df['League'] = league
                    df['Season'] = season
                    
                    logger.info("Loaded {0} rows from {1} - Season {2}".format(len(df), filename, season))
                    all_data.append(df)
                    
                except Exception as e:
                    logger.warning("Error reading file {0}: {1}".format(file, str(e)))
                    continue
        
        if not all_data:
            raise ValueError("No valid data files could be loaded")
        
        data = pd.concat(all_data, ignore_index=True)
        logger.info("Combined data shape: {0}".format(data.shape))
        
        data = self._preprocess_data(data)
        logger.info("After preprocessing shape: {0}".format(data.shape))
        
        return data

    def split_data(self, data, test_mode=False):
        """Split data into training and test sets."""
        if test_mode:
            # Sort by date to ensure chronological order
            data = data.sort_values('Date')
            
            # Use the most recent seasons for testing
            unique_seasons = sorted(data['Season'].unique())
            num_test_seasons = max(1, int(len(unique_seasons) * self.config.test_size))
            test_seasons = unique_seasons[-num_test_seasons:]
            train_seasons = unique_seasons[:-num_test_seasons]
            
            logger.info(f"All seasons: {unique_seasons}")
            logger.info(f"Test seasons: {test_seasons}")
            logger.info(f"Train seasons: {train_seasons}")
            
            # Split data
            train_data = data[data['Season'].isin(train_seasons)].copy()
            test_data = data[data['Season'].isin(test_seasons)].copy()
        else:
            # For production mode, use all data for training
            train_data = data.copy()
            test_data = pd.DataFrame(columns=data.columns)  # Empty DataFrame with same structure
        
        logger.info(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")
        logger.info(f"Train seasons: {sorted(train_data['Season'].unique())}")
        logger.info(f"Test seasons: {sorted(test_data['Season'].unique())}")
        
        return train_data, test_data

    def _preprocess_data(self, data):
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = data[numeric_columns].fillna(0)
        
        critical_columns = ['HomeTeam', 'AwayTeam', 'FTR', 'B365H', 'B365D', 'B365A']
        data = data.dropna(subset=critical_columns)
        
        data = data[
            (data['B365H'] >= self.config.min_odds) & 
            (data['B365H'] <= self.config.max_odds) &
            (data['B365D'] >= self.config.min_odds) & 
            (data['B365D'] <= self.config.max_odds) &
            (data['B365A'] >= self.config.min_odds) & 
            (data['B365A'] <= self.config.max_odds)
        ]
        
        return data

    def load_seasons(self, seasons):
        """Load data only for specific seasons."""
        data_dir = os.path.join(self.project_root, self.config.data_dir)
        logger.info(f"Loading data for seasons: {seasons}")
        
        all_data = []
        for season in seasons:
            season_path = os.path.join(data_dir, season)
            data_files = sorted(glob.glob(os.path.join(season_path, '*.csv')))
            
            for file in data_files:
                try:
                    logger.info(f"Reading {file}")
                    
                    df = pd.read_csv(file)
                    if len(df) == 0:
                        continue
                    
                    filename = os.path.basename(file)
                    league = filename.split('.')[0]
                    
                    df['League'] = league
                    df['Season'] = season
                    
                    logger.info(f"Loaded {len(df)} rows from {filename} - Season {season}")
                    all_data.append(df)
                    
                except Exception as e:
                    logger.warning(f"Error reading file {file}: {str(e)}")
                    continue
        
        if not all_data:
            raise ValueError("No valid data files could be loaded")
        
        data = pd.concat(all_data, ignore_index=True)
        logger.info(f"Combined data shape: {data.shape}")
        
        data = self._preprocess_data(data)
        logger.info(f"After preprocessing shape: {data.shape}")
        
        return data