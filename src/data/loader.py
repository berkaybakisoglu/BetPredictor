"""Data loading module for the betting prediction system."""
import pandas as pd
import numpy as np
import logging
import os
import glob

logger = logging.getLogger(__name__)

class DataLoader:
    """Class for loading and preprocessing betting data."""
    
    def __init__(self, config):
        self.config = config
        # Get the project root directory (two levels up from this file)
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    def load_data(self, test_mode=False):
        """Load data from Excel files and split into train/test sets."""
        # Construct absolute path to data directory
        data_dir = os.path.join(self.project_root, self.config.data_dir)
        logger.info("Loading data from {0}".format(data_dir))
        
        # Get all Excel files in the data directory
        data_files = sorted(glob.glob(os.path.join(data_dir, '*.xls*')))
        
        if not data_files:
            raise ValueError("No Excel files found in {0}".format(data_dir))
        
        logger.info("Found {0} Excel files".format(len(data_files)))
        
        # Read and combine all files
        all_data = []
        for file in data_files:
            try:
                # Choose engine based on file extension
                engine = 'xlrd' if file.endswith('.xls') else 'openpyxl'
                logger.info("Reading {0} using {1} engine".format(file, engine))
        
                # Read all sheets
            excel_file = pd.ExcelFile(file)
            
                # Extract season from filename (e.g., "all-euro-data-2024-2025.xlsx" -> "2024-2025")
                filename = os.path.basename(file)
                season = filename.split('-')[-1].split('.')[0]
                
                # Process each sheet (league)
                for sheet_name in excel_file.sheet_names:
                    try:
                        df = pd.read_excel(excel_file, sheet_name=sheet_name)
                        if len(df) == 0:
                            continue
                        
                        # Add season and league information
                        df['Season'] = season
                        df['League'] = sheet_name
                        
                        logger.info("Loaded {0} rows from {1} - {2}".format(
                            len(df), filename, sheet_name))
                        all_data.append(df)
                        
                    except Exception as e:
                        logger.warning("Error reading sheet {0} from {1}: {2}".format(
                            sheet_name, filename, str(e)))
                        continue
                
            except Exception as e:
                logger.warning("Error reading file {0}: {1}".format(file, str(e)))
                        continue
        
        if not all_data:
            raise ValueError("No valid data files could be loaded")
        
        # Combine all dataframes
        data = pd.concat(all_data, ignore_index=True)
        logger.info("Combined data shape: {0}".format(data.shape))
        
        # Basic preprocessing
        data = self._preprocess_data(data)
        logger.info("After preprocessing shape: {0}".format(data.shape))
        
        # Split into train and test sets
        train_data, test_data = self._split_data(data, test_mode)
        
        logger.info("Loaded {0} training samples and {1} test samples".format(
            len(train_data), len(test_data)))
        
        return train_data, test_data
    
    def _preprocess_data(self, data):
        """Apply basic preprocessing to the data."""
        # Convert date column
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
        
        # Handle missing values
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = data[numeric_columns].fillna(0)
        
        # Drop rows with missing critical values
        critical_columns = ['HomeTeam', 'AwayTeam', 'FTR', 'B365H', 'B365D', 'B365A']
        data = data.dropna(subset=critical_columns)
        
        # Filter odds within reasonable range
        data = data[
            (data['B365H'] >= self.config.min_odds) & 
            (data['B365H'] <= self.config.max_odds) &
            (data['B365D'] >= self.config.min_odds) & 
            (data['B365D'] <= self.config.max_odds) &
            (data['B365A'] >= self.config.min_odds) & 
            (data['B365A'] <= self.config.max_odds)
        ]
        
        return data
    
    def _split_data(self, data, test_mode):
        """Split data into training and test sets."""
        if test_mode:
            # In test mode, use a simple random split
            train_size = int(len(data) * (1 - self.config.test_size))
            train_data = data.iloc[:train_size]
            test_data = data.iloc[train_size:]
        else:
            # In production mode, split by date
            cutoff_date = data['Date'].max() - pd.DateOffset(months=3)
            train_data = data[data['Date'] <= cutoff_date]
            test_data = data[data['Date'] > cutoff_date]
        
        return train_data, test_data