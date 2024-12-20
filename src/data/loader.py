"""Data loading and preprocessing module."""
from pathlib import Path
from typing import List, Optional, Set, Tuple
import pandas as pd
import logging
from tqdm import tqdm
from src.config.config import DataConfig

logger = logging.getLogger(__name__)

class DataLoader:
    """Handles data loading, validation, and preprocessing."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and preprocess all data files.
        
        Returns:
            Tuple of (train_data, test_data)
        """
        all_data = []
        
        # Load all data files
        data_files = sorted(self.config.data_dir.glob('*.xls*'))
        logger.info(f"Found {len(data_files)} data files in {self.config.data_dir}")
        
        # Add progress bar for files
        for file_path in tqdm(data_files, desc="Processing data files", unit="file"):
            df = self._load_and_process_file(file_path)
            if df is not None:
                df = self._process_all_leagues(df)
                all_data.append(df)
        
        # Combine all data
        logger.info("Combining all datasets...")
        df = pd.concat(all_data, ignore_index=True)
        df = df.drop_duplicates()
        logger.info(f"Combined dataset has {len(df)} total rows after removing duplicates")
        
        # Extract season from Date
        df['Season'] = df['Date'].dt.year
        df.loc[df['Date'].dt.month < 8, 'Season'] -= 1  # Adjust for season spanning calendar years
        
        # Split into training and test sets by complete seasons
        logger.info("Splitting data into train and test sets...")
        train_data = df[df['Season'] < 2022].copy()  # Up to 2021-22 season
        test_data = df[df['Season'] >= 2022].copy()  # 2022-23 season onwards
        
        logger.info(f"Training data: {len(train_data)} rows ({train_data['Date'].min()} to {train_data['Date'].max()})")
        logger.info(f"Test data: {len(test_data)} rows ({test_data['Date'].min()} to {test_data['Date'].max()})")
        
        # Log data info per league
        logger.info("\nData distribution across leagues:")
        for league in sorted(df['League'].unique()):
            train_league = train_data[train_data['League'] == league]
            test_league = test_data[test_data['League'] == league]
            logger.info(f"\n{league}:")
            logger.info(f"Training data: {len(train_league)} matches")
            logger.info(f"Test data: {len(test_league)} matches")
        
        train_seasons = sorted(train_data['Season'].unique())
        test_seasons = sorted(test_data['Season'].unique())
        logger.info(f"\nTraining seasons: {train_seasons}")
        logger.info(f"Test seasons: {test_seasons}")
        
        # Verify no season overlap
        if set(train_seasons) & set(test_seasons):
            logger.warning("WARNING: Some seasons appear in both training and test sets!")
        
        return train_data, test_data
    
    def _validate_data(self, df: pd.DataFrame) -> bool:
        """Validate DataFrame quality."""
        # Check required columns
        missing_cols = self.config.required_columns - set(df.columns)
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            return False
        
        # Check for duplicate matches
        df['Match'] = df['Date'].astype(str) + '_' + df['HomeTeam'] + '_' + df['AwayTeam']
        duplicates = df['Match'].duplicated()
        if duplicates.any():
            logger.warning(f"Found {duplicates.sum()} duplicate matches - will be handled during preprocessing")
        
        # Check for invalid odds (<=1.0)
        odds_columns = [col for col in df.columns if 'B365' in col]
        invalid_odds_count = (df[odds_columns] <= 1.0).sum().sum()
        total_odds = len(df) * len(odds_columns)
        invalid_odds_percentage = (invalid_odds_count / total_odds) * 100
        
        # Only reject if more than 5% of odds are invalid
        if invalid_odds_percentage > 5:
            logger.warning(f"Too many invalid odds ({invalid_odds_percentage:.2f}% of all odds are <=1.0)")
            return False
        elif invalid_odds_count > 0:
            logger.warning(f"Found {invalid_odds_count} invalid odds (will be handled during preprocessing)")
        
        # Log data information
        logger.info(f"Data range: {df['Date'].min()} to {df['Date'].max()}")
        logger.info(f"Number of unique teams: {len(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))}")
        
        return True
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess raw data."""
        # Convert date column to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Sort by date to ensure chronological order
        df = df.sort_values('Date')
        
        # Add season column (season starts in August)
        df['Season'] = df.apply(lambda x: 
            x['Date'].year if x['Date'].month >= 8 
            else x['Date'].year - 1, axis=1)
        
        # Validate season consistency
        season_counts = df.groupby('Season').size()
        expected_matches = 380  # Standard season length
        for season, count in season_counts.items():
            if count > expected_matches:
                logger.warning(f"Season {season} has {count} matches (expected {expected_matches})")
            elif count < expected_matches and season != df['Season'].max():  # Exclude current season
                logger.warning(f"Season {season} has only {count} matches (expected {expected_matches})")
        
        # Log season information
        logger.info(f"Number of matches per season: {season_counts}")
        
        # Fill missing values with appropriate defaults
        df = self._fill_missing_values(df)
        
        # Add derived columns
        df = self._add_derived_columns(df)
        
        return df
    
    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with appropriate defaults."""
        # Fill missing odds with high values to prevent betting
        odds_columns = [col for col in df.columns if 'B365' in col]
        df[odds_columns] = df[odds_columns].fillna(100.0)
        
        # Replace invalid odds (<=1.0) with high values to prevent betting
        for col in odds_columns:
            df.loc[df[col] <= 1.0, col] = 100.0
        
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
    
    def _load_and_process_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load and process a single data file.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Processed DataFrame or None if file is invalid
        """
        logger.info(f"\nProcessing file: {file_path}")
        
        try:
            # Read all sheets from the Excel file
            excel_file = pd.ExcelFile(file_path)
            all_sheets_data = []
            
            # Add progress bar for sheets
            for sheet_name in tqdm(excel_file.sheet_names, desc="Processing sheets", unit="sheet", leave=False):
                logger.info(f"Reading sheet: {sheet_name}")
                # Load data from this sheet
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Skip empty sheets or sheets without required structure
                if df.empty or 'Div' not in df.columns:
                    logger.warning(f"Skipping invalid sheet: {sheet_name}")
                    continue
                
                logger.info(f"Loaded {len(df)} rows from sheet {sheet_name}")
                
                # Basic validation
                if not self._validate_data(df):
                    logger.warning(f"Skipping invalid sheet: {sheet_name}")
                    continue
                
                # Process data
                df = self._preprocess_data(df)
                all_sheets_data.append(df)
            
            if not all_sheets_data:
                logger.warning(f"No valid data found in file: {file_path}")
                return None
            
            # Combine all sheets' data
            logger.info("Combining sheets data...")
            combined_df = pd.concat(all_sheets_data, ignore_index=True)
            
            # Log combined data info
            logger.info(f"Successfully processed {len(combined_df)} total rows from {file_path}")
            logger.info(f"Data range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")
            logger.info(f"Number of unique teams: {len(set(combined_df['HomeTeam'].unique()) | set(combined_df['AwayTeam'].unique()))}")
            
            # Log matches per season
            season_counts = combined_df.groupby('Season').size()
            logger.info(f"Number of matches per season:\n{season_counts}")
            
            return combined_df
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return None
    
    def _process_all_leagues(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process data for all leagues.
        
        Args:
            df: DataFrame containing data for all leagues
            
        Returns:
            Processed DataFrame with all leagues
        """
        # Map league codes to names
        league_mapping = {
            'E0': 'English Premier League',
            'E1': 'English Championship',
            'E2': 'English League One',
            'E3': 'English League Two',
            'EC': 'English Conference',
            'SC0': 'Scottish Premiership',
            'SC1': 'Scottish Championship',
            'SC2': 'Scottish League One',
            'SC3': 'Scottish League Two',
            'D1': 'German Bundesliga',
            'D2': 'German 2. Bundesliga',
            'SP1': 'Spanish La Liga',
            'SP2': 'Spanish Segunda División',
            'I1': 'Italian Serie A',
            'I2': 'Italian Serie B',
            'F1': 'French Ligue 1',
            'F2': 'French Ligue 2',
            'B1': 'Belgian First Division',
            'N1': 'Dutch Eredivisie',
            'P1': 'Portuguese Primeira Liga',
            'T1': 'Turkish Super Lig',
            'G1': 'Greek Super League'
        }
        
        # Filter for all specified leagues
        df = df[df['Div'].isin(league_mapping.keys())].copy()
        
        # Add league name column
        df['League'] = df['Div'].map(league_mapping)
        
        logger.info(f"Processed data for leagues: {sorted(df['League'].unique())}")
        logger.info(f"Number of matches per league:\n{df.groupby('League').size()}")
        
        return df