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
        
    def load_data(self, test_mode: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and preprocess all data files.
        
        Returns:
            Tuple of (training_data, test_data)
        """
        logger = logging.getLogger(__name__)
        
        # Get list of data files
        data_files = sorted(self.config.data_dir.glob('*.xls*'))  # Include .xls and .xlsx
        logger.info(f"Found {len(data_files)} data files in {self.config.data_dir}")
        
        # In test mode, we only need 2020-2024 data
        if test_mode:
            required_years = set(range(2020, 2024))  # 2020-2023 for test mode
            data_files = [f for f in data_files if any(str(year) in f.name for year in required_years)]
            logger.info(f"Test mode: Using {len(data_files)} files for years 2020-2023")
        
        all_data = []
        
        # Process each file
        for file in tqdm(data_files, desc="Processing data files", unit="file"):
            logger.info(f"\nProcessing file: {file}")
            
            # Read Excel file
            excel_file = pd.ExcelFile(file)
            
            # Process each sheet
            for sheet in excel_file.sheet_names:
                if sheet in self.config.leagues_to_include:
                    logger.info(f"Reading sheet: {sheet}")
                    
                    try:
                        # Read data from sheet
                        df = pd.read_excel(excel_file, sheet_name=sheet)
                        
                        if len(df) == 0:
                            logger.warning(f"Empty sheet: {sheet}")
                            continue
                        
                        # Add league identifier
                        df['League'] = sheet
                        
                        # Basic data validation
                        if not all(col in df.columns for col in self.config.required_columns):
                            missing = set(self.config.required_columns) - set(df.columns)
                            logger.warning(f"Missing required columns in {sheet}: {missing}")
                            continue
                        
                        # Convert date column
                        df['Date'] = pd.to_datetime(df['Date'])
                        
                        # Extract season
                        df['Season'] = df['Date'].dt.year
                        # Adjust season for matches in latter half of year
                        df.loc[df['Date'].dt.month > 6, 'Season'] += 1
                        
                        # Log data info
                        logger.info(f"Loaded {len(df)} rows from sheet {sheet}")
                        logger.info(f"Data range: {df['Date'].min()} to {df['Date'].max()}")
                        logger.info(f"Number of unique teams: {len(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))}")
                        
                        # Verify number of matches per season
                        matches_per_season = df.groupby('Season').size()
                        logger.info(f"Number of matches per season: {matches_per_season}")
                        
                        # Check for expected number of matches
                        expected = self.config.expected_matches.get(sheet)
                        if expected is not None:
                            for season, count in matches_per_season.items():
                                if count > expected:
                                    logger.warning(f"Season {season} has {count} matches (expected {expected})")
                                elif count < expected * 0.8:  # Allow for some missing matches (80% threshold)
                                    logger.warning(f"Season {season} has too few matches: {count} (expected {expected})")
                                    
                        all_data.append(df)
                        
                    except Exception as e:
                        logger.error(f"Error processing sheet {sheet}: {str(e)}")
                        continue
        
        if not all_data:
            raise ValueError("No valid data loaded")
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Sort by date
        combined_data = combined_data.sort_values('Date').reset_index(drop=True)
        
        if test_mode:
            # Get unique seasons
            seasons = sorted(combined_data['Season'].unique())
            if len(seasons) >= 3:
                # Use last 3 seasons: 2 for training, 1 for testing
                test_season = seasons[-1]  # Last season for testing
                train_seasons = seasons[-3:-1]  # Previous 2 seasons for training
                logger.info(f"Test mode - Training seasons: {train_seasons}, Test season: {test_season}")
                
                # Split data
                train_data = combined_data[combined_data['Season'].isin(train_seasons)].copy()
                test_data = combined_data[combined_data['Season'] == test_season].copy()
            else:
                raise ValueError(f"Not enough seasons for test mode. Need at least 3 seasons, found {len(seasons)}")
        else:
            # Split into training and test sets using date
            split_date = pd.Timestamp('2022-01-01')  # Use 2022 as cutoff for non-test mode
            train_data = combined_data[combined_data['Date'] < split_date].copy()
            test_data = combined_data[combined_data['Date'] >= split_date].copy()
        
        # Log data splits
        for league in train_data['League'].unique():
            logger.info(f"\n{league}:")
            logger.info(f"Training data: {len(train_data[train_data['League'] == league])} matches")
            logger.info(f"Test data: {len(test_data[test_data['League'] == league])} matches")
        
        # Log seasons
        train_seasons = sorted(train_data['Season'].unique())
        test_seasons = sorted(test_data['Season'].unique())
        logger.info(f"Training seasons: {train_seasons}")
        logger.info(f"Test seasons: {test_seasons}")
        
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