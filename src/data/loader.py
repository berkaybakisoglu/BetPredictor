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
        """Load and preprocess all data files."""
        logger = logging.getLogger(__name__)
        
        data_files = sorted(self.config.data_dir.glob('*.xls*'))
        logger.info(f"Found {len(data_files)} data files in {self.config.data_dir}")
        
        if test_mode:
            required_years = set(range(2020, 2024))
            data_files = [f for f in data_files if any(str(year) in f.name for year in required_years)]
            logger.info(f"Test mode: Using {len(data_files)} files for years 2020-2024")
        
        all_data = []
        
        for file in tqdm(data_files, desc="Processing data files", unit="file"):
            logger.info(f"\nProcessing file: {file}")
            
            excel_file = pd.ExcelFile(file)
            
            for sheet in excel_file.sheet_names:
                if sheet in self.config.leagues_to_include:
                    logger.info(f"Reading sheet: {sheet}")
                    
                    try:
                        df = pd.read_excel(excel_file, sheet_name=sheet)
                        
                        if len(df) == 0:
                            logger.warning(f"Empty sheet: {sheet}")
                            continue
                        
                        df['League'] = sheet
                        
                        if not all(col in df.columns for col in self.config.required_columns):
                            missing = set(self.config.required_columns) - set(df.columns)
                            logger.warning(f"Missing required columns in {sheet}: {missing}")
                            continue
                        
                        df['Date'] = pd.to_datetime(df['Date'])
                        df['Season'] = df['Date'].dt.year
                        df.loc[df['Date'].dt.month > 6, 'Season'] += 1
                        
                        logger.info(f"Loaded {len(df)} rows from sheet {sheet}")
                        logger.info(f"Data range: {df['Date'].min()} to {df['Date'].max()}")
                        logger.info(f"Number of unique teams: {len(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))}")
                        
                        matches_per_season = df.groupby('Season').size()
                        logger.info(f"Number of matches per season: {matches_per_season}")
                        
                        expected = self.config.expected_matches.get(sheet)
                        if expected is not None:
                            for season, count in matches_per_season.items():
                                if count > expected:
                                    logger.warning(f"Season {season} has {count} matches (expected {expected})")
                                elif count < expected * 0.8:
                                    logger.warning(f"Season {season} has too few matches: {count} (expected {expected})")
                                    
                        all_data.append(df)
                        
                    except Exception as e:
                        logger.error(f"Error processing sheet {sheet}: {str(e)}")
                        continue
        
        if not all_data:
            raise ValueError("No valid data loaded")
        
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data = combined_data.sort_values('Date').reset_index(drop=True)
        
        if test_mode:
            seasons = sorted(combined_data['Season'].unique())
            if len(seasons) >= 3:
                test_season = seasons[-1]
                train_seasons = seasons[-3:-1]
                logger.info(f"Test mode - Training seasons: {train_seasons}, Test season: {test_season}")
                
                train_data = combined_data[combined_data['Season'].isin(train_seasons)].copy()
                test_data = combined_data[combined_data['Season'] == test_season].copy()
            else:
                raise ValueError(f"Not enough seasons for test mode. Need at least 3 seasons, found {len(seasons)}")
        else:
            split_date = pd.Timestamp('2022-01-01')
            train_data = combined_data[combined_data['Date'] < split_date].copy()
            test_data = combined_data[combined_data['Date'] >= split_date].copy()
        
        for league in train_data['League'].unique():
            logger.info(f"\n{league}:")
            logger.info(f"Training data: {len(train_data[train_data['League'] == league])} matches")
            logger.info(f"Test data: {len(test_data[test_data['League'] == league])} matches")
        
        train_seasons = sorted(train_data['Season'].unique())
        test_seasons = sorted(test_data['Season'].unique())
        logger.info(f"Training seasons: {train_seasons}")
        logger.info(f"Test seasons: {test_seasons}")
        
        return train_data, test_data
    
    def _validate_data(self, df: pd.DataFrame) -> bool:
        """Validate DataFrame quality."""
        missing_cols = self.config.required_columns - set(df.columns)
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            return False
        
        df['Match'] = df['Date'].astype(str) + '_' + df['HomeTeam'] + '_' + df['AwayTeam']
        duplicates = df['Match'].duplicated()
        if duplicates.any():
            logger.warning(f"Found {duplicates.sum()} duplicate matches - will be handled during preprocessing")
        
        odds_columns = [col for col in df.columns if 'B365' in col]
        invalid_odds_count = (df[odds_columns] <= 1.0).sum().sum()
        total_odds = len(df) * len(odds_columns)
        invalid_odds_percentage = (invalid_odds_count / total_odds) * 100
        
        if invalid_odds_percentage > 5:
            logger.warning(f"Too many invalid odds ({invalid_odds_percentage:.2f}% of all odds are <=1.0)")
            return False
        elif invalid_odds_count > 0:
            logger.warning(f"Found {invalid_odds_count} invalid odds (will be handled during preprocessing)")
        
        logger.info(f"Data range: {df['Date'].min()} to {df['Date'].max()}")
        logger.info(f"Number of unique teams: {len(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))}")
        
        return True
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess raw data."""
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        df['Season'] = df.apply(lambda x: 
            x['Date'].year if x['Date'].month >= 8 
            else x['Date'].year - 1, axis=1)
        
        season_counts = df.groupby('Season').size()
        expected_matches = 380
        for season, count in season_counts.items():
            if count > expected_matches:
                logger.warning(f"Season {season} has {count} matches (expected {expected_matches})")
            elif count < expected_matches and season != df['Season'].max():
                logger.warning(f"Season {season} has only {count} matches (expected {expected_matches})")
        
        logger.info(f"Number of matches per season: {season_counts}")
        
        df = self._fill_missing_values(df)
        df = self._add_derived_columns(df)
        
        return df
    
    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with appropriate defaults."""
        odds_columns = [col for col in df.columns if 'B365' in col]
        df[odds_columns] = df[odds_columns].fillna(100.0)
        
        for col in odds_columns:
            df.loc[df[col] <= 1.0, col] = 100.0
        
        stats_columns = ['FTHG', 'FTAG', 'HTHG', 'HTAG', 'HC', 'AC', 'HY', 'AY']
        df[stats_columns] = df[stats_columns].fillna(0)
        
        return df
    
    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived columns useful for feature engineering."""
        df['TotalGoals'] = df['FTHG'] + df['FTAG']
        df['HTTotalGoals'] = df['HTHG'] + df['HTAG']
        df['TotalCorners'] = df['HC'] + df['AC']
        df['TotalCards'] = df['HY'] + df['AY']
        df['GoalDiff'] = df['FTHG'] - df['FTAG']
        df['HTGoalDiff'] = df['HTHG'] - df['HTAG']
        
        return df 
    
    def _load_and_process_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load and process a single data file."""
        logger.info(f"\nProcessing file: {file_path}")
        
        try:
            excel_file = pd.ExcelFile(file_path)
            all_sheets_data = []
            
            for sheet_name in tqdm(excel_file.sheet_names, desc="Processing sheets", unit="sheet", leave=False):
                logger.info(f"Reading sheet: {sheet_name}")
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                if df.empty or 'Div' not in df.columns:
                    logger.warning(f"Skipping invalid sheet: {sheet_name}")
                    continue
                
                logger.info(f"Loaded {len(df)} rows from sheet {sheet_name}")
                
                if not self._validate_data(df):
                    logger.warning(f"Skipping invalid sheet: {sheet_name}")
                    continue
                
                df = self._preprocess_data(df)
                all_sheets_data.append(df)
            
            if not all_sheets_data:
                logger.warning(f"No valid data found in file: {file_path}")
                return None
            
            logger.info("Combining sheets data...")
            combined_df = pd.concat(all_sheets_data, ignore_index=True)
            
            logger.info(f"Successfully processed {len(combined_df)} total rows from {file_path}")
            logger.info(f"Data range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")
            logger.info(f"Number of unique teams: {len(set(combined_df['HomeTeam'].unique()) | set(combined_df['AwayTeam'].unique()))}")
            
            season_counts = combined_df.groupby('Season').size()
            logger.info(f"Number of matches per season:\n{season_counts}")
            
            return combined_df
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return None
    
    def _process_all_leagues(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process data for all leagues."""
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
        
        df = df[df['Div'].isin(league_mapping.keys())].copy()
        df['League'] = df['Div'].map(league_mapping)
        
        logger.info(f"Processed data for leagues: {sorted(df['League'].unique())}")
        logger.info(f"Number of matches per league:\n{df.groupby('League').size()}")
        
        return df