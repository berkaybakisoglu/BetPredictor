import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler
import logging
from src.utils.config import FEATURES, FORM_WINDOW, PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.processed_data_dir = PROCESSED_DATA_DIR
        self._setup_directories()

    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

    def calculate_team_form(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling form for each team."""
        logger.info("Calculating team form...")
        
        # Create a copy to avoid modifying the original
        df = df.copy()
        
        # Convert date to datetime and sort
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Initialize form columns
        df['home_team_form'] = 0.0
        df['away_team_form'] = 0.0
        
        # Calculate form for each team
        teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
        
        for team in teams:
            # Get all matches for this team
            home_matches = df[df['HomeTeam'] == team].copy()
            away_matches = df[df['AwayTeam'] == team].copy()
            
            # Calculate points
            home_matches['points'] = home_matches['FTR'].map({'H': 3, 'D': 1, 'A': 0})
            away_matches['points'] = away_matches['FTR'].map({'H': 0, 'D': 1, 'A': 3})
            
            # Calculate rolling form
            if not home_matches.empty:
                home_form = home_matches['points'].rolling(window=FORM_WINDOW, min_periods=1).mean()
                for idx, form in zip(home_matches.index, home_form):
                    df.loc[idx, 'home_team_form'] = form
            
            if not away_matches.empty:
                away_form = away_matches['points'].rolling(window=FORM_WINDOW, min_periods=1).mean()
                for idx, form in zip(away_matches.index, away_form):
                    df.loc[idx, 'away_team_form'] = form
        
        logger.info("Team form calculation completed")
        return df

    def add_goal_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling goal statistics."""
        logger.info("Calculating goal statistics...")
        
        df = df.copy()
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Initialize goal stats columns
        df['home_goals_scored_avg'] = 0.0
        df['home_goals_conceded_avg'] = 0.0
        df['away_goals_scored_avg'] = 0.0
        df['away_goals_conceded_avg'] = 0.0
        
        teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
        
        for team in teams:
            # Home games
            home_matches = df[df['HomeTeam'] == team].copy()
            if not home_matches.empty:
                goals_scored = home_matches['FTHG'].rolling(window=FORM_WINDOW, min_periods=1).mean()
                goals_conceded = home_matches['FTAG'].rolling(window=FORM_WINDOW, min_periods=1).mean()
                for idx, (scored, conceded) in zip(home_matches.index, zip(goals_scored, goals_conceded)):
                    df.loc[idx, 'home_goals_scored_avg'] = scored
                    df.loc[idx, 'home_goals_conceded_avg'] = conceded
            
            # Away games
            away_matches = df[df['AwayTeam'] == team].copy()
            if not away_matches.empty:
                goals_scored = away_matches['FTAG'].rolling(window=FORM_WINDOW, min_periods=1).mean()
                goals_conceded = away_matches['FTHG'].rolling(window=FORM_WINDOW, min_periods=1).mean()
                for idx, (scored, conceded) in zip(away_matches.index, zip(goals_scored, goals_conceded)):
                    df.loc[idx, 'away_goals_scored_avg'] = scored
                    df.loc[idx, 'away_goals_conceded_avg'] = conceded
        
        logger.info("Goal statistics calculation completed")
        return df

    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variable from full time result."""
        logger.info("Creating target variable...")
        df = df.copy()
        df['match_outcome'] = df['FTR'].map({'H': 1, 'D': 0, 'A': -1})
        return df

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Main preprocessing pipeline."""
        try:
            logger.info("Starting preprocessing pipeline...")
            
            # Add form features
            df = self.calculate_team_form(df)
            logger.info("Added team form features")
            
            # Add goal statistics
            df = self.add_goal_stats(df)
            logger.info("Added goal statistics")
            
            # Create target variable
            df = self.create_target_variable(df)
            logger.info("Created target variable")
            
            # Select features from config
            X = df[FEATURES].copy()
            y = df['match_outcome']
            
            # Handle missing values
            X = X.fillna(X.mean())
            logger.info("Handled missing values")
            
            # Scale features
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            logger.info("Scaled features")
            
            # Save processed data
            output_path = self.processed_data_dir / "processed_data.csv"
            processed_df = pd.concat([X_scaled, y], axis=1)
            processed_df.to_csv(output_path, index=False)
            logger.info(f"Saved processed data to {output_path}")
            
            # Print some statistics
            logger.info(f"\nFeature statistics:")
            logger.info(f"Number of features: {X_scaled.shape[1]}")
            logger.info(f"Number of samples: {X_scaled.shape[0]}")
            logger.info(f"\nClass distribution:")
            logger.info(y.value_counts(normalize=True))
            
            return X_scaled, y
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise