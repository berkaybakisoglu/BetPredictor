"""Feature engineering module for match prediction."""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from src.config.config import FeatureConfig

class FeatureEngineer:
    """Handles feature engineering for match prediction."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        
    def create_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Create all features for model training or prediction.
        
        Args:
            df: Preprocessed DataFrame with match data
            is_training: Whether this is for training (True) or prediction (False)
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Process each season separately to prevent look-ahead bias
        seasons = df['Season'].unique()
        processed_dfs = []
        
        for season in seasons:
            season_df = df[df['Season'] == season].copy()
            
            # Add team performance features
            season_df = self._add_team_performance_features(season_df)
            
            # Add head-to-head features
            season_df = self._add_h2h_features(season_df)
            
            # Add form features
            season_df = self._add_form_features(season_df)
            
            # Add market features
            season_df = self._add_market_features(season_df)
            
            processed_dfs.append(season_df)
        
        # Combine all processed data
        df = pd.concat(processed_dfs, ignore_index=True)
        
        # Sort chronologically
        df = df.sort_values('Date')
        
        # For training, remove first N matches of each team to ensure reliable features
        if is_training:
            reliable_mask = self._get_reliable_matches_mask(df)
            df = df[reliable_mask].copy()
        
        return df
    
    def _get_reliable_matches_mask(self, df: pd.DataFrame) -> pd.Series:
        """Get mask for matches with reliable features.
        
        Args:
            df: DataFrame with matches
            
        Returns:
            Boolean mask for reliable matches
        """
        mask = pd.Series(True, index=df.index)
        
        for team in pd.concat([df['HomeTeam'], df['AwayTeam']]).unique():
            team_matches = (df['HomeTeam'] == team) | (df['AwayTeam'] == team)
            team_match_indices = df[team_matches].index
            if len(team_match_indices) > self.config.min_matches_required:
                mask.loc[team_match_indices[:self.config.min_matches_required]] = False
        
        return mask
    
    def _add_team_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add team performance features.
        
        Args:
            df: DataFrame without team performance features
            
        Returns:
            DataFrame with team performance features
        """
        teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
        
        for team in teams:
            # Home performance
            home_matches = df[df['HomeTeam'] == team].sort_values('Date')
            
            # Calculate expanding averages for key metrics
            metrics = {
                'Goals_Scored': 'FTHG',
                'Goals_Conceded': 'FTAG',
                'Corners_For': 'HC',
                'Corners_Against': 'AC',
                'Cards': 'HY'
            }
            
            for name, col in metrics.items():
                df.loc[df['HomeTeam'] == team, f'Home_{name}_Avg'] = (
                    home_matches[col].expanding(min_periods=1)
                    .mean()
                    .shift(1)  # Shift to prevent look-ahead
                    .fillna(0)
                )
            
            # Away performance
            away_matches = df[df['AwayTeam'] == team].sort_values('Date')
            
            metrics = {
                'Goals_Scored': 'FTAG',
                'Goals_Conceded': 'FTHG',
                'Corners_For': 'AC',
                'Corners_Against': 'HC',
                'Cards': 'AY'
            }
            
            for name, col in metrics.items():
                df.loc[df['AwayTeam'] == team, f'Away_{name}_Avg'] = (
                    away_matches[col].expanding(min_periods=1)
                    .mean()
                    .shift(1)  # Shift to prevent look-ahead
                    .fillna(0)
                )
        
        return df
    
    def _add_h2h_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add head-to-head features.
        
        Args:
            df: DataFrame without H2H features
            
        Returns:
            DataFrame with H2H features
        """
        df = df.copy()
        
        def get_h2h_stats(row: pd.Series) -> Dict[str, float]:
            """Calculate H2H stats for a match."""
            # Get previous matches between these teams
            prev_matches = df[
                ((df['HomeTeam'] == row['HomeTeam']) & (df['AwayTeam'] == row['AwayTeam']) |
                 (df['HomeTeam'] == row['AwayTeam']) & (df['AwayTeam'] == row['HomeTeam'])) &
                (df['Date'] < row['Date'])  # Only use matches before current date
            ].sort_values('Date').tail(self.config.h2h_window)
            
            if len(prev_matches) == 0:
                return {
                    'H2H_Home_Wins': 0,
                    'H2H_Away_Wins': 0,
                    'H2H_Draws': 0,
                    'H2H_Avg_Goals': 0,
                    'H2H_Avg_Corners': 0
                }
            
            stats = {
                'H2H_Home_Wins': sum((prev_matches['HomeTeam'] == row['HomeTeam']) & 
                                   (prev_matches['FTR'] == 'H')) / len(prev_matches),
                'H2H_Away_Wins': sum((prev_matches['HomeTeam'] == row['AwayTeam']) & 
                                   (prev_matches['FTR'] == 'A')) / len(prev_matches),
                'H2H_Draws': sum(prev_matches['FTR'] == 'D') / len(prev_matches),
                'H2H_Avg_Goals': prev_matches['TotalGoals'].mean(),
                'H2H_Avg_Corners': prev_matches['TotalCorners'].mean()
            }
            
            return stats
        
        # Apply H2H calculations
        h2h_stats = df.apply(get_h2h_stats, axis=1)
        for col in h2h_stats.iloc[0].keys():
            df[col] = h2h_stats.apply(lambda x: x[col])
        
        return df
    
    def _add_form_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add form-based features with exponential decay.
        
        Args:
            df: DataFrame without form features
            
        Returns:
            DataFrame with form features
        """
        def calculate_form(results: List[str]) -> float:
            """Calculate form with exponential decay."""
            if not results:
                return 0.0
                
            weights = [np.exp(-self.config.decay_factor * i) for i in range(len(results))]
            points = [3 if r == 'W' else 1 if r == 'D' else 0 for r in results]
            
            return sum(w * p for w, p in zip(weights, points)) / sum(weights)
        
        teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
        
        for team in teams:
            # Home form
            home_matches = df[df['HomeTeam'] == team].sort_values('Date')
            home_results = ['W' if r == 'H' else 'D' if r == 'D' else 'L' 
                          for r in home_matches['FTR']]
            
            # Calculate form using only past matches
            df.loc[df['HomeTeam'] == team, 'Home_Form'] = [
                calculate_form(home_results[:i][-self.config.form_window:])
                for i in range(1, len(home_results) + 1)
            ]
            
            # Away form
            away_matches = df[df['AwayTeam'] == team].sort_values('Date')
            away_results = ['W' if r == 'A' else 'D' if r == 'D' else 'L' 
                          for r in away_matches['FTR']]
            
            # Calculate form using only past matches
            df.loc[df['AwayTeam'] == team, 'Away_Form'] = [
                calculate_form(away_results[:i][-self.config.form_window:])
                for i in range(1, len(away_results) + 1)
            ]
        
        return df
    
    def _add_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market-based features.
        
        Args:
            df: DataFrame without market features
            
        Returns:
            DataFrame with market features
        """
        # Calculate implied probabilities from odds
        df['Home_ImpliedProb'] = 1 / df['B365H']
        df['Draw_ImpliedProb'] = 1 / df['B365D']
        df['Away_ImpliedProb'] = 1 / df['B365A']
        
        # Calculate market overround
        df['Market_Overround'] = (
            df['Home_ImpliedProb'] + 
            df['Draw_ImpliedProb'] + 
            df['Away_ImpliedProb']
        )
        
        # Normalize probabilities
        df['Home_FairProb'] = df['Home_ImpliedProb'] / df['Market_Overround']
        df['Draw_FairProb'] = df['Draw_ImpliedProb'] / df['Market_Overround']
        df['Away_FairProb'] = df['Away_ImpliedProb'] / df['Market_Overround']
        
        return df 