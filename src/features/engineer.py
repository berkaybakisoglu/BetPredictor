"""Feature engineering module for match prediction."""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
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
        
        for season in tqdm(seasons, desc="Processing seasons", unit="season"):
            season_df = df[df['Season'] == season].copy()
            
            # Add standings features first
            season_df = self._add_standings_features(season_df)
            
            # Add other features
            season_df = self._add_team_performance_features(season_df)
            season_df = self._add_h2h_features(season_df)
            season_df = self._add_form_features(season_df)
            season_df = self._add_market_features(season_df)
            
            processed_dfs.append(season_df)
        
        # Combine all processed data
        df = pd.concat(processed_dfs, ignore_index=True)
        
        # Sort chronologically
        df = df.sort_values('Date')
        
        return df
    
    def _add_team_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add team performance features."""
        teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
        
        # Define multiple windows for rolling averages
        windows = [3, 5, 10]
        
        for team in tqdm(teams, desc="Processing team performance", unit="team", leave=False):
            # Home performance
            home_matches = df[df['HomeTeam'] == team].sort_values('Date')
            
            # Calculate expanding and rolling averages for key metrics
            metrics = {
                'Goals_Scored': 'FTHG',
                'Goals_Conceded': 'FTAG',
                'Corners_For': 'HC',
                'Corners_Against': 'AC',
                'Cards': 'HY'
            }
            
            for name, col in metrics.items():
                # Expanding average (all-time)
                df.loc[df['HomeTeam'] == team, f'Home_{name}_Avg'] = (
                    home_matches[col].expanding(min_periods=1)
                    .mean()
                    .shift(1)  # Shift to prevent look-ahead
                    .fillna(0)
                )
                
                # Rolling averages for different windows
                for window in windows:
                    df.loc[df['HomeTeam'] == team, f'Home_{name}_Last{window}'] = (
                        home_matches[col].rolling(window, min_periods=1)
                        .mean()
                        .shift(1)  # Shift to prevent look-ahead
                        .fillna(0)
                    )
            
            # Calculate trend indicators (comparing recent to longer-term performance)
            df.loc[df['HomeTeam'] == team, 'Home_Form_Trend'] = (
                df.loc[df['HomeTeam'] == team, 'Home_Goals_Scored_Last3'] -
                df.loc[df['HomeTeam'] == team, 'Home_Goals_Scored_Last10']
            )
            
            # Away performance (similar calculations)
            away_matches = df[df['AwayTeam'] == team].sort_values('Date')
            
            metrics = {
                'Goals_Scored': 'FTAG',
                'Goals_Conceded': 'FTHG',
                'Corners_For': 'AC',
                'Corners_Against': 'HC',
                'Cards': 'AY'
            }
            
            for name, col in metrics.items():
                # Expanding average
                df.loc[df['AwayTeam'] == team, f'Away_{name}_Avg'] = (
                    away_matches[col].expanding(min_periods=1)
                    .mean()
                    .shift(1)
                    .fillna(0)
                )
                
                # Rolling averages
                for window in windows:
                    df.loc[df['AwayTeam'] == team, f'Away_{name}_Last{window}'] = (
                        away_matches[col].rolling(window, min_periods=1)
                        .mean()
                        .shift(1)
                        .fillna(0)
                    )
            
            # Calculate trend indicators
            df.loc[df['AwayTeam'] == team, 'Away_Form_Trend'] = (
                df.loc[df['AwayTeam'] == team, 'Away_Goals_Scored_Last3'] -
                df.loc[df['AwayTeam'] == team, 'Away_Goals_Scored_Last10']
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
        """Add form-based features with exponential decay."""
        teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
        
        for team in tqdm(teams, desc="Processing team form", unit="team", leave=False):
            # Home form
            home_matches = df[df['HomeTeam'] == team].sort_values('Date')
            home_results = ['W' if r == 'H' else 'D' if r == 'D' else 'L' 
                          for r in home_matches['FTR']]
            
            # Away form
            away_matches = df[df['AwayTeam'] == team].sort_values('Date')
            away_results = ['W' if r == 'A' else 'D' if r == 'D' else 'L' 
                          for r in away_matches['FTR']]
            
            # General form (combining home and away matches chronologically)
            all_matches = pd.concat([
                home_matches[['Date', 'FTR']].assign(IsHome=True),
                away_matches[['Date', 'FTR']].assign(IsHome=False)
            ]).sort_values('Date')
            
            general_results = []
            for _, match in all_matches.iterrows():
                if match['IsHome']:
                    result = 'W' if match['FTR'] == 'H' else 'D' if match['FTR'] == 'D' else 'L'
                else:
                    result = 'W' if match['FTR'] == 'A' else 'D' if match['FTR'] == 'D' else 'L'
                general_results.append(result)
            
            # Calculate form features
            df.loc[df['HomeTeam'] == team, 'Home_Form'] = [
                self._calculate_form(home_results[:i][-self.config.form_window:])
                for i in range(1, len(home_results) + 1)
            ]
            
            df.loc[df['AwayTeam'] == team, 'Away_Form'] = [
                self._calculate_form(away_results[:i][-self.config.form_window:])
                for i in range(1, len(away_results) + 1)
            ]
            
            # Calculate general form for both home and away matches
            general_form = [
                self._calculate_form(general_results[:i][-self.config.form_window:])
                for i in range(1, len(general_results) + 1)
            ]
            
            # Assign general form to both home and away matches
            home_indices = df[df['HomeTeam'] == team].index
            away_indices = df[df['AwayTeam'] == team].index
            
            for idx, form in zip(home_indices, general_form[:len(home_indices)]):
                df.loc[idx, 'General_Form'] = form
            for idx, form in zip(away_indices, general_form[:len(away_indices)]):
                df.loc[idx, 'General_Form'] = form
        
        return df
    
    def _calculate_form(self, results: List[str]) -> float:
        """Calculate form with exponential decay weights."""
        if not results:
            return 0.0
        
        # Exponential decay weights
        weights = [np.exp(-self.config.decay_factor * i) for i in range(len(results))]
        points = [3 if r == 'W' else 1 if r == 'D' else 0 for r in results]
        return sum(w * p for w, p in zip(weights, points)) / sum(weights)
    
    def _add_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market-based features."""
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
        
        # Calculate odds ratios and value indicators
        df['Home_Value'] = df['Home_FairProb'] * df['B365H']
        df['Draw_Value'] = df['Draw_FairProb'] * df['B365D']
        df['Away_Value'] = df['Away_FairProb'] * df['B365A']
        
        # Add market confidence indicators
        df['Market_Confidence'] = 1 - (df['Market_Overround'] - 1)  # Lower overround indicates higher confidence
        
        # Calculate favorite/underdog status
        df['Home_Is_Favorite'] = (df['B365H'] < df['B365A']).astype(int)
        df['Favorite_Odds'] = df.apply(lambda x: min(x['B365H'], x['B365A']), axis=1)
        df['Underdog_Odds'] = df.apply(lambda x: max(x['B365H'], x['B365A']), axis=1)
        
        return df
    
    def _calculate_standings(self, df: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
        """Calculate league standings up to a given date.
        
        Args:
            df: DataFrame with matches
            date: Date up to which to calculate standings
            
        Returns:
            DataFrame with standings
        """
        # Filter matches up to the given date
        past_matches = df[df['Date'] < date].copy()
        
        # Initialize standings dictionary
        standings = {}
        
        # Process matches chronologically
        for _, match in past_matches.sort_values('Date').iterrows():
            # Initialize teams if not in standings
            for team in [match['HomeTeam'], match['AwayTeam']]:
                if team not in standings:
                    standings[team] = {
                        'points': 0,
                        'played': 0,
                        'won': 0,
                        'drawn': 0,
                        'lost': 0,
                        'goals_for': 0,
                        'goals_against': 0,
                        'goal_diff': 0
                    }
            
            # Update home team stats
            standings[match['HomeTeam']]['played'] += 1
            standings[match['HomeTeam']]['goals_for'] += match['FTHG']
            standings[match['HomeTeam']]['goals_against'] += match['FTAG']
            standings[match['HomeTeam']]['goal_diff'] = (
                standings[match['HomeTeam']]['goals_for'] - 
                standings[match['HomeTeam']]['goals_against']
            )
            
            # Update away team stats
            standings[match['AwayTeam']]['played'] += 1
            standings[match['AwayTeam']]['goals_for'] += match['FTAG']
            standings[match['AwayTeam']]['goals_against'] += match['FTHG']
            standings[match['AwayTeam']]['goal_diff'] = (
                standings[match['AwayTeam']]['goals_for'] - 
                standings[match['AwayTeam']]['goals_against']
            )
            
            # Update points and results
            if match['FTR'] == 'H':
                standings[match['HomeTeam']]['points'] += 3
                standings[match['HomeTeam']]['won'] += 1
                standings[match['AwayTeam']]['lost'] += 1
            elif match['FTR'] == 'A':
                standings[match['AwayTeam']]['points'] += 3
                standings[match['AwayTeam']]['won'] += 1
                standings[match['HomeTeam']]['lost'] += 1
            else:  # Draw
                standings[match['HomeTeam']]['points'] += 1
                standings[match['AwayTeam']]['points'] += 1
                standings[match['HomeTeam']]['drawn'] += 1
                standings[match['AwayTeam']]['drawn'] += 1
        
        # Convert to DataFrame and sort by points and goal difference
        standings_df = pd.DataFrame.from_dict(standings, orient='index')
        
        # Ensure all required columns exist
        required_columns = ['points', 'goal_diff', 'goals_for']
        for col in required_columns:
            if col not in standings_df.columns:
                standings_df[col] = 0
        
        # Sort by points, goal difference, and goals for
        standings_df = standings_df.sort_values(
            by=['points', 'goal_diff', 'goals_for'],
            ascending=[False, False, False]
        )
        
        # Add position column
        standings_df['position'] = range(1, len(standings_df) + 1)
        
        return standings_df

    def _add_standings_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add standings-based features."""
        df = df.copy()
        
        # Process each match with progress bar
        for idx, row in tqdm(df.iterrows(), desc="Processing standings", total=len(df), unit="match", leave=False):
            # Calculate standings before this match
            standings = self._calculate_standings(df, row['Date'])
            
            # Add features for both teams
            for team, prefix in [(row['HomeTeam'], 'Home'), (row['AwayTeam'], 'Away')]:
                if team in standings.index:
                    team_stats = standings.loc[team]
                    df.loc[idx, f'{prefix}_Position'] = team_stats['position']
                    df.loc[idx, f'{prefix}_Points'] = team_stats['points']
                    df.loc[idx, f'{prefix}_Played'] = team_stats['played']
                    df.loc[idx, f'{prefix}_WinRatio'] = team_stats['won'] / team_stats['played'] if team_stats['played'] > 0 else 0
                    df.loc[idx, f'{prefix}_GoalDiff'] = team_stats['goal_diff']
                    df.loc[idx, f'{prefix}_AvgGoalsFor'] = team_stats['goals_for'] / team_stats['played'] if team_stats['played'] > 0 else 0
                    df.loc[idx, f'{prefix}_AvgGoalsAgainst'] = team_stats['goals_against'] / team_stats['played'] if team_stats['played'] > 0 else 0
                else:
                    # Fill with defaults for new teams
                    df.loc[idx, [
                        f'{prefix}_Position',
                        f'{prefix}_Points',
                        f'{prefix}_Played',
                        f'{prefix}_WinRatio',
                        f'{prefix}_GoalDiff',
                        f'{prefix}_AvgGoalsFor',
                        f'{prefix}_AvgGoalsAgainst'
                    ]] = [0, 0, 0, 0, 0, 0, 0]
            
            # Add position difference and points difference
            df.loc[idx, 'Position_Diff'] = df.loc[idx, 'Away_Position'] - df.loc[idx, 'Home_Position']
            df.loc[idx, 'Points_Diff'] = df.loc[idx, 'Away_Points'] - df.loc[idx, 'Home_Points']
        
        return df