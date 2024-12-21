"""Feature engineering module for match prediction."""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from src.config.config import FeatureConfig
import logging

class FeatureEngineer:
    """Handles feature engineering for match prediction."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        
    def create_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Create all features for model training or prediction."""
        logger = logging.getLogger(__name__)
        
        # Initial chronological sort
        logger.info("Initial chronological sort...")
        df = df.sort_values(['Date', 'HomeTeam', 'AwayTeam']).reset_index(drop=True)
        
        # Process each season separately
        seasons = df['Season'].unique()
        processed_dfs = []
        
        for season in tqdm(seasons, desc="Processing seasons", unit="season"):
            logger.info(f"Processing season: {season}")
            season_df = df[df['Season'] == season].copy()
            
            # Initialize required features
            logger.info("Initializing required features...")
            required_features = {
                'Goals_Diff_Home': 0.0, 
                'Goals_Diff_Away': 0.0,
                'Home_Points': 0.0, 
                'Away_Points': 0.0,
                'Position_Diff': 0.0, 
                'Points_Diff': 0.0,
                'Form_Diff': 0.0, 
                'Home_Form': 0.0, 
                'Away_Form': 0.0,
                'General_Form': 0.0,
                # Corner-related features
                'Home_Corners_For_Last5_Avg': 0.0,
                'Home_Corners_Against_Last5_Avg': 0.0,
                'Away_Corners_For_Last5_Avg': 0.0,
                'Away_Corners_Against_Last5_Avg': 0.0,
                'Home_Corner_Diff_Avg': 0.0,
                'Away_Corner_Diff_Avg': 0.0,
                'Home_Home_Corner_Avg': 0.0,
                'Away_Away_Corner_Avg': 0.0,
                'Home_Corner_Std': 0.0,
                'Away_Corner_Std': 0.0,
                'Home_Scoring_First_Ratio': 0.0,
                'Away_Scoring_First_Ratio': 0.0,
                'Home_Clean_Sheets_Ratio': 0.0,
                'Away_Clean_Sheets_Ratio': 0.0
            }
            
            for feature, default_value in required_features.items():
                if feature not in season_df.columns:
                    season_df[feature] = default_value
            
            # Add features in correct order
            logger.info("Adding standings features...")
            season_df = self._add_standings_features(season_df)
            
            logger.info("Adding team performance features...")
            season_df = self._add_team_performance_features(season_df)
            
            logger.info("Adding H2H features...")
            season_df = self._add_h2h_features(season_df)
            
            logger.info("Adding form features...")
            season_df = self._add_form_features(season_df)
            
            logger.info("Adding market features...")
            season_df = self._add_market_features(season_df)
            
            logger.info("Adding corner features...")
            season_df = self._add_corner_features(season_df)
            
            # Verify all features are present
            logger.info("Verifying features...")
            missing_features = set(required_features.keys()) - set(season_df.columns)
            if missing_features:
                logger.error(f"Missing features after processing: {missing_features}")
                raise ValueError(f"Failed to generate required features: {missing_features}")
            
            processed_dfs.append(season_df)
            logger.info(f"Completed processing season: {season}")
        
        # Combine all processed data
        logger.info("Combining processed seasons...")
        df = pd.concat(processed_dfs, ignore_index=True)
        
        # Final sort
        df = df.sort_values(['Date', 'HomeTeam', 'AwayTeam']).reset_index(drop=True)
        
        return df
    
    def _add_team_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add team performance features."""
        logger = logging.getLogger(__name__)
        teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
        
        # Define multiple windows for rolling averages
        windows = [3, 5, 10]
        
        for team in tqdm(teams, desc="Processing team performance", unit="team", leave=False):
            # Home performance
            home_matches = df[df['HomeTeam'] == team].sort_values('Date')
            away_matches = df[df['AwayTeam'] == team].sort_values('Date')
            
            if not home_matches.empty:
                # Days since last game
                df.loc[df['HomeTeam'] == team, 'Days_Since_Last_Game_Home'] = (
                    home_matches['Date'].diff().dt.days.fillna(7.0)
                )
                
                # Clean sheets and scoring
                df.loc[df['HomeTeam'] == team, 'Home_Clean_Sheets'] = (
                    (home_matches['FTAG'] == 0).astype(float).rolling(window=5, min_periods=1).mean()
                ).shift(1).fillna(0.0)
                
                df.loc[df['HomeTeam'] == team, 'Home_Failed_Score'] = (
                    (home_matches['FTHG'] == 0).astype(float).rolling(window=5, min_periods=1).mean()
                ).shift(1).fillna(0.0)
                
                df.loc[df['HomeTeam'] == team, 'Home_Win_Rate'] = (
                    (home_matches['FTR'] == 'H').astype(float).rolling(window=5, min_periods=1).mean()
                ).shift(1).fillna(0.0)
                
                # Goals difference
                df.loc[df['HomeTeam'] == team, 'Goals_Diff_Home'] = (
                    home_matches['FTHG'].astype(float).rolling(window=5, min_periods=1).mean() -
                    home_matches['FTAG'].astype(float).rolling(window=5, min_periods=1).mean()
                ).shift(1).fillna(0.0)
                
                # Calculate other metrics
                for name, col in {
                    'Goals_Scored': 'FTHG',
                    'Goals_Conceded': 'FTAG',
                    'Corners_For': 'HC',
                    'Corners_Against': 'AC',
                    'Cards': 'HY'
                }.items():
                    # Expanding average
                    df.loc[df['HomeTeam'] == team, f'Home_{name}_Avg'] = (
                        home_matches[col].astype(float).expanding(min_periods=1)
                        .mean()
                        .shift(1)
                        .fillna(0.0)
                    )
                    
                    # Rolling averages
                    for window in windows:
                        df.loc[df['HomeTeam'] == team, f'Home_{name}_Last{window}'] = (
                            home_matches[col].astype(float).rolling(window, min_periods=1)
                            .mean()
                            .shift(1)
                            .fillna(0.0)
                        )
            
            if not away_matches.empty:
                # Days since last game
                df.loc[df['AwayTeam'] == team, 'Days_Since_Last_Game_Away'] = (
                    away_matches['Date'].diff().dt.days.fillna(7.0)
                )
                
                # Clean sheets and scoring
                df.loc[df['AwayTeam'] == team, 'Away_Clean_Sheets'] = (
                    (away_matches['FTHG'] == 0).astype(float).rolling(window=5, min_periods=1).mean()
                ).shift(1).fillna(0.0)
                
                df.loc[df['AwayTeam'] == team, 'Away_Failed_Score'] = (
                    (away_matches['FTAG'] == 0).astype(float).rolling(window=5, min_periods=1).mean()
                ).shift(1).fillna(0.0)
                
                df.loc[df['AwayTeam'] == team, 'Away_Win_Rate'] = (
                    (away_matches['FTR'] == 'A').astype(float).rolling(window=5, min_periods=1).mean()
                ).shift(1).fillna(0.0)
                
                # Goals difference
                df.loc[df['AwayTeam'] == team, 'Goals_Diff_Away'] = (
                    away_matches['FTAG'].astype(float).rolling(window=5, min_periods=1).mean() -
                    away_matches['FTHG'].astype(float).rolling(window=5, min_periods=1).mean()
                ).shift(1).fillna(0.0)
                
                # Calculate other metrics
                for name, col in {
                    'Goals_Scored': 'FTAG',
                    'Goals_Conceded': 'FTHG',
                    'Corners_For': 'AC',
                    'Corners_Against': 'HC',
                    'Cards': 'AY'
                }.items():
                    # Expanding average
                    df.loc[df['AwayTeam'] == team, f'Away_{name}_Avg'] = (
                        away_matches[col].astype(float).expanding(min_periods=1)
                        .mean()
                        .shift(1)
                        .fillna(0.0)
                    )
                    
                    # Rolling averages
                    for window in windows:
                        df.loc[df['AwayTeam'] == team, f'Away_{name}_Last{window}'] = (
                            away_matches[col].astype(float).rolling(window, min_periods=1)
                            .mean()
                            .shift(1)
                            .fillna(0.0)
                        )
        
        # Verify all required features are present
        required_features = [
            'Days_Since_Last_Game_Home', 'Days_Since_Last_Game_Away',
            'Home_Clean_Sheets', 'Away_Clean_Sheets',
            'Goals_Diff_Home', 'Goals_Diff_Away',
            'Home_Failed_Score', 'Away_Failed_Score',
            'Home_Win_Rate', 'Away_Win_Rate'
        ]
        
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            logger.error(f"Missing required features after processing: {missing_features}")
            for feature in missing_features:
                df[feature] = 0.0
                logger.warning(f"Initialized missing feature {feature} with zeros")
        
        return df
    
    def _add_h2h_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add head-to-head features.
        
        Args:
            df: DataFrame without H2H features
            
        Returns:
            DataFrame with H2H features
        """
        df = df.copy()
        
        # Calculate total goals and corners
        df['TotalGoals'] = df['FTHG'] + df['FTAG']
        df['TotalCorners'] = df['HC'] + df['AC']
        
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
    
    def _calculate_standings(self, df: pd.DataFrame, current_date: pd.Timestamp) -> pd.DataFrame:
        """Calculate league standings up to a given date."""
        # Get matches up to current date
        past_matches = df[df['Date'] < current_date].copy()
        
        if past_matches.empty:
            return pd.DataFrame(columns=['points', 'position', 'played', 'won', 'drawn', 'lost',
                                       'goals_for', 'goals_against', 'goal_diff', 'league'])
        
        # Initialize standings for each league
        standings_dict = {}
        
        # Process matches league by league
        for league in past_matches['League'].unique():
            league_matches = past_matches[past_matches['League'] == league]
            teams = pd.concat([league_matches['HomeTeam'], league_matches['AwayTeam']]).unique()
            
            # Initialize standings for this league
            league_standings = pd.DataFrame(index=teams)
            league_standings['points'] = 0
            league_standings['played'] = 0
            league_standings['won'] = 0
            league_standings['drawn'] = 0
            league_standings['lost'] = 0
            league_standings['goals_for'] = 0
            league_standings['goals_against'] = 0
            league_standings['goal_diff'] = 0
            league_standings['league'] = league  # Add league column
            
            # Calculate standings
            for _, match in league_matches.iterrows():
                home_team = match['HomeTeam']
                away_team = match['AwayTeam']
                
                # Update matches played
                league_standings.at[home_team, 'played'] += 1
                league_standings.at[away_team, 'played'] += 1
                
                # Update goals
                league_standings.at[home_team, 'goals_for'] += match['FTHG']
                league_standings.at[home_team, 'goals_against'] += match['FTAG']
                league_standings.at[away_team, 'goals_for'] += match['FTAG']
                league_standings.at[away_team, 'goals_against'] += match['FTHG']
                
                # Update results based on FTR (Full Time Result)
                if match['FTR'] == 'H':  # Home win
                    league_standings.at[home_team, 'won'] += 1
                    league_standings.at[away_team, 'lost'] += 1
                    league_standings.at[home_team, 'points'] += 3
                elif match['FTR'] == 'A':  # Away win
                    league_standings.at[away_team, 'won'] += 1
                    league_standings.at[home_team, 'lost'] += 1
                    league_standings.at[away_team, 'points'] += 3
                else:  # Draw
                    league_standings.at[home_team, 'drawn'] += 1
                    league_standings.at[away_team, 'drawn'] += 1
                    league_standings.at[home_team, 'points'] += 1
                    league_standings.at[away_team, 'points'] += 1
            
            # Calculate goal difference
            league_standings['goal_diff'] = league_standings['goals_for'] - league_standings['goals_against']
            
            # Sort and assign positions within league
            league_standings = league_standings.sort_values(['points', 'goal_diff', 'goals_for'], 
                                                          ascending=[False, False, False])
            league_standings['position'] = range(1, len(league_standings) + 1)
            
            # Create unique index by combining team and league
            league_standings.index = [f"{team}_{league}" for team in league_standings.index]
            
            # Store in dictionary
            standings_dict[league] = league_standings
        
        # Combine all league standings
        final_standings = pd.concat(standings_dict.values())
        
        # Create a lookup dictionary for quick team-league position access
        position_lookup = {}
        for idx in final_standings.index:
            team, league = idx.rsplit('_', 1)
            position_lookup[(team, league)] = {
                'position': final_standings.loc[idx, 'position'],
                'points': final_standings.loc[idx, 'points']
            }
        
        return position_lookup

    def _add_standings_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add standings-based features."""
        df = df.copy()
        logger = logging.getLogger(__name__)
        
        # Process each match with progress bar
        for idx, row in tqdm(df.iterrows(), desc="Processing standings", total=len(df), unit="match", leave=False):
            # Calculate standings before this match
            standings = self._calculate_standings(df, row['Date'])
            
            # Debug logging
            logger.debug(f"Processing match {idx}: {row['HomeTeam']} vs {row['AwayTeam']}")
            logger.debug(f"Standings keys: {list(standings.keys())}")
            
            # Add features for both teams
            for team, prefix in [(row['HomeTeam'], 'Home'), (row['AwayTeam'], 'Away')]:
                # Get team stats using team and league
                team_key = (team, row['League'])
                logger.debug(f"Looking up team {team} in league {row['League']}")
                
                if team_key in standings:
                    team_stats = standings[team_key]
                    logger.debug(f"Found stats for {team}: {team_stats}")
                    
                    # Ensure we're getting numeric values
                    position = float(team_stats['position'])
                    points = float(team_stats['points'])
                    
                    # Verify values are reasonable
                    if position <= 0 or position > 20:
                        logger.warning(f"Invalid position {position} for {team}, using default")
                        position = 20.0
                    
                    if points < 0:
                        logger.warning(f"Invalid points {points} for {team}, using default")
                        points = 0.0
                    
                    df.at[idx, f'{prefix}_League_Position'] = position
                    df.at[idx, f'{prefix}_Points'] = points
                    
                    logger.debug(f"Set {prefix}_League_Position={position}, {prefix}_Points={points}")
                else:
                    logger.warning(f"No standings found for {team} in {row['League']}")
                    # Fill with defaults for new teams
                    df.at[idx, f'{prefix}_League_Position'] = 20.0  # Default to bottom position
                    df.at[idx, f'{prefix}_Points'] = 0.0
            
            # Add position and points differences
            home_pos = df.at[idx, 'Home_League_Position']
            away_pos = df.at[idx, 'Away_League_Position']
            home_pts = df.at[idx, 'Home_Points']
            away_pts = df.at[idx, 'Away_Points']
            
            df.at[idx, 'Position_Diff'] = away_pos - home_pos
            df.at[idx, 'Points_Diff'] = away_pts - home_pts
            
            # Verify values were set correctly
            logger.debug(f"Final values for match {idx}:")
            logger.debug(f"Home_League_Position: {df.at[idx, 'Home_League_Position']}")
            logger.debug(f"Away_League_Position: {df.at[idx, 'Away_League_Position']}")
            logger.debug(f"Home_Points: {df.at[idx, 'Home_Points']}")
            logger.debug(f"Away_Points: {df.at[idx, 'Away_Points']}")
            logger.debug(f"Position_Diff: {df.at[idx, 'Position_Diff']}")
            logger.debug(f"Points_Diff: {df.at[idx, 'Points_Diff']}")
        
        # Verify final values
        for col in ['Home_League_Position', 'Away_League_Position', 'Home_Points', 'Away_Points', 'Position_Diff', 'Points_Diff']:
            stats = df[col].describe()
            logger.info(f"\nStats for {col}:")
            logger.info(f"  Mean: {stats['mean']:.4f}")
            logger.info(f"  Std: {stats['std']:.4f}")
            logger.info(f"  Min: {stats['min']:.4f}")
            logger.info(f"  Max: {stats['max']:.4f}")
            if df[col].isna().any():
                logger.error(f"Found {df[col].isna().sum()} NaN values in {col}")
        
        return df
    
    def _add_corner_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add corner-related features."""
        logger = logging.getLogger(__name__)
        df = df.copy()
        
        for team in df['HomeTeam'].unique():
            # Get team's matches
            home_matches = df[df['HomeTeam'] == team].sort_values('Date')
            away_matches = df[df['AwayTeam'] == team].sort_values('Date')
            
            # Basic corner averages
            df.loc[df['HomeTeam'] == team, 'Home_Corners_For_Avg'] = (
                home_matches['HC'].expanding().mean().shift(1).fillna(0)
            )
            df.loc[df['HomeTeam'] == team, 'Home_Corners_Against_Avg'] = (
                home_matches['AC'].expanding().mean().shift(1).fillna(0)
            )
            df.loc[df['AwayTeam'] == team, 'Away_Corners_For_Avg'] = (
                away_matches['AC'].expanding().mean().shift(1).fillna(0)
            )
            df.loc[df['AwayTeam'] == team, 'Away_Corners_Against_Avg'] = (
                away_matches['HC'].expanding().mean().shift(1).fillna(0)
            )
            
            # Last 3 and 5 matches corner averages
            for window in [3, 5]:
                df.loc[df['HomeTeam'] == team, f'Home_Corners_For_Last{window}'] = (
                    home_matches['HC'].rolling(window=window, min_periods=1).mean().shift(1).fillna(0)
                )
                df.loc[df['HomeTeam'] == team, f'Home_Corners_Against_Last{window}'] = (
                    home_matches['AC'].rolling(window=window, min_periods=1).mean().shift(1).fillna(0)
                )
                df.loc[df['AwayTeam'] == team, f'Away_Corners_For_Last{window}'] = (
                    away_matches['AC'].rolling(window=window, min_periods=1).mean().shift(1).fillna(0)
                )
                df.loc[df['AwayTeam'] == team, f'Away_Corners_Against_Last{window}'] = (
                    away_matches['HC'].rolling(window=window, min_periods=1).mean().shift(1).fillna(0)
                )
            
            # Corner differentials
            df.loc[df['HomeTeam'] == team, 'Home_Corner_Diff_Avg'] = (
                (home_matches['HC'] - home_matches['AC']).expanding().mean().shift(1).fillna(0)
            )
            df.loc[df['AwayTeam'] == team, 'Away_Corner_Diff_Avg'] = (
                (away_matches['AC'] - away_matches['HC']).expanding().mean().shift(1).fillna(0)
            )
            
            # Consistency metrics (standard deviation)
            df.loc[df['HomeTeam'] == team, 'Home_Corner_Std'] = (
                home_matches['HC'].expanding().std().shift(1).fillna(0)
            )
            df.loc[df['AwayTeam'] == team, 'Away_Corner_Std'] = (
                away_matches['AC'].expanding().std().shift(1).fillna(0)
            )
        
        # Calculate H2H corner averages if not already present
        if 'H2H_Avg_Corners' not in df.columns:
            df['H2H_Avg_Corners'] = self._calculate_h2h_stat(df, ['HC', 'AC'], 'sum')
        
        # Verify all corner features are present
        corner_features = [
            'Home_Corners_For_Avg', 'Home_Corners_Against_Avg',
            'Away_Corners_For_Avg', 'Away_Corners_Against_Avg',
            'Home_Corners_For_Last3', 'Home_Corners_Against_Last3',
            'Away_Corners_For_Last3', 'Away_Corners_Against_Last3',
            'Home_Corners_For_Last5', 'Home_Corners_Against_Last5',
            'Away_Corners_For_Last5', 'Away_Corners_Against_Last5',
            'Home_Corner_Diff_Avg', 'Away_Corner_Diff_Avg',
            'Home_Corner_Std', 'Away_Corner_Std',
            'H2H_Avg_Corners'
        ]
        
        missing_features = [f for f in corner_features if f not in df.columns]
        if missing_features:
            logger.error(f"Missing corner features: {missing_features}")
            raise ValueError(f"Failed to generate corner features: {missing_features}")
        
        return df