import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from config.config import FeatureConfig
import logging

class FeatureEngineer:
    
    def __init__(self, config, test_seasons=None):
        self.config = config
        self.test_seasons = test_seasons if test_seasons else []
        self.logger = logging.getLogger(__name__)

    def create_features(self, df, historical_data=None, is_training=True):
        """
        Create features for the dataset.
        Args:
            df: DataFrame to create features for
            historical_data: Optional DataFrame containing historical data for test predictions
            is_training: Whether this is training data or test data
        """
        logger = logging.getLogger(__name__)
        logger.info("Creating features...")
        
        if len(df) == 0:
            logger.warning("Empty DataFrame provided, returning as is")
            return df
        
        df = df.copy()
        total_rows = len(df)
        logger.info(f"Processing {total_rows} rows of data")
        
        # Sort data chronologically
        df = df.sort_values('Date')
        
        # For test data, use historical data for feature calculation
        if not is_training and historical_data is not None:
            data_for_features = pd.concat([historical_data, df]).sort_values('Date')
            logger.info(f"Combined historical and test data shape: {data_for_features.shape}")
            logger.info(f"Test data seasons: {sorted(df['Season'].unique())}")
            logger.info(f"Historical data seasons: {sorted(historical_data['Season'].unique())}")
        else:
            data_for_features = df.copy()
        
        logger.info(f"Data for features shape: {data_for_features.shape}")
        logger.info(f"Seasons in data: {sorted(data_for_features['Season'].unique())}")
        
        # Process each season separately
        seasons = df['Season'].unique()  # Always use the seasons from the input DataFrame
        logger.info(f"Processing {len(seasons)} seasons: {sorted(seasons)}")
        
        for season in tqdm(seasons, desc="Processing seasons", unit="season"):
            logger.info(f"\nProcessing season {season}")
            
            # Get all data up to and including the current season for feature calculation
            if not is_training:
                season_data = data_for_features[data_for_features['Season'] <= season]
            else:
                season_data = data_for_features[data_for_features['Season'] == season]
            
            # Process each league separately within the season
            leagues = df[df['Season'] == season]['League'].unique()
            logger.info(f"Found {len(leagues)} leagues in season {season}: {sorted(leagues)}")
            
            for league in tqdm(leagues, desc=f"Processing leagues for {season}", unit="league", leave=False):
                league_mask = season_data['League'] == league
                league_data = season_data[league_mask].copy()
                
                logger.info(f"Processing league {league} for season {season}")
                
                # For test data, calculate stats using only past matches
                if not is_training:
                    league_data = league_data.sort_values('Date')
                    test_matches = df[(df['Season'] == season) & (df['League'] == league)]
                    logger.info(f"Processing {len(test_matches)} test matches for {league} in {season}")
                    
                    for idx, match in test_matches.iterrows():
                        past_matches = league_data[league_data['Date'] < match['Date']]
                        if len(past_matches) > 0:
                            stats = self._calculate_team_stats(past_matches, season)
                            home_team = match['HomeTeam']
                            away_team = match['AwayTeam']
                            
                            if home_team in stats:
                                for stat, value in stats[home_team].items():
                                    df.loc[idx, stat] = value
                            if away_team in stats:
                                for stat, value in stats[away_team].items():
                                    df.loc[idx, stat] = value
                        else:
                            logger.warning(f"No past matches found for {league} {season} match on {match['Date']}")
                else:
                    stats = self._calculate_team_stats(league_data, season)
                    for team in stats:
                        home_mask = (df['Season'] == season) & (df['League'] == league) & (df['HomeTeam'] == team)
                        away_mask = (df['Season'] == season) & (df['League'] == league) & (df['AwayTeam'] == team)
                        
                        if team in stats:
                            for stat, value in stats[team].items():
                                df.loc[home_mask, stat] = value
                                df.loc[away_mask, stat] = value
                
                logger.info(f"Completed processing league {league} for season {season}")
            logger.info(f"Completed processing season {season}")
        
        # Add head-to-head features
        logger.info("\nCalculating head-to-head statistics...")
        if not is_training:
            for idx, match in tqdm(df.iterrows(), desc="Adding H2H features", total=len(df)):
                past_matches = data_for_features[data_for_features['Date'] < match['Date']]
                if len(past_matches) > 0:
                    h2h_stats = self._calculate_h2h_stats(past_matches)
                    home_team = match['HomeTeam']
                    away_team = match['AwayTeam']
                    h2h_key = f"{home_team}_{away_team}"
                    
                    if h2h_key in h2h_stats:
                        stats = h2h_stats[h2h_key]
                        for stat, value in stats.items():
                            df.loc[idx, f'h2h_{stat}'] = value
                    else:
                        # Use historical data averages for default values
                        df.loc[idx, ['h2h_home_win_rate', 'h2h_draw_rate', 'h2h_away_win_rate']] = 1/3
                        df.loc[idx, ['h2h_home_goals', 'h2h_away_goals', 'h2h_total_goals']] = data_for_features[['FTHG', 'FTAG']].mean().mean()
        else:
            h2h_stats = self._calculate_h2h_stats(df)
            for idx, row in tqdm(df.iterrows(), desc="Adding H2H features", total=len(df)):
                home_team = row['HomeTeam']
                away_team = row['AwayTeam']
                h2h_key = f"{home_team}_{away_team}"
                
                if h2h_key in h2h_stats:
                    stats = h2h_stats[h2h_key]
                    for stat, value in stats.items():
                        df.loc[idx, f'h2h_{stat}'] = value
                else:
                    df.loc[idx, ['h2h_home_win_rate', 'h2h_draw_rate', 'h2h_away_win_rate']] = 1/3
                    df.loc[idx, ['h2h_home_goals', 'h2h_away_goals', 'h2h_total_goals']] = df[['FTHG', 'FTAG']].mean().mean()
        
        if self.config.use_market_features:
            logger.info("\nCreating market features...")
            with tqdm(total=8, desc="Market features", unit="step") as pbar:
                # Preserve original betting odds columns
                df['B365H_Original'] = df['B365H']
                df['B365D_Original'] = df['B365D']
                df['B365A_Original'] = df['B365A']
                
                df['Home_ImpliedProb'] = 1 / df['B365H']
                df['Draw_ImpliedProb'] = 1 / df['B365D']
                df['Away_ImpliedProb'] = 1 / df['B365A']
                pbar.update(1)
                
                # Only create over/under market features if the columns exist
                if 'BbMx>2.5' in df.columns and 'BbMx<2.5' in df.columns:
                    df['Over_ImpliedProb'] = 1 / df['BbMx>2.5']
                    df['Under_ImpliedProb'] = 1 / df['BbMx<2.5']
                    df['OU_Market_Overround'] = df['Over_ImpliedProb'] + df['Under_ImpliedProb']
                    df['OU_Market_Confidence'] = df[['Over_ImpliedProb', 'Under_ImpliedProb']].max(axis=1)
                    df['Over_Value'] = df['Over_ImpliedProb'] < 1/df['BbMx>2.5']
                    df['Under_Value'] = df['Under_ImpliedProb'] < 1/df['BbMx<2.5']
                    df['Over_Is_Favorite'] = df[['Over_ImpliedProb', 'Under_ImpliedProb']].idxmax(axis=1) == 'Over_ImpliedProb'
                pbar.update(3)
                
                df['Market_Overround'] = df['Home_ImpliedProb'] + df['Draw_ImpliedProb'] + df['Away_ImpliedProb']
                df['Market_Confidence'] = df[['Home_ImpliedProb', 'Draw_ImpliedProb', 'Away_ImpliedProb']].max(axis=1)
                pbar.update(1)
                
                df['Home_Value'] = df['Home_ImpliedProb'] < 1/df['B365H']
                df['Draw_Value'] = df['Draw_ImpliedProb'] < 1/df['B365D']
                df['Away_Value'] = df['Away_ImpliedProb'] < 1/df['B365A']
                pbar.update(2)
                
                df['Home_Is_Favorite'] = df[['Home_ImpliedProb', 'Draw_ImpliedProb', 'Away_ImpliedProb']].idxmax(axis=1) == 'Home_ImpliedProb'
                
                # Add favorite and underdog odds
                df['Favorite_Odds'] = df.apply(lambda x: min(x['B365H'], x['B365D'], x['B365A']), axis=1)
                df['Underdog_Odds'] = df.apply(lambda x: max(x['B365H'], x['B365D'], x['B365A']), axis=1)
                pbar.update(1)
            
            logger.info("Completed market features")
        
        if self.config.use_position_features:
            logger.info("\nCreating position features...")
            for season in tqdm(df['Season'].unique(), desc="Calculating league positions", unit="season"):
                logger.info(f"Processing season {season}")
                season_mask = df['Season'] == season
                season_data = df[season_mask].copy()
                
                # Process each league separately
                leagues = season_data['League'].unique()
                for league in tqdm(leagues, desc=f"Processing leagues for {season}", unit="league", leave=False):
                    logger.info(f"Calculating standings for league {league}")
                    league_mask = season_data['League'] == league
                    league_data = season_data[league_mask].sort_values('Date')
                    
                    # Initialize standings dictionary for this league
                    standings = {}
                    
                    # Process each match chronologically to update standings
                    for idx, match in tqdm(league_data.iterrows(), desc=f"Processing matches for {league}", unit="match", leave=False):
                        match_date = match['Date']
                        home_team = match['HomeTeam']
                        away_team = match['AwayTeam']
                        
                        # Initialize teams in standings if not present
                        for team in [home_team, away_team]:
                            if team not in standings:
                                standings[team] = {
                                    'points': 0,
                                    'matches_played': 0,
                                    'wins': 0,
                                    'draws': 0,
                                    'losses': 0,
                                    'goals_for': 0,
                                    'goals_against': 0
                                }
                        
                        # Record current standings before the match
                        home_mask = (season_mask) & (df['League'] == league) & (df['HomeTeam'] == home_team) & (df['Date'] == match_date)
                        away_mask = (season_mask) & (df['League'] == league) & (df['AwayTeam'] == away_team) & (df['Date'] == match_date)
                        
                        # Sort teams by points (desc), goal difference, goals scored
                        sorted_teams = sorted(
                            standings.items(),
                            key=lambda x: (-x[1]['points'], 
                                         -(x[1]['goals_for'] - x[1]['goals_against']),
                                         -x[1]['goals_for'])
                        )
                        
                        # Assign positions
                        positions = {team: pos+1 for pos, (team, _) in enumerate(sorted_teams)}
                        
                        # Update dataframe with current standings
                        if home_team in positions:
                            df.loc[home_mask, 'Home_League_Position'] = positions[home_team]
                            df.loc[home_mask, 'Home_Points'] = standings[home_team]['points']
                            df.loc[home_mask, 'Home_Matches_Played'] = standings[home_team]['matches_played']
                        
                        if away_team in positions:
                            df.loc[away_mask, 'Away_League_Position'] = positions[away_team]
                            df.loc[away_mask, 'Away_Points'] = standings[away_team]['points']
                            df.loc[away_mask, 'Away_Matches_Played'] = standings[away_team]['matches_played']
                        
                        if not is_training:
                            # For test data, only update standings after recording the current positions
                            if match['FTR'] == 'H':  # Home win
                                standings[home_team]['points'] += 3
                                standings[home_team]['wins'] += 1
                                standings[away_team]['losses'] += 1
                            elif match['FTR'] == 'A':  # Away win
                                standings[away_team]['points'] += 3
                                standings[away_team]['wins'] += 1
                                standings[home_team]['losses'] += 1
                            else:  # Draw
                                standings[home_team]['points'] += 1
                                standings[away_team]['points'] += 1
                                standings[home_team]['draws'] += 1
                                standings[away_team]['draws'] += 1
                            
                            # Update goals
                            standings[home_team]['goals_for'] += match['FTHG']
                            standings[home_team]['goals_against'] += match['FTAG']
                            standings[away_team]['goals_for'] += match['FTAG']
                            standings[away_team]['goals_against'] += match['FTHG']
                            
                            # Update matches played
                            standings[home_team]['matches_played'] += 1
                            standings[away_team]['matches_played'] += 1
                    
                    # Calculate final differences for this league
                    league_mask = (season_mask) & (df['League'] == league)
                    df.loc[league_mask, 'Position_Diff'] = df.loc[league_mask, 'Home_League_Position'] - df.loc[league_mask, 'Away_League_Position']
                    df.loc[league_mask, 'Points_Diff'] = df.loc[league_mask, 'Home_Points'] - df.loc[league_mask, 'Away_Points']
                    df.loc[league_mask, 'Form_Diff'] = df.loc[league_mask, 'Home_Matches_Played'] - df.loc[league_mask, 'Away_Matches_Played']
                    
                    logger.info(f"Completed standings calculation for league {league}")
                logger.info(f"Completed position features for season {season}")
        
        if self.config.use_corner_features:
            logger.info("\nCreating corner features...")
            with tqdm(total=4, desc="Corner features", unit="step") as pbar:
                if not is_training:
                    df = df.sort_values('Date')
                    for idx, match in df.iterrows():
                        past_matches = data_for_features[data_for_features['Date'] < match['Date']]
                        if len(past_matches) > 0:
                            home_team = match['HomeTeam']
                            away_team = match['AwayTeam']
                            
                            # Basic Corner Stats
                            home_matches = past_matches[past_matches['HomeTeam'] == home_team]
                            away_matches = past_matches[past_matches['AwayTeam'] == away_team]
                            
                            df.loc[idx, 'Home_Corners_For_Avg'] = home_matches['HC'].mean() if len(home_matches) > 0 else 0
                            df.loc[idx, 'Home_Corners_Against_Avg'] = home_matches['AC'].mean() if len(home_matches) > 0 else 0
                            df.loc[idx, 'Away_Corners_For_Avg'] = away_matches['AC'].mean() if len(away_matches) > 0 else 0
                            df.loc[idx, 'Away_Corners_Against_Avg'] = away_matches['HC'].mean() if len(away_matches) > 0 else 0
                            
                            # Recent Performance
                            for window in [3, 5]:
                                df.loc[idx, f'Home_Corners_For_Last{window}'] = home_matches['HC'].tail(window).mean() if len(home_matches) > 0 else 0
                                df.loc[idx, f'Home_Corners_Against_Last{window}'] = home_matches['AC'].tail(window).mean() if len(home_matches) > 0 else 0
                                df.loc[idx, f'Away_Corners_For_Last{window}'] = away_matches['AC'].tail(window).mean() if len(away_matches) > 0 else 0
                                df.loc[idx, f'Away_Corners_Against_Last{window}'] = away_matches['HC'].tail(window).mean() if len(away_matches) > 0 else 0
                            
                            # Derived Metrics
                            df.loc[idx, 'Home_Corner_Diff_Avg'] = df.loc[idx, 'Home_Corners_For_Avg'] - df.loc[idx, 'Home_Corners_Against_Avg']
                            df.loc[idx, 'Away_Corner_Diff_Avg'] = df.loc[idx, 'Away_Corners_For_Avg'] - df.loc[idx, 'Away_Corners_Against_Avg']
                            df.loc[idx, 'Home_Corner_Std'] = home_matches['HC'].std() if len(home_matches) > 1 else 0
                            df.loc[idx, 'Away_Corner_Std'] = away_matches['AC'].std() if len(away_matches) > 1 else 0
                else:
                    # Basic Corner Stats
                    df['Home_Corners_For_Avg'] = df.groupby('HomeTeam')['HC'].transform('mean')
                    df['Home_Corners_Against_Avg'] = df.groupby('HomeTeam')['AC'].transform('mean')
                    df['Away_Corners_For_Avg'] = df.groupby('AwayTeam')['AC'].transform('mean')
                    df['Away_Corners_Against_Avg'] = df.groupby('AwayTeam')['HC'].transform('mean')
                    pbar.update(1)
                    
                    # Recent Performance
                    for window in [3, 5]:
                        df[f'Home_Corners_For_Last{window}'] = df.groupby('HomeTeam')['HC'].transform(lambda x: x.rolling(window, min_periods=1).mean())
                        df[f'Home_Corners_Against_Last{window}'] = df.groupby('HomeTeam')['AC'].transform(lambda x: x.rolling(window, min_periods=1).mean())
                        df[f'Away_Corners_For_Last{window}'] = df.groupby('AwayTeam')['AC'].transform(lambda x: x.rolling(window, min_periods=1).mean())
                        df[f'Away_Corners_Against_Last{window}'] = df.groupby('AwayTeam')['HC'].transform(lambda x: x.rolling(window, min_periods=1).mean())
                    pbar.update(2)
                    
                    # Derived Metrics
                    df['Home_Corner_Diff_Avg'] = df['Home_Corners_For_Avg'] - df['Home_Corners_Against_Avg']
                    df['Away_Corner_Diff_Avg'] = df['Away_Corners_For_Avg'] - df['Away_Corners_Against_Avg']
                    df['Home_Corner_Std'] = df.groupby('HomeTeam')['HC'].transform('std')
                    df['Away_Corner_Std'] = df.groupby('AwayTeam')['AC'].transform('std')
                    pbar.update(1)
        
        if self.config.use_card_features:
            logger.info("\nCreating card features...")
            with tqdm(total=2, desc="Card features", unit="step") as pbar:
                if not is_training:
                    df = df.sort_values('Date')
                    for idx, match in df.iterrows():
                        past_matches = data_for_features[data_for_features['Date'] < match['Date']]
                        if len(past_matches) > 0:
                            home_team = match['HomeTeam']
                            away_team = match['AwayTeam']
                            
                            # Basic Card Stats
                            home_matches = past_matches[past_matches['HomeTeam'] == home_team]
                            away_matches = past_matches[past_matches['AwayTeam'] == away_team]
                            
                            df.loc[idx, 'Home_Cards_Avg'] = home_matches['HY'].mean() if len(home_matches) > 0 else 0
                            df.loc[idx, 'Away_Cards_Avg'] = away_matches['AY'].mean() if len(away_matches) > 0 else 0
                            
                            # Recent Performance
                            for window in [3, 5]:
                                df.loc[idx, f'Home_Cards_Last{window}'] = home_matches['HY'].tail(window).mean() if len(home_matches) > 0 else 0
                                df.loc[idx, f'Away_Cards_Last{window}'] = away_matches['AY'].tail(window).mean() if len(away_matches) > 0 else 0
                else:
                    # Basic Card Stats
                    df['Home_Cards_Avg'] = df.groupby('HomeTeam')['HY'].transform('mean')
                    df['Away_Cards_Avg'] = df.groupby('AwayTeam')['AY'].transform('mean')
                    pbar.update(1)
                    
                    # Recent Performance
                    for window in [3, 5]:
                        df[f'Home_Cards_Last{window}'] = df.groupby('HomeTeam')['HY'].transform(lambda x: x.rolling(window, min_periods=1).mean())
                        df[f'Away_Cards_Last{window}'] = df.groupby('AwayTeam')['AY'].transform(lambda x: x.rolling(window, min_periods=1).mean())
                    pbar.update(1)
        
        logger.info(f"\nFeature creation completed. Final shape: {df.shape}")
        return df
    
    def _calculate_h2h_stats(self, df):
        """Calculate head-to-head statistics for each team pair."""
        h2h_stats = {}
        
        for _, row in df.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            h2h_key = f"{home_team}_{away_team}"
            
            if h2h_key not in h2h_stats:
                # Get all matches between these teams
                h2h_matches = df[
                    ((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
                    ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))
                ]
                
                if len(h2h_matches) > 0:
                    home_wins = len(h2h_matches[h2h_matches['FTR'] == 'H'])
                    draws = len(h2h_matches[h2h_matches['FTR'] == 'D'])
                    away_wins = len(h2h_matches[h2h_matches['FTR'] == 'A'])
                    total_matches = len(h2h_matches)
                    
                    h2h_stats[h2h_key] = {
                        'home_win_rate': home_wins / total_matches,
                        'draw_rate': draws / total_matches,
                        'away_win_rate': away_wins / total_matches,
                        'home_goals': h2h_matches['FTHG'].mean(),
                        'away_goals': h2h_matches['FTAG'].mean(),
                        'total_goals': (h2h_matches['FTHG'] + h2h_matches['FTAG']).mean()
                    }
        
        return h2h_stats
    
    def _calculate_team_stats(self, df, season):
        """Calculate team statistics for both training and test data."""
        stats = {}
        df = df.sort_values('Date')
        
        for team in df['HomeTeam'].unique():
            home_matches = df[df['HomeTeam'] == team]
            away_matches = df[df['AwayTeam'] == team]
            
            if len(home_matches) == 0 and len(away_matches) == 0:
                continue
            
            # Initialize team stats with correct column names
            stats[team] = {
                'Home_Goals_Scored_Avg': 0,
                'Home_Goals_Conceded_Avg': 0,
                'Away_Goals_Scored_Avg': 0,
                'Away_Goals_Conceded_Avg': 0,
                'Home_Clean_Sheets': 0,
                'Away_Clean_Sheets': 0,
                'Home_Failed_Score': 0,
                'Away_Failed_Score': 0,
                'Goals_Diff_Home': 0,
                'Goals_Diff_Away': 0,
                'Home_Form': 0,
                'Away_Form': 0,
                'General_Form': 0,
                'Home_Win_Rate': 0,
                'Away_Win_Rate': 0
            }
            
            # Calculate home stats
            if len(home_matches) > 0:
                stats[team]['Home_Goals_Scored_Avg'] = home_matches['FTHG'].mean()
                stats[team]['Home_Goals_Conceded_Avg'] = home_matches['FTAG'].mean()
                stats[team]['Home_Clean_Sheets'] = (home_matches['FTAG'] == 0).mean()
                stats[team]['Home_Failed_Score'] = (home_matches['FTHG'] == 0).mean()
                stats[team]['Goals_Diff_Home'] = (home_matches['FTHG'] - home_matches['FTAG']).mean()
                stats[team]['Home_Win_Rate'] = (home_matches['FTR'] == 'H').mean()
                
                # Calculate home form (last 5 matches)
                home_form = []
                for _, match in home_matches.iterrows():
                    if match['FTR'] == 'H':
                        home_form.append(3)
                    elif match['FTR'] == 'D':
                        home_form.append(1)
                    else:
                        home_form.append(0)
                stats[team]['Home_Form'] = sum(home_form[-5:]) / min(5, len(home_form))
            
            # Calculate away stats
            if len(away_matches) > 0:
                stats[team]['Away_Goals_Scored_Avg'] = away_matches['FTAG'].mean()
                stats[team]['Away_Goals_Conceded_Avg'] = away_matches['FTHG'].mean()
                stats[team]['Away_Clean_Sheets'] = (away_matches['FTHG'] == 0).mean()
                stats[team]['Away_Failed_Score'] = (away_matches['FTAG'] == 0).mean()
                stats[team]['Goals_Diff_Away'] = (away_matches['FTAG'] - away_matches['FTHG']).mean()
                stats[team]['Away_Win_Rate'] = (away_matches['FTR'] == 'A').mean()
                
                # Calculate away form (last 5 matches)
                away_form = []
                for _, match in away_matches.iterrows():
                    if match['FTR'] == 'A':
                        away_form.append(3)
                    elif match['FTR'] == 'D':
                        away_form.append(1)
                    else:
                        away_form.append(0)
                stats[team]['Away_Form'] = sum(away_form[-5:]) / min(5, len(away_form))
            
            # Calculate general form
            all_matches = pd.concat([
                home_matches[['Date', 'FTR']].assign(is_home=True),
                away_matches[['Date', 'FTR']].assign(is_home=False)
            ]).sort_values('Date')
            
            general_form = []
            for _, match in all_matches.iterrows():
                if (match['is_home'] and match['FTR'] == 'H') or (not match['is_home'] and match['FTR'] == 'A'):
                    general_form.append(3)
                elif match['FTR'] == 'D':
                    general_form.append(1)
                else:
                    general_form.append(0)
            stats[team]['General_Form'] = sum(general_form[-5:]) / min(5, len(general_form))
        
        return stats

    def _calculate_form(self, matches, team_type, window=5):
        """Calculate form for a team based on their recent match results.
        
        Args:
            matches (pd.DataFrame): DataFrame containing the team's matches
            team_type (str): Either 'HomeTeam' or 'AwayTeam'
            window (int): Number of matches to consider for form calculation
            
        Returns:
            pd.Series: Form values for each match
        """
        if len(matches) == 0:
            return pd.Series([0])
            
        # Sort matches by date to ensure chronological order
        matches = matches.sort_values('Date')
        
        # Calculate points for each match
        if team_type == 'HomeTeam':
            points = matches['FTR'].map({'H': 3, 'D': 1, 'A': 0})
        else:  # AwayTeam
            points = matches['FTR'].map({'A': 3, 'D': 1, 'H': 0})
        
        # Calculate rolling average of points
        form = points.rolling(window=window, min_periods=1).mean()
        
        # Normalize form to be between 0 and 1
        form = form / 3
        
        return form