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

    def create_features(self, df, is_training=True):
        logger = logging.getLogger(__name__)
        logger.info("Creating features...")
        
        df = df.copy()
        total_rows = len(df)
        logger.info(f"Processing {total_rows} rows of data")
        
        if self.config.use_team_stats:
            logger.info("Calculating team statistics...")
            seasons = df['Season'].unique()
            logger.info(f"Processing {len(seasons)} seasons: {sorted(seasons)}")
            
            for season in tqdm(seasons, desc="Processing seasons", unit="season"):
                logger.info(f"\nProcessing season {season}")
                season_mask = df['Season'] == season
                season_data = df[season_mask]
                
                # Process each league separately within the season
                leagues = season_data['League'].unique()
                logger.info(f"Found {len(leagues)} leagues in season {season}: {sorted(leagues)}")
                
                for league in tqdm(leagues, desc=f"Processing leagues for {season}", unit="league", leave=False):
                    league_mask = season_data['League'] == league
                    league_data = season_data[league_mask]
                    
                    logger.info(f"Processing league {league} for season {season}")
                    stats = self._calculate_team_stats(league_data, season)
                    
                    # Get all teams that appear in either home or away positions for this league
                    league_teams = pd.concat([
                        league_data['HomeTeam'],
                        league_data['AwayTeam']
                    ]).unique()
                    
                    logger.info(f"Processing {len(league_teams)} teams for league {league} in season {season}")
                    for team in tqdm(league_teams, desc=f"Processing teams for {league}", unit="team", leave=False):
                        home_mask = (season_mask) & (df['League'] == league) & (df['HomeTeam'] == team)
                        away_mask = (season_mask) & (df['League'] == league) & (df['AwayTeam'] == team)
                        
                        if team in stats:  # Only try to access stats if we have them for this team
                            logger.debug(f"Updating stats for {team}")
                            df.loc[home_mask, 'Home_Goals_Scored_Avg'] = stats[team]['home_goals_scored_avg']
                            df.loc[home_mask, 'Home_Goals_Conceded_Avg'] = stats[team]['home_goals_conceded_avg']
                            df.loc[away_mask, 'Away_Goals_Scored_Avg'] = stats[team]['away_goals_scored_avg']
                            df.loc[away_mask, 'Away_Goals_Conceded_Avg'] = stats[team]['away_goals_conceded_avg']
                            
                            df.loc[home_mask, 'Home_Clean_Sheets'] = stats[team]['home_clean_sheets']
                            df.loc[away_mask, 'Away_Clean_Sheets'] = stats[team]['away_clean_sheets']
                            df.loc[home_mask, 'Home_Failed_To_Score'] = stats[team]['home_failed_to_score']
                            df.loc[away_mask, 'Away_Failed_To_Score'] = stats[team]['away_failed_to_score']
                            
                            df.loc[home_mask, 'Goals_Diff_Home'] = stats[team]['goals_diff_home']
                            df.loc[away_mask, 'Goals_Diff_Away'] = stats[team]['goals_diff_away']
                            
                            df.loc[home_mask, 'Home_Win_Rate'] = stats[team]['home_win_rate']
                            df.loc[away_mask, 'Away_Win_Rate'] = stats[team]['away_win_rate']
                            
                            df.loc[home_mask, 'Home_Form'] = stats[team]['home_form']
                            df.loc[away_mask, 'Away_Form'] = stats[team]['away_form']
                        else:
                            logger.warning(f"No stats found for team {team} in league {league}, season {season}")
                    logger.info(f"Completed processing league {league} for season {season}")
                logger.info(f"Completed processing season {season}")
        
        if self.config.use_market_features:
            logger.info("\nCreating market features...")
            with tqdm(total=8, desc="Market features", unit="step") as pbar:
                df['Home_ImpliedProb'] = 1 / df['B365H']
                df['Draw_ImpliedProb'] = 1 / df['B365D']
                df['Away_ImpliedProb'] = 1 / df['B365A']
                pbar.update(1)
                
                # Use BbMx odds for over/under markets (maximum odds across bookmakers)
                df['Over_ImpliedProb'] = 1 / df['BbMx>2.5']
                df['Under_ImpliedProb'] = 1 / df['BbMx<2.5']
                pbar.update(1)
                
                df['Market_Overround'] = df['Home_ImpliedProb'] + df['Draw_ImpliedProb'] + df['Away_ImpliedProb']
                df['OU_Market_Overround'] = df['Over_ImpliedProb'] + df['Under_ImpliedProb']
                pbar.update(1)
                
                df['Market_Confidence'] = df[['Home_ImpliedProb', 'Draw_ImpliedProb', 'Away_ImpliedProb']].max(axis=1)
                df['OU_Market_Confidence'] = df[['Over_ImpliedProb', 'Under_ImpliedProb']].max(axis=1)
                pbar.update(1)
                
                df['Home_Value'] = df['Home_ImpliedProb'] < 1/df['B365H']
                df['Draw_Value'] = df['Draw_ImpliedProb'] < 1/df['B365D']
                df['Away_Value'] = df['Away_ImpliedProb'] < 1/df['B365A']
                pbar.update(2)
                
                df['Over_Value'] = df['Over_ImpliedProb'] < 1/df['BbMx>2.5']
                df['Under_Value'] = df['Under_ImpliedProb'] < 1/df['BbMx<2.5']
                pbar.update(1)
                
                df['Home_Is_Favorite'] = df[['Home_ImpliedProb', 'Draw_ImpliedProb', 'Away_ImpliedProb']].idxmax(axis=1) == 'Home_ImpliedProb'
                df['Over_Is_Favorite'] = df[['Over_ImpliedProb', 'Under_ImpliedProb']].idxmax(axis=1) == 'Over_ImpliedProb'
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
                    for _, match in tqdm(league_data.iterrows(), desc=f"Processing matches for {league}", unit="match", leave=False):
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
                        
                        # Update standings based on match result
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
        h2h_df = pd.DataFrame()
        for _, match in df.iterrows():
            home_team, away_team = match['HomeTeam'], match['AwayTeam']
            h2h_matches = df[((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
                           ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))]
            
            if len(h2h_matches) > 0:
                stats = {
                    'h2h_home_win_rate': (h2h_matches['FTR'] == 'H').mean(),
                    'h2h_draw_rate': (h2h_matches['FTR'] == 'D').mean(),
                    'h2h_away_win_rate': (h2h_matches['FTR'] == 'A').mean(),
                    'h2h_home_goals': h2h_matches['FTHG'].mean(),
                    'h2h_away_goals': h2h_matches['FTAG'].mean(),
                    'h2h_total_goals': h2h_matches['FTHG'].sum() + h2h_matches['FTAG'].sum()
                }
            else:
                stats = {
                    'h2h_home_win_rate': 0.33,
                    'h2h_draw_rate': 0.33,
                    'h2h_away_win_rate': 0.34,
                    'h2h_home_goals': 0,
                    'h2h_away_goals': 0,
                    'h2h_total_goals': 0
                }
            
            h2h_df = pd.concat([h2h_df, pd.DataFrame([{**{'HomeTeam': home_team, 'AwayTeam': away_team}, **stats}])])
        
        return h2h_df
    
    def _create_corner_features(self, df):
        # Basic Corner Stats
        df['Home_Corners_For_Avg'] = df.groupby('HomeTeam')['HC'].transform('mean')
        df['Home_Corners_Against_Avg'] = df.groupby('HomeTeam')['AC'].transform('mean')
        df['Away_Corners_For_Avg'] = df.groupby('AwayTeam')['AC'].transform('mean')
        df['Away_Corners_Against_Avg'] = df.groupby('AwayTeam')['HC'].transform('mean')
        
        # Recent Performance
        for window in [3, 5]:
            df[f'Home_Corners_For_Last{window}'] = df.groupby('HomeTeam')['HC'].transform(lambda x: x.rolling(window, min_periods=1).mean())
            df[f'Home_Corners_Against_Last{window}'] = df.groupby('HomeTeam')['AC'].transform(lambda x: x.rolling(window, min_periods=1).mean())
            df[f'Away_Corners_For_Last{window}'] = df.groupby('AwayTeam')['AC'].transform(lambda x: x.rolling(window, min_periods=1).mean())
            df[f'Away_Corners_Against_Last{window}'] = df.groupby('AwayTeam')['HC'].transform(lambda x: x.rolling(window, min_periods=1).mean())
        
        # Derived Metrics
        df['Home_Corner_Diff_Avg'] = df['Home_Corners_For_Avg'] - df['Home_Corners_Against_Avg']
        df['Away_Corner_Diff_Avg'] = df['Away_Corners_For_Avg'] - df['Away_Corners_Against_Avg']
        df['Home_Corner_Std'] = df.groupby('HomeTeam')['HC'].transform('std')
        df['Away_Corner_Std'] = df.groupby('AwayTeam')['AC'].transform('std')
        
        return df
    
    def _create_card_features(self, df):
        # Basic Card Stats
        df['Home_Cards_Avg'] = df.groupby('HomeTeam')['HY'].transform('mean')
        df['Away_Cards_Avg'] = df.groupby('AwayTeam')['AY'].transform('mean')
        
        # Recent Performance
        for window in [3, 5]:
            df[f'Home_Cards_Last{window}'] = df.groupby('HomeTeam')['HY'].transform(lambda x: x.rolling(window, min_periods=1).mean())
            df[f'Away_Cards_Last{window}'] = df.groupby('AwayTeam')['AY'].transform(lambda x: x.rolling(window, min_periods=1).mean())
        
        return df
    
    def _calculate_form(self, df, team_col, window):
        # Convert match results to points first
        points = pd.Series(0, index=df.index)
        if team_col == 'HomeTeam':
            points.loc[df['FTR'] == 'H'] = 3  # Home win
            points.loc[df['FTR'] == 'D'] = 1  # Draw
        else:  # AwayTeam
            points.loc[df['FTR'] == 'A'] = 3  # Away win
            points.loc[df['FTR'] == 'D'] = 1  # Draw
        
        # Calculate rolling form (average points over last n matches)
        form = points.rolling(window=window, min_periods=1).mean()
        return form
    
    def _calculate_points(self, df, team_col):
        points = df.groupby(team_col)['FTR'].transform(
            lambda x: x.expanding().apply(
                lambda s: sum([(s == 'H').sum() * 3 + (s == 'D').sum()])
            )
        )
        return points
    
    def _calculate_league_standings(self, df):
        # Sort by date to ensure chronological order
        df = df.sort_values('Date')
        
        # Calculate points for each match
        df['Points'] = 0
        df.loc[df['FTR'] == 'H', 'Points'] = 3  # Home win
        df.loc[df['FTR'] == 'A', 'Points'] = 0  # Home loss
        df.loc[df['FTR'] == 'D', 'Points'] = 1  # Draw
        
        # Calculate cumulative points and position for each team in their league
        standings = {}
        for league in df['League'].unique():
            league_matches = df[df['League'] == league].copy()
            
            for date in league_matches['Date'].unique():
                date_matches = league_matches[league_matches['Date'] <= date]
                
                # Calculate home team points
                home_points = date_matches.groupby('HomeTeam')['Points'].sum()
                # Calculate away team points (3 for away win, 1 for draw)
                away_points = date_matches.groupby('AwayTeam').apply(
                    lambda x: (3 * (x['FTR'] == 'A')).sum() + (x['FTR'] == 'D').sum()
                )
                
                # Combine points
                total_points = pd.concat([home_points, away_points]).groupby(level=0).sum()
                
                # Calculate positions (higher points = lower position number)
                positions = total_points.rank(method='min', ascending=False)
                
                standings[(league, date)] = {
                    'points': total_points,
                    'positions': positions
                }
        
        # Add standings to dataframe
        df['Home_Points'] = 0
        df['Away_Points'] = 0
        df['Home_League_Position'] = 0
        df['Away_League_Position'] = 0
        
        for idx, row in df.iterrows():
            league, date = row['League'], row['Date']
            if (league, date) in standings:
                current_standings = standings[(league, date)]
                
                df.at[idx, 'Home_Points'] = current_standings['points'].get(row['HomeTeam'], 0)
                df.at[idx, 'Away_Points'] = current_standings['points'].get(row['AwayTeam'], 0)
                df.at[idx, 'Home_League_Position'] = current_standings['positions'].get(row['HomeTeam'], 0)
                df.at[idx, 'Away_League_Position'] = current_standings['positions'].get(row['AwayTeam'], 0)
        
        # Calculate differences
        df['Points_Diff'] = df['Home_Points'] - df['Away_Points']
        df['Position_Diff'] = df['Away_League_Position'] - df['Home_League_Position']
        
        return df
    
    def _calculate_team_stats(self, df, season):
        season_data = df[df['Season'] == season]
        stats = {}
        
        # Get all unique teams from both home and away positions
        all_teams = pd.concat([season_data['HomeTeam'], season_data['AwayTeam']]).unique()
        
        for team in all_teams:
            home_matches = season_data[season_data['HomeTeam'] == team]
            away_matches = season_data[season_data['AwayTeam'] == team]
            
            stats[team] = {
                'home_goals_scored_avg': home_matches['FTHG'].mean() if len(home_matches) > 0 else 0,
                'away_goals_scored_avg': away_matches['FTAG'].mean() if len(away_matches) > 0 else 0,
                'home_goals_conceded_avg': home_matches['FTAG'].mean() if len(home_matches) > 0 else 0,
                'away_goals_conceded_avg': away_matches['FTHG'].mean() if len(away_matches) > 0 else 0,
                'home_clean_sheets': ((home_matches['FTAG'] == 0).sum() / len(home_matches)) if len(home_matches) > 0 else 0,
                'away_clean_sheets': ((away_matches['FTHG'] == 0).sum() / len(away_matches)) if len(away_matches) > 0 else 0,
                'home_failed_to_score': ((home_matches['FTHG'] == 0).sum() / len(home_matches)) if len(home_matches) > 0 else 0,
                'away_failed_to_score': ((away_matches['FTAG'] == 0).sum() / len(away_matches)) if len(away_matches) > 0 else 0,
                'goals_diff_home': (home_matches['FTHG'] - home_matches['FTAG']).mean() if len(home_matches) > 0 else 0,
                'goals_diff_away': (away_matches['FTAG'] - away_matches['FTHG']).mean() if len(away_matches) > 0 else 0,
                'home_win_rate': ((home_matches['FTR'] == 'H').sum() / len(home_matches)) if len(home_matches) > 0 else 0,
                'away_win_rate': ((away_matches['FTR'] == 'A').sum() / len(away_matches)) if len(away_matches) > 0 else 0,
                'home_form': self._calculate_form(home_matches, 'HomeTeam', 5).iloc[-1] if len(home_matches) > 0 else 0,
                'away_form': self._calculate_form(away_matches, 'AwayTeam', 5).iloc[-1] if len(away_matches) > 0 else 0
            }
        
        return stats