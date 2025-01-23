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

    def _validate_no_leakage(self, match_date, used_data, context=""):
        """Validate that we're not using any future data."""
        if len(used_data) > 0:
            future_data = used_data[used_data['Date'] >= match_date]
            if len(future_data) > 0:
                future_matches = future_data[['Date', 'HomeTeam', 'AwayTeam']].head()
                raise ValueError(
                    f"DATA LEAKAGE DETECTED in {context}:\n"
                    f"Found {len(future_data)} matches from the future!\n"
                    f"Current match date: {match_date}\n"
                    f"Future matches used:\n{future_matches}"
                )
    
    def _validate_date_order(self, df, context=""):
        """Validate that dates are in chronological order."""
        if not df['Date'].is_monotonic_increasing:
            raise ValueError(
                f"DATA LEAKAGE RISK in {context}:\n"
                f"Dates are not in chronological order!\n"
                f"Please sort the data by date before processing."
            )

    def create_features(self, df, historical_data=None, is_training=True):
        """Create features using only historical data before each match."""
        logger = logging.getLogger(__name__)
        logger.info("Creating features...")
        
        if len(df) == 0:
            logger.warning("Empty DataFrame provided, returning as is")
            return df
        
        # Validate input data
        self._validate_date_order(df, "input data")
        if historical_data is not None:
            self._validate_date_order(historical_data, "historical data")
        
        df = df.copy()
        df = df.sort_values('Date')
        
        # Store original betting odds
        df['B365H_Original'] = df['B365H']
        df['B365D_Original'] = df['B365D']
        df['B365A_Original'] = df['B365A']
        
        # Combine with historical data if provided
        if historical_data is not None:
            data_for_features = pd.concat([historical_data, df]).sort_values('Date')
            logger.info(f"Combined historical and current data shape: {data_for_features.shape}")
        else:
            data_for_features = df.copy()
        
        logger.info(f"Processing {len(df)} matches...")
        
        # Process each match chronologically
        for idx, match in tqdm(df.iterrows(), desc="Processing matches", total=len(df)):
            # Get all matches before current match
            past_matches = data_for_features[data_for_features['Date'] < match['Date']]
            
            # Validate no data leakage in past_matches
            self._validate_no_leakage(match['Date'], past_matches, "initial past matches filter")
            
            if len(past_matches) > 0:
                # 1. League Standings
                standings = self._calculate_league_standings(match, past_matches)
                for key, value in standings.items():
                    df.loc[idx, key] = value
                
                # 2. Team Performance Stats
                team_stats = self._calculate_team_stats(match, past_matches)
                for key, value in team_stats.items():
                    df.loc[idx, key] = value
                
                # 3. Head-to-Head Stats
                h2h_stats = self._calculate_h2h_stats(match, past_matches)
                for key, value in h2h_stats.items():
                    df.loc[idx, f'h2h_{key}'] = value
                
                # 4. Market Features (if enabled)
                if self.config.use_market_features:
                    market_features = self._calculate_market_features(match)
                    for key, value in market_features.items():
                        df.loc[idx, key] = value
                
                # 5. Corner Features (if enabled)
                if self.config.use_corner_features:
                    corner_stats = self._calculate_corner_stats(match, past_matches)
                    for key, value in corner_stats.items():
                        df.loc[idx, key] = value
                
                # 6. Card Features (if enabled)
                if self.config.use_card_features:
                    card_stats = self._calculate_card_stats(match, past_matches)
                    for key, value in card_stats.items():
                        df.loc[idx, key] = value
                
                # 7. Additional Performance Stats
                additional_stats = self._calculate_additional_stats(match, past_matches)
                for key, value in additional_stats.items():
                    df.loc[idx, key] = value
            else:
                # For first matches, set default values
                self._set_default_values(df, idx)
        
        logger.info(f"Feature creation completed. Final shape: {df.shape}")
        return df

    def _calculate_league_standings(self, match, past_matches):
        """Calculate league standings up to the current match."""
        season = match['Season']
        league = match['League']
        match_date = match['Date']
        
        # Get only matches from same season/league
        season_matches = past_matches[
            (past_matches['Season'] == season) & 
            (past_matches['League'] == league)
        ].sort_values('Date')
        
        # Validate no future matches in season data
        self._validate_no_leakage(match_date, season_matches, "league standings calculation")
        
        standings = {}
        
        # Process each match to build standings
        for _, past_match in season_matches.iterrows():
            home_team = past_match['HomeTeam']
            away_team = past_match['AwayTeam']
            
            # Initialize teams if not in standings
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
            
            # Update standings based on result
            if past_match['FTR'] == 'H':
                standings[home_team]['points'] += 3
                standings[home_team]['wins'] += 1
                standings[away_team]['losses'] += 1
            elif past_match['FTR'] == 'A':
                standings[away_team]['points'] += 3
                standings[away_team]['wins'] += 1
                standings[home_team]['losses'] += 1
            else:  # Draw
                standings[home_team]['points'] += 1
                standings[away_team]['points'] += 1
                standings[home_team]['draws'] += 1
                standings[away_team]['draws'] += 1
            
            # Update goals
            standings[home_team]['goals_for'] += past_match['FTHG']
            standings[home_team]['goals_against'] += past_match['FTAG']
            standings[away_team]['goals_for'] += past_match['FTAG']
            standings[away_team]['goals_against'] += past_match['FTHG']
            
            standings[home_team]['matches_played'] += 1
            standings[away_team]['matches_played'] += 1
        
        # Sort teams by points, goal difference, goals scored
        sorted_teams = sorted(
            standings.items(),
            key=lambda x: (
                -x[1]['points'],
                -(x[1]['goals_for'] - x[1]['goals_against']),
                -x[1]['goals_for']
            )
        )
        
        positions = {team: pos+1 for pos, (team, _) in enumerate(sorted_teams)}
        
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        
        return {
            'Home_League_Position': positions.get(home_team, len(positions) + 1),
            'Away_League_Position': positions.get(away_team, len(positions) + 1),
            'Home_Points': standings.get(home_team, {'points': 0})['points'],
            'Away_Points': standings.get(away_team, {'points': 0})['points'],
            'Position_Diff': positions.get(home_team, len(positions) + 1) - positions.get(away_team, len(positions) + 1),
            'Points_Diff': standings.get(home_team, {'points': 0})['points'] - standings.get(away_team, {'points': 0})['points']
        }

    def _calculate_team_stats(self, match, past_matches):
        """Calculate team statistics using only past matches."""
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        match_date = match['Date']
        
        # Get team's previous matches
        home_past = past_matches[
            (past_matches['HomeTeam'] == home_team) | 
            (past_matches['AwayTeam'] == home_team)
        ].sort_values('Date')
        
        # Validate no future matches in home team data
        self._validate_no_leakage(match_date, home_past, "home team stats calculation")
        
        away_past = past_matches[
            (past_matches['HomeTeam'] == away_team) | 
            (past_matches['AwayTeam'] == away_team)
        ].sort_values('Date')
        
        # Validate no future matches in away team data
        self._validate_no_leakage(match_date, away_past, "away team stats calculation")
        
        stats = {}
        
        # Home team stats
        if len(home_past) > 0:
            home_goals_scored = []
            home_goals_conceded = []
            for _, past_match in home_past.iterrows():
                if past_match['HomeTeam'] == home_team:
                    home_goals_scored.append(past_match['FTHG'])
                    home_goals_conceded.append(past_match['FTAG'])
                else:
                    home_goals_scored.append(past_match['FTAG'])
                    home_goals_conceded.append(past_match['FTHG'])
            
            stats['Home_Goals_Scored_Avg'] = np.mean(home_goals_scored)
            stats['Home_Goals_Conceded_Avg'] = np.mean(home_goals_conceded)
            stats['Home_Form'] = self._calculate_form(home_past, home_team)
        else:
            stats['Home_Goals_Scored_Avg'] = 0
            stats['Home_Goals_Conceded_Avg'] = 0
            stats['Home_Form'] = 0
        
        # Away team stats
        if len(away_past) > 0:
            away_goals_scored = []
            away_goals_conceded = []
            for _, past_match in away_past.iterrows():
                if past_match['HomeTeam'] == away_team:
                    away_goals_scored.append(past_match['FTHG'])
                    away_goals_conceded.append(past_match['FTAG'])
                else:
                    away_goals_scored.append(past_match['FTAG'])
                    away_goals_conceded.append(past_match['FTHG'])
            
            stats['Away_Goals_Scored_Avg'] = np.mean(away_goals_scored)
            stats['Away_Goals_Conceded_Avg'] = np.mean(away_goals_conceded)
            stats['Away_Form'] = self._calculate_form(away_past, away_team)
        else:
            stats['Away_Goals_Scored_Avg'] = 0
            stats['Away_Goals_Conceded_Avg'] = 0
            stats['Away_Form'] = 0
        
        return stats

    def _calculate_h2h_stats(self, match, past_matches):
        """Calculate head-to-head statistics using only past matches."""
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        match_date = match['Date']
        
        h2h_matches = past_matches[
            ((past_matches['HomeTeam'] == home_team) & (past_matches['AwayTeam'] == away_team)) |
            ((past_matches['HomeTeam'] == away_team) & (past_matches['AwayTeam'] == home_team))
        ]
        
        # Validate no future matches in h2h data
        self._validate_no_leakage(match_date, h2h_matches, "head-to-head calculation")
        
        if len(h2h_matches) > 0:
            home_wins = len(h2h_matches[h2h_matches['FTR'] == 'H'])
            draws = len(h2h_matches[h2h_matches['FTR'] == 'D'])
            away_wins = len(h2h_matches[h2h_matches['FTR'] == 'A'])
            total_matches = len(h2h_matches)
            
            return {
                'home_win_rate': home_wins / total_matches,
                'draw_rate': draws / total_matches,
                'away_win_rate': away_wins / total_matches,
                'home_goals': h2h_matches['FTHG'].mean(),
                'away_goals': h2h_matches['FTAG'].mean(),
                'total_goals': (h2h_matches['FTHG'] + h2h_matches['FTAG']).mean()
            }
        else:
            return {
                'home_win_rate': 1/3,
                'draw_rate': 1/3,
                'away_win_rate': 1/3,
                'home_goals': 0,
                'away_goals': 0,
                'total_goals': 0
            }

    def _calculate_market_features(self, match):
        """Calculate betting market features."""
        features = {}
        
        # Basic implied probabilities
        features['Home_ImpliedProb'] = 1 / match['B365H']
        features['Draw_ImpliedProb'] = 1 / match['B365D']
        features['Away_ImpliedProb'] = 1 / match['B365A']
        
        # Market overround and confidence
        features['Market_Overround'] = sum([
            features['Home_ImpliedProb'],
            features['Draw_ImpliedProb'],
            features['Away_ImpliedProb']
        ])
        
        features['Market_Confidence'] = max([
            features['Home_ImpliedProb'],
            features['Draw_ImpliedProb'],
            features['Away_ImpliedProb']
        ])
        
        # Value bets
        features['Home_Value'] = features['Home_ImpliedProb'] < 1/match['B365H']
        features['Draw_Value'] = features['Draw_ImpliedProb'] < 1/match['B365D']
        features['Away_Value'] = features['Away_ImpliedProb'] < 1/match['B365A']
        
        # Favorite status
        features['Home_Is_Favorite'] = features['Home_ImpliedProb'] == features['Market_Confidence']
        
        return features

    def _calculate_corner_stats(self, match, past_matches):
        """Calculate corner statistics using only past matches."""
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        match_date = match['Date']
        
        home_past = past_matches[past_matches['HomeTeam'] == home_team]
        away_past = past_matches[past_matches['AwayTeam'] == away_team]
        
        # Validate no future matches
        self._validate_no_leakage(match_date, home_past, "home corner stats calculation")
        self._validate_no_leakage(match_date, away_past, "away corner stats calculation")
        
        stats = {}
        
        if len(home_past) > 0:
            stats['Home_Corners_For_Avg'] = home_past['HC'].mean()
            stats['Home_Corners_Against_Avg'] = home_past['AC'].mean()
            
            # Recent performance
            for window in [3, 5]:
                stats[f'Home_Corners_For_Last{window}'] = home_past['HC'].tail(window).mean()
                stats[f'Home_Corners_Against_Last{window}'] = home_past['AC'].tail(window).mean()
        else:
            stats['Home_Corners_For_Avg'] = 0
            stats['Home_Corners_Against_Avg'] = 0
            for window in [3, 5]:
                stats[f'Home_Corners_For_Last{window}'] = 0
                stats[f'Home_Corners_Against_Last{window}'] = 0
        
        if len(away_past) > 0:
            stats['Away_Corners_For_Avg'] = away_past['AC'].mean()
            stats['Away_Corners_Against_Avg'] = away_past['HC'].mean()
            
            for window in [3, 5]:
                stats[f'Away_Corners_For_Last{window}'] = away_past['AC'].tail(window).mean()
                stats[f'Away_Corners_Against_Last{window}'] = away_past['HC'].tail(window).mean()
        else:
            stats['Away_Corners_For_Avg'] = 0
            stats['Away_Corners_Against_Avg'] = 0
            for window in [3, 5]:
                stats[f'Away_Corners_For_Last{window}'] = 0
                stats[f'Away_Corners_Against_Last{window}'] = 0
        
        return stats

    def _calculate_card_stats(self, match, past_matches):
        """Calculate card statistics using only past matches."""
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        match_date = match['Date']
        
        home_past = past_matches[past_matches['HomeTeam'] == home_team]
        away_past = past_matches[past_matches['AwayTeam'] == away_team]
        
        # Validate no future matches
        self._validate_no_leakage(match_date, home_past, "home card stats calculation")
        self._validate_no_leakage(match_date, away_past, "away card stats calculation")
        
        stats = {}
        
        if len(home_past) > 0:
            stats['Home_Cards_Avg'] = home_past['HY'].mean()
            for window in [3, 5]:
                stats[f'Home_Cards_Last{window}'] = home_past['HY'].tail(window).mean()
        else:
            stats['Home_Cards_Avg'] = 0
            for window in [3, 5]:
                stats[f'Home_Cards_Last{window}'] = 0
        
        if len(away_past) > 0:
            stats['Away_Cards_Avg'] = away_past['AY'].mean()
            for window in [3, 5]:
                stats[f'Away_Cards_Last{window}'] = away_past['AY'].tail(window).mean()
        else:
            stats['Away_Cards_Avg'] = 0
            for window in [3, 5]:
                stats[f'Away_Cards_Last{window}'] = 0
        
        return stats

    def _calculate_form(self, matches, team, window=5):
        """Calculate team's form based on recent results."""
        if len(matches) == 0:
            return 0
        
        recent_matches = matches.tail(window)
        points = []
        
        for _, match in recent_matches.iterrows():
            if match['HomeTeam'] == team:
                points.append(3 if match['FTR'] == 'H' else 1 if match['FTR'] == 'D' else 0)
            else:
                points.append(3 if match['FTR'] == 'A' else 1 if match['FTR'] == 'D' else 0)
        
        return sum(points) / len(points) if points else 0

    def _set_default_values(self, df, idx):
        """Set default values for first matches with no history."""
        df.loc[idx, [
            'Home_League_Position', 'Away_League_Position',
            'Home_Points', 'Away_Points',
            'Position_Diff', 'Points_Diff',
            'Home_Goals_Scored_Avg', 'Home_Goals_Conceded_Avg',
            'Away_Goals_Scored_Avg', 'Away_Goals_Conceded_Avg',
            'Home_Form', 'Away_Form',
            'Home_Clean_Sheets', 'Away_Clean_Sheets',
            'Home_Failed_Score', 'Away_Failed_Score',
            'Goals_Diff_Home', 'Goals_Diff_Away',
            'Home_Win_Rate', 'Away_Win_Rate',
            'General_Form'
        ]] = 0
        
        df.loc[idx, [
            'h2h_home_win_rate', 'h2h_draw_rate', 'h2h_away_win_rate'
        ]] = 1/3
        
        if self.config.use_corner_features:
            df.loc[idx, [col for col in df.columns if 'Corner' in col]] = 0
        
        if self.config.use_card_features:
            df.loc[idx, [col for col in df.columns if 'Card' in col]] = 0
            
        # Original odds are already set in create_features

    def _calculate_additional_stats(self, match, past_matches):
        """Calculate additional performance statistics."""
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        
        # Get team's previous matches
        home_past = past_matches[
            (past_matches['HomeTeam'] == home_team) | 
            (past_matches['AwayTeam'] == home_team)
        ].sort_values('Date')
        
        away_past = past_matches[
            (past_matches['HomeTeam'] == away_team) | 
            (past_matches['AwayTeam'] == away_team)
        ].sort_values('Date')
        
        stats = {}
        
        # Clean sheets and failed to score
        if len(home_past) > 0:
            home_clean_sheets = 0
            home_failed_score = 0
            for _, past_match in home_past.iterrows():
                if past_match['HomeTeam'] == home_team:
                    if past_match['FTAG'] == 0:
                        home_clean_sheets += 1
                    if past_match['FTHG'] == 0:
                        home_failed_score += 1
                else:
                    if past_match['FTHG'] == 0:
                        home_clean_sheets += 1
                    if past_match['FTAG'] == 0:
                        home_failed_score += 1
            
            stats['Home_Clean_Sheets'] = home_clean_sheets / len(home_past)
            stats['Home_Failed_Score'] = home_failed_score / len(home_past)
            
            # Goal differences
            home_goals_diff = []
            for _, past_match in home_past.iterrows():
                if past_match['HomeTeam'] == home_team:
                    home_goals_diff.append(past_match['FTHG'] - past_match['FTAG'])
                else:
                    home_goals_diff.append(past_match['FTAG'] - past_match['FTHG'])
            stats['Goals_Diff_Home'] = np.mean(home_goals_diff)
            
            # Win rates
            home_wins = len(home_past[
                ((home_past['HomeTeam'] == home_team) & (home_past['FTR'] == 'H')) |
                ((home_past['AwayTeam'] == home_team) & (home_past['FTR'] == 'A'))
            ])
            stats['Home_Win_Rate'] = home_wins / len(home_past)
        else:
            stats['Home_Clean_Sheets'] = 0
            stats['Home_Failed_Score'] = 0
            stats['Goals_Diff_Home'] = 0
            stats['Home_Win_Rate'] = 0
        
        # Away team stats
        if len(away_past) > 0:
            away_clean_sheets = 0
            away_failed_score = 0
            for _, past_match in away_past.iterrows():
                if past_match['HomeTeam'] == away_team:
                    if past_match['FTAG'] == 0:
                        away_clean_sheets += 1
                    if past_match['FTHG'] == 0:
                        away_failed_score += 1
                else:
                    if past_match['FTHG'] == 0:
                        away_clean_sheets += 1
                    if past_match['FTAG'] == 0:
                        away_failed_score += 1
            
            stats['Away_Clean_Sheets'] = away_clean_sheets / len(away_past)
            stats['Away_Failed_Score'] = away_failed_score / len(away_past)
            
            # Goal differences
            away_goals_diff = []
            for _, past_match in away_past.iterrows():
                if past_match['HomeTeam'] == away_team:
                    away_goals_diff.append(past_match['FTHG'] - past_match['FTAG'])
                else:
                    away_goals_diff.append(past_match['FTAG'] - past_match['FTHG'])
            stats['Goals_Diff_Away'] = np.mean(away_goals_diff)
            
            # Win rates
            away_wins = len(away_past[
                ((away_past['HomeTeam'] == away_team) & (away_past['FTR'] == 'H')) |
                ((away_past['AwayTeam'] == away_team) & (away_past['FTR'] == 'A'))
            ])
            stats['Away_Win_Rate'] = away_wins / len(away_past)
        else:
            stats['Away_Clean_Sheets'] = 0
            stats['Away_Failed_Score'] = 0
            stats['Goals_Diff_Away'] = 0
            stats['Away_Win_Rate'] = 0
        
        # General form (combined win rate of both teams)
        stats['General_Form'] = (stats['Home_Win_Rate'] + stats['Away_Win_Rate']) / 2
        
        return stats