import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from typing import Dict, List, Set, Optional
from datetime import datetime
from src.models.unified_predictor import UnifiedPredictor
from src.evaluation.unified_betting_simulator import UnifiedBettingSimulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    """Configuration for the predictor test."""
    train_cutoff_year = 2023
    
    # Separate required columns into pre-match and in-game data
    required_columns = {
        'Date', 'HomeTeam', 'AwayTeam',
        'B365H', 'B365D', 'B365A',  # Pre-match odds
        'FTHG', 'FTAG', 'FTR',  # Full-time results
        'HTHG', 'HTAG', 'HTR',  # Half-time results
        'HC', 'AC',  # Historical corners data
        'HY', 'AY',  # Historical cards data
    }
    
    # Optional columns for additional markets
    optional_columns = {
        'B365>2.5', 'B365<2.5',  # Over/under odds
    }
    
    min_training_samples = 1000
    initial_bankroll = 1000.0
    form_window = 5  # Last N matches for form
    h2h_window = 5   # Last N head-to-head matches

def validate_dataframe(df: pd.DataFrame, required_columns: Set[str]) -> None:
    """Validate that dataframe contains required columns."""
    missing_cols = required_columns - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

def load_training_data(data_dir: Path = Path('data/raw')) -> pd.DataFrame:
    """Load and combine historical training data with validation."""
    all_data = []
    failed_files = []
    
    for excel_file in data_dir.glob('*.xls*'):
        try:
            df = pd.read_excel(excel_file)
            validate_dataframe(df, Config.required_columns)
            all_data.append(df)
            logger.info(f"Loaded {len(df)} rows from {excel_file}")
        except Exception as e:
            failed_files.append(str(excel_file))
            logger.error(f"Error loading {excel_file}: {str(e)}")
    
    if failed_files:
        logger.warning(f"Failed to load {len(failed_files)} files: {failed_files}")
    
    if not all_data:
        raise ValueError("No valid training data found")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    if len(combined_df) < Config.min_training_samples:
        raise ValueError(f"Insufficient training data: {len(combined_df)} samples")
    
    logger.info(f"Combined dataset has {len(combined_df)} rows")
    return combined_df

def prepare_labels(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Prepare labels for each market."""
    labels = {}
    df = df.copy()
    
    # Match result labels
    df['FTR'] = df['FTR'].fillna('Unknown')
    result_map = {'H': 2, 'D': 1, 'A': 0}  # Home=2, Draw=1, Away=0
    labels['match_result'] = df['FTR'].map(result_map)
    
    # Over/Under 2.5 goals labels (only if odds are available)
    if 'B365>2.5' in df.columns and 'B365<2.5' in df.columns:
        valid_ou_mask = (
            ~df['B365>2.5'].isna() & 
            ~df['B365<2.5'].isna() & 
            (df['B365>2.5'] > 1.0) & 
            (df['B365<2.5'] > 1.0)
        )
        if valid_ou_mask.any():
            df['FTHG'] = df['FTHG'].fillna(0)
            df['FTAG'] = df['FTAG'].fillna(0)
            total_goals = df['FTHG'] + df['FTAG']
            labels['over_under'] = (total_goals > 2.5).astype(int)
    
    # Half-time score labels
    df['HTScore'] = df['HTHG'].astype(str) + '-' + df['HTAG'].astype(str)
    ht_scores = ['0-0', '1-0', '0-1', '2-0', '1-1', '0-2', '2-1', '1-2']
    ht_score_map = {score: idx for idx, score in enumerate(ht_scores)}
    ht_score_map['other'] = len(ht_scores)
    labels['ht_score'] = df['HTScore'].apply(lambda x: ht_score_map[x if x in ht_scores else 'other'])
    
    # Full-time score labels
    df['FTScore'] = df['FTHG'].astype(str) + '-' + df['FTAG'].astype(str)
    ft_scores = ['0-0', '1-0', '0-1', '2-0', '1-1', '0-2', '2-1', '1-2', '2-2']
    ft_score_map = {score: idx for idx, score in enumerate(ft_scores)}
    ft_score_map['other'] = len(ft_scores)
    labels['ft_score'] = df['FTScore'].apply(lambda x: ft_score_map[x if x in ft_scores else 'other'])
    
    # Corners and cards labels (using historical data)
    if 'HC' in df.columns and 'AC' in df.columns:
        labels['corners'] = df['HC'].fillna(0) + df['AC'].fillna(0)
    
    if 'HY' in df.columns and 'AY' in df.columns:
        labels['cards'] = df['HY'].fillna(0) + df['AY'].fillna(0)
    
    return labels

def prepare_actuals(df: pd.DataFrame) -> List[Dict]:
    """Prepare actual results for betting simulation."""
    actuals = []
    
    for _, row in df.iterrows():
        total_goals = row['FTHG'] + row['FTAG']
        actual = {
            'result': 'Home' if row['FTR'] == 'H' else 'Draw' if row['FTR'] == 'D' else 'Away',
            'over_under': 'Over' if total_goals > 2.5 else 'Under',
            'ht_score': f"{row['HTHG']}-{row['HTAG']}",
            'ft_score': f"{row['FTHG']}-{row['FTAG']}",
            'corners': row['HC'] + row['AC'] if 'HC' in row and 'AC' in row else None,
            'cards': row['HY'] + row['AY'] if 'HY' in row and 'AY' in row else None
        }
        actuals.append(actual)
    
    return actuals

def calculate_form(df: pd.DataFrame, team: str, match_date: datetime, window: int = 5) -> Dict:
    """Calculate team's recent form before the match date."""
    
    recent_matches = df[
        (df['Date'] < match_date) &
        ((df['HomeTeam'] == team) | (df['AwayTeam'] == team))
    ].sort_values('Date', ascending=False).head(window)
    
    if len(recent_matches) == 0:
        return {
            f'recent_wins': 0,
            f'recent_draws': 0,
            f'recent_losses': 0,
            f'recent_goals_scored': 0,
            f'recent_goals_conceded': 0,
            f'recent_clean_sheets': 0
        }
    
    stats = {
        'wins': 0,
        'draws': 0,
        'losses': 0,
        'goals_scored': 0,
        'goals_conceded': 0,
        'clean_sheets': 0
    }
    
    # Apply exponential decay weights
    weights = np.exp(-np.arange(len(recent_matches)) * 0.2)
    
    for i, (_, match) in enumerate(recent_matches.iterrows()):
        is_home = match['HomeTeam'] == team
        team_goals = match['FTHG'] if is_home else match['FTAG']
        opp_goals = match['FTAG'] if is_home else match['FTHG']
        weight = weights[i]
        
        # Update weighted stats
        if team_goals > opp_goals:
            stats['wins'] += weight
        elif team_goals == opp_goals:
            stats['draws'] += weight
        else:
            stats['losses'] += weight
            
        stats['goals_scored'] += team_goals * weight
        stats['goals_conceded'] += opp_goals * weight
        stats['clean_sheets'] += (1 if opp_goals == 0 else 0) * weight
    
    # Normalize by sum of weights
    weight_sum = np.sum(weights[:len(recent_matches)])
    return {
        f'recent_wins': stats['wins'] / weight_sum,
        f'recent_draws': stats['draws'] / weight_sum,
        f'recent_losses': stats['losses'] / weight_sum,
        f'recent_goals_scored': stats['goals_scored'] / weight_sum,
        f'recent_goals_conceded': stats['goals_conceded'] / weight_sum,
        f'recent_clean_sheets': stats['clean_sheets'] / weight_sum
    }

def calculate_season_stats(df: pd.DataFrame, team: str, match_date: datetime) -> Dict:
    """Calculate season statistics up to (but not including) the match date."""
    
    # Get season start date (assuming season starts in August)
    season_year = match_date.year if match_date.month >= 8 else match_date.year - 1
    season_start = datetime(season_year, 8, 1)
    
    # Get all matches for this team in current season up to match_date
    season_matches = df[
        (df['Date'] >= season_start) &
        (df['Date'] < match_date) &
        ((df['HomeTeam'] == team) | (df['AwayTeam'] == team))
    ].sort_values('Date')
    
    if len(season_matches) == 0:
        return {
            'ppg': 0,
            'goals_per_game': 0,
            'goals_conceded_per_game': 0,
            'clean_sheet_ratio': 0,
            'win_ratio': 0
        }
    
    stats = {
        'points': 0,
        'goals_scored': 0,
        'goals_conceded': 0,
        'wins': 0,
        'clean_sheets': 0
    }
    
    for _, match in season_matches.iterrows():
        is_home = match['HomeTeam'] == team
        team_goals = match['FTHG'] if is_home else match['FTAG']
        opp_goals = match['FTAG'] if is_home else match['FTHG']
        
        # Update stats
        if team_goals > opp_goals:
            stats['points'] += 3
            stats['wins'] += 1
        elif team_goals == opp_goals:
            stats['points'] += 1
            
        stats['goals_scored'] += team_goals
        stats['goals_conceded'] += opp_goals
        stats['clean_sheets'] += 1 if opp_goals == 0 else 0
    
    matches_played = len(season_matches)
    return {
        'ppg': stats['points'] / matches_played,
        'goals_per_game': stats['goals_scored'] / matches_played,
        'goals_conceded_per_game': stats['goals_conceded'] / matches_played,
        'clean_sheet_ratio': stats['clean_sheets'] / matches_played,
        'win_ratio': stats['wins'] / matches_played
    }

def calculate_h2h_stats(df: pd.DataFrame, home_team: str, away_team: str, match_date: datetime, window: int = 5) -> Dict:
    """Calculate head-to-head statistics before the match date."""
    
    h2h_matches = df[
        (df['Date'] < match_date) &
        (
            ((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
            ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))
        )
    ].sort_values('Date', ascending=False).head(window)
    
    if len(h2h_matches) == 0:
        return {
            'h2h_home_wins': 0,
            'h2h_away_wins': 0,
            'h2h_draws': 0,
            'h2h_avg_goals': 0,
            'h2h_home_team_avg_goals': 0,
            'h2h_away_team_avg_goals': 0
        }
    
    stats = {
        'home_wins': 0,
        'away_wins': 0,
        'draws': 0,
        'total_goals': 0,
        'home_team_goals': 0,
        'away_team_goals': 0
    }
    
    for _, match in h2h_matches.iterrows():
        home_goals = match['FTHG']
        away_goals = match['FTAG']
        is_same_home = match['HomeTeam'] == home_team
        
        if is_same_home:
            stats['home_team_goals'] += home_goals
            stats['away_team_goals'] += away_goals
        else:
            stats['home_team_goals'] += away_goals
            stats['away_team_goals'] += home_goals
            
        stats['total_goals'] += home_goals + away_goals
        
        if home_goals > away_goals:
            stats['home_wins'] += 1 if is_same_home else 0
            stats['away_wins'] += 0 if is_same_home else 1
        elif away_goals > home_goals:
            stats['home_wins'] += 0 if is_same_home else 1
            stats['away_wins'] += 1 if is_same_home else 0
        else:
            stats['draws'] += 1
    
    matches = len(h2h_matches)
    return {
        'h2h_home_wins': stats['home_wins'] / matches,
        'h2h_away_wins': stats['away_wins'] / matches,
        'h2h_draws': stats['draws'] / matches,
        'h2h_avg_goals': stats['total_goals'] / matches,
        'h2h_home_team_avg_goals': stats['home_team_goals'] / matches,
        'h2h_away_team_avg_goals': stats['away_team_goals'] / matches
    }

def validate_odds(row: pd.Series) -> bool:
    """Validate that odds are present and valid."""
    # Required odds
    required_odds = ['B365H', 'B365D', 'B365A']
    for col in required_odds:
        if col not in row or pd.isna(row[col]) or row[col] <= 1.0:
            return False
    return True

def calculate_features(past_data: pd.DataFrame, row: pd.Series, is_training: bool = True) -> Optional[Dict]:
    """Calculate features for a match, avoiding look-ahead bias."""
    # Validate required odds first
    if not validate_odds(row):
        return None
        
    try:
        # Get historical matches for both teams
        home_team_matches = past_data[past_data['HomeTeam'] == row['HomeTeam']]
        away_team_matches = past_data[past_data['AwayTeam'] == row['AwayTeam']]
        
        # Require more historical matches for better feature calculation
        min_matches_required = 5  # Increased from 3
        if len(home_team_matches) < min_matches_required or len(away_team_matches) < min_matches_required:
            return None
            
        features = {
            # Form features (using only historical data)
            **calculate_form(past_data, row['HomeTeam'], row['Date'], Config.form_window),
            **calculate_form(past_data, row['AwayTeam'], row['Date'], Config.form_window),
            # Season stats (using only historical data)
            **calculate_season_stats(past_data, row['HomeTeam'], row['Date']),
            **calculate_season_stats(past_data, row['AwayTeam'], row['Date']),
            # H2H stats (using only historical data)
            **calculate_h2h_stats(past_data, row['HomeTeam'], row['AwayTeam'], row['Date'], Config.h2h_window),
        }
        
        # Add required pre-match odds
        odds_features = {
            'B365H': float(row['B365H']),
            'B365D': float(row['B365D']),
            'B365A': float(row['B365A']),
        }
        
        # Add optional over/under odds if available
        if 'B365>2.5' in row and 'B365<2.5' in row:
            if not pd.isna(row['B365>2.5']) and not pd.isna(row['B365<2.5']):
                if row['B365>2.5'] > 1.0 and row['B365<2.5'] > 1.0:
                    odds_features.update({
                        'B365>2.5': float(row['B365>2.5']),
                        'B365<2.5': float(row['B365<2.5']),
                    })
        
        features.update(odds_features)
        
        # Calculate historical averages for corners and cards
        home_corners_avg = home_team_matches['HC'].mean()
        away_corners_avg = away_team_matches['AC'].mean()
        home_cards_avg = home_team_matches['HY'].mean()
        away_cards_avg = away_team_matches['AY'].mean()
        
        additional_features = {
            'home_corners_avg': home_corners_avg,
            'away_corners_avg': away_corners_avg,
            'home_cards_avg': home_cards_avg,
            'away_cards_avg': away_cards_avg,
        }
        features.update(additional_features)
        
        return features
    except Exception as e:
        logger.warning(f"Error calculating features: {str(e)}")
        return None

def main():
    """Main execution with granular error handling."""
    output_dir = Path('output') / datetime.now().strftime('%Y%m%d_%H%M%S')
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create output directory: {e}")
        return

    try:
        all_data = load_training_data()
    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        return

    try:
        # Convert dates and sort chronologically
        all_data['Date'] = pd.to_datetime(all_data['Date'])
        all_data = all_data.sort_values('Date')
        
        # Split data
        train_data = all_data[all_data['Date'].dt.year < Config.train_cutoff_year].copy()
        test_data = all_data[all_data['Date'].dt.year >= Config.train_cutoff_year].copy()
        
        if len(train_data) < Config.min_training_samples:
            raise ValueError(f"Insufficient training data: {len(train_data)} samples")
        
        logger.info(f"Training set: {len(train_data)} matches")
        logger.info(f"Test set: {len(test_data)} matches")
        
        # Calculate features for training data
        logger.info("Calculating features for training data...")
        train_features = []
        train_indices = []  # Keep track of valid indices
        for i, (idx, row) in enumerate(train_data.iterrows()):
            # Calculate features using only past data
            past_data = train_data.iloc[:i]
            features = calculate_features(past_data, row, is_training=True)
            if features is not None:
                train_features.append(features)
                train_indices.append(idx)
            if i % 100 == 0:
                logger.info(f"Processed {i}/{len(train_data)} training matches")
        
        # Filter training data to only include rows with valid features
        train_data = train_data.loc[train_indices]
        
        # Calculate features for test data
        logger.info("Calculating features for test data...")
        test_features = []
        test_indices = []  # Keep track of valid indices
        for i, (idx, row) in enumerate(test_data.iterrows()):
            # Calculate features using only training data as history
            past_data = train_data.copy()  # Use only training data for historical calculations
            features = calculate_features(past_data, row, is_training=False)
            if features is not None:
                test_features.append(features)
                test_indices.append(idx)
            if i % 100 == 0:
                logger.info(f"Processed {i}/{len(test_data)} test matches")
        
        # Filter test data to only include rows with valid features
        test_data = test_data.loc[test_indices]
        
        if len(train_features) < Config.min_training_samples:
            raise ValueError(f"Insufficient valid training samples after feature calculation: {len(train_features)}")
        
        # Convert features to DataFrames
        train_features_df = pd.DataFrame(train_features)
        test_features_df = pd.DataFrame(test_features)
        
        # Prepare labels
        labels = prepare_labels(train_data)
        
        # Initialize and train model
        logger.info("Training model...")
        predictor = UnifiedPredictor()
        evaluation_metrics = predictor.train(train_features_df, labels)
        
        # Save trained model
        predictor.save()
        
        # Make predictions on test set
        logger.info("Making predictions...")
        predictions = predictor.predict(test_features_df)
        formatted_predictions = predictor.format_predictions(predictions, test_data)
        
        # Run betting simulation
        logger.info("Running betting simulation...")
        simulator = UnifiedBettingSimulator(initial_bankroll=Config.initial_bankroll)
        results_df, summary = simulator.simulate_bets(formatted_predictions, prepare_actuals(test_data))
        
        # Save results
        simulator.save_results(results_df, summary, output_dir)
        
        # Display summary
        logger.info("\nBetting Simulation Summary:")
        logger.info(f"Initial Bankroll: ${summary['initial_bankroll']:.2f}")
        logger.info(f"Final Bankroll: ${summary['final_bankroll']:.2f}")
        logger.info(f"Total Profit: ${summary['total_profit']:.2f}")
        logger.info(f"ROI: {summary['roi']:.2f}%")
        logger.info(f"Total Bets: {summary['total_bets']}")
        logger.info(f"Win Rate: {summary['win_rate']:.2f}%")
        
        logger.info("\nMarket Performance:")
        for market, stats in summary['market_performance'].items():
            logger.info(f"\n{market.upper()}:")
            logger.info(f"Bets: {stats['bets']}")
            logger.info(f"Won: {stats['won']}")
            logger.info(f"Win Rate: {stats['win_rate']:.2f}%")
            logger.info(f"Profit: ${stats['profit']:.2f}")
            logger.info(f"ROI: {stats['roi']:.2f}%")
        
    except Exception as e:
        logger.error(f"Error in processing: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 