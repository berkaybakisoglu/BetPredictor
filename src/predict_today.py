"""Script to fetch and predict today's matches."""
import os
from datetime import datetime
import pandas as pd
from src.fetch_matches import (
    fetch_fixtures,
    fetch_odds,
    format_match_data,
    save_data,
    LEAGUE_MAPPINGS,
    FOOTBALL_API_KEY
)
from src.models.predictor import UnifiedPredictor
from src.config.config import Config, DataConfig, FeatureConfig, ModelConfig, BettingConfig
from pathlib import Path
import json

def fetch_todays_matches():
    """Fetch today's matches and return processed DataFrame."""
    date_str = datetime.now().strftime('%Y-%m-%d')
    date_folder = date_str.replace('-', '')
    
    # Define data directory and file paths
    data_dir = f'data/api_data/{date_folder}'
    fixtures_file = f'{data_dir}/fixtures.json'
    odds_file = f'{data_dir}/odds.json'
    
    # Check if we already have today's data
    if os.path.exists(fixtures_file) and os.path.exists(odds_file):
        print(f"\nFound existing data for {date_str}")
        try:
            # Load existing data
            with open(fixtures_file, 'r') as f:
                supported_fixtures = json.load(f)
            with open(odds_file, 'r') as f:
                all_odds = json.load(f)
                
            print(f"Loaded {len(supported_fixtures)} fixtures from existing data")
            
            # Format data with all required features
            df = format_match_data(supported_fixtures, all_odds)
            return df
            
        except Exception as e:
            print(f"Error loading existing data: {str(e)}")
            print("Will fetch fresh data instead")
    
    print(f"\nStarting data fetch for date: {date_str}")
    print(f"Using API key: {FOOTBALL_API_KEY[:5]}...")  # Only show first 5 chars for security
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Fetch fixtures
    fixtures = fetch_fixtures(date_str)
    if not fixtures:
        print("Failed to fetch fixtures")
        return None
        
    # Filter for supported leagues
    supported_fixtures = [f for f in fixtures if f['league']['id'] in [ids['api_football'] for ids in LEAGUE_MAPPINGS.values()]]
    print(f"\nFound {len(supported_fixtures)} fixtures in supported leagues:")
    
    # Print fixtures details
    for fixture in supported_fixtures:
        print(f"\n{fixture['league']['name']}")
        print(f"{fixture['teams']['home']['name']} vs {fixture['teams']['away']['name']}")
        print(f"Kickoff: {datetime.fromtimestamp(fixture['fixture']['timestamp']).strftime('%H:%M')}")
    
    # Fetch odds for each fixture
    all_odds = []
    total_fixtures = len(supported_fixtures)
    
    for i, fixture in enumerate(supported_fixtures, 1):
        fixture_id = fixture['fixture']['id']
        print(f"\nFetching odds ({i}/{total_fixtures}): {fixture['teams']['home']['name']} vs {fixture['teams']['away']['name']}")
        odds = fetch_odds(fixture_id)
        if odds:
            all_odds.append(odds)
    
    # Format data with all required features
    df = format_match_data(supported_fixtures, all_odds)
    
    # Save raw data for reference
    save_data(supported_fixtures, fixtures_file)
    save_data(all_odds, odds_file)
    
    return df

def format_predictions(predictions: dict, df: pd.DataFrame) -> None:
    """Format and print predictions in a readable way."""
    print("\n=== PREDICTIONS ===")
    
    # Standard confidence thresholds for different betting systems
    CONFIDENCE_THRESHOLDS = {
        'Conservative': 0.70,  # 70% confidence required
        'Moderate': 0.60,     # 60% confidence required
        'Aggressive': 0.50    # 50% confidence required
    }
    
    VALUE_THRESHOLD = 1.05    # 5% edge required for value bets
    
    for match_idx in range(len(df)):
        match = df.iloc[match_idx]
        print(f"\n{match['League']}")
        print(f"{match['HomeTeam']} vs {match['AwayTeam']}")
        print(f"Kickoff: {match['Date'].strftime('%H:%M')}")
        print("-" * 40)
        
        # Match Result
        if 'match_result' in predictions:
            match_result = predictions['match_result']
            if isinstance(match_result, pd.DataFrame) and len(match_result) > match_idx:
                probs = match_result.iloc[match_idx]
                print("\nMatch Result:")
                try:
                    # Get probabilities from named columns
                    home_prob = float(probs['Home_Prob'])
                    draw_prob = float(probs['Draw_Prob'])
                    away_prob = float(probs['Away_Prob'])
                    confidence = float(probs['Confidence'])
                    predicted = probs['Predicted']
                    
                    print(f"Home Win: {home_prob:.1%}")
                    print(f"Draw: {draw_prob:.1%}")
                    print(f"Away Win: {away_prob:.1%}")
                    
                    # Map prediction code to readable text
                    outcome_map = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}
                    outcome = outcome_map.get(predicted, predicted)
                    
                    print(f"Prediction: {outcome} (Confidence: {confidence:.1%})")
                    
                    # Compare with standard betting systems
                    print("\nConfidence Analysis:")
                    for system, threshold in CONFIDENCE_THRESHOLDS.items():
                        if confidence >= threshold:
                            print(f"✓ {system} ({threshold:.0%} required)")
                        else:
                            print(f"✗ {system} ({threshold:.0%} required) - Need {(threshold - confidence):.1%} more confidence")
                    
                    # Value bet analysis
                    value_bets = []
                    value_analysis = []
                    
                    # Calculate value ratios
                    home_value = float(probs['Home_Value'])
                    draw_value = float(probs['Draw_Value'])
                    away_value = float(probs['Away_Value'])
                    
                    if home_value > VALUE_THRESHOLD:
                        value_bets.append(f"Home @ {probs['B365H']}")
                        value_analysis.append(f"Home: {(home_value - 1):.1%} edge")
                    if draw_value > VALUE_THRESHOLD:
                        value_bets.append(f"Draw @ {probs['B365D']}")
                        value_analysis.append(f"Draw: {(draw_value - 1):.1%} edge")
                    if away_value > VALUE_THRESHOLD:
                        value_bets.append(f"Away @ {probs['B365A']}")
                        value_analysis.append(f"Away: {(away_value - 1):.1%} edge")
                    
                    if value_bets:
                        print("\nValue Analysis:")
                        print("Value Bets: " + ", ".join(value_bets))
                        print("Edge Analysis: " + ", ".join(value_analysis))
                    
                except Exception as e:
                    print(f"Unable to parse match result probabilities: {str(e)}")
        
        # Over/Under
        if 'over_under' in predictions:
            over_under = predictions['over_under']
            if isinstance(over_under, pd.DataFrame) and len(over_under) > match_idx:
                probs = over_under.iloc[match_idx]
                print("\nOver/Under 2.5:")
                try:
                    # Get probabilities from named columns
                    under_prob = float(probs['Under_Prob'])
                    over_prob = float(probs['Over_Prob'])
                    
                    print(f"Under: {under_prob:.1%}")
                    print(f"Over: {over_prob:.1%}")
                    
                    # Calculate confidence
                    confidence = abs(over_prob - under_prob)
                    prediction = "Over" if over_prob > under_prob else "Under"
                    print(f"Prediction: {prediction} (Confidence: {confidence:.1%})")
                    
                    # Compare with standard betting systems
                    print("\nConfidence Analysis:")
                    for system, threshold in CONFIDENCE_THRESHOLDS.items():
                        if confidence >= threshold:
                            print(f"✓ {system} ({threshold:.0%} required)")
                        else:
                            print(f"✗ {system} ({threshold:.0%} required) - Need {(threshold - confidence):.1%} more confidence")
                    
                    # Value bet analysis
                    try:
                        implied_over = 1/float(probs['B365>2.5'])
                        implied_under = 1/float(probs['B365<2.5'])
                        
                        value_bets = []
                        value_analysis = []
                        
                        over_edge = over_prob - implied_over
                        under_edge = under_prob - implied_under
                        
                        if over_edge > 0.05:
                            value_bets.append(f"Over @ {probs['B365>2.5']}")
                            value_analysis.append(f"Over: {over_edge:.1%} edge")
                        if under_edge > 0.05:
                            value_bets.append(f"Under @ {probs['B365<2.5']}")
                            value_analysis.append(f"Under: {under_edge:.1%} edge")
                        
                        if value_bets:
                            print("\nValue Analysis:")
                            print("Value Bets: " + ", ".join(value_bets))
                            print("Edge Analysis: " + ", ".join(value_analysis))
                            
                    except Exception as e:
                        pass
                    
                except Exception as e:
                    print(f"Unable to parse over/under probabilities: {str(e)}")
        
        print("\nBetting Odds:")
        print(f"Home: {match.get('B365H', 'N/A')}")
        print(f"Draw: {match.get('B365D', 'N/A')}")
        print(f"Away: {match.get('B365A', 'N/A')}")
        print(f"Over 2.5: {match.get('B365>2.5', 'N/A')}")
        print(f"Under 2.5: {match.get('B365<2.5', 'N/A')}")

def main():
    """Fetch today's matches and make predictions."""
    try:
        # Initialize config
        config = Config(
            data=DataConfig(),
            features=FeatureConfig(),
            model=ModelConfig(),
            betting=BettingConfig(),
            output_dir=Path('output'),
            models_dir=Path('models')
        )
        
        # Initialize predictor
        predictor = UnifiedPredictor(config.model)
        
        # Load trained models
        print("\nLoading trained models...")
        predictor.load_models(config.models_dir)
        
        # Check if we already have processed data for today
        date_str = datetime.now().strftime('%Y%m%d')
        predictions_file = f'predictions/{date_str}/matches_with_features.csv'
        
        if os.path.exists(predictions_file):
            print(f"\nFound existing processed data for today at {predictions_file}")
            df = pd.read_csv(predictions_file)
            df['Date'] = pd.to_datetime(df['Date'])  # Convert date back to datetime
        else:
            # Fetch today's matches
            df = fetch_todays_matches()
            if df is None or df.empty:
                print("No matches to predict")
                return
                
            # Save processed data
            os.makedirs(os.path.dirname(predictions_file), exist_ok=True)
            df.to_csv(predictions_file, index=False)
            print(f"\nSaved processed data to {predictions_file}")
        
        print(f"\nMaking predictions for {len(df)} matches...")
        
        # Make predictions
        predictions = predictor.predict(df)
        
        # Format and display predictions
        format_predictions(predictions, df)
        
    except Exception as e:
        print(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 