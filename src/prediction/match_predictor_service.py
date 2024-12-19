import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional
from src.models.unified_predictor import UnifiedPredictor
from src.data_collection.current_matches_collector import CurrentMatchesCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MatchPredictorService:
    """Service class to handle current match predictions."""
    
    def __init__(self, predictor: Optional[UnifiedPredictor] = None):
        self.predictor = predictor or UnifiedPredictor()
        self.collector = CurrentMatchesCollector()
    
    def get_current_matches(self, date_str: str) -> pd.DataFrame:
        """Fetch current matches and their odds."""
        # Get fixtures
        current_matches = self.collector.test_date_request(date_str)
        if not current_matches:
            logger.warning(f"No matches found for {date_str}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        current_df = pd.DataFrame([{
            'HomeTeam': match['teams']['home']['name'],
            'AwayTeam': match['teams']['away']['name'],
            'Date': pd.to_datetime(match['fixture']['date']).strftime('%Y-%m-%d'),
            'Time': pd.to_datetime(match['fixture']['date']).strftime('%H:%M'),
            'League': match['league']['name']
        } for match in current_matches])
        
        # Get odds
        odds_data = self.collector.test_odds_request(date_str, bookmaker=8)
        
        if odds_data:
            # Update odds in current_df
            for match_odds in odds_data:
                try:
                    fixture_id = match_odds.get('fixture', {}).get('id')
                    if not fixture_id:
                        continue
                    
                    # Find corresponding match
                    match_info = next((m for m in current_matches if m['fixture']['id'] == fixture_id), None)
                    if not match_info:
                        continue
                    
                    home_team = match_info['teams']['home']['name']
                    away_team = match_info['teams']['away']['name']
                    
                    # Find match in current_df
                    match_idx = current_df[
                        (current_df['HomeTeam'] == home_team) & 
                        (current_df['AwayTeam'] == away_team)
                    ].index
                    
                    if len(match_idx) > 0:
                        idx = match_idx[0]
                        # Update odds
                        for bookmaker in match_odds.get('bookmakers', []):
                            if bookmaker['name'] == 'Bet365':
                                for bet in bookmaker['bets']:
                                    if bet['name'] == 'Match Winner':
                                        for value in bet['values']:
                                            if value['value'] == 'Home':
                                                current_df.loc[idx, 'B365H'] = float(value['odd'])
                                            elif value['value'] == 'Draw':
                                                current_df.loc[idx, 'B365D'] = float(value['odd'])
                                            elif value['value'] == 'Away':
                                                current_df.loc[idx, 'B365A'] = float(value['odd'])
                                    elif bet['name'] == 'Goals Over/Under':
                                        for value in bet['values']:
                                            if value['value'] == 'Over 2.5':
                                                current_df.loc[idx, 'B365>2.5'] = float(value['odd'])
                                            elif value['value'] == 'Under 2.5':
                                                current_df.loc[idx, 'B365<2.5'] = float(value['odd'])
                except Exception as e:
                    logger.warning(f"Error processing odds for a match: {str(e)}")
                    continue
        
        return current_df
    
    def predict_matches(self, date_str: str) -> List[Dict]:
        """Get predictions for matches on a specific date."""
        # Get current matches
        current_df = self.get_current_matches(date_str)
        if current_df.empty:
            return []
        
        # Make predictions
        predictions = self.predictor.predict(current_df)
        
        # Format predictions
        formatted_predictions = self.predictor.format_predictions(predictions, current_df)
        
        return formatted_predictions
    
    def save_predictions(self, predictions: List[Dict], output_dir: Optional[Path] = None) -> Path:
        """Save predictions to a file."""
        if output_dir is None:
            output_dir = Path('predictions') / datetime.now().strftime('%Y%m%d_%H%M%S')
        
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions_file = output_dir / 'predictions.json'
        
        with open(predictions_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        logger.info(f"Predictions saved to {predictions_file}")
        return predictions_file 