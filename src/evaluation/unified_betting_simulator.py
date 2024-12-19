import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedBettingSimulator:
    """Simulates betting on multiple markets using predictions from the unified predictor."""
    
    def __init__(self, initial_bankroll: float = 1000.0):
        self.initial_bankroll = initial_bankroll
        self.markets = {
            'match_result': {
                'confidence_threshold': 0.55,
                'min_odds': 1.3,
                'max_stake_pct': 0.03
            },
            'over_under': {
                'confidence_threshold': 0.55,
                'min_odds': 1.3,
                'max_stake_pct': 0.03
            },
            'ht_score': {
                'confidence_threshold': 0.15,
                'min_odds': 2.0,
                'max_stake_pct': 0.02
            },
            'ft_score': {
                'confidence_threshold': 0.20,
                'min_odds': 3.0,
                'max_stake_pct': 0.01
            },
            'corners': {
                'confidence_threshold': 0.50,
                'min_odds': 1.5,
                'max_stake_pct': 0.02
            },
            'cards': {
                'confidence_threshold': 0.50,
                'min_odds': 1.5,
                'max_stake_pct': 0.02
            }
        }
    
    def calculate_kelly_stake(self, odds: float, prob: float, bankroll: float, market: str) -> float:
        """Calculate stake using Kelly Criterion with maximum stake limit."""
        max_stake = bankroll * self.markets[market]['max_stake_pct']
        q = 1 - prob
        kelly_fraction = (odds * prob - q) / odds
        stake = bankroll * kelly_fraction
        
        # Apply fractional Kelly (1/4) and maximum stake limit
        return min(stake * 0.25, max_stake)
    
    def evaluate_bet(self, prediction: Dict, actual: Dict, market: str) -> Tuple[bool, float]:
        """Evaluate if a bet won and calculate profit multiplier."""
        if market == 'match_result':
            actual_result = actual['result']
            if actual_result == prediction['best_outcome']:
                return True, prediction['odds'][prediction['best_outcome']] - 1
            return False, -1
        
        elif market == 'over_under':
            actual_result = actual['over_under']
            if actual_result == prediction['best_outcome']:
                return True, prediction['odds'][prediction['best_outcome']] - 1
            return False, -1
        
        elif market in ['ht_score', 'ft_score']:
            actual_score = actual[market]
            if actual_score == prediction['best_outcome']:
                return True, prediction['odds'][prediction['best_outcome']] - 1
            return False, -1
        
        elif market == 'corners':
            actual_corners = actual['corners']
            pred_corners = prediction['prediction']
            margin = 2  # Allow 2 corners margin of error
            
            if abs(actual_corners - pred_corners) <= margin:
                return True, prediction['odds'] - 1
            return False, -1
        
        elif market == 'cards':
            actual_cards = actual['cards']
            pred_cards = prediction['prediction']
            margin = 1  # Allow 1 card margin of error
            
            if abs(actual_cards - pred_cards) <= margin:
                return True, prediction['odds'] - 1
            return False, -1
        
        return False, -1
    
    def simulate_bets(self, predictions: List[Dict], actuals: List[Dict]) -> Tuple[pd.DataFrame, Dict]:
        """Simulate betting on multiple markets."""
        results = []
        bankroll = self.initial_bankroll
        
        for pred, actual in zip(predictions, actuals):
            match_info = {
                'date': pred['date'],
                'match': pred['match'],
                'league': pred.get('league', '')
            }
            
            # Evaluate each market
            for market in self.markets:
                if market not in pred['predictions']:
                    continue
                
                market_config = self.markets[market]
                market_pred = pred['predictions'][market]
                
                # Get highest confidence prediction for the market
                if market in ['match_result', 'ht_score', 'ft_score', 'over_under']:
                    # Remove odds from probabilities for finding best outcome
                    probs = {k: float(v.strip('%'))/100 for k, v in market_pred.items() if k != 'odds'}
                    best_outcome = max(probs.items(), key=lambda x: x[1])[0]
                    confidence = probs[best_outcome]
                    odds = float(market_pred['odds'][best_outcome])
                    
                    # Store best outcome and odds for evaluation
                    market_pred = {
                        'best_outcome': best_outcome,
                        'confidence': confidence,
                        'odds': market_pred['odds']
                    }
                else:
                    confidence = market_pred['confidence']
                    odds = market_pred['odds']
                
                # Check confidence threshold
                if confidence < market_config['confidence_threshold']:
                    continue
                
                # Check minimum odds
                if odds < market_config['min_odds']:
                    continue
                
                # Calculate stake using Kelly criterion
                stake = self.calculate_kelly_stake(odds, confidence, bankroll, market)
                if stake <= 0:
                    continue
                
                # Evaluate bet
                won, profit_mult = self.evaluate_bet(market_pred, actual, market)
                profit = stake * profit_mult
                bankroll += profit
                
                # Record result
                result = {
                    **match_info,
                    'market': market,
                    'prediction': market_pred.get('best_outcome', market_pred.get('prediction')),
                    'confidence': f"{confidence:.1%}",
                    'odds': odds,
                    'stake': stake,
                    'won': won,
                    'profit': profit,
                    'bankroll': bankroll
                }
                results.append(result)
        
        # Create DataFrame
        results_df = pd.DataFrame(results) if results else pd.DataFrame(columns=[
            'date', 'match', 'league', 'market', 'prediction', 
            'confidence', 'odds', 'stake', 'won', 'profit', 'bankroll'
        ])
        
        # Calculate summary
        summary = {
            'initial_bankroll': self.initial_bankroll,
            'final_bankroll': bankroll,
            'total_profit': bankroll - self.initial_bankroll,
            'roi': ((bankroll - self.initial_bankroll) / self.initial_bankroll) * 100,
            'total_bets': len(results_df),
            'total_won': int(results_df['won'].sum()) if not results_df.empty else 0,
            'win_rate': (results_df['won'].mean() * 100) if not results_df.empty else 0,
            'avg_odds': results_df['odds'].mean() if not results_df.empty else 0,
            'market_performance': {}
        }
        
        # Add per-market statistics
        if not results_df.empty:
            for market in results_df['market'].unique():
                market_df = results_df[results_df['market'] == market]
                summary['market_performance'][market] = {
                    'bets': len(market_df),
                    'won': int(market_df['won'].sum()),
                    'win_rate': market_df['won'].mean() * 100,
                    'profit': market_df['profit'].sum(),
                    'roi': (market_df['profit'].sum() / market_df['stake'].sum()) * 100
                }
        
        return results_df, summary
    
    def save_results(self, results_df: pd.DataFrame, summary: Dict, output_dir: Path):
        """Save betting simulation results to files."""
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results to CSV
        results_path = output_dir / 'betting_results.csv'
        results_df.to_csv(results_path, index=False)
        logger.info(f"Results saved to {results_path}")
        
        # Convert numpy types to Python native types
        def convert_to_native(obj):
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_native(x) for x in obj]
            elif isinstance(obj, (np.int8, np.int16, np.int32, np.int64,
                                np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Convert summary to native Python types
        summary = convert_to_native(summary)
        
        # Save summary to JSON
        summary_path = output_dir / 'betting_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        logger.info(f"Summary saved to {summary_path}") 