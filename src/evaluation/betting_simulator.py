import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BettingSimulator:
    def __init__(self, initial_bankroll: float = 1000.0, stake_percent: float = 2.0):
        """
        Initialize betting simulator.
        
        Args:
            initial_bankroll: Starting bankroll
            stake_percent: Percentage of bankroll to bet (default 2%)
        """
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.stake_percent = stake_percent
        self.bets_history: List[Dict] = []
        
    def calculate_kelly_stake(self, prob: float, odds: float) -> float:
        """Calculate Kelly Criterion stake size with a more conservative fraction."""
        if odds <= 1.0 or prob <= 0:
            return 0.0
        q = 1.0 - prob
        f = (prob * (odds - 1) - q) / (odds - 1)
        # Use 1/4 Kelly for more conservative betting
        return max(0, f * 0.25)
    
    def simulate_bets(self, 
                     predictions: pd.DataFrame,
                     confidence_threshold: float = 0.65,  # Increased threshold
                     min_odds: float = 1.8,              # Increased minimum odds
                     max_stake_percent: float = 1.0      # Reduced max stake
                     ) -> Tuple[pd.DataFrame, Dict]:
        """
        Simulate betting based on predictions and odds.
        
        Args:
            predictions: DataFrame with predictions and actual odds
            confidence_threshold: Minimum prediction confidence to place bet
            min_odds: Minimum odds to consider for betting
            max_stake_percent: Maximum stake as percentage of bankroll
        """
        results = []
        self.bankroll = self.initial_bankroll
        
        for idx, row in predictions.iterrows():
            try:
                # Check if we have all required data
                required_cols = ['pred_home_win', 'pred_draw', 'pred_away_win', 
                               'AvgH', 'AvgD', 'AvgA', 'FTR']
                if not all(col in row.index for col in required_cols):
                    logger.warning(f"Missing required columns for match {idx}")
                    continue
                
                # Get prediction probabilities and odds
                home_prob = row['pred_home_win']
                draw_prob = row['pred_draw']
                away_prob = row['pred_away_win']
                
                # Use average odds if available, otherwise skip
                try:
                    home_odds = float(row['AvgH'])
                    draw_odds = float(row['AvgD'])
                    away_odds = float(row['AvgA'])
                except (ValueError, TypeError):
                    logger.warning(f"Invalid odds for match {idx}")
                    continue
                
                # Skip if any odds are missing or invalid
                if any(pd.isna([home_odds, draw_odds, away_odds])) or \
                   any(x <= 1.0 for x in [home_odds, draw_odds, away_odds]):
                    logger.warning(f"Invalid odds for match {idx}")
                    continue
                
                # Find best betting opportunity
                opportunities = [
                    ('Home', home_prob, home_odds),
                    ('Draw', draw_prob, draw_odds),
                    ('Away', away_prob, away_odds)
                ]
                
                best_bet = max(opportunities, key=lambda x: x[1])  # Choose highest probability
                pred_outcome, prob, odds = best_bet
                
                # Calculate implied probability from odds
                implied_prob = 1 / odds
                
                # Only bet if our predicted probability is significantly higher than implied
                if prob > implied_prob * 1.1 and prob >= confidence_threshold and odds >= min_odds:
                    # Calculate stake using Kelly Criterion
                    kelly_fraction = self.calculate_kelly_stake(prob, odds)
                    max_stake = self.bankroll * (max_stake_percent / 100)
                    stake = min(max_stake, self.bankroll * kelly_fraction)
                    
                    # Record bet details
                    bet_result = {
                        'date': row.get('Date', 'Unknown'),
                        'match': f"{row.get('HomeTeam', 'Home')} vs {row.get('AwayTeam', 'Away')}",
                        'prediction': pred_outcome,
                        'confidence': prob,
                        'implied_prob': implied_prob,
                        'edge': (prob - implied_prob) * 100,  # Edge as percentage
                        'odds': odds,
                        'stake': stake,
                        'bankroll_before': self.bankroll
                    }
                    
                    # Check if bet won
                    actual_result = row['FTR']  # Full Time Result
                    won = (
                        (pred_outcome == 'Home' and actual_result == 'H') or
                        (pred_outcome == 'Draw' and actual_result == 'D') or
                        (pred_outcome == 'Away' and actual_result == 'A')
                    )
                    
                    # Update bankroll
                    if won:
                        profit = stake * (odds - 1)
                        self.bankroll += profit
                        bet_result['profit'] = profit
                    else:
                        self.bankroll -= stake
                        bet_result['profit'] = -stake
                    
                    bet_result['bankroll_after'] = self.bankroll
                    bet_result['won'] = won
                    results.append(bet_result)
            
            except Exception as e:
                logger.error(f"Error processing bet for match {idx}: {str(e)}")
                continue
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate summary statistics
        summary = {
            'initial_bankroll': self.initial_bankroll,
            'final_bankroll': self.bankroll,
            'total_profit': self.bankroll - self.initial_bankroll,
            'roi': ((self.bankroll - self.initial_bankroll) / self.initial_bankroll) * 100,
            'total_bets': len(results),
            'winning_bets': sum(1 for r in results if r['won']),
            'win_rate': sum(1 for r in results if r['won']) / len(results) if results else 0,
            'avg_odds': results_df['odds'].mean() if not results_df.empty else 0,
            'avg_stake': results_df['stake'].mean() if not results_df.empty else 0,
            'avg_edge': results_df['edge'].mean() if not results_df.empty else 0
        }
        
        return results_df, summary
    
    def save_results(self, results_df: pd.DataFrame, summary: Dict, output_dir: Path):
        """Save simulation results and summary."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results_path = output_dir / 'betting_results.csv'
        results_df.to_csv(results_path, index=False)
        
        # Save summary
        summary_path = output_dir / 'betting_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("Betting Simulation Summary\n")
            f.write("=========================\n\n")
            for key, value in summary.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.2f}\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        logger.info(f"Results saved to {output_dir}")
        
    def plot_bankroll_over_time(self, results_df: pd.DataFrame, output_dir: Path):
        """Plot bankroll evolution over time."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(12, 6))
            sns.set_style("whitegrid")
            
            # Plot bankroll evolution
            plt.plot(range(len(results_df)), results_df['bankroll_after'], 
                    label='Bankroll', color='blue')
            plt.axhline(y=self.initial_bankroll, color='r', linestyle='--', 
                       label='Initial Bankroll')
            
            plt.title('Bankroll Evolution Over Time')
            plt.xlabel('Number of Bets')
            plt.ylabel('Bankroll')
            plt.legend()
            
            # Save plot
            plot_path = output_dir / 'bankroll_evolution.png'
            plt.savefig(plot_path)
            plt.close()
            
            logger.info(f"Bankroll evolution plot saved to {plot_path}")
            
        except Exception as e:
            logger.error(f"Error creating plot: {str(e)}") 