import pandas as pd
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class BettingEvaluator:
    def __init__(self, config):
        self.config = config
        self.initial_bankroll = config.initial_bankroll
        self.current_bankroll = self.initial_bankroll
        self.bet_history = []
        self.kelly_fraction = config.kelly_fraction  # Usually 0.25 for conservative betting
    
    def _calculate_kelly_stake(self, prob, odds, bankroll):
        """Calculate Kelly Criterion stake."""
        q = 1 - prob  # Probability of losing
        p = prob  # Probability of winning
        b = odds - 1  # Decimal odds minus 1
        
        kelly = (b * p - q) / b
        kelly = max(0, kelly)  # No negative bets
        kelly = min(kelly, 0.2)  # Cap at 20% of bankroll
        
        return kelly * self.kelly_fraction * bankroll
    
    def evaluate(self, predictions, actual_results):
        evaluation_results = {}
        self.current_bankroll = self.initial_bankroll
        self.bet_history = []
        
        for market, pred_df in predictions.items():
            logger.info(f"\nProcessing market: {market}")
            
            # Handle both Series and DataFrame inputs
            if isinstance(pred_df, pd.Series):
                logger.info(f"Prediction data shape: {pred_df.shape}")
                logger.info("Input is a Series - skipping column analysis")
                continue  # Skip non-betting markets
            else:
                logger.info(f"Prediction data shape: {pred_df.shape}")
                logger.info(f"Available columns: {pred_df.columns.tolist()}")
                
                # Log some basic statistics about the predictions
                if 'Confidence' in pred_df.columns:
                    conf_stats = pred_df['Confidence'].describe()
                    logger.info(f"\nConfidence statistics:")
                    logger.info(f"- Mean: {conf_stats['mean']:.4f}")
                    logger.info(f"- Min: {conf_stats['min']:.4f}")
                    logger.info(f"- Max: {conf_stats['max']:.4f}")
                    logger.info(f"- High confidence predictions (>= {self.config.confidence_threshold}): {(pred_df['Confidence'] >= self.config.confidence_threshold).sum()}")
                
                # Map match_result to match_odds for evaluation
                eval_market = 'match_odds' if market == 'match_result' else market
                logger.info(f"Evaluating as market: {eval_market}")
                
                if eval_market not in self.config.markets:
                    logger.info(f"Skipping market {eval_market} - not in configured markets: {self.config.markets}")
                    continue
                
                try:
                    if market in ['corners', 'cards']:
                        results = self._evaluate_regression(pred_df, actual_results, market)
                    else:
                        results = self._evaluate_classification(pred_df, actual_results, market)
                    
                    evaluation_results[market] = results
                    
                except Exception as e:
                    logger.error(f"Error evaluating {market} predictions: {str(e)}", exc_info=True)
                    continue
        
        # Add bankroll metrics
        evaluation_results['bankroll'] = {
            'final_balance': self.current_bankroll,
            'max_drawdown': self._calculate_max_drawdown(),
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'profit_factor': self._calculate_profit_factor()
        }
        
        return evaluation_results
    
    def _evaluate_regression(self, predictions, actuals, market):
        merged = predictions.merge(
            actuals[['Date', 'HomeTeam', 'AwayTeam', market]],
            on=['Date', 'HomeTeam', 'AwayTeam']
        )
        
        true_values = merged[market]
        pred_values = merged['prediction']
        
        mse = np.mean((true_values - pred_values) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(true_values - pred_values))
        
        within_margin = np.mean(np.abs(true_values - pred_values) <= self.config.regression_margin)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'within_margin': within_margin
        }
    
    def _evaluate_classification(self, predictions, actual_results, market):
        logger.info(f"\nEvaluating classification predictions for {market}")
        
        # Merge predictions with actual results and betting odds
        merged_data = predictions.merge(
            actual_results[['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'B365H', 'B365D', 'B365A']],
            on=['Date', 'HomeTeam', 'AwayTeam']
        )
        logger.info(f"Merged data shape: {merged_data.shape}")
        
        # Initialize metrics
        total_bets = 0
        total_wins = 0
        total_stake = 0
        total_returns = 0
        
        logger.info("\nAnalyzing betting opportunities...")
        logger.info(f"Confidence threshold: {self.config.confidence_threshold}")
        logger.info(f"Minimum edge: {self.config.min_edge}")
        logger.info(f"Kelly fraction: {self.config.kelly_fraction}")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"MATCH PREDICTIONS AND ANALYSIS")
        logger.info(f"{'='*80}")
        
        for idx, row in merged_data.iterrows():
            try:
                # Calculate implied probabilities from odds
                home_implied_prob = 1 / row['B365H']
                draw_implied_prob = 1 / row['B365D']
                away_implied_prob = 1 / row['B365A']
                
                # Get predicted probabilities
                home_pred = row['Home_Prob'] if 'Home_Prob' in row else 0
                draw_pred = row['Draw_Prob'] if 'Draw_Prob' in row else 0
                away_pred = row['Away_Prob'] if 'Away_Prob' in row else 0
                
                # Determine predicted outcome
                probs = {'H': home_pred, 'D': draw_pred, 'A': away_pred}
                predicted_outcome = max(probs, key=probs.get)
                max_prob = max(probs.values())
                
                # Calculate edges
                home_edge = home_pred - home_implied_prob
                draw_edge = draw_pred - draw_implied_prob
                away_edge = away_pred - away_implied_prob
                
                # Log prediction for every match
                logger.info(f"\nMatch: {row['HomeTeam']} vs {row['AwayTeam']} ({row['Date']})")
                logger.info(f"Prediction: {predicted_outcome} (Confidence: {row['Confidence']:.3f})")
                logger.info(f"Predicted probabilities - H: {home_pred:.3f}, D: {draw_pred:.3f}, A: {away_pred:.3f}")
                logger.info(f"Implied probabilities - H: {home_implied_prob:.3f}, D: {draw_implied_prob:.3f}, A: {away_implied_prob:.3f}")
                logger.info(f"Edges - H: {home_edge:.3f}, D: {draw_edge:.3f}, A: {away_edge:.3f}")
                logger.info(f"Actual result: {row['FTR']}")
                
                # Log why bet was not placed if confidence threshold not met
                if row['Confidence'] < self.config.confidence_threshold:
                    logger.info(f"No bet placed - Confidence {row['Confidence']:.3f} below threshold {self.config.confidence_threshold}")
                    continue
                
                # Check for value bets and log decisions
                bet_placed = False
                
                if home_edge > self.config.min_edge and row['B365H'] >= self.config.min_odds and row['B365H'] <= self.config.max_odds:
                    bet_placed = True
                    stake = self._calculate_kelly_stake(home_pred, row['B365H'], self.current_bankroll)
                    total_stake += stake
                    total_bets += 1
                    
                    profit_loss = stake * (row['B365H'] - 1) if row['FTR'] == 'H' else -stake
                    if row['FTR'] == 'H':
                        total_wins += 1
                        total_returns += stake * row['B365H']
                        self.current_bankroll += stake * (row['B365H'] - 1)
                        result = 'WIN'
                    else:
                        self.current_bankroll -= stake
                        result = 'LOSS'
                        
                    logger.info(f"Bet placed - Home win")
                    logger.info(f"Odds: {row['B365H']:.2f}, Edge: {home_edge:.3f}")
                    logger.info(f"Stake: {stake:.2f}, Result: {result}, Profit/Loss: {profit_loss:.2f}")
                    
                    self.bet_history.append({
                        'date': row['Date'],
                        'match': f"{row['HomeTeam']} vs {row['AwayTeam']}",
                        'bet_type': 'Home',
                        'odds': row['B365H'],
                        'stake': stake,
                        'result': result,
                        'profit_loss': profit_loss
                    })
                elif predicted_outcome == 'H':
                    if row['B365H'] < self.config.min_odds:
                        logger.info(f"No home bet - Odds {row['B365H']:.2f} below minimum {self.config.min_odds}")
                    elif row['B365H'] > self.config.max_odds:
                        logger.info(f"No home bet - Odds {row['B365H']:.2f} above maximum {self.config.max_odds}")
                    elif home_edge <= self.config.min_edge:
                        logger.info(f"No home bet - Edge {home_edge:.3f} below minimum {self.config.min_edge}")
                
                if draw_edge > self.config.min_edge and row['B365D'] >= self.config.min_odds and row['B365D'] <= self.config.max_odds:
                    bet_placed = True
                    stake = self._calculate_kelly_stake(draw_pred, row['B365D'], self.current_bankroll)
                    total_stake += stake
                    total_bets += 1
                    
                    profit_loss = stake * (row['B365D'] - 1) if row['FTR'] == 'D' else -stake
                    if row['FTR'] == 'D':
                        total_wins += 1
                        total_returns += stake * row['B365D']
                        self.current_bankroll += stake * (row['B365D'] - 1)
                        result = 'WIN'
                    else:
                        self.current_bankroll -= stake
                        result = 'LOSS'
                        
                    logger.info(f"Bet placed - Draw")
                    logger.info(f"Odds: {row['B365D']:.2f}, Edge: {draw_edge:.3f}")
                    logger.info(f"Stake: {stake:.2f}, Result: {result}, Profit/Loss: {profit_loss:.2f}")
                    
                    self.bet_history.append({
                        'date': row['Date'],
                        'match': f"{row['HomeTeam']} vs {row['AwayTeam']}",
                        'bet_type': 'Draw',
                        'odds': row['B365D'],
                        'stake': stake,
                        'result': result,
                        'profit_loss': profit_loss
                    })
                elif predicted_outcome == 'D':
                    if row['B365D'] < self.config.min_odds:
                        logger.info(f"No draw bet - Odds {row['B365D']:.2f} below minimum {self.config.min_odds}")
                    elif row['B365D'] > self.config.max_odds:
                        logger.info(f"No draw bet - Odds {row['B365D']:.2f} above maximum {self.config.max_odds}")
                    elif draw_edge <= self.config.min_edge:
                        logger.info(f"No draw bet - Edge {draw_edge:.3f} below minimum {self.config.min_edge}")
                
                if away_edge > self.config.min_edge and row['B365A'] >= self.config.min_odds and row['B365A'] <= self.config.max_odds:
                    bet_placed = True
                    stake = self._calculate_kelly_stake(away_pred, row['B365A'], self.current_bankroll)
                    total_stake += stake
                    total_bets += 1
                    
                    profit_loss = stake * (row['B365A'] - 1) if row['FTR'] == 'A' else -stake
                    if row['FTR'] == 'A':
                        total_wins += 1
                        total_returns += stake * row['B365A']
                        self.current_bankroll += stake * (row['B365A'] - 1)
                        result = 'WIN'
                    else:
                        self.current_bankroll -= stake
                        result = 'LOSS'
                        
                    logger.info(f"Bet placed - Away win")
                    logger.info(f"Odds: {row['B365A']:.2f}, Edge: {away_edge:.3f}")
                    logger.info(f"Stake: {stake:.2f}, Result: {result}, Profit/Loss: {profit_loss:.2f}")
                    
                    self.bet_history.append({
                        'date': row['Date'],
                        'match': f"{row['HomeTeam']} vs {row['AwayTeam']}",
                        'bet_type': 'Away',
                        'odds': row['B365A'],
                        'stake': stake,
                        'result': result,
                        'profit_loss': profit_loss
                    })
                elif predicted_outcome == 'A':
                    if row['B365A'] < self.config.min_odds:
                        logger.info(f"No away bet - Odds {row['B365A']:.2f} below minimum {self.config.min_odds}")
                    elif row['B365A'] > self.config.max_odds:
                        logger.info(f"No away bet - Odds {row['B365A']:.2f} above maximum {self.config.max_odds}")
                    elif away_edge <= self.config.min_edge:
                        logger.info(f"No away bet - Edge {away_edge:.3f} below minimum {self.config.min_edge}")
                
                if not bet_placed and row['Confidence'] >= self.config.confidence_threshold:
                    logger.info("No bet placed despite high confidence - check edges and odds limits")
                
            except Exception as e:
                logger.error(f"Error processing bet for match {row['HomeTeam']} vs {row['AwayTeam']}: {str(e)}")
                continue
        
        logger.info(f"\nBetting evaluation complete")
        logger.info(f"Total bets placed: {total_bets}")
        logger.info(f"Total stake: {total_stake:.2f}")
        logger.info(f"Total returns: {total_returns:.2f}")
        logger.info(f"ROI: {((total_returns - total_stake) / total_stake * 100):.2f}% if total_stake > 0 else '0.00%'")
        
        roi = ((total_returns - total_stake) / total_stake) if total_stake > 0 else 0
        
        return {
            'total_bets': total_bets,
            'total_stake': total_stake,
            'total_returns': total_returns,
            'roi': roi,
            'win_rate': total_wins / total_bets if total_bets > 0 else 0
        }
    
    def _calculate_max_drawdown(self):
        """Calculate maximum drawdown from bet history."""
        if not self.bet_history:
            return 0
            
        balances = [self.initial_bankroll] + [bet['starting_bankroll'] for bet in self.bet_history]
        peak = self.initial_bankroll
        max_drawdown = 0
        
        for balance in balances:
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak
            max_drawdown = max(max_drawdown, drawdown)
            
        return max_drawdown
    
    def _calculate_sharpe_ratio(self):
        """Calculate Sharpe ratio of betting returns."""
        if not self.bet_history:
            return 0
            
        returns = []
        for bet in self.bet_history:
            if bet['outcome'] == 'WIN':
                returns.append((bet['stake'] * (bet['odds'] - 1)) / bet['starting_bankroll'])
            else:
                returns.append(-bet['stake'] / bet['starting_bankroll'])
                
        if not returns:
            return 0
            
        return np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)  # Annualized
    
    def _calculate_profit_factor(self):
        """Calculate profit factor (gross wins / gross losses)."""
        if not self.bet_history:
            return 0
            
        gross_wins = sum(bet['stake'] * (bet['odds'] - 1) 
                        for bet in self.bet_history 
                        if bet['outcome'] == 'WIN')
        gross_losses = sum(bet['stake'] 
                          for bet in self.bet_history 
                          if bet['outcome'] == 'LOSS')
                          
        return gross_wins / gross_losses if gross_losses > 0 else float('inf')