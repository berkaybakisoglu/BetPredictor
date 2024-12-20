"""Betting performance evaluation module."""
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config.config import BettingConfig

logger = logging.getLogger(__name__)

class BettingEvaluator:
    """Evaluates betting performance across markets."""
    
    def __init__(self, config: BettingConfig):
        self.config = config
        self.bankroll = self.config.initial_bankroll
        self.bets_history: List[Dict] = []
        self.market_pnl: Dict[str, float] = {
            market: 0.0 for market in self.config.market_loss_limits
        }
    
    def evaluate_predictions(self, predictions: Dict[str, pd.DataFrame], test_data: pd.DataFrame) -> Dict[str, Dict]:
        """Evaluate predictions across all markets."""
        metrics = {}
        
        logger.info("\n" + "="*50)
        logger.info("PREDICTION ACCURACY METRICS")
        logger.info("="*50)
        
        for market in predictions:
            metrics[market] = {}
            
            if market == 'match_result':
                # Calculate pure prediction accuracy
                y_true = test_data['FTR']
                y_pred = predictions[market]['Predicted']
                correct_predictions = (y_pred == y_true).sum()
                total_predictions = len(y_true)
                accuracy = correct_predictions / total_predictions
                
                # Log detailed predictions
                logger.info("\nDetailed Match Result Predictions:")
                logger.info("=" * 100)
                logger.info(f"{'Date':<12} {'Home':<20} {'Away':<20} {'Score':<8} {'Pred':<6} {'Actual':<6} {'Odds':<6} {'Conf':<6}")
                logger.info("-" * 100)
                
                for idx, row in predictions[market].iterrows():
                    actual = test_data.loc[idx, 'FTR']
                    date = test_data.loc[idx, 'Date'].strftime('%Y-%m-%d')
                    home_team = test_data.loc[idx, 'HomeTeam']
                    away_team = test_data.loc[idx, 'AwayTeam']
                    score = f"{test_data.loc[idx, 'FTHG']}-{test_data.loc[idx, 'FTAG']}"
                    pred = row['Predicted']
                    confidence = row['Confidence']
                    
                    # Get corresponding odds
                    if pred == 'H':
                        odds = row['B365H']
                    elif pred == 'D':
                        odds = row['B365D']
                    else:
                        odds = row['B365A']
                    
                    logger.info(f"{date:<12} {home_team:<20} {away_team:<20} {score:<8} {pred:<6} {actual:<6} {odds:<6.2f} {confidence:<6.2f}")
                    
                    if pred == actual:
                        logger.info(f"✓ Correct prediction! Odds: {odds:.2f}, Profit: {odds - 1:.2f}")
                    else:
                        logger.info(f"✗ Wrong prediction. Loss: -1.00")
                    logger.info("-" * 100)
                
                # Log aggregate metrics
                logger.info(f"\nMatch Result Accuracy:")
                logger.info(f"Total Predictions: {total_predictions}")
                logger.info(f"Correct Predictions: {correct_predictions}")
                logger.info(f"Accuracy: {accuracy:.2%}")
                
                metrics[market]['prediction_accuracy'] = accuracy
                
            elif market == 'over_under':
                # Calculate over/under prediction accuracy
                y_true = (test_data['FTHG'] + test_data['FTAG'] > 2.5).astype(int)
                y_pred = (predictions[market]['Over_Prob'] > 0.5).astype(int)
                correct_predictions = (y_pred == y_true).sum()
                total_predictions = len(y_true)
                accuracy = correct_predictions / total_predictions
                
                # Log detailed predictions
                logger.info("\nDetailed Over/Under Predictions:")
                logger.info("=" * 100)
                logger.info(f"{'Date':<12} {'Home':<20} {'Away':<20} {'Score':<8} {'Goals':<6} {'Pred':<6} {'Actual':<6} {'Odds':<6} {'Conf':<6}")
                logger.info("-" * 100)
                
                for idx, row in predictions[market].iterrows():
                    date = test_data.loc[idx, 'Date'].strftime('%Y-%m-%d')
                    home_team = test_data.loc[idx, 'HomeTeam']
                    away_team = test_data.loc[idx, 'AwayTeam']
                    total_goals = test_data.loc[idx, 'FTHG'] + test_data.loc[idx, 'FTAG']
                    score = f"{test_data.loc[idx, 'FTHG']}-{test_data.loc[idx, 'FTAG']}"
                    pred = 'Over' if row['Over_Prob'] > 0.5 else 'Under'
                    actual = 'Over' if total_goals > 2.5 else 'Under'
                    confidence = max(row['Over_Prob'], 1 - row['Over_Prob'])
                    odds = row['B365>2.5'] if pred == 'Over' else row['B365<2.5']
                    
                    logger.info(f"{date:<12} {home_team:<20} {away_team:<20} {score:<8} {total_goals:<6} {pred:<6} {actual:<6} {odds:<6.2f} {confidence:<6.2f}")
                    
                    if pred == actual:
                        logger.info(f"✓ Correct prediction! Odds: {odds:.2f}, Profit: {odds - 1:.2f}")
                    else:
                        logger.info(f"✗ Wrong prediction. Loss: -1.00")
                    logger.info("-" * 100)
                
                # Log aggregate metrics
                logger.info(f"\nOver/Under Accuracy:")
                logger.info(f"Total Predictions: {total_predictions}")
                logger.info(f"Correct Predictions: {correct_predictions}")
                logger.info(f"Accuracy: {accuracy:.2%}")
                
                metrics[market]['prediction_accuracy'] = accuracy
                
            elif market in ['corners', 'cards']:
                # Calculate regression accuracy metrics
                y_true = test_data['HC'] + test_data['AC'] if market == 'corners' else test_data['HY'] + test_data['AY']
                y_pred = predictions[market]['prediction']
                
                mae = mean_absolute_error(y_true, y_pred)
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                
                # Calculate percentage of predictions within different margins
                margin_1 = np.mean(abs(y_true - y_pred) <= 1)
                margin_2 = np.mean(abs(y_true - y_pred) <= 2)
                margin_3 = np.mean(abs(y_true - y_pred) <= 3)
                
                # Log detailed predictions
                logger.info(f"\nDetailed {market.title()} Predictions:")
                logger.info("=" * 100)
                logger.info(f"{'Date':<12} {'Home':<20} {'Away':<20} {'Score':<8} {'Pred':<8} {'Actual':<8} {'Diff':<8} {'Conf':<6}")
                logger.info("-" * 100)
                
                for idx, row in predictions[market].iterrows():
                    date = test_data.loc[idx, 'Date'].strftime('%Y-%m-%d')
                    home_team = test_data.loc[idx, 'HomeTeam']
                    away_team = test_data.loc[idx, 'AwayTeam']
                    score = f"{test_data.loc[idx, 'FTHG']}-{test_data.loc[idx, 'FTAG']}"
                    pred = row['prediction']
                    actual = y_true[idx]
                    diff = abs(pred - actual)
                    confidence = row['confidence']
                    
                    logger.info(f"{date:<12} {home_team:<20} {away_team:<20} {score:<8} {pred:<8.1f} {actual:<8} {diff:<8.1f} {confidence:<6.2f}")
                    
                    margin = 1 if market == 'cards' else 2
                    if diff <= margin:
                        logger.info(f"✓ Good prediction! Predicted: {pred:.1f}, Actual: {actual}")
                    else:
                        logger.info(f"✗ Prediction off by {diff:.1f} {market}")
                    logger.info("-" * 100)
                
                # Log aggregate metrics
                logger.info(f"\n{market.title()} Prediction Accuracy:")
                logger.info(f"Mean Absolute Error: {mae:.2f}")
                logger.info(f"Root Mean Square Error: {rmse:.2f}")
                logger.info(f"Within ±1 {market}: {margin_1:.2%}")
                logger.info(f"Within ±2 {market}: {margin_2:.2%}")
                logger.info(f"Within ±3 {market}: {margin_3:.2%}")
                
                metrics[market].update({
                    'mae': mae,
                    'rmse': rmse,
                    'within_1': margin_1,
                    'within_2': margin_2,
                    'within_3': margin_3
                })
            
            # Add betting performance metrics
            betting_metrics = self._evaluate_market(predictions[market], test_data, market)
            metrics[market].update(betting_metrics)
            
            if metrics[market].get('bets', 0) > 0:
                logger.info(f"\n{market.upper()} Betting Performance:")
                logger.info(f"Total Bets: {metrics[market]['bets']}")
                logger.info(f"Wins: {metrics[market]['wins']}")
                logger.info(f"Win Rate: {metrics[market]['win_rate']:.2%}")
                logger.info(f"ROI: {metrics[market]['roi']:.2%}")
                logger.info(f"P&L: {metrics[market]['pnl']:.2f}")
                logger.info(f"Max Drawdown: {metrics[market]['max_drawdown']:.2f}")
        
        return metrics
    
    def _evaluate_market(self, predictions: pd.DataFrame, test_data: pd.DataFrame, market: str) -> Dict[str, float]:
        """Evaluate predictions for a specific market.
        
        Args:
            predictions: Predictions for the market
            test_data: Actual results
            market: Market being evaluated
            
        Returns:
            Dictionary with market-specific metrics
        """
        bets = self._place_bets(market, predictions)
        
        if bets.empty:
            return {
                'bets': 0,
                'wins': 0,
                'win_rate': 0.0,
                'roi': 0.0,
                'pnl': 0.0,
                'max_drawdown': 0.0
            }
        
        # Merge bets with results
        result_columns = [
            'Date', 'HomeTeam', 'AwayTeam', 'FTR', 'HTR',
            'FTHG', 'FTAG', 'HTHG', 'HTAG',
            'HC', 'AC', 'HY', 'AY'
        ]
        bets = bets.merge(test_data[result_columns], 
                         on=['Date', 'HomeTeam', 'AwayTeam'])
        
        # Calculate bet outcomes
        bets['Won'] = self._calculate_wins(market, bets)
        bets['PnL'] = bets.apply(
            lambda x: x['Stake'] * (x['Odds'] - 1) if x['Won'] else -x['Stake'],
            axis=1
        )
        
        # Calculate metrics
        total_bets = len(bets)
        wins = bets['Won'].sum()
        pnl = bets['PnL'].sum()
        roi = pnl / bets['Stake'].sum() if total_bets > 0 else 0.0
        
        # Calculate drawdown
        cumulative_pnl = bets['PnL'].cumsum()
        rolling_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - rolling_max
        max_drawdown = abs(drawdown.min())
        
        # Store bet history
        self.bets_history.extend(bets.to_dict('records'))
        
        return {
            'bets': total_bets,
            'wins': wins,
            'win_rate': wins / total_bets if total_bets > 0 else 0.0,
            'roi': roi,
            'pnl': pnl,
            'max_drawdown': max_drawdown
        }
    
    def _place_bets(self, market: str, predictions: pd.DataFrame) -> pd.DataFrame:
        """Place bets based on predictions and thresholds.
        
        Args:
            market: Market to bet on
            predictions: Predictions for the market
            
        Returns:
            DataFrame with placed bets
        """
        threshold = self.config.thresholds[market]['confidence']
        min_odds = self.config.thresholds[market]['min_odds']
        max_stake_pct = self.config.thresholds[market]['max_stake_pct']
        
        bets = []
        
        if market == 'match_result':
            for _, pred in predictions.iterrows():
                # Get highest probability and corresponding outcome
                probs = {
                    'Home': pred['Home_Prob'],
                    'Draw': pred['Draw_Prob'],
                    'Away': pred['Away_Prob']
                }
                best_outcome = max(probs.items(), key=lambda x: x[1])
                
                # Get corresponding odds
                odds_map = {
                    'Home': pred['B365H'],
                    'Draw': pred['B365D'],
                    'Away': pred['B365A']
                }
                
                # Calculate value
                value_map = {
                    outcome: prob * odds_map[outcome]
                    for outcome, prob in probs.items()
                }
                
                # Only bet if we have both high confidence and good value
                outcome, confidence = best_outcome
                odds = odds_map[outcome]
                value = value_map[outcome]
                
                if (confidence > threshold and 
                    odds >= min_odds and 
                    value > 1.1):  # Value threshold
                    
                    bets.append({
                        'Date': pred['Date'],
                        'HomeTeam': pred['HomeTeam'],
                        'AwayTeam': pred['AwayTeam'],
                        'Market': market,
                        'Bet': outcome,
                        'Confidence': confidence,
                        'Odds': odds,
                        'Value': value,
                        'Stake': self._calculate_stake(confidence, odds, max_stake_pct)
                    })
        
        elif market == 'over_under':
            for _, pred in predictions.iterrows():
                if ('B365>2.5' in pred and 
                    pred['Over_Prob'] > threshold and 
                    pred['B365>2.5'] >= min_odds):
                    bets.append({
                        'Date': pred['Date'],
                        'HomeTeam': pred['HomeTeam'],
                        'AwayTeam': pred['AwayTeam'],
                        'Market': market,
                        'Bet': 'Over',
                        'Confidence': pred['Over_Prob'],
                        'Odds': pred['B365>2.5'],
                        'Stake': self._calculate_stake(pred['Over_Prob'],
                                                     pred['B365>2.5'],
                                                     max_stake_pct)
                    })
                elif ('B365<2.5' in pred and 
                      pred['Under_Prob'] > threshold and 
                      pred['B365<2.5'] >= min_odds):
                    bets.append({
                        'Date': pred['Date'],
                        'HomeTeam': pred['HomeTeam'],
                        'AwayTeam': pred['AwayTeam'],
                        'Market': market,
                        'Bet': 'Under',
                        'Confidence': pred['Under_Prob'],
                        'Odds': pred['B365<2.5'],
                        'Stake': self._calculate_stake(pred['Under_Prob'],
                                                     pred['B365<2.5'],
                                                     max_stake_pct)
                    })
        
        return pd.DataFrame(bets) if bets else pd.DataFrame()
    
    def _calculate_stake(self, probability: float, odds: float, max_stake_pct: float) -> float:
        """Calculate stake using Kelly Criterion.
        
        Args:
            probability: Predicted probability
            odds: Betting odds
            max_stake_pct: Maximum stake as percentage of bankroll
            
        Returns:
            Stake amount
        """
        # Kelly stake = (bp - q) / b
        # where: b = odds - 1, p = probability of win, q = probability of loss
        b = odds - 1
        p = probability
        q = 1 - p
        
        kelly = (b * p - q) / b
        
        # Apply Kelly fraction and max stake limit
        stake = min(
            kelly * self.config.kelly_fraction,
            max_stake_pct
        ) * self.bankroll
        
        return max(0, stake)
    
    def _calculate_wins(self, market: str, bets: pd.DataFrame) -> pd.Series:
        """Calculate which bets won.
        
        Args:
            market: Market being evaluated
            bets: DataFrame with bets
            
        Returns:
            Series indicating if each bet won
        """
        if market == 'match_result':
            return ((bets['Bet'] == 'Home') & (bets['FTR'] == 'H') |
                   (bets['Bet'] == 'Draw') & (bets['FTR'] == 'D') |
                   (bets['Bet'] == 'Away') & (bets['FTR'] == 'A'))
        elif market == 'over_under':
            total_goals = bets['FTHG'] + bets['FTAG']
            return ((bets['Bet'] == 'Over') & (total_goals > 2.5) |
                   (bets['Bet'] == 'Under') & (total_goals < 2.5))
        else:
            return pd.Series([False] * len(bets))
    
    def plot_performance(self, save_path: Optional[Path] = None) -> None:
        """Plot betting performance metrics.
        
        Args:
            save_path: Optional path to save plots
        """
        if not self.bets_history:
            print("No bets to plot")
            return
            
        bets_df = pd.DataFrame(self.bets_history)
        bets_df['Cumulative_PnL'] = bets_df['PnL'].cumsum()
        
        # Set up the plot style
        plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Cumulative P&L
        ax1.plot(range(len(bets_df)), bets_df['Cumulative_PnL'])
        ax1.set_title('Cumulative P&L')
        ax1.set_xlabel('Number of Bets')
        ax1.set_ylabel('Profit/Loss')
        ax1.grid(True)
        
        # Plot 2: P&L by market
        market_pnl = bets_df.groupby('Market')['PnL'].sum()
        market_pnl.plot(kind='bar', ax=ax2)
        ax2.set_title('P&L by Market')
        ax2.set_xlabel('Market')
        ax2.set_ylabel('Profit/Loss')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            save_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                save_path / f'performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            )
        else:
            plt.show() 