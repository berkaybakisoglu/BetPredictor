"""Betting performance evaluation module."""
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config.config import BettingConfig
from visualization.visualizer import ResultVisualizer

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
        self.default_stake = 1.0  # Default stake size for each bet
        
        # Detailed tracking of bets and PnL
        self.bet_details: List[Dict] = []
        
        # Define odds ranges for analysis
        self.odds_ranges = [
            (1.0, 1.5),
            (1.5, 2.0),
            (2.0, 2.5),
            (2.5, 3.0),
            (3.0, 4.0),
            (4.0, float('inf'))
        ]
        
        # Initialize odds range statistics
        self.odds_stats = {
            'selective': {range_: {'wins': 0, 'total': 0, 'pnl': 0.0} 
                        for range_ in self.odds_ranges},
            'all_matches': {range_: {'wins': 0, 'total': 0, 'pnl': 0.0} 
                          for range_ in self.odds_ranges}
        }
        
        # Initialize visualizer
        self.visualizer = ResultVisualizer(Path('output/visualizations'))
    
    def _calculate_bet_profit(self, bet_amount: float, odds: float, is_winner: bool) -> float:
        """Calculate profit/loss for a single bet.
        
        Args:
            bet_amount: Stake amount
            odds: Betting odds
            is_winner: Whether the bet won
            
        Returns:
            Profit (positive) or loss (negative)
        """
        if is_winner:
            return bet_amount * (odds - 1)  # (odds * stake) - stake
        return -bet_amount  # Lost stake
    
    def evaluate(self, predictions: Dict[str, pd.DataFrame], actual_results: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Evaluate predictions for all markets.
        
        Args:
            predictions: Dictionary of predictions for each market
            actual_results: DataFrame with actual results
            
        Returns:
            Dictionary with evaluation metrics for each market
        """
        metrics = {}
        
        # Add league information to actual_results if not present
        if 'League' not in actual_results.columns:
            actual_results['League'] = actual_results['Division']
        
        # Evaluate match result predictions
        if 'match_result' in predictions:
            logger.info("\nMatch Result Accuracy:")
            accuracy = self._evaluate_classification(
                predictions['match_result']['Predicted'],
                actual_results['FTR']
            )
            betting_metrics = self.evaluate_betting_performance(predictions, actual_results, 'match_result')
            metrics['match_result'] = {
                'prediction_accuracy': accuracy,
                **betting_metrics['selective'],
                'all_matches': betting_metrics['all_matches']
            }
            
            # Plot feature importance if available
            if hasattr(self, 'feature_importances') and 'match_result' in self.feature_importances:
                self.visualizer.plot_feature_importance(
                    self.feature_importances['match_result'],
                    'match_result'
                )
        
        # Evaluate over/under predictions
        if 'over_under' in predictions:
            logger.info("\nOver/Under Accuracy:")
            over_under_preds = (predictions['over_under']['Over_Prob'] > 
                              predictions['over_under']['Under_Prob']).astype(int)
            actual_over = (actual_results['FTHG'] + 
                          actual_results['FTAG'] > 2.5).astype(int)
            accuracy = self._evaluate_classification(over_under_preds, actual_over)
            betting_metrics = self.evaluate_betting_performance(predictions, actual_results, 'over_under')
            metrics['over_under'] = {
                'prediction_accuracy': accuracy,
                **betting_metrics['selective'],
                'all_matches': betting_metrics['all_matches']
            }
            
            # Plot feature importance if available
            if hasattr(self, 'feature_importances') and 'over_under' in self.feature_importances:
                self.visualizer.plot_feature_importance(
                    self.feature_importances['over_under'],
                    'over_under'
                )
        
        # Evaluate corners predictions
        if 'corners' in predictions:
            logger.info("\nCorners Prediction Accuracy:")
            actual_corners = actual_results['HC'] + actual_results['AC']
            regression_metrics = self._evaluate_regression(
                predictions['corners']['prediction'],
                actual_corners
            )
            betting_metrics = self.evaluate_betting_performance(predictions, actual_results, 'corners')
            metrics['corners'] = {
                **regression_metrics,
                **betting_metrics['selective'],
                'all_matches': betting_metrics['all_matches']
            }
            
            # Plot feature importance if available
            if hasattr(self, 'feature_importances') and 'corners' in self.feature_importances:
                self.visualizer.plot_feature_importance(
                    self.feature_importances['corners'],
                    'corners'
                )
        
        # Evaluate cards predictions
        if 'cards' in predictions:
            logger.info("\nCards Prediction Accuracy:")
            actual_cards = actual_results['HY'] + actual_results['AY']
            regression_metrics = self._evaluate_regression(
                predictions['cards']['prediction'],
                actual_cards
            )
            betting_metrics = self.evaluate_betting_performance(predictions, actual_results, 'cards')
            metrics['cards'] = {
                **regression_metrics,
                **betting_metrics['selective'],
                'all_matches': betting_metrics['all_matches']
            }
            
            # Plot feature importance if available
            if hasattr(self, 'feature_importances') and 'cards' in self.feature_importances:
                self.visualizer.plot_feature_importance(
                    self.feature_importances['cards'],
                    'cards'
                )
        
        # Create bet history DataFrame
        if self.bet_details:
            bet_history = pd.DataFrame(self.bet_details)
            
            # Create comprehensive visualizations
            self.visualizer.plot_pnl_evolution(bet_history)
            self.visualizer.plot_win_rate_by_odds(bet_history)
            self.visualizer.plot_roi_by_league(bet_history)
            self.visualizer.plot_confidence_analysis(predictions)
            
            # Create summary report
            self.visualizer.create_summary_report(bet_history, predictions)
        
        return metrics
    
    def _evaluate_classification(self, y_pred: pd.Series, y_true: pd.Series) -> float:
        """Evaluate classification accuracy.
        
        Args:
            y_pred: Predicted labels
            y_true: Actual labels
            
        Returns:
            Accuracy score
        """
        correct_predictions = (y_pred == y_true).sum()
        total_predictions = len(y_true)
        accuracy = correct_predictions / total_predictions
        return accuracy
    
    def _evaluate_regression(self, y_pred: pd.Series, y_true: pd.Series) -> Dict[str, float]:
        """Evaluate regression metrics.
        
        Args:
            y_pred: Predicted values
            y_true: Actual values
            
        Returns:
            Dictionary with regression metrics
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # Calculate percentage of predictions within different margins
        margin_1 = np.mean(abs(y_true - y_pred) <= 1)
        margin_2 = np.mean(abs(y_true - y_pred) <= 2)
        margin_3 = np.mean(abs(y_true - y_pred) <= 3)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'within_1': margin_1,
            'within_2': margin_2,
            'within_3': margin_3
        }
    
    def _get_odds_range(self, odds: float) -> Tuple[float, float]:
        """Get the odds range for given odds."""
        for odds_range in self.odds_ranges:
            if odds_range[0] <= odds < odds_range[1]:
                return odds_range
        return self.odds_ranges[-1]  # Return highest range if no match found
    
    def _update_odds_stats(self, odds: float, is_winner: bool, profit: float, strategy: str):
        """Update statistics for odds ranges."""
        odds_range = self._get_odds_range(odds)
        self.odds_stats[strategy][odds_range]['total'] += 1
        if is_winner:
            self.odds_stats[strategy][odds_range]['wins'] += 1
        self.odds_stats[strategy][odds_range]['pnl'] += profit
    
    def evaluate_betting_performance(self, predictions: Dict[str, pd.DataFrame], 
                                   actual_results: pd.DataFrame, market: str) -> Dict[str, float]:
        """Evaluate betting performance for a market."""
        if market not in predictions:
            return {
                'bets': 0, 'wins': 0, 'win_rate': 0.0,
                'roi': 0.0, 'pnl': 0.0, 'max_drawdown': 0.0,
                'avg_odds': 0.0, 'avg_winning_odds': 0.0
            }
        
        # Get predictions for this market
        market_preds = predictions[market]
        
        # Initialize metrics for both strategies
        metrics = {
            'selective': {
                'bets': 0, 'wins': 0, 'win_rate': 0.0,
                'roi': 0.0, 'pnl': 0.0, 'max_drawdown': 0.0,
                'avg_odds': 0.0, 'avg_winning_odds': 0.0
            },
            'all_matches': {
                'bets': 0, 'wins': 0, 'win_rate': 0.0,
                'roi': 0.0, 'pnl': 0.0, 'max_drawdown': 0.0,
                'avg_odds': 0.0, 'avg_winning_odds': 0.0
            }
        }
        
        # Track running PnL and odds
        running_pnl_selective = [0]
        running_pnl_all = [0]
        odds_history = {'selective': [], 'all_matches': []}
        winning_odds = {'selective': [], 'all_matches': []}
        
        if market == 'match_result':
            # Selective betting strategy
            value_bets = (
                (market_preds['Home_Value'] > self.config.value_threshold) |
                (market_preds['Draw_Value'] > self.config.value_threshold) |
                (market_preds['Away_Value'] > self.config.value_threshold)
            )
            high_conf_bets = market_preds['Confidence'] > self.config.confidence_threshold
            bet_mask_selective = value_bets & high_conf_bets
            
            # All matches strategy
            bet_mask_all = pd.Series(True, index=market_preds.index)
            
            for bet_mask, strategy in [(bet_mask_selective, 'selective'), (bet_mask_all, 'all_matches')]:
                bets = market_preds[bet_mask]
                metrics[strategy]['bets'] = len(bets)
                
                if len(bets) > 0:
                    # Get actual results for bet matches
                    results = actual_results.loc[bets.index, 'FTR']
                    
                    # Calculate wins and PnL
                    pnl = []
                    for idx, bet in bets.iterrows():
                        result = results.loc[idx]
                        pred = bet['Predicted']
                        
                        # Get odds for the predicted outcome
                        if pred == 'H':
                            odds = bet['B365H']
                        elif pred == 'D':
                            odds = bet['B365D']
                        else:  # 'A'
                            odds = bet['B365A']
                        
                        odds_history[strategy].append(odds)
                        is_winner = (result == pred)
                        
                        if is_winner:
                            metrics[strategy]['wins'] += 1
                            winning_odds[strategy].append(odds)
                        
                        # Calculate profit/loss
                        profit = self._calculate_bet_profit(
                            bet_amount=self.default_stake,
                            odds=odds,
                            is_winner=is_winner
                        )
                        
                        # Store bet details
                        self.bet_details.append({
                            'date': bet['Date'],
                            'market': market,
                            'strategy': strategy,
                            'home_team': bet['HomeTeam'],
                            'away_team': bet['AwayTeam'],
                            'prediction': pred,
                            'actual': result,
                            'odds': odds,
                            'stake': self.default_stake,
                            'profit': profit,
                            'is_winner': is_winner,
                            'league': actual_results.loc[idx, 'League']
                        })
                        
                        pnl.append(profit)
                        if strategy == 'selective':
                            running_pnl_selective.append(running_pnl_selective[-1] + profit)
                        else:
                            running_pnl_all.append(running_pnl_all[-1] + profit)
                        
                        # Update odds range statistics
                        self._update_odds_stats(odds, is_winner, profit, strategy)
                    
                    # Calculate metrics
                    metrics[strategy]['win_rate'] = metrics[strategy]['wins'] / metrics[strategy]['bets']
                    metrics[strategy]['pnl'] = sum(pnl)
                    metrics[strategy]['roi'] = metrics[strategy]['pnl'] / (metrics[strategy]['bets'] * self.default_stake)
                    metrics[strategy]['avg_odds'] = np.mean(odds_history[strategy])
                    metrics[strategy]['avg_winning_odds'] = np.mean(winning_odds[strategy]) if winning_odds[strategy] else 0
                    
                    # Calculate max drawdown
                    if strategy == 'selective':
                        peak = max(running_pnl_selective)
                        drawdowns = [peak - val for val in running_pnl_selective]
                        metrics[strategy]['max_drawdown'] = max(drawdowns)
                    else:
                        peak = max(running_pnl_all)
                        drawdowns = [peak - val for val in running_pnl_all]
                        metrics[strategy]['max_drawdown'] = max(drawdowns)
        
        elif market == 'over_under':
            # Selective betting strategy
            value_bets = (
                (market_preds['Over_Value'] > self.config.value_threshold) |
                (market_preds['Under_Value'] > self.config.value_threshold)
            )
            high_conf_bets = market_preds['Confidence'] > self.config.thresholds['over_under']['confidence']
            bet_mask_selective = value_bets & high_conf_bets
            
            # All matches strategy
            bet_mask_all = pd.Series(True, index=market_preds.index)
            
            for bet_mask, strategy in [(bet_mask_selective, 'selective'), (bet_mask_all, 'all_matches')]:
                bets = market_preds[bet_mask]
                metrics[strategy]['bets'] = len(bets)
                
                if len(bets) > 0:
                    # Get actual results for bet matches
                    actual_goals = actual_results.loc[bets.index, ['FTHG', 'FTAG']].sum(axis=1)
                    actual_over = (actual_goals > 2.5).astype(str).map({
                        'True': 'Over',
                        'False': 'Under'
                    })
                    
                    # Calculate wins and PnL
                    pnl = []
                    for idx, bet in bets.iterrows():
                        result = actual_over.loc[idx]
                        pred = bet['Predicted']
                        
                        # Get odds for the predicted outcome
                        if pred == 'Over':
                            odds = bet['B365>2.5']
                        else:  # Under
                            odds = bet['B365<2.5']
                        
                        odds_history[strategy].append(odds)
                        is_winner = (result == pred)
                        
                        if is_winner:
                            metrics[strategy]['wins'] += 1
                            winning_odds[strategy].append(odds)
                        
                        # Calculate profit/loss
                        profit = self._calculate_bet_profit(
                            bet_amount=self.default_stake,
                            odds=odds,
                            is_winner=is_winner
                        )
                        
                        # Store bet details
                        self.bet_details.append({
                            'date': bet['Date'],
                            'market': market,
                            'strategy': strategy,
                            'home_team': bet['HomeTeam'],
                            'away_team': bet['AwayTeam'],
                            'prediction': pred,
                            'actual': result,
                            'odds': odds,
                            'stake': self.default_stake,
                            'profit': profit,
                            'is_winner': is_winner,
                            'league': actual_results.loc[idx, 'League']
                        })
                        
                        pnl.append(profit)
                        if strategy == 'selective':
                            running_pnl_selective.append(running_pnl_selective[-1] + profit)
                        else:
                            running_pnl_all.append(running_pnl_all[-1] + profit)
                        
                        # Update odds range statistics
                        self._update_odds_stats(odds, is_winner, profit, strategy)
                    
                    # Calculate metrics
                    metrics[strategy]['win_rate'] = metrics[strategy]['wins'] / metrics[strategy]['bets']
                    metrics[strategy]['pnl'] = sum(pnl)
                    metrics[strategy]['roi'] = metrics[strategy]['pnl'] / (metrics[strategy]['bets'] * self.default_stake)
                    metrics[strategy]['avg_odds'] = np.mean(odds_history[strategy])
                    metrics[strategy]['avg_winning_odds'] = np.mean(winning_odds[strategy]) if winning_odds[strategy] else 0
                    
                    # Calculate max drawdown
                    if strategy == 'selective':
                        peak = max(running_pnl_selective)
                        drawdowns = [peak - val for val in running_pnl_selective]
                        metrics[strategy]['max_drawdown'] = max(drawdowns)
                    else:
                        peak = max(running_pnl_all)
                        drawdowns = [peak - val for val in running_pnl_all]
                        metrics[strategy]['max_drawdown'] = max(drawdowns)
        
        # Log detailed performance metrics
        logger.info(f"\nBetting Performance Comparison for {market}:")
        for strategy in ['selective', 'all_matches']:
            logger.info(f"\n{strategy.upper()} STRATEGY:")
            logger.info(f"Number of bets: {metrics[strategy]['bets']}")
            logger.info(f"Wins: {metrics[strategy]['wins']}")
            logger.info(f"Win rate: {metrics[strategy]['win_rate']:.2%}")
            logger.info(f"Average odds: {metrics[strategy]['avg_odds']:.2f}")
            logger.info(f"Average winning odds: {metrics[strategy]['avg_winning_odds']:.2f}")
            logger.info(f"Total PnL: {metrics[strategy]['pnl']:.2f}")
            logger.info(f"ROI: {metrics[strategy]['roi']:.2%}")
            logger.info(f"Max Drawdown: {metrics[strategy]['max_drawdown']:.2f}")
            
            # Log odds range analysis
            logger.info("\nOdds Range Analysis:")
            for odds_range in self.odds_ranges:
                stats = self.odds_stats[strategy][odds_range]
                if stats['total'] > 0:
                    win_rate = stats['wins'] / stats['total']
                    roi = stats['pnl'] / (stats['total'] * self.default_stake)
                    logger.info(
                        f"Odds {odds_range[0]:.2f}-{odds_range[1]:.2f}: "
                        f"{stats['wins']}/{stats['total']} "
                        f"({win_rate:.1%}) | "
                        f"ROI: {roi:.1%} | "
                        f"PnL: {stats['pnl']:.2f}"
                    )
        
        return metrics
    
    def save_bet_details(self, save_path: Path) -> None:
        """Save detailed betting history and odds analysis to CSV."""
        if self.bet_details:
            # Save individual bet details
            df = pd.DataFrame(self.bet_details)
            save_path.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path / 'betting_history.csv', index=False)
            
            # Save odds range analysis
            odds_analysis = []
            for strategy in ['selective', 'all_matches']:
                for odds_range, stats in self.odds_stats[strategy].items():
                    if stats['total'] > 0:
                        odds_analysis.append({
                            'strategy': strategy,
                            'odds_range_start': odds_range[0],
                            'odds_range_end': odds_range[1],
                            'total_bets': stats['total'],
                            'wins': stats['wins'],
                            'win_rate': stats['wins'] / stats['total'],
                            'pnl': stats['pnl'],
                            'roi': stats['pnl'] / (stats['total'] * self.default_stake)
                        })
            
            odds_df = pd.DataFrame(odds_analysis)
            odds_df.to_csv(save_path / 'odds_analysis.csv', index=False)
            
            logger.info(f"\nBetting history saved to {save_path}/betting_history.csv")
            logger.info(f"Odds analysis saved to {save_path}/odds_analysis.csv")