"""Betting performance evaluation module."""
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from src.config.config import BettingConfig

class BettingEvaluator:
    """Evaluates betting performance across markets."""
    
    def __init__(self, config: BettingConfig):
        self.config = config
        self.bankroll = self.config.initial_bankroll
        self.bets_history: List[Dict] = []
        self.market_pnl: Dict[str, float] = {
            market: 0.0 for market in self.config.market_loss_limits
        }
    
    def evaluate_predictions(self, predictions: Dict[str, pd.DataFrame], 
                           results: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Evaluate predictions and simulate betting.
        
        Args:
            predictions: Dictionary of predictions for each market
            results: DataFrame with actual results
            
        Returns:
            Dictionary with performance metrics
        """
        metrics = {}
        
        for market, preds in predictions.items():
            market_metrics = self._evaluate_market(market, preds, results)
            metrics[market] = market_metrics
            
            # Update market P&L
            self.market_pnl[market] += market_metrics['pnl']
            
            # Check market-specific stop loss
            if (self.market_pnl[market] < 
                -self.config.market_loss_limits[market] * self.config.initial_bankroll):
                print(f"\nWarning: {market} market reached stop loss limit")
                
        return metrics
    
    def _evaluate_market(self, market: str, predictions: pd.DataFrame,
                        results: pd.DataFrame) -> Dict[str, float]:
        """Evaluate predictions for a specific market.
        
        Args:
            market: Market being evaluated
            predictions: Predictions for the market
            results: Actual results
            
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
        bets = bets.merge(results[result_columns], 
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
                # Check home win
                if (pred['Home_Prob'] > threshold and 
                    pred['B365H'] >= min_odds):
                    bets.append({
                        'Date': pred['Date'],
                        'HomeTeam': pred['HomeTeam'],
                        'AwayTeam': pred['AwayTeam'],
                        'Market': market,
                        'Bet': 'Home',
                        'Confidence': pred['Home_Prob'],
                        'Odds': pred['B365H'],
                        'Stake': self._calculate_stake(pred['Home_Prob'], 
                                                     pred['B365H'],
                                                     max_stake_pct)
                    })
                # Check away win
                elif (pred['Away_Prob'] > threshold and 
                      pred['B365A'] >= min_odds):
                    bets.append({
                        'Date': pred['Date'],
                        'HomeTeam': pred['HomeTeam'],
                        'AwayTeam': pred['AwayTeam'],
                        'Market': market,
                        'Bet': 'Away',
                        'Confidence': pred['Away_Prob'],
                        'Odds': pred['B365A'],
                        'Stake': self._calculate_stake(pred['Away_Prob'],
                                                     pred['B365A'],
                                                     max_stake_pct)
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