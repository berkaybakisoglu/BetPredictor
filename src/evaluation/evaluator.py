import pandas as pd
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class BettingEvaluator:
    def __init__(self, config):
        self.config = config
    
    def evaluate(self, predictions, actual_results):
        evaluation_results = {}
        
        for market, pred_df in predictions.items():
            if not self.config.markets[market]:
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
    
    def _evaluate_classification(self, predictions, actuals, market='match_odds'):
        merged = predictions.merge(
            actuals[['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG']],
            on=['Date', 'HomeTeam', 'AwayTeam']
        )
        
        if market == 'match_odds':
            true_values = merged['FTR']
            pred_values = merged['Predicted']
            
            accuracy = np.mean(true_values == pred_values)
            
            high_conf_mask = merged['Confidence'] >= self.config.confidence_threshold
            high_conf_accuracy = np.mean(true_values[high_conf_mask] == pred_values[high_conf_mask])
            high_conf_pct = np.mean(high_conf_mask)
            
            value_bets = 0
            value_wins = 0
            
            for _, row in merged.iterrows():
                if row['Predicted'] == 'H' and row['Home_Value'] > 1:
                    value_bets += 1
                    if row['FTR'] == 'H':
                        value_wins += row['B365H']
                elif row['Predicted'] == 'D' and row['Draw_Value'] > 1:
                    value_bets += 1
                    if row['FTR'] == 'D':
                        value_wins += row['B365D']
                elif row['Predicted'] == 'A' and row['Away_Value'] > 1:
                    value_bets += 1
                    if row['FTR'] == 'A':
                        value_wins += row['B365A']
        
        elif market == 'over_under':
            true_values = (merged['FTHG'] + merged['FTAG']).gt(2.5).astype(int)
            pred_values = merged['Predicted']
            
            accuracy = np.mean(true_values == pred_values)
            
            high_conf_mask = merged['Confidence'] >= self.config.confidence_threshold
            high_conf_accuracy = np.mean(true_values[high_conf_mask] == pred_values[high_conf_mask])
            high_conf_pct = np.mean(high_conf_mask)
            
            value_bets = 0
            value_wins = 0
            
            for _, row in merged.iterrows():
                total_goals = row['FTHG'] + row['FTAG']
                if row['Predicted'] == 1 and row['Over_Value'] > 1:  # Over 2.5
                    value_bets += 1
                    if total_goals > 2.5:
                        value_wins += row['B365O']
                elif row['Predicted'] == 0 and row['Under_Value'] > 1:  # Under 2.5
                    value_bets += 1
                    if total_goals <= 2.5:
                        value_wins += row['B365U']
        
        roi = ((value_wins - value_bets) / value_bets) if value_bets > 0 else 0
        
        return {
            'accuracy': accuracy,
            'high_conf_accuracy': high_conf_accuracy,
            'high_conf_pct': high_conf_pct,
            'value_bets': value_bets,
            'roi': roi
        }