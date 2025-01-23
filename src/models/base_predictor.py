import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class BasePredictor:
    def __init__(self, config):
        self.config = config
        self.scalers = {}
        self.feature_importances = {}
    
    def _define_base_features(self):
        return [
            # Team Performance
            'Home_Goals_Scored_Avg', 'Away_Goals_Scored_Avg',
            'Home_Goals_Conceded_Avg', 'Away_Goals_Conceded_Avg',
            'Home_Clean_Sheets', 'Away_Clean_Sheets',
            'Home_Failed_Score', 'Away_Failed_Score',
            'Goals_Diff_Home', 'Goals_Diff_Away',
            
            # Form and Recent Performance
            'Home_Form', 'Away_Form', 'General_Form',
            'Home_Win_Rate', 'Away_Win_Rate',
            
            # League Position
            'Home_League_Position', 'Away_League_Position',
            'Position_Diff', 'Points_Diff',
            'Home_Points', 'Away_Points',
            
            # Head to Head
            'h2h_home_win_rate', 'h2h_draw_rate', 'h2h_away_win_rate',
            'h2h_home_goals', 'h2h_away_goals', 'h2h_total_goals',
            
            # Market Features
            'Home_ImpliedProb', 'Draw_ImpliedProb', 'Away_ImpliedProb',
            'Market_Confidence', 'Market_Overround',
            'Home_Value', 'Draw_Value', 'Away_Value',
            'Home_Is_Favorite', 'Favorite_Odds', 'Underdog_Odds',
            
            # Original Betting Odds
            'B365H_Original', 'B365D_Original', 'B365A_Original'
        ]
    
    def _define_corner_features(self):
        return [
            # Basic Corner Stats
            'Home_Corners_For_Avg', 'Home_Corners_Against_Avg',
            'Away_Corners_For_Avg', 'Away_Corners_Against_Avg',
            
            # Recent Performance
            'Home_Corners_For_Last5', 'Home_Corners_Against_Last5',
            'Away_Corners_For_Last5', 'Away_Corners_Against_Last5',
            'Home_Corners_For_Last3', 'Home_Corners_Against_Last3',
            'Away_Corners_For_Last3', 'Away_Corners_Against_Last3',
            
            # Derived Metrics
            'Home_Corner_Diff_Avg', 'Away_Corner_Diff_Avg',
            'Home_Corner_Std', 'Away_Corner_Std'
        ]
    
    def _define_card_features(self):
        return [
            # Basic Card Stats
            'Home_Cards_Avg', 'Away_Cards_Avg',
            
            # Recent Performance
            'Home_Cards_Last5', 'Away_Cards_Last5',
            'Home_Cards_Last3', 'Away_Cards_Last3',
            
            # Additional Context
            'Home_Form', 'Away_Form',  # Form can indicate match intensity
            'Position_Diff',  # Position difference can suggest match importance
            'Home_Is_Favorite'  # Favorite status can influence card frequency
        ]
    
    def _get_target(self, df, market):
        if market == 'match_result':
            result_map = {'H': 0, 'D': 1, 'A': 2}
            valid_results = df['FTR'].isin(result_map.keys())
            if not valid_results.all():
                invalid_values = df.loc[~valid_results, 'FTR'].unique()
                logger.warning(f"Found {(~valid_results).sum()} invalid match results with values {invalid_values}. These will be excluded from training.")
            
            # Ensure we have all classes in the data
            unique_results = df.loc[valid_results, 'FTR'].unique()
            missing_classes = set(result_map.keys()) - set(unique_results)
            if missing_classes:
                logger.warning(f"Missing classes in the data: {missing_classes}")
            
            return df.loc[valid_results, 'FTR'].map(result_map)
        elif market == 'over_under':
            valid_goals = df['FTHG'].notna() & df['FTAG'].notna()
            if not valid_goals.all():
                logger.warning(f"Found {(~valid_goals).sum()} matches with missing goals. These will be excluded from training.")
            return df.loc[valid_goals, ['FTHG', 'FTAG']].sum(axis=1).gt(2.5).astype(int)
        elif market == 'corners':
            valid_corners = df['HC'].notna() & df['AC'].notna()
            if not valid_corners.all():
                logger.warning(f"Found {(~valid_corners).sum()} matches with missing corner data. These will be excluded from training.")
            return df.loc[valid_corners, ['HC', 'AC']].sum(axis=1)
        elif market == 'cards':
            valid_cards = df['HY'].notna() & df['AY'].notna()
            if not valid_cards.all():
                logger.warning(f"Found {(~valid_cards).sum()} matches with missing card data. These will be excluded from training.")
            return df.loc[valid_cards, ['HY', 'AY']].sum(axis=1)
        else:
            raise ValueError(f"Unknown market: {market}")
    
    def _calculate_confidence(self, probabilities):
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10), axis=1)
        if probabilities.shape[1] == 2:  # binary classification
            max_entropy = 1.0
        else:  # multi-class
            max_entropy = -np.log2(1/probabilities.shape[1])
        return 1 - (entropy / max_entropy)
    
    def _format_match_predictions(self, df, probas):
        predictions = pd.DataFrame({
            'Date': df['Date'],
            'HomeTeam': df['HomeTeam'],
            'AwayTeam': df['AwayTeam'],
            'Home_Prob': probas[:, 0],
            'Draw_Prob': probas[:, 1],
            'Away_Prob': probas[:, 2],
            'B365H': df['B365H_Original'],
            'B365D': df['B365D_Original'],
            'B365A': df['B365A_Original']
        })
        
        predictions['Predicted'] = predictions[['Home_Prob', 'Draw_Prob', 'Away_Prob']].idxmax(axis=1).map({
            'Home_Prob': 'H',
            'Draw_Prob': 'D',
            'Away_Prob': 'A'
        })
        
        predictions['Confidence'] = self._calculate_confidence(
            predictions[['Home_Prob', 'Draw_Prob', 'Away_Prob']].values
        )
        
        predictions['Home_Value'] = predictions['Home_Prob'] * df['B365H_Original']
        predictions['Draw_Value'] = predictions['Draw_Prob'] * df['B365D_Original']
        predictions['Away_Value'] = predictions['Away_Prob'] * df['B365A_Original']
        
        return predictions
    
    def _format_regression_predictions(self, df, predictions, market):
        return pd.DataFrame({
            'Date': df['Date'],
            'HomeTeam': df['HomeTeam'],
            'AwayTeam': df['AwayTeam'],
            'prediction': predictions.round(1)
        })
    
    def _analyze_seasonal_progression(self, df, y_true, y_pred, market):
        """Analyze prediction performance throughout the season."""
        analysis_df = pd.DataFrame({
            'Date': df['Date'],
            'Season': df['Season'],
            'HomeTeam': df['HomeTeam'],
            'True': y_true,
            'Pred': y_pred
        })
        
        analysis_df = analysis_df.sort_values('Date')
        analysis_df['Match_Number'] = analysis_df.groupby(['Season', 'HomeTeam']).cumcount() + 1
        
        match_accuracies = []
        max_matches = analysis_df['Match_Number'].max()
        
        for match_num in range(1, max_matches + 1):
            matches_at_num = analysis_df[analysis_df['Match_Number'] == match_num]
            if len(matches_at_num) > 0:
                if market in ['corners', 'cards']:
                    accuracy = np.sqrt(mean_squared_error(
                        matches_at_num['True'],
                        matches_at_num['Pred']
                    ))
                else:
                    accuracy = accuracy_score(
                        matches_at_num['True'],
                        matches_at_num['Pred']
                    )
                match_accuracies.append(accuracy)
        
        return match_accuracies
    
    def _plot_seasonal_progression(self, seasonal_progression, save_path=None):
        plt.style.use('bmh')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12), facecolor='#f0f0f0')
        fig.suptitle('Prediction Performance Throughout Season', fontsize=16, y=1.02)
        
        axes = axes.flatten()
        line_color = '#2F5373'
        trend_color = '#C44E52'
        
        for idx, market in enumerate(seasonal_progression.keys()):
            ax = axes[idx]
            match_numbers = list(seasonal_progression[market].keys())
            accuracies = list(seasonal_progression[market].values())
            
            ax.plot(match_numbers, accuracies, marker='o', linewidth=2, markersize=6,
                   color=line_color, markerfacecolor='white', markeredgecolor=line_color)
            
            z = np.polyfit(match_numbers, accuracies, 1)
            p = np.poly1d(z)
            ax.plot(match_numbers, p(match_numbers), linestyle='--', color=trend_color,
                   alpha=0.8, label='Trend', linewidth=2)
            
            metric_name = "RMSE" if market in ['corners', 'cards'] else "Accuracy"
            ax.set_title(f'{market.upper()} {metric_name} by Match Number', 
                        fontsize=12, pad=10)
            ax.set_xlabel('Match Number in Season', fontsize=10)
            ax.set_ylabel(metric_name, fontsize=10)
            
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.set_axisbelow(True)
            
            ax.legend(['Actual', 'Trend'], frameon=True, facecolor='white', edgecolor='none')
            
            if market in ['corners', 'cards']:
                ax.invert_yaxis()
            
            ax.annotate(f'Start: {accuracies[0]:.3f}', 
                       xy=(match_numbers[0], accuracies[0]),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
            ax.annotate(f'End: {accuracies[-1]:.3f}',
                       xy=(match_numbers[-1], accuracies[-1]),
                       xytext=(-10, 10), textcoords='offset points',
                       ha='right',
                       bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
            
            ax.set_facecolor('white')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, 'seasonal_progression.png'), 
                       dpi=300, bbox_inches='tight',
                       facecolor='white')
        else:
            plt.show()
        
        plt.close()
    
    def _analyze_league_performance(self, df, y_true, y_pred, market):
        analysis_df = pd.DataFrame({
            'League': df['League'],
            'True': y_true,
            'Pred': y_pred
        })
        
        league_accuracies = {}
        for league in analysis_df['League'].unique():
            league_data = analysis_df[analysis_df['League'] == league]
            if len(league_data) > 0:
                if market in ['corners', 'cards']:
                    accuracy = np.sqrt(mean_squared_error(
                        league_data['True'],
                        league_data['Pred']
                    ))
                else:
                    accuracy = accuracy_score(
                        league_data['True'],
                        league_data['Pred']
                    )
                league_accuracies[league] = accuracy
        
        return league_accuracies

    def _plot_league_performance(self, league_performance, save_path=None):
        plt.style.use('bmh')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12), facecolor='#f0f0f0')
        fig.suptitle('Prediction Performance by League', fontsize=16, y=1.02)
        
        axes = axes.flatten()
        colors = plt.cm.Set3(np.linspace(0, 1, 10))
        
        for idx, (market, league_accuracies) in enumerate(league_performance.items()):
            ax = axes[idx]
            
            sorted_leagues = sorted(league_accuracies.items(), key=lambda x: x[1], reverse=True)
            leagues, accuracies = zip(*sorted_leagues)
            
            bars = ax.bar(range(len(leagues)), accuracies, color=colors)
            
            metric_name = "RMSE" if market in ['corners', 'cards'] else "Accuracy"
            ax.set_title(f'{market.upper()} {metric_name} by League', fontsize=12, pad=10)
            ax.set_xlabel('League', fontsize=10)
            ax.set_ylabel(metric_name, fontsize=10)
            
            ax.set_xticks(range(len(leagues)))
            ax.set_xticklabels(leagues, rotation=45, ha='right')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom')
            
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.set_axisbelow(True)
            
            if market in ['corners', 'cards']:
                ax.invert_yaxis()
            
            ax.set_facecolor('white')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, 'league_performance.png'), 
                       dpi=300, bbox_inches='tight',
                       facecolor='white')
        else:
            plt.show()
        
        plt.close()
    
    def _average_metrics(self, window_metrics):
        if not window_metrics:
            return {}
        
        avg_metrics = {k: 0.0 for k in window_metrics[0].keys()}
        
        for metrics in window_metrics:
            for k, v in metrics.items():
                avg_metrics[k] += v
        
        n_windows = len(window_metrics)
        return {k: v / n_windows for k, v in avg_metrics.items()}
    
    def _average_progression(self, window_progressions):
        if not window_progressions:
            return {}
        
        all_match_nums = set()
        for prog in window_progressions:
            all_match_nums.update(prog.keys())
        
        avg_progression = {match_num: 0.0 for match_num in all_match_nums}
        counts = {match_num: 0 for match_num in all_match_nums}
        
        for prog in window_progressions:
            for match_num, value in prog.items():
                avg_progression[match_num] += value
                counts[match_num] += 1
        
        return {
            match_num: value / counts[match_num]
            for match_num, value in avg_progression.items()
        }
    
    def _average_league_performance(self, window_performances):
        if not window_performances:
            return {}
        
        all_leagues = set()
        for perf in window_performances:
            all_leagues.update(perf.keys())
        
        avg_performance = {league: 0.0 for league in all_leagues}
        counts = {league: 0 for league in all_leagues}
        
        for perf in window_performances:
            for league, value in perf.items():
                avg_performance[league] += value
                counts[league] += 1
        
        return {
            league: value / counts[league]
            for league, value in avg_performance.items()
        }
    
    def save(self, model_dir):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        for name, scaler in self.scalers.items():
            scaler_path = os.path.join(model_dir, f'{name}_scaler.joblib')
            joblib.dump(scaler, scaler_path)
            
            if name in self.feature_importances and self.feature_importances[name] is not None:
                importance_path = os.path.join(model_dir, f'{name}_importance.csv')
                self.feature_importances[name].to_csv(importance_path, index=False)
            
        logger.info(f"Model saved to {model_dir}")
    
    def load(self, model_dir):
        for name in self.scalers.keys():
            scaler_path = os.path.join(model_dir, f'{name}_scaler.joblib')
            self.scalers[name] = joblib.load(scaler_path)
            
            importance_path = os.path.join(model_dir, f'{name}_importance.csv')
            if os.path.exists(importance_path):
                self.feature_importances[name] = pd.read_csv(importance_path)
            
        logger.info(f"Model loaded from {model_dir}")
    
    def _log_results(self, market, metrics):
        logger.info(f"\n{market} metrics:")
        for metric, value in metrics.items():
            logger.info(f"- {metric}: {value:.4f}")
        
        if market in self.feature_importances:
            logger.info(f"\nTop 10 important features for {market}:")
            for _, row in self.feature_importances[market].head(10).iterrows():
                logger.info(f"- {row['feature']}: {row['importance']:.4f}") 