"""Unified predictor module for all betting markets."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import joblib
from lightgbm import LGBMRegressor
from src.config.config import ModelConfig
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class UnifiedPredictor:
    """Unified predictor for all betting markets.
    
    This class handles both classification (match results, over/under) and 
    regression (corners, cards) predictions using Random Forest models.
    
    Attributes:
        config: Model configuration settings
        models: Dictionary of classification models
        regression_models: Dictionary of regression models
        scalers: Dictionary of feature scalers
        feature_sets: Dictionary of feature sets for each market
        feature_importances: Dictionary storing feature importance for each market
    """
    
    def __init__(self, config: ModelConfig):
        """Initialize the predictor.
        
        Args:
            config: Model configuration settings
        """
        self.config = config
        self.models: Dict[str, RandomForestClassifier] = {}
        self.regression_models: Dict[str, RandomForestRegressor] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_sets: Dict[str, List[str]] = self._define_feature_sets()
        self.feature_importances: Dict[str, pd.DataFrame] = {}
    
    def _define_feature_sets(self) -> Dict[str, List[str]]:
        """Define feature sets for each market."""
        # Base features used by all markets
        base_features = [
            # Team performance metrics
            'Home_Goals_Scored_Avg', 'Home_Goals_Conceded_Avg',
            'Away_Goals_Scored_Avg', 'Away_Goals_Conceded_Avg',
            'Home_Form', 'Away_Form',
            
            # Head-to-head statistics
            'H2H_Home_Wins', 'H2H_Away_Wins', 'H2H_Draws',
            'H2H_Avg_Goals',
            
            # Market implied probabilities
            'Home_ImpliedProb', 'Draw_ImpliedProb', 'Away_ImpliedProb'
        ]
        
        # Enhanced corner features (only those we can reliably calculate)
        corner_features = [
            # Basic corner stats
            'Home_Corners_For_Avg', 'Home_Corners_Against_Avg',
            'Away_Corners_For_Avg', 'Away_Corners_Against_Avg',
            'H2H_Avg_Corners',
            
            # Recent form corner stats
            'Home_Corners_For_Last5', 'Home_Corners_Against_Last5',
            'Away_Corners_For_Last5', 'Away_Corners_Against_Last5',
            
            # Corner differentials
            'Home_Corner_Diff_Avg',  # Corners For - Against
            'Away_Corner_Diff_Avg',
            
            # Home/Away specific performance
            'Home_Corners_For_Last3', 'Home_Corners_Against_Last3',
            'Away_Corners_For_Last3', 'Away_Corners_Against_Last3',
            
            # Consistency metrics
            'Home_Corner_Std', 'Away_Corner_Std'
        ]
        
        card_features = [
            'Home_Cards_Avg', 'Away_Cards_Avg',
            'Home_Cards_Last5', 'Away_Cards_Last5'
        ]
        
        # Define feature sets for each market
        return {
            'match_result': base_features + [
                'Home_League_Position', 'Away_League_Position',
                'Position_Diff', 'Points_Diff'
            ],
            'over_under': base_features + [
                'Goals_Diff_Home', 'Goals_Diff_Away',
                'Home_Clean_Sheets', 'Away_Clean_Sheets'
            ],
            'corners': base_features + corner_features,
            'cards': base_features + card_features
        }
    
    def _prepare_training_data(self, df: pd.DataFrame, market: str, train_seasons: List[int], test_season: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training and test data for a market.
        
        Args:
            df: Input DataFrame
            market: Market name
            train_seasons: List of seasons for training
            test_season: Season for testing
            
        Returns:
            Tuple of (X_train_scaled, X_test_scaled, y_train, y_test)
        """
        # Get features and target
        X = df[self.feature_sets[market]].copy()
        y = self._get_target(df, market)
        
        # Ensure X and y are aligned after filtering invalid targets
        valid_indices = y.index
        X = X.loc[valid_indices]
        
        # Log feature information
        logger.info(f"Features used ({len(self.feature_sets[market])}):")
        for feature in self.feature_sets[market]:
            logger.info(f"- {feature}")
        
        logger.info(f"\nTraining data size after filtering:")
        logger.info(f"Total samples: {len(X)}")
        logger.info(f"Valid samples: {len(valid_indices)}")
        
        if len(X) < self.config.min_training_samples:
            raise ValueError(f"Insufficient training data for {market} after filtering")
        
        # Split data
        X_train = X[df['Season'].isin(train_seasons)]
        y_train = y[df['Season'].isin(train_seasons)]
        X_test = X[df['Season'] == test_season]
        y_test = y[df['Season'] == test_season]
        
        # Scale features
        self.scalers[market] = StandardScaler()
        X_train_scaled = self.scalers[market].fit_transform(X_train)
        X_test_scaled = self.scalers[market].transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns
    
    def _train_regression_model(self, market: str, X_train: np.ndarray, X_test: np.ndarray, 
                              y_train: np.ndarray, y_test: np.ndarray, feature_names: pd.Index) -> Dict[str, float]:
        """Train regression model for corners/cards markets.
        
        Args:
            market: Market name
            X_train: Training features
            X_test: Test features
            y_train: Training targets
            y_test: Test targets
            feature_names: Names of features
            
        Returns:
            Dictionary of metrics
        """
        if market == 'corners':
            # Use LightGBM for corners prediction
            # Initialize model with optimized parameters
            self.regression_models[market] = LGBMRegressor(
                n_estimators=500,
                learning_rate=0.01,
                num_leaves=31,
                max_depth=8,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42
            )
        else:
            # Use Random Forest for other regression tasks
            self.regression_models[market] = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            )
        
        # Train model
        self.regression_models[market].fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.regression_models[market].predict(X_test)
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        # Calculate feature importance
        if isinstance(self.regression_models[market], LGBMRegressor):
            importance = self.regression_models[market].feature_importances_
        else:
            importance = self.regression_models[market].feature_importances_
            
        self.feature_importances[market] = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return metrics
    
    def _train_classification_model(self, market: str, X_train: np.ndarray, X_test: np.ndarray, 
                                  y_train: np.ndarray, y_test: np.ndarray, feature_names: pd.Index) -> Dict[str, float]:
        """Train classification model for match result/over-under markets.
        
        Args:
            market: Market name
            X_train: Training features
            X_test: Test features
            y_train: Training targets
            y_test: Test targets
            feature_names: Names of features
            
        Returns:
            Dictionary of metrics
        """
        # Initialize model
        self.models[market] = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42
        )
        
        # Train model
        self.models[market].fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.models[market].predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        # Calculate feature importance
        self.feature_importances[market] = pd.DataFrame({
            'feature': feature_names,
            'importance': self.models[market].feature_importances_
        }).sort_values('importance', ascending=False)
        
        return metrics
    
    def _save_model(self, market: str, save_path: Path) -> None:
        """Save model, scaler, and feature importance for a market.
        
        Args:
            market: Market name
            save_path: Path to save files
        """
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if market in ['corners', 'cards']:
            joblib.dump(self.regression_models[market], save_path / f"{market}_model.joblib")
        else:
            joblib.dump(self.models[market], save_path / f"{market}_model.joblib")
        
        # Save scaler and feature importance
        joblib.dump(self.scalers[market], save_path / f"{market}_scaler.joblib")
        importance_path = save_path / f"{market}_importance.csv"
        self.feature_importances[market].to_csv(importance_path, index=False)
    
    def _log_results(self, market: str, metrics: Dict[str, float]) -> None:
        """Log training results for a market.
        
        Args:
            market: Market name
            metrics: Dictionary of metrics
        """
        logger.info(f"\n{market} metrics:")
        for metric, value in metrics.items():
            logger.info(f"- {metric}: {value:.4f}")
        
        logger.info(f"\nTop 10 important features for {market}:")
        for _, row in self.feature_importances[market].head(10).iterrows():
            logger.info(f"- {row['feature']}: {row['importance']:.4f}")
    
    def _analyze_seasonal_progression(self, df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, market: str) -> Dict[str, List[float]]:
        """Analyze how prediction accuracy changes as season progresses.
        
        Args:
            df: DataFrame with match information
            y_true: True values
            y_pred: Predicted values
            market: Market being analyzed
            
        Returns:
            Dictionary with match number and corresponding accuracies
        """
        # Create DataFrame with predictions and actual results
        analysis_df = pd.DataFrame({
            'Date': df['Date'],
            'Season': df['Season'],
            'HomeTeam': df['HomeTeam'],
            'True': y_true,
            'Pred': y_pred
        })
        
        # Sort by date to ensure chronological order
        analysis_df = analysis_df.sort_values('Date')
        
        # Add match number in season for each team
        analysis_df['Match_Number'] = analysis_df.groupby(['Season', 'HomeTeam']).cumcount() + 1
        
        # Calculate accuracy for each match number
        match_accuracies = {}
        max_matches = analysis_df['Match_Number'].max()
        
        for match_num in range(1, max_matches + 1):
            matches_at_num = analysis_df[analysis_df['Match_Number'] == match_num]
            if len(matches_at_num) > 0:
                if market in ['corners', 'cards']:
                    # For regression tasks, calculate RMSE
                    accuracy = np.sqrt(mean_squared_error(
                        matches_at_num['True'],
                        matches_at_num['Pred']
                    ))
                else:
                    # For classification tasks, calculate accuracy
                    accuracy = accuracy_score(
                        matches_at_num['True'],
                        matches_at_num['Pred']
                    )
                match_accuracies[match_num] = accuracy
        
        return match_accuracies
    
    def _plot_seasonal_progression(self, seasonal_progression: Dict[str, Dict[int, float]], save_path: Optional[Path] = None) -> None:
        """Create plots showing how prediction accuracy changes throughout the season.
        
        Args:
            seasonal_progression: Dictionary with match numbers and accuracies
            save_path: Optional path to save plots
        """
        # Use a built-in style
        plt.style.use('bmh')  # Alternative clean style
        
        # Set up the figure with a light gray background
        fig, axes = plt.subplots(2, 2, figsize=(15, 12), facecolor='#f0f0f0')
        fig.suptitle('Prediction Performance Throughout Season', fontsize=16, y=1.02)
        
        # Flatten axes for easier iteration
        axes = axes.flatten()
        
        # Define colors for better visibility
        line_color = '#2F5373'
        trend_color = '#C44E52'
        
        for idx, market in enumerate(seasonal_progression.keys()):
            ax = axes[idx]
            match_numbers = list(seasonal_progression[market].keys())
            accuracies = list(seasonal_progression[market].values())
            
            # Create line plot with custom styling
            ax.plot(match_numbers, accuracies, marker='o', linewidth=2, markersize=6,
                   color=line_color, markerfacecolor='white', markeredgecolor=line_color)
            
            # Add trend line
            z = np.polyfit(match_numbers, accuracies, 1)
            p = np.poly1d(z)
            ax.plot(match_numbers, p(match_numbers), linestyle='--', color=trend_color,
                   alpha=0.8, label='Trend', linewidth=2)
            
            # Customize plot
            metric_name = "RMSE" if market in ['corners', 'cards'] else "Accuracy"
            ax.set_title(f'{market.upper()} {metric_name} by Match Number', 
                        fontsize=12, pad=10)
            ax.set_xlabel('Match Number in Season', fontsize=10)
            ax.set_ylabel(metric_name, fontsize=10)
            
            # Customize grid
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.set_axisbelow(True)  # Put grid behind data
            
            # Add legend with custom styling
            ax.legend(['Actual', 'Trend'], frameon=True, facecolor='white', edgecolor='none')
            
            # For RMSE (lower is better), invert y-axis
            if market in ['corners', 'cards']:
                ax.invert_yaxis()
            
            # Add annotations with improved styling
            ax.annotate(f'Start: {accuracies[0]:.3f}', 
                       xy=(match_numbers[0], accuracies[0]),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
            ax.annotate(f'End: {accuracies[-1]:.3f}',
                       xy=(match_numbers[-1], accuracies[-1]),
                       xytext=(-10, 10), textcoords='offset points',
                       ha='right',
                       bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
            
            # Set background color
            ax.set_facecolor('white')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show plot
        if save_path:
            save_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path / 'seasonal_progression.png', 
                       dpi=300, bbox_inches='tight',
                       facecolor='white')
        else:
            plt.show()
        
        plt.close()
    
    def _analyze_league_performance(self, df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, market: str) -> Dict[str, float]:
        """Analyze prediction performance for each league.
        
        Args:
            df: DataFrame with match information
            y_true: True values
            y_pred: Predicted values
            market: Market being analyzed
            
        Returns:
            Dictionary with league names and corresponding accuracies
        """
        # Create DataFrame with predictions and actual results
        analysis_df = pd.DataFrame({
            'League': df['League'],
            'True': y_true,
            'Pred': y_pred
        })
        
        # Calculate accuracy for each league
        league_accuracies = {}
        for league in analysis_df['League'].unique():
            league_data = analysis_df[analysis_df['League'] == league]
            if len(league_data) > 0:
                if market in ['corners', 'cards']:
                    # For regression tasks, calculate RMSE
                    accuracy = np.sqrt(mean_squared_error(
                        league_data['True'],
                        league_data['Pred']
                    ))
                else:
                    # For classification tasks, calculate accuracy
                    accuracy = accuracy_score(
                        league_data['True'],
                        league_data['Pred']
                    )
                league_accuracies[league] = accuracy
        
        return league_accuracies

    def _plot_league_performance(self, league_performance: Dict[str, Dict[str, float]], save_path: Optional[Path] = None) -> None:
        """Create plots showing prediction performance by league.
        
        Args:
            league_performance: Dictionary with leagues and accuracies for each market
            save_path: Optional path to save plots
        """
        plt.style.use('bmh')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12), facecolor='#f0f0f0')
        fig.suptitle('Prediction Performance by League', fontsize=16, y=1.02)
        
        # Flatten axes for easier iteration
        axes = axes.flatten()
        
        # Color palette for bars
        colors = plt.cm.Set3(np.linspace(0, 1, 10))
        
        for idx, (market, league_accuracies) in enumerate(league_performance.items()):
            ax = axes[idx]
            
            # Sort leagues by performance
            sorted_leagues = sorted(league_accuracies.items(), key=lambda x: x[1], reverse=True)
            leagues, accuracies = zip(*sorted_leagues)
            
            # Create bar plot
            bars = ax.bar(range(len(leagues)), accuracies, color=colors)
            
            # Customize plot
            metric_name = "RMSE" if market in ['corners', 'cards'] else "Accuracy"
            ax.set_title(f'{market.upper()} {metric_name} by League', fontsize=12, pad=10)
            ax.set_xlabel('League', fontsize=10)
            ax.set_ylabel(metric_name, fontsize=10)
            
            # Rotate x-axis labels for better readability
            ax.set_xticks(range(len(leagues)))
            ax.set_xticklabels(leagues, rotation=45, ha='right')
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom')
            
            # Customize grid
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.set_axisbelow(True)
            
            # For RMSE (lower is better), invert y-axis
            if market in ['corners', 'cards']:
                ax.invert_yaxis()
            
            # Set background color
            ax.set_facecolor('white')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show plot
        if save_path:
            save_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path / 'league_performance.png', 
                       dpi=300, bbox_inches='tight',
                       facecolor='white')
        else:
            plt.show()
        
        plt.close()
    
    def train(self, df: pd.DataFrame, save_path: Optional[Path] = None, test_mode: bool = False) -> Dict[str, Dict[str, float]]:
        """Train models for all enabled markets using time-series validation.
        
        Args:
            df: DataFrame with features and targets
            save_path: Optional path to save trained models
            test_mode: Whether running in test mode (allows fewer seasons)
            
        Returns:
            Dictionary with performance metrics for each market
        """
        metrics = {}
        all_feature_importances = {}
        seasonal_progression = {}
        league_performance = {}
        logger = logging.getLogger(__name__)
        
        # Sort by date and validate seasons
        df = df.sort_values('Date')
        seasons = df['Season'].unique()
        logger.info(f"Training data spans {len(seasons)} seasons: {sorted(seasons)}")
        
        if not test_mode and len(seasons) < 3:
            raise ValueError("Need at least 3 seasons for proper validation")
        elif test_mode and len(seasons) < 2:
            raise ValueError("Need at least 2 seasons even in test mode")
        
        # Define training and test seasons
        train_seasons = seasons[:-1]
        test_season = seasons[-1]
        logger.info(f"Using seasons {train_seasons} for training and season {test_season} for testing")
        
        # Train models for each market
        for market in self.config.markets:
            if not self.config.markets[market]:
                logger.info(f"Skipping {market} market (disabled in config)")
                continue
            
            try:
                # Prepare data
                X_train, X_test, y_train, y_test, feature_names = self._prepare_training_data(
                    df, market, train_seasons, test_season
                )
                
                # Train model and get metrics
                if market in ['corners', 'cards']:
                    metrics[market] = self._train_regression_model(
                        market, X_train, X_test, y_train, y_test, feature_names
                    )
                    # Get predictions for seasonal progression analysis
                    y_pred = self.regression_models[market].predict(X_test)
                else:
                    metrics[market] = self._train_classification_model(
                        market, X_train, X_test, y_train, y_test, feature_names
                    )
                    # Get predictions for seasonal progression analysis
                    y_pred = self.models[market].predict(X_test)
                
                # Analyze seasonal progression
                test_df = df[df['Season'] == test_season]
                seasonal_progression[market] = self._analyze_seasonal_progression(
                    test_df, y_test, y_pred, market
                )
                
                # Analyze league performance
                test_df = df[df['Season'] == test_season]
                league_performance[market] = self._analyze_league_performance(
                    test_df, y_test, y_pred, market
                )
                
                # Save model if path provided
                if save_path:
                    self._save_model(market, save_path)
                
                # Store feature importances for later
                all_feature_importances[market] = self.feature_importances[market].head(10)
                
            except Exception as e:
                logger.error(f"Error training {market} model: {str(e)}", exc_info=True)
                continue
        
        # Log consolidated metrics at the end
        logger.info("\n" + "="*50)
        logger.info("TRAINING RESULTS SUMMARY")
        logger.info("="*50)
        
        for market in metrics:
            logger.info(f"\n{market.upper()} METRICS:")
            for metric, value in metrics[market].items():
                logger.info(f"- {metric}: {value:.4f}")
        
        logger.info("\n" + "="*50)
        logger.info("TOP FEATURE IMPORTANCES")
        logger.info("="*50)
        
        for market in all_feature_importances:
            logger.info(f"\n{market.upper()} TOP 10 FEATURES:")
            for _, row in all_feature_importances[market].iterrows():
                logger.info(f"- {row['feature']}: {row['importance']:.4f}")
        
        logger.info("\n" + "="*50)
        logger.info("SEASONAL PROGRESSION ANALYSIS")
        logger.info("="*50)
        
        for market in seasonal_progression:
            logger.info(f"\n{market.upper()} ACCURACY BY MATCH NUMBER:")
            for match_num, accuracy in seasonal_progression[market].items():
                metric_name = "RMSE" if market in ['corners', 'cards'] else "Accuracy"
                logger.info(f"Match {match_num}: {metric_name} = {accuracy:.4f}")
        
        logger.info("\n" + "="*50)
        logger.info("LEAGUE PERFORMANCE ANALYSIS")
        logger.info("="*50)
        
        for market in league_performance:
            logger.info(f"\n{market.upper()} PERFORMANCE BY LEAGUE:")
            metric_name = "RMSE" if market in ['corners', 'cards'] else "Accuracy"
            for league, accuracy in sorted(league_performance[market].items(), 
                                        key=lambda x: x[1], 
                                        reverse=market not in ['corners', 'cards']):
                logger.info(f"{league}: {metric_name} = {accuracy:.4f}")
        
        # Create and save visualization
        logger.info("\nCreating seasonal progression visualization...")
        self._plot_seasonal_progression(seasonal_progression, save_path)
        if save_path:
            logger.info(f"Saved seasonal progression plot to {save_path}/seasonal_progression.png")
        
        logger.info("\nCreating league performance visualization...")
        self._plot_league_performance(league_performance, save_path)
        if save_path:
            logger.info(f"Saved league performance plot to {save_path}/league_performance.png")
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Make predictions for all enabled markets.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Dictionary with predictions for each market
        """
        predictions = {}
        
        for market in self.config.markets:
            if not self.config.markets[market]:
                continue
            
            X = df[self.feature_sets[market]]
            X_scaled = self.scalers[market].transform(X)
            
            if market in ['corners', 'cards']:
                # Regression predictions
                if market in self.regression_models:
                    pred_values = self.regression_models[market].predict(X_scaled)
                    predictions[market] = pd.DataFrame({
                        'Date': df['Date'],
                        'HomeTeam': df['HomeTeam'],
                        'AwayTeam': df['AwayTeam'],
                        'prediction': pred_values.round(1),  # Round to 1 decimal
                        'confidence': self._get_prediction_confidence(market, X_scaled)
                    })
            else:
                # Classification predictions
                if market in self.models:
                    probs = self.models[market].predict_proba(X_scaled)
                    predictions[market] = self._format_predictions(df, market, probs)
        
        return predictions
    
    def load_models(self, load_path: Path) -> None:
        """Load trained models from disk.
        
        Args:
            load_path: Path to load models from
        """
        for market in self.config.markets:
            if not self.config.markets[market]:
                continue
                
            model_path = load_path / f"{market}_model.joblib"
            scaler_path = load_path / f"{market}_scaler.joblib"
            
            if model_path.exists() and scaler_path.exists():
                self.models[market] = joblib.load(model_path)
                self.scalers[market] = joblib.load(scaler_path)
                print(f"Loaded {market} model and scaler")
    
    def _get_target(self, df: pd.DataFrame, market: str) -> pd.Series:
        """Get target variable for a market.
        
        Args:
            df: DataFrame with targets
            market: Market to get target for
            
        Returns:
            Target series
        """
        if market == 'match_result':
            result_map = {'H': 0, 'D': 1, 'A': 2}
            # Filter out NaN values and invalid results
            valid_results = df['FTR'].isin(result_map.keys())
            if not valid_results.all():
                logger = logging.getLogger(__name__)
                logger.warning(f"Found {(~valid_results).sum()} invalid match results. These will be excluded from training.")
            return df.loc[valid_results, 'FTR'].map(result_map)
        elif market == 'over_under':
            # Filter out NaN values in goals
            valid_goals = df['FTHG'].notna() & df['FTAG'].notna()
            if not valid_goals.all():
                logger = logging.getLogger(__name__)
                logger.warning(f"Found {(~valid_goals).sum()} matches with missing goals. These will be excluded from training.")
            return df.loc[valid_goals, ['FTHG', 'FTAG']].sum(axis=1).gt(2.5).astype(int)
        elif market == 'corners':
            # Filter out NaN values in corners
            valid_corners = df['HC'].notna() & df['AC'].notna()
            if not valid_corners.all():
                logger = logging.getLogger(__name__)
                logger.warning(f"Found {(~valid_corners).sum()} matches with missing corner data. These will be excluded from training.")
            return df.loc[valid_corners, ['HC', 'AC']].sum(axis=1)
        elif market == 'cards':
            # Filter out NaN values in cards
            valid_cards = df['HY'].notna() & df['AY'].notna()
            if not valid_cards.all():
                logger = logging.getLogger(__name__)
                logger.warning(f"Found {(~valid_cards).sum()} matches with missing card data. These will be excluded from training.")
            return df.loc[valid_cards, ['HY', 'AY']].sum(axis=1)
        else:
            raise ValueError(f"Unknown market: {market}")
    
    def _format_predictions(self, df: pd.DataFrame, market: str, probs: np.ndarray) -> pd.DataFrame:
        """Format predictions for a market.
        
        Args:
            df: Original DataFrame
            market: Market being predicted
            probs: Model probabilities
            
        Returns:
            DataFrame with formatted predictions
        """
        if market == 'match_result':
            predictions = pd.DataFrame({
                'Date': df['Date'],
                'HomeTeam': df['HomeTeam'],
                'AwayTeam': df['AwayTeam'],
                'Home_Prob': probs[:, 0],
                'Draw_Prob': probs[:, 1],
                'Away_Prob': probs[:, 2],
                'B365H': df['B365H'],
                'B365D': df['B365D'],
                'B365A': df['B365A']
            })
            
            # Add predicted outcome based on highest probability
            predictions['Predicted'] = predictions[['Home_Prob', 'Draw_Prob', 'Away_Prob']].idxmax(axis=1).map({
                'Home_Prob': 'H',
                'Draw_Prob': 'D',
                'Away_Prob': 'A'
            })
            
            # Add confidence (highest probability)
            predictions['Confidence'] = predictions[['Home_Prob', 'Draw_Prob', 'Away_Prob']].max(axis=1)
            
            # Add value indicators
            predictions['Home_Value'] = predictions['Home_Prob'] * df['B365H']
            predictions['Draw_Value'] = predictions['Draw_Prob'] * df['B365D']
            predictions['Away_Value'] = predictions['Away_Prob'] * df['B365A']
            
            return predictions
        elif market == 'over_under':
            predictions = pd.DataFrame({
                'Date': df['Date'],
                'HomeTeam': df['HomeTeam'],
                'AwayTeam': df['AwayTeam'],
                'Under_Prob': probs[:, 0],
                'Over_Prob': probs[:, 1],
                'B365<2.5': df['B365<2.5'],
                'B365>2.5': df['B365>2.5']
            })
            
            # Add predicted outcome based on highest probability
            predictions['Predicted'] = predictions[['Under_Prob', 'Over_Prob']].idxmax(axis=1).map({
                'Under_Prob': 'Under',
                'Over_Prob': 'Over'
            })
            
            # Add confidence (highest probability)
            predictions['Confidence'] = predictions[['Under_Prob', 'Over_Prob']].max(axis=1)
            
            # Add value indicators
            predictions['Under_Value'] = predictions['Under_Prob'] * df['B365<2.5']
            predictions['Over_Value'] = predictions['Over_Prob'] * df['B365>2.5']
            
            return predictions
        elif market == 'ht_score':
            return pd.DataFrame({
                'Date': df['Date'],
                'HomeTeam': df['HomeTeam'],
                'AwayTeam': df['AwayTeam'],
                'HT_Home_Prob': probs[:, 0],
                'HT_Draw_Prob': probs[:, 1],
                'HT_Away_Prob': probs[:, 2]
            })
        elif market in ['corners', 'cards']:
            return pd.DataFrame({
                'Date': df['Date'],
                'HomeTeam': df['HomeTeam'],
                'AwayTeam': df['AwayTeam'],
                f'{market.title()}_Low_Prob': probs[:, 0],
                f'{market.title()}_Medium_Prob': probs[:, 1],
                f'{market.title()}_High_Prob': probs[:, 2]
            })
        else:
            raise ValueError(f"Unknown market: {market}") 
    
    def _get_prediction_confidence(self, market: str, X_scaled: np.ndarray) -> np.ndarray:
        """Calculate prediction confidence based on model type."""
        if market not in self.regression_models:
            return np.zeros(len(X_scaled))
        
        model = self.regression_models[market]
        
        if hasattr(model, 'estimators_'):  # RandomForest
            # Get predictions from all trees
            trees = model.estimators_
            tree_preds = np.array([tree.predict(X_scaled) for tree in trees])
            
            # Calculate standard deviation across trees (lower std = higher confidence)
            std = np.std(tree_preds, axis=0)
            max_std = np.max(std) if len(std) > 0 else 1
            
            # Convert to confidence score (1 - normalized std)
            confidence = 1 - (std / max_std)
            
        else:  # LightGBM
            # For LightGBM, we'll use the prediction standard deviation
            # from built-in method if available, otherwise return uniform confidence
            try:
                # Make predictions with standard deviation
                pred_mean = model.predict(X_scaled)
                
                # Calculate relative deviation from mean as confidence
                abs_dev = np.abs(pred_mean - np.mean(pred_mean))
                max_dev = np.max(abs_dev) if len(abs_dev) > 0 else 1
                confidence = 1 - (abs_dev / max_dev)
                
            except Exception as e:
                logger.warning(f"Could not calculate confidence for LightGBM model: {str(e)}")
                confidence = np.ones(len(X_scaled)) * 0.5  # Default medium confidence
        
        return confidence