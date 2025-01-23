import pandas as pd
import numpy as np
import os
import joblib
import json
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score, log_loss, precision_score, recall_score, f1_score
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.ensemble import RandomForestRegressor
from .base_predictor import BasePredictor
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class UnifiedPredictor(BasePredictor):
    def __init__(self, config):
        super().__init__(config)
        self.models = {}
        self.regression_models = {}
        self.feature_sets = self._define_feature_sets()
    
    def _define_feature_sets(self):
        base_features = self._define_base_features()
        
        return {
            'match_result': base_features + [
                'Home_League_Position', 'Away_League_Position',
                'Position_Diff', 'Points_Diff'
            ],
            'over_under': base_features + [
                'Goals_Diff_Home', 'Goals_Diff_Away',
                'Home_Clean_Sheets', 'Away_Clean_Sheets'
            ],
            'corners': base_features + self._define_corner_features(),
            'cards': base_features + self._define_card_features()
        }
    
    def _prepare_training_data(self, df, market, train_seasons, test_season):
        X = df[self.feature_sets[market]].copy()
        y = self._get_target(df, market)
        
        valid_indices = y.index
        X = X.loc[valid_indices]
        
        logger.info(f"Features used ({len(self.feature_sets[market])}):")
        for feature in self.feature_sets[market]:
            logger.info(f"- {feature}")
        
        logger.info(f"\nTraining data size after filtering:")
        logger.info(f"Total samples: {len(X)}")
        logger.info(f"Valid samples: {len(valid_indices)}")
        
        if len(X) < self.config.min_training_samples:
            raise ValueError(f"Insufficient training data for {market} after filtering")
        
        X_train = X[df['Season'].isin(train_seasons)]
        y_train = y[df['Season'].isin(train_seasons)]
        X_test = X[df['Season'] == test_season]
        y_test = y[df['Season'] == test_season]
        
        self.scalers[market] = StandardScaler()
        X_train_scaled = self.scalers[market].fit_transform(X_train)
        X_test_scaled = self.scalers[market].transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns
    
    def _train_regression_model(self, market, X_train, X_test, y_train, y_test, feature_names):
        if market == 'corners':
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
            self.regression_models[market] = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            )
        
        self.regression_models[market].fit(X_train, y_train)
        y_pred = self.regression_models[market].predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        if isinstance(self.regression_models[market], LGBMRegressor):
            importance = self.regression_models[market].feature_importances_
        else:
            importance = self.regression_models[market].feature_importances_
            
        self.feature_importances[market] = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return metrics
    
    def _train_classification_model(self, market, X_train, X_test, y_train, y_test, feature_names):
        if market == 'match_result':
            # For match results, use LightGBM with optimized parameters for better confidence
            self.models[market] = LGBMClassifier(
                n_estimators=1000,          # More trees
                learning_rate=0.005,        # Slower learning rate
                num_leaves=63,              # More leaves for complex patterns
                max_depth=10,               # Deeper trees
                min_child_samples=50,       # More samples per leaf for stability
                subsample=0.7,              # Sample 70% of data per tree
                colsample_bytree=0.7,       # Sample 70% of features per tree
                reg_alpha=0.05,             # L1 regularization
                reg_lambda=0.05,            # L2 regularization
                random_state=42,
                class_weight='balanced',
                n_jobs=-1,
                verbose=-1,
                boosting_type='gbdt',       # Traditional gradient boosting
                feature_fraction=0.8,       # Use 80% of features in each iteration
                bagging_freq=5,            # Perform bagging every 5 iterations
                bagging_fraction=0.8,      # Use 80% of data for bagging
                min_data_in_leaf=30,       # Minimum samples in leaf for stability
                max_bin=255               # More bins for continuous features
            )
        else:
            # For binary classification (like over/under), keep RandomForest
            self.models[market] = RandomForestClassifier(
                n_estimators=200,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        
        # Convert target to numpy array to ensure proper shape
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        
        # Check unique classes and their distribution
        unique_classes = np.unique(y_train)
        class_counts = {str(c): np.sum(y_train == c) for c in unique_classes}
        if market == 'match_result':
            if len(unique_classes) != 3:
                logger.warning(f"Expected 3 classes for match_result, but found {len(unique_classes)}")
                logger.warning(f"Unique classes: {unique_classes}")
                logger.warning(f"Class distribution: {class_counts}")
            else:
                logger.info("Class distribution in training data:")
                for cls, count in class_counts.items():
                    logger.info(f"Class {cls}: {count} samples ({count/len(y_train)*100:.2f}%)")
        
        # Feature selection for match_result
        if market == 'match_result':
            # First fit a quick model to get feature importances
            quick_model = LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
            quick_model.fit(X_train, y_train)
            
            # Select top features
            importances = pd.DataFrame({
                'feature': feature_names,
                'importance': quick_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            top_features = importances.head(30)['feature'].tolist()  # Keep top 30 features
            logger.info(f"\nSelected top {len(top_features)} features for match_result model:")
            for f in top_features:
                logger.info(f"- {f}")
            
            # Filter features
            feature_mask = [f in top_features for f in feature_names]
            X_train = X_train[:, feature_mask]
            X_test = X_test[:, feature_mask]
            feature_names = [f for f in feature_names if f in top_features]
        
        self.models[market].fit(X_train, y_train)
        y_pred = self.models[market].predict(X_test)
        
        # Ensure we have all classes in the predictions
        if market == 'match_result':
            proba = self.models[market].predict_proba(X_test)
            if proba.shape[1] != 3:
                logger.error(f"Model predicting {proba.shape[1]} classes instead of 3")
                raise ValueError("Model not properly configured for 3-class prediction")
            
            # Log probability distribution statistics
            prob_means = proba.mean(axis=0)
            prob_stds = proba.std(axis=0)
            logger.info("\nProbability distribution statistics:")
            logger.info(f"Mean probabilities: Home={prob_means[0]:.3f}, Draw={prob_means[1]:.3f}, Away={prob_means[2]:.3f}")
            logger.info(f"Std probabilities: Home={prob_stds[0]:.3f}, Draw={prob_stds[1]:.3f}, Away={prob_stds[2]:.3f}")
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        if isinstance(self.models[market], LGBMClassifier):
            importance = self.models[market].feature_importances_
        else:
            importance = self.models[market].feature_importances_
            
        self.feature_importances[market] = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return metrics, y_pred
    
    def train(self, df, save_path=None, test_mode=False):
        metrics = {}
        all_feature_importances = {}
        seasonal_progression = {}
        league_performance = {}
        
        df = df.sort_values('Date')
        seasons = sorted(df['Season'].unique())
        logger.info(f"Training data spans {len(seasons)} seasons: {seasons}")
        
        validation_windows = []
        if test_mode:
            train_size = int(len(df) * 0.8)
            train_data = df.iloc[:train_size]
            test_data = df.iloc[train_size:]
            train_seasons = sorted(train_data['Season'].unique())
            test_seasons = sorted(test_data['Season'].unique())
            validation_windows.append((train_seasons, test_seasons[-1]))
        else:
            # Use all but the last season for training, last season for testing
            train_seasons = seasons[:-1]
            test_season = seasons[-1]
            validation_windows.append((train_seasons, test_season))
        
        logger.info(f"Created {len(validation_windows)} validation windows:")
        for train_seasons, test_season in validation_windows:
            logger.info(f"Train seasons: {train_seasons}, Test season: {test_season}")
        
        for market in self.config.markets:
            if not self.config.markets[market]:
                logger.info(f"Skipping {market} market (disabled in config)")
                continue
            
            try:
                logger.info(f"\nTraining {market} model...")
                
                # Check if we have all required features
                required_features = self.feature_sets[market]
                missing_features = [f for f in required_features if f not in df.columns]
                if missing_features:
                    raise ValueError(f"Missing required features for {market}: {missing_features}")
                
                market_metrics = []
                market_progressions = []
                market_league_perfs = []
                
                for train_seasons, test_season in validation_windows:
                    logger.info(f"\nValidation window - Train: {train_seasons}, Test: {test_season}")
                    
                    try:
                        X_train, X_test, y_train, y_test, feature_names = self._prepare_training_data(
                            df, market, train_seasons, test_season
                        )
                        
                        if market in ['corners', 'cards']:
                            window_metrics = self._train_regression_model(
                                market, X_train, X_test, y_train, y_test, feature_names
                            )
                            y_pred = self.regression_models[market].predict(X_test)
                        else:
                            window_metrics, y_pred = self._train_classification_model(
                                market, X_train, X_test, y_train, y_test, feature_names
                            )
                        
                        market_metrics.append(window_metrics)
                        
                        test_df = df[df['Season'] == test_season]
                        window_progression = self._analyze_seasonal_progression(
                            test_df, y_test, y_pred, market
                        )
                        market_progressions.append(window_progression)
                        
                        window_league_perf = self._analyze_league_performance(
                            test_df, y_test, y_pred, market
                        )
                        market_league_perfs.append(window_league_perf)
                        
                    except Exception as e:
                        logger.error(f"Error in validation window for {market} model: {str(e)}", exc_info=True)
                        raise
                
                metrics[market] = self._average_metrics(market_metrics)
                
                # Convert list of progressions to averaged dictionary
                progression_dict = {}
                if market_progressions:
                    max_matches = max(len(prog) for prog in market_progressions)
                    for match_num in range(max_matches):
                        values = [prog[match_num] for prog in market_progressions if match_num < len(prog)]
                        if values:
                            progression_dict[match_num + 1] = sum(values) / len(values)
                seasonal_progression[market] = progression_dict
                
                # Convert list of league performances to averaged dictionary
                league_dict = {}
                if market_league_perfs:
                    all_leagues = set()
                    for perf in market_league_perfs:
                        all_leagues.update(perf.keys())
                    
                    for league in all_leagues:
                        values = [perf.get(league, 0) for perf in market_league_perfs]
                        league_dict[league] = sum(values) / len(values)
                league_performance[market] = league_dict
                
                all_feature_importances[market] = self.feature_importances[market].head(10)
                
                if save_path:
                    self._save_model(market, save_path)
                
            except Exception as e:
                logger.error(f"Error training {market} model: {str(e)}", exc_info=True)
                logger.error(f"Skipping {market} market due to training error")
                continue
        
        logger.info("\n" + "="*50)
        logger.info("TRAINING RESULTS SUMMARY")
        logger.info("="*50)
        
        for market in metrics:
            logger.info(f"\n{market.upper()} METRICS:")
            for metric, value in metrics[market].items():
                logger.info(f"- {metric}: {value:.4f}")
            
            if market in seasonal_progression and seasonal_progression[market]:
                logger.info("\nSeasonal Progression:")
                for match_num, acc in seasonal_progression[market].items():
                    logger.info(f"Match {match_num}: {acc:.4f}")
            
            if market in league_performance and league_performance[market]:
                logger.info("\nLeague Performance:")
                for league, perf in league_performance[market].items():
                    logger.info(f"{league}: {perf:.4f}")
        
        return {
            'metrics': metrics,
            'seasonal_progression': seasonal_progression,
            'league_performance': league_performance,
            'feature_importances': all_feature_importances
        }
    
    def predict(self, df):
        predictions = {}
        
        for market in self.config.markets:
            if not self.config.markets[market]:
                logger.info(f"Skipping {market} market (disabled in config)")
                continue
            
            if market not in self.scalers:
                logger.warning(f"No trained model found for {market} market. Skipping predictions.")
                continue
                
            try:
                X = df[self.feature_sets[market]]
                X_scaled = self.scalers[market].transform(X)
                
                if market in ['corners', 'cards']:
                    if market in self.regression_models:
                        pred_values = self.regression_models[market].predict(X_scaled)
                        predictions[market] = self._format_regression_predictions(df, pred_values, market)
                    else:
                        logger.warning(f"No regression model found for {market} market")
                else:
                    if market in self.models:
                        probs = self.models[market].predict_proba(X_scaled)
                        if market == 'match_result':
                            # Ensure we include original odds in the output
                            predictions[market] = pd.DataFrame({
                                'Date': df['Date'],
                                'HomeTeam': df['HomeTeam'],
                                'AwayTeam': df['AwayTeam'],
                                'Home_Prob': probs[:, 0],
                                'Draw_Prob': probs[:, 1],
                                'Away_Prob': probs[:, 2],
                                'B365H': df['B365H'],  # Include original odds
                                'B365D': df['B365D'],
                                'B365A': df['B365A'],
                                'Predicted': ['H' if p[0] > max(p[1], p[2]) else 'D' if p[1] > p[2] else 'A' for p in probs],
                                'Confidence': self._calculate_confidence(probs)
                            })
                            
                            # Calculate betting value
                            predictions[market]['Home_Value'] = predictions[market]['Home_Prob'] * predictions[market]['B365H']
                            predictions[market]['Draw_Value'] = predictions[market]['Draw_Prob'] * predictions[market]['B365D']
                            predictions[market]['Away_Value'] = predictions[market]['Away_Prob'] * predictions[market]['B365A']
                            
                        else:  # over_under
                            predictions[market] = pd.DataFrame({
                                'Date': df['Date'],
                                'HomeTeam': df['HomeTeam'],
                                'AwayTeam': df['AwayTeam'],
                                'Under_Prob': probs[:, 0],
                                'Over_Prob': probs[:, 1],
                                'Predicted': ['Under' if p[0] > p[1] else 'Over' for p in probs],
                                'Confidence': self._calculate_confidence(probs)
                            })
                    else:
                        logger.warning(f"No classification model found for {market} market")
            except Exception as e:
                logger.error(f"Error making predictions for {market} market: {str(e)}")
                continue
        
        return predictions
    
    def save(self, model_dir):
        super().save(model_dir)
        
        for market in self.models:
            joblib.dump(self.models[market], os.path.join(model_dir, f"{market}_model.joblib"))
        
        for market in self.regression_models:
            joblib.dump(self.regression_models[market], os.path.join(model_dir, f"{market}_model.joblib"))
    
    def load(self, model_dir):
        super().load(model_dir)
        
        for market in self.config.markets:
            if not self.config.markets[market]:
                continue
                
            model_path = os.path.join(model_dir, f"{market}_model.joblib")
            if os.path.exists(model_path):
                if market in ['corners', 'cards']:
                    self.regression_models[market] = joblib.load(model_path)
                else:
                    self.models[market] = joblib.load(model_path)
    
    def plot_feature_importance(self, save_path=None):
        """Plot feature importance for each market."""
        if not self.feature_importances:
            logger.warning("No feature importance data available.")
            return
        
        plt.style.use('bmh')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12), facecolor='#f0f0f0')
        fig.suptitle('Feature Importance by Market', fontsize=16, y=1.02)
        
        axes = axes.flatten()
        colors = plt.cm.Set3(np.linspace(0, 1, 10))
        
        for idx, (market, importance_df) in enumerate(self.feature_importances.items()):
            if importance_df is None:
                continue
                
            ax = axes[idx]
            importance_df = importance_df.sort_values('importance', ascending=True)
            
            bars = ax.barh(range(len(importance_df)), 
                          importance_df['importance'],
                          color=colors[idx % len(colors)])
            
            ax.set_yticks(range(len(importance_df)))
            ax.set_yticklabels(importance_df['feature'], fontsize=8)
            
            ax.set_title(f'{market.upper()} Feature Importance', fontsize=12, pad=10)
            ax.set_xlabel('Importance', fontsize=10)
            
            # Add value labels
            for i, v in enumerate(importance_df['importance']):
                ax.text(v, i, f'{v:.3f}', va='center', fontsize=8)
            
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.set_axisbelow(True)
            ax.set_facecolor('white')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, 'feature_importance.png'),
                       dpi=300, bbox_inches='tight',
                       facecolor='white')
        else:
            plt.show()
        
        plt.close()