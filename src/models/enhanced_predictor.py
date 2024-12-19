import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import joblib
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.ensemble import VotingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import warnings
import lightgbm

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedPredictor:
    """Enhanced predictor with ensemble models and advanced features."""
    
    def __init__(self, model_dir: Path = Path('models')):
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model configurations
        self.model_configs = {
            'match_result': {
                'model': LGBMClassifier(
                    n_estimators=1000,
                    learning_rate=0.01,
                    num_leaves=31,
                    class_weight='balanced',
                    random_state=42,
                    verbose=-1
                ),
                'type': 'multiclass',
                'n_classes': 3,
                'class_labels': ['Away', 'Draw', 'Home'],
                'eval_metric': ['multi_logloss', 'multi_error'],
                'required_features': [
                    # Form features
                    'home_recent_wins', 'home_recent_draws', 'home_recent_losses',
                    'home_recent_goals_scored', 'home_recent_goals_conceded',
                    'home_recent_clean_sheets',
                    'away_recent_wins', 'away_recent_draws', 'away_recent_losses',
                    'away_recent_goals_scored', 'away_recent_goals_conceded',
                    'away_recent_clean_sheets',
                    # Team strength features
                    'home_ppg', 'home_goals_per_game', 'home_conceded_per_game',
                    'home_clean_sheet_ratio', 'home_win_ratio', 'home_league_position',
                    'away_ppg', 'away_goals_per_game', 'away_conceded_per_game',
                    'away_clean_sheet_ratio', 'away_win_ratio', 'away_league_position',
                    # Form trends
                    'home_form_trend', 'away_form_trend',
                    'home_goals_trend', 'away_goals_trend',
                    # H2H stats
                    'h2h_home_wins', 'h2h_away_wins', 'h2h_draws',
                    'h2h_avg_goals', 'h2h_home_team_avg_goals',
                    'h2h_away_team_avg_goals',
                    # Market features
                    'B365H', 'B365D', 'B365A',
                    'implied_home_prob', 'implied_draw_prob', 'implied_away_prob',
                    'market_overround'
                ]
            }
        }
        
        # Initialize processors
        self.scalers = {}
        self.calibrators = {}
        self.feature_importances = {}
        
        # Initialize validation strategy
        self.cv_strategy = TimeSeriesSplit(n_splits=5)
    
    @staticmethod
    def calculate_market_features(odds: Dict[str, float]) -> Dict[str, float]:
        """Calculate market-derived features from odds."""
        home_odds = odds['B365H']
        draw_odds = odds['B365D']
        away_odds = odds['B365A']
        
        # Calculate implied probabilities
        implied_home = 1 / home_odds
        implied_draw = 1 / draw_odds
        implied_away = 1 / away_odds
        
        # Calculate overround
        overround = implied_home + implied_draw + implied_away - 1
        
        # Normalize probabilities
        total_prob = implied_home + implied_draw + implied_away
        implied_home_norm = implied_home / total_prob
        implied_draw_norm = implied_draw / total_prob
        implied_away_norm = implied_away / total_prob
        
        return {
            'implied_home_prob': implied_home_norm,
            'implied_draw_prob': implied_draw_norm,
            'implied_away_prob': implied_away_norm,
            'market_overround': overround
        }
    
    def preprocess_features(self, data: pd.DataFrame, market: str) -> pd.DataFrame:
        """Preprocess features for a specific market."""
        config = self.model_configs[market]
        features = config['required_features']
        
        # Select required features
        X = data[features].copy()
        
        # Handle missing values with more sophisticated approach
        numeric_features = X.select_dtypes(include=[np.number]).columns
        categorical_features = X.select_dtypes(exclude=[np.number]).columns
        
        # Fill numeric missing values with median
        X[numeric_features] = X[numeric_features].fillna(X[numeric_features].median())
        
        # Fill categorical missing values with mode
        for col in categorical_features:
            X[col] = X[col].fillna(X[col].mode()[0])
        
        # Scale features if not already fit
        if market not in self.scalers:
            self.scalers[market] = StandardScaler()
            X_scaled = self.scalers[market].fit_transform(X)
        else:
            X_scaled = self.scalers[market].transform(X)
        
        return pd.DataFrame(X_scaled, columns=features, index=X.index)
    
    def train(self, train_data: pd.DataFrame, labels: Dict[str, pd.Series]) -> Dict[str, Dict]:
        """Train models with proper validation."""
        evaluation_metrics = {}
        
        for market, config in self.model_configs.items():
            if market not in labels:
                logger.warning(f"No labels provided for {market}, skipping training")
                continue
            
            logger.info(f"\nTraining {market} model...")
            
            # Preprocess features
            X = self.preprocess_features(train_data, market)
            y = labels[market]
            
            # Initialize metrics collection
            fold_metrics = []
            
            # Time series cross-validation
            for fold, (train_idx, val_idx) in enumerate(self.cv_strategy.split(X)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train model
                model = config['model']
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric=config['eval_metric'],
                    callbacks=[
                        lightgbm.early_stopping(stopping_rounds=50),
                        lightgbm.log_evaluation(period=0)
                    ]
                )
                
                # Make predictions
                y_pred = model.predict(X_val)
                y_pred_proba = model.predict_proba(X_val)
                
                # Calculate metrics
                fold_metric = {
                    'accuracy': accuracy_score(y_val, y_pred),
                    'precision': precision_score(y_val, y_pred, average='weighted'),
                    'recall': recall_score(y_val, y_pred, average='weighted'),
                    'f1': f1_score(y_val, y_pred, average='weighted'),
                    'log_loss': log_loss(y_val, y_pred_proba)
                }
                fold_metrics.append(fold_metric)
                
                logger.info(f"Fold {fold + 1} metrics:")
                for metric, value in fold_metric.items():
                    logger.info(f"{metric}: {value:.4f}")
            
            # Calculate average metrics
            avg_metrics = {
                metric: np.mean([fold[metric] for fold in fold_metrics])
                for metric in fold_metrics[0].keys()
            }
            
            # Train final model on all data
            final_model = config['model']
            
            # Create a small validation set for early stopping
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.1, shuffle=False  # Keep chronological order
            )
            
            final_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric=config['eval_metric'],
                callbacks=[
                    lightgbm.early_stopping(stopping_rounds=50),
                    lightgbm.log_evaluation(period=0)
                ]
            )
            
            # Store model and metrics
            self.model_configs[market]['model'] = final_model
            evaluation_metrics[market] = avg_metrics
            
            # Calculate and store feature importance
            importance = pd.DataFrame({
                'feature': X.columns,
                'importance': final_model.feature_importances_
            })
            self.feature_importances[market] = importance.sort_values('importance', ascending=False)
            
            # Log top features
            logger.info("\nTop 10 important features:")
            for _, row in importance.head(10).iterrows():
                logger.info(f"{row['feature']}: {row['importance']:.4f}")
        
        return evaluation_metrics
    
    def predict(self, features_df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Make predictions for all markets."""
        predictions = {}
        
        for market, config in self.model_configs.items():
            logger.info(f"\nMaking predictions for {market}...")
            
            # Preprocess features
            X = self.preprocess_features(features_df, market)
            
            # Make predictions
            model = config['model']
            probas = model.predict_proba(X)
            
            # Format predictions based on market type
            if config['type'] == 'multiclass':
                pred_list = []
                for row_probas in probas:
                    pred_dict = {
                        label: float(prob)
                        for label, prob in zip(config['class_labels'], row_probas)
                    }
                    pred_list.append(pred_dict)
                predictions[market] = pred_list
            
            logger.info(f"Made {len(pred_list)} predictions")
            logger.info(f"Sample prediction: {pred_list[0]}")
        
        return predictions
    
    def save(self) -> None:
        """Save models and preprocessors."""
        try:
            for market, config in self.model_configs.items():
                # Save model
                model_path = self.model_dir / f"{market}_model.joblib"
                joblib.dump(config['model'], model_path)
                
                # Save scaler
                scaler_path = self.model_dir / f"{market}_scaler.joblib"
                joblib.dump(self.scalers[market], scaler_path)
                
                # Save feature importance
                importance_path = self.model_dir / f"{market}_importance.json"
                self.feature_importances[market].to_json(importance_path)
            
            logger.info(f"Models and preprocessors saved to {self.model_dir}")
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
    
    def load(self) -> None:
        """Load models and preprocessors."""
        try:
            for market in self.model_configs.keys():
                # Load model
                model_path = self.model_dir / f"{market}_model.joblib"
                if model_path.exists():
                    self.model_configs[market]['model'] = joblib.load(model_path)
                
                # Load scaler
                scaler_path = self.model_dir / f"{market}_scaler.joblib"
                if scaler_path.exists():
                    self.scalers[market] = joblib.load(scaler_path)
                
                # Load feature importance
                importance_path = self.model_dir / f"{market}_importance.json"
                if importance_path.exists():
                    self.feature_importances[market] = pd.read_json(importance_path)
            
            logger.info(f"Models and preprocessors loaded from {self.model_dir}")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}") 