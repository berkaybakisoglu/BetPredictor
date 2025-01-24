"""Enhanced prediction module using LightGBM and ensemble methods."""
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from pathlib import Path
import logging

from src.config.config import ModelConfig

class LightGBMPredictor:
    """Enhanced predictor using LightGBM models."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.models: Dict[str, lgb.Booster] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_sets: Dict[str, List[str]] = self._define_feature_sets()
        self.cv = TimeSeriesSplit(n_splits=5)
        
    def _define_feature_sets(self) -> Dict[str, List[str]]:
        """Define feature sets for each market."""
        # Reuse the same feature sets as UnifiedPredictor
        base_features = [
            'Home_Goals_Scored_Avg', 'Home_Goals_Conceded_Avg',
            'Away_Goals_Scored_Avg', 'Away_Goals_Conceded_Avg',
            'Home_Form', 'Away_Form',
            'H2H_Home_Wins', 'H2H_Away_Wins', 'H2H_Draws',
            'H2H_Avg_Goals', 'Home_ImpliedProb', 'Draw_ImpliedProb',
            'Away_ImpliedProb'
        ]
        
        corner_features = [
            'Home_Corners_For_Avg', 'Home_Corners_Against_Avg',
            'Away_Corners_For_Avg', 'Away_Corners_Against_Avg',
            'H2H_Avg_Corners'
        ]
        
        card_features = [
            'Home_Cards_Avg', 'Away_Cards_Avg'
        ]
        
        return {
            'match_result': base_features,
            'over_under': base_features,
            'corners': base_features + corner_features,
            'cards': base_features + card_features
        }
    
    def train(self, df: pd.DataFrame, save_path: Optional[Path] = None, test_mode: bool = False) -> Dict[str, Dict[str, float]]:
        """Train models using LightGBM with time-series cross-validation."""
        metrics = {}
        logger = logging.getLogger(__name__)
        
        # Sort by date for time-series validation
        df = df.sort_values('Date')
        
        # Get unique seasons
        seasons = df['Season'].unique()
        logger.info(f"Training data spans {len(seasons)} seasons: {sorted(seasons)}")
        
        if not test_mode and len(seasons) < 3:
            raise ValueError("Need at least 3 seasons for proper validation")
        
        # Use the last season as final test set
        train_seasons = seasons[:-1]
        test_season = seasons[-1]
        
        for market in self.config.markets:
            if not self.config.markets[market]:
                continue
                
            logger.info(f"\nTraining {market} model...")
            
            try:
                # Prepare data
                X = df[self.feature_sets[market]].copy()
                y = self._get_target(df, market)
                
                # Split data
                X_train = X[df['Season'].isin(train_seasons)]
                y_train = y[df['Season'].isin(train_seasons)]
                X_test = X[df['Season'] == test_season]
                y_test = y[df['Season'] == test_season]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                self.scalers[market] = scaler
                
                # Create LightGBM datasets
                train_data = lgb.Dataset(
                    X_train_scaled, 
                    label=y_train,
                    feature_name=self.feature_sets[market]
                )
                
                # Define parameters based on market type
                params = self._get_lgb_params(market)
                
                # Train model with early stopping
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=1000,
                    valid_sets=[train_data],
                    callbacks=[
                        lgb.early_stopping(50),
                        lgb.log_evaluation(100)
                    ]
                )
                
                # Store model
                self.models[market] = model
                
                # Make predictions
                y_pred = self._predict_market(X_test_scaled, market)
                
                # Calculate metrics
                metrics[market] = self._calculate_metrics(y_test, y_pred, market)
                
                # Save model if path provided
                if save_path:
                    save_path.mkdir(parents=True, exist_ok=True)
                    model.save_model(str(save_path / f"{market}_lgb_model.txt"))
                    joblib.dump(scaler, save_path / f"{market}_scaler.joblib")
                
                # Log feature importance
                importance = pd.DataFrame({
                    'feature': self.feature_sets[market],
                    'importance': model.feature_importance()
                }).sort_values('importance', ascending=False)
                
                logger.info("\nTop 10 important features:")
                for _, row in importance.head(10).iterrows():
                    logger.info(f"{row['feature']}: {row['importance']}")
                
            except Exception as e:
                logger.error(f"Error training {market} model: {str(e)}", exc_info=True)
                continue
        
        return metrics
    
    def _get_lgb_params(self, market: str) -> Dict:
        """Get LightGBM parameters based on market type."""
        base_params = {
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'max_depth': -1,
            'learning_rate': 0.01,
            'n_estimators': 1000,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbose': -1
        }
        
        if market == 'match_result':
            return {
                **base_params,
                'objective': 'multiclass',
                'num_class': 3,
                'metric': 'multi_logloss'
            }
        elif market == 'over_under':
            return {
                **base_params,
                'objective': 'binary',
                'metric': 'binary_logloss'
            }
        else:  # corners, cards
            return {
                **base_params,
                'objective': 'regression',
                'metric': 'rmse'
            }
    
    def predict(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Make predictions for all enabled markets."""
        predictions = {}
        
        for market in self.config.markets:
            if not self.config.markets[market] or market not in self.models:
                continue
            
            X = df[self.feature_sets[market]]
            X_scaled = self.scalers[market].transform(X)
            
            predictions[market] = self._format_predictions(
                df,
                market,
                self._predict_market(X_scaled, market)
            )
        
        return predictions
    
    def _predict_market(self, X: np.ndarray, market: str) -> np.ndarray:
        """Make predictions for a specific market."""
        model = self.models[market]
        
        if market == 'match_result':
            return model.predict(X)
        elif market == 'over_under':
            return model.predict(X)
        else:  # corners, cards
            return model.predict(X)
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, market: str) -> Dict[str, float]:
        """Calculate metrics based on market type."""
        if market in ['corners', 'cards']:
            return {
                'rmse': np.sqrt(np.mean((y_true - y_pred) ** 2)),
                'mae': np.mean(np.abs(y_true - y_pred))
            }
        else:
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1': f1_score(y_true, y_pred, average='weighted')
            }
    
    def _get_target(self, df: pd.DataFrame, market: str) -> pd.Series:
        """Get target variable for a market."""
        if market == 'match_result':
            result_map = {'H': 0, 'D': 1, 'A': 2}
            valid_results = df['FTR'].isin(result_map.keys())
            return df.loc[valid_results, 'FTR'].map(result_map)
        elif market == 'over_under':
            valid_goals = df['FTHG'].notna() & df['FTAG'].notna()
            return df.loc[valid_goals, ['FTHG', 'FTAG']].sum(axis=1).gt(2.5).astype(int)
        elif market == 'corners':
            valid_corners = df['HC'].notna() & df['AC'].notna()
            return df.loc[valid_corners, ['HC', 'AC']].sum(axis=1)
        elif market == 'cards':
            valid_cards = df['HY'].notna() & df['AY'].notna()
            return df.loc[valid_cards, ['HY', 'AY']].sum(axis=1)
        else:
            raise ValueError(f"Unknown market: {market}")
    
    def _format_predictions(self, df: pd.DataFrame, market: str, preds: np.ndarray) -> pd.DataFrame:
        """Format predictions based on market type."""
        base_cols = {
            'Date': df['Date'],
            'HomeTeam': df['HomeTeam'],
            'AwayTeam': df['AwayTeam']
        }
        
        if market == 'match_result':
            result = pd.DataFrame(base_cols)
            result['Predicted'] = ['H', 'D', 'A'][np.argmax(preds, axis=1)]
            result['Confidence'] = np.max(preds, axis=1)
            return result
        elif market == 'over_under':
            result = pd.DataFrame(base_cols)
            result['Predicted'] = preds > 0.5
            result['Confidence'] = np.maximum(preds, 1 - preds)
            return result
        else:  # corners, cards
            result = pd.DataFrame(base_cols)
            result['Predicted'] = preds.round(1)
            return result 