"""Enhanced ensemble predictor module."""
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.config.config import ModelConfig

class EnsemblePredictor:
    """Enhanced predictor using ensemble of multiple models."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.models: Dict[str, VotingClassifier] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_sets: Dict[str, List[str]] = self._define_feature_sets()
        
    def _define_feature_sets(self) -> Dict[str, List[str]]:
        """Define feature sets for each market - using same features as UnifiedPredictor."""
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
    
    def _create_ensemble(self, n_classes: int) -> VotingClassifier:
        """Create an ensemble of different models."""
        models = [
            ('lgbm', LGBMClassifier(
                n_estimators=500,
                learning_rate=0.01,
                num_leaves=31,
                class_weight='balanced',
                random_state=42
            )),
            ('xgb', XGBClassifier(
                n_estimators=500,
                learning_rate=0.01,
                max_depth=7,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )),
            ('catboost', CatBoostClassifier(
                iterations=500,
                learning_rate=0.01,
                depth=7,
                loss_function='MultiClass' if n_classes > 2 else 'Logloss',
                verbose=False,
                random_state=42
            )),
            ('gbm', GradientBoostingClassifier(
                n_estimators=500,
                learning_rate=0.01,
                max_depth=7,
                subsample=0.8,
                random_state=42
            ))
        ]
        
        return VotingClassifier(
            estimators=models,
            voting='soft',  # Use probability predictions
            n_jobs=-1  # Use all available cores
        )
    
    def train(self, df: pd.DataFrame, save_path: Optional[Path] = None, test_mode: bool = False) -> Dict[str, Dict[str, float]]:
        """Train ensemble models for all enabled markets."""
        metrics = {}
        logger = logging.getLogger(__name__)
        
        # Sort by date for time-series validation
        df = df.sort_values('Date')
        
        # Get unique seasons
        seasons = df['Season'].unique()
        logger.info(f"Training data spans {len(seasons)} seasons: {sorted(seasons)}")
        
        if not test_mode and len(seasons) < 3:
            raise ValueError("Need at least 3 seasons for proper validation")
        elif test_mode and len(seasons) < 2:
            raise ValueError("Need at least 2 seasons even in test mode")
        
        # Use last season as test set
        train_seasons = seasons[:-1]
        test_season = seasons[-1]
        
        for market in self.config.markets:
            if not self.config.markets[market]:
                continue
            
            logger.info(f"\nTraining {market} model...")
            
            try:
                # Get features and target
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
                
                # Create and train ensemble
                n_classes = len(np.unique(y))
                ensemble = self._create_ensemble(n_classes)
                ensemble.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = ensemble.predict(X_test_scaled)
                
                # Calculate metrics
                metrics[market] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted'),
                    'recall': recall_score(y_test, y_pred, average='weighted'),
                    'f1': f1_score(y_test, y_pred, average='weighted')
                }
                
                # Store model
                self.models[market] = ensemble
                
                # Save if path provided
                if save_path:
                    save_path.mkdir(parents=True, exist_ok=True)
                    joblib.dump(ensemble, save_path / f"{market}_ensemble.joblib")
                    joblib.dump(scaler, save_path / f"{market}_scaler.joblib")
                
                logger.info(f"{market} metrics:")
                for metric, value in metrics[market].items():
                    logger.info(f"- {metric}: {value:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {market} model: {str(e)}", exc_info=True)
                continue
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Make predictions using ensemble models."""
        predictions = {}
        
        for market in self.config.markets:
            if not self.config.markets[market] or market not in self.models:
                continue
            
            # Get and scale features
            X = df[self.feature_sets[market]]
            X_scaled = self.scalers[market].transform(X)
            
            # Get probabilities from ensemble
            probs = self.models[market].predict_proba(X_scaled)
            
            # Format predictions
            predictions[market] = self._format_predictions(df, market, probs)
        
        return predictions
    
    def load_models(self, load_path: Path) -> None:
        """Load trained ensemble models."""
        for market in self.config.markets:
            if not self.config.markets[market]:
                continue
            
            model_path = load_path / f"{market}_ensemble.joblib"
            scaler_path = load_path / f"{market}_scaler.joblib"
            
            if model_path.exists() and scaler_path.exists():
                self.models[market] = joblib.load(model_path)
                self.scalers[market] = joblib.load(scaler_path)
                print(f"Loaded {market} ensemble model and scaler")
    
    def _get_target(self, df: pd.DataFrame, market: str) -> pd.Series:
        """Get target variable - same as UnifiedPredictor."""
        if market == 'match_result':
            result_map = {'H': 0, 'D': 1, 'A': 2}
            valid_results = df['FTR'].isin(result_map.keys())
            return df.loc[valid_results, 'FTR'].map(result_map)
        elif market == 'over_under':
            valid_goals = df['FTHG'].notna() & df['FTAG'].notna()
            return df.loc[valid_goals, ['FTHG', 'FTAG']].sum(axis=1).gt(2.5).astype(int)
        elif market == 'corners':
            valid_corners = df['HC'].notna() & df['AC'].notna()
            return pd.qcut(df.loc[valid_corners, ['HC', 'AC']].sum(axis=1), q=3, labels=[0, 1, 2])
        elif market == 'cards':
            valid_cards = df['HY'].notna() & df['AY'].notna()
            return pd.qcut(df.loc[valid_cards, ['HY', 'AY']].sum(axis=1), q=3, labels=[0, 1, 2])
        else:
            raise ValueError(f"Unknown market: {market}")
    
    def _format_predictions(self, df: pd.DataFrame, market: str, probs: np.ndarray) -> pd.DataFrame:
        """Format predictions - similar to UnifiedPredictor but with ensemble probabilities."""
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
            
            predictions['Predicted'] = predictions[['Home_Prob', 'Draw_Prob', 'Away_Prob']].idxmax(axis=1).map({
                'Home_Prob': 'H',
                'Draw_Prob': 'D',
                'Away_Prob': 'A'
            })
            
            predictions['Confidence'] = predictions[['Home_Prob', 'Draw_Prob', 'Away_Prob']].max(axis=1)
            
            # Add ensemble-based value indicators
            predictions['Home_Value'] = predictions['Home_Prob'] * df['B365H']
            predictions['Draw_Value'] = predictions['Draw_Prob'] * df['B365D']
            predictions['Away_Value'] = predictions['Away_Prob'] * df['B365A']
            
            return predictions
        else:
            return pd.DataFrame({
                'Date': df['Date'],
                'HomeTeam': df['HomeTeam'],
                'AwayTeam': df['AwayTeam'],
                **{f'Class_{i}_Prob': probs[:, i] for i in range(probs.shape[1])}
            }) 