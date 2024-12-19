"""Match prediction module."""
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from pathlib import Path

from src.config.config import ModelConfig

class UnifiedPredictor:
    """Unified predictor for all betting markets."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.models: Dict[str, RandomForestClassifier] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_sets: Dict[str, List[str]] = self._define_feature_sets()
        
    def _define_feature_sets(self) -> Dict[str, List[str]]:
        """Define feature sets for each market."""
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
            'ht_score': base_features,
            'corners': base_features + corner_features,
            'cards': base_features + card_features
        }
    
    def train(self, df: pd.DataFrame, save_path: Optional[Path] = None) -> Dict[str, Dict[str, float]]:
        """Train models for all enabled markets using time-series validation.
        
        Args:
            df: DataFrame with features and targets
            save_path: Optional path to save trained models
            
        Returns:
            Dictionary with performance metrics for each market
        """
        metrics = {}
        
        # Sort by date to ensure chronological order
        df = df.sort_values('Date')
        
        # Get unique seasons for time-series validation
        seasons = df['Season'].unique()
        if len(seasons) < 3:
            raise ValueError("Need at least 3 seasons for proper validation")
        
        # Use the last season as final test set
        train_seasons = seasons[:-1]
        test_season = seasons[-1]
        
        for market in self.config.markets:
            if not self.config.markets[market]:
                continue
                
            print(f"\nTraining {market} model...")
            
            # Get features and target
            X = df[self.feature_sets[market]]
            y = self._get_target(df, market)
            
            # Split data by season
            X_train = X[df['Season'].isin(train_seasons)]
            y_train = y[df['Season'].isin(train_seasons)]
            X_test = X[df['Season'] == test_season]
            y_test = y[df['Season'] == test_season]
            
            if len(y_train) < self.config.min_training_samples:
                print(f"Insufficient samples for {market} model")
                continue
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model with optimized parameters
            model = RandomForestClassifier(
                n_estimators=200,  # Increased from 100
                max_depth=15,      # Increased from 10
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',  # Handle class imbalance
                random_state=42
            )
            model.fit(X_train_scaled, y_train)
            
            # Save model and scaler
            self.models[market] = model
            self.scalers[market] = scaler
            
            if save_path:
                save_path.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, save_path / f"{market}_model.joblib")
                joblib.dump(scaler, save_path / f"{market}_scaler.joblib")
            
            # Evaluate on test set
            y_pred = model.predict(X_test_scaled)
            metrics[market] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted')
            }
            
            # Print feature importance
            feature_importance = pd.DataFrame({
                'feature': self.feature_sets[market],
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            print(f"\nTop 5 important features for {market}:")
            print(feature_importance.head())
            
            print(f"\n{market} metrics:", metrics[market])
        
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
            if not self.config.markets[market] or market not in self.models:
                continue
            
            X = df[self.feature_sets[market]]
            X_scaled = self.scalers[market].transform(X)
            
            # Get probabilities
            probs = self.models[market].predict_proba(X_scaled)
            
            # Format predictions
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
            return df['FTR']
        elif market == 'over_under':
            return (df['TotalGoals'] > 2.5).astype(int)
        elif market == 'ht_score':
            return df['HTR']
        elif market == 'corners':
            return pd.qcut(df['TotalCorners'], q=3, labels=['Low', 'Medium', 'High'])
        elif market == 'cards':
            return pd.qcut(df['TotalCards'], q=3, labels=['Low', 'Medium', 'High'])
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
            return pd.DataFrame({
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
        elif market == 'over_under':
            return pd.DataFrame({
                'Date': df['Date'],
                'HomeTeam': df['HomeTeam'],
                'AwayTeam': df['AwayTeam'],
                'Under_Prob': probs[:, 0],
                'Over_Prob': probs[:, 1],
                'B365<2.5': df['B365<2.5'],
                'B365>2.5': df['B365>2.5']
            })
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