import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import logging
from pathlib import Path
import joblib
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiMarketPredictor:
    """Predictor for multiple betting markets (corners, scores, etc.)"""
    
    def __init__(self, model_dir: Path = Path('models')):
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Dictionary to store different models
        self.models = {
            'ht_score': RandomForestClassifier(n_estimators=50, max_depth=5),
            'ft_score': RandomForestClassifier(n_estimators=50, max_depth=5),
            'corners': RandomForestRegressor(n_estimators=50, max_depth=5),
            'cards': RandomForestRegressor(n_estimators=50, max_depth=5)
        }
        
        self.scalers = {market: StandardScaler() for market in self.models.keys()}
        self.feature_importance = {}
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction."""
        df = df.copy()
        
        # Pre-match odds and market features
        base_features = [
            # Main odds
            'B365H', 'B365D', 'B365A',
            'AvgH', 'AvgD', 'AvgA',
            
            # Over/Under odds if available
            'B365>2.5', 'B365<2.5',
            'Avg>2.5', 'Avg<2.5',
            
            # Asian Handicap odds if available
            'AHh', 'B365AHH', 'B365AHA',
            'AvgAHH', 'AvgAHA'
        ]
        
        # Get available numeric columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        available_features = [col for col in base_features if col in numeric_cols]
        
        # Select features and handle missing values
        X = df[available_features].copy()
        X = X.fillna(X.median())
        
        logger.info(f"Selected {len(available_features)} features: {available_features}")
        return X
    
    def prepare_targets(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Prepare different target variables."""
        targets = {}
        
        try:
            logger.info("Preparing target variables...")
            
            # Half-time score combinations (most common)
            if 'HTHG' in df.columns and 'HTAG' in df.columns:
                ht_scores = ['0-0', '1-0', '0-1', '1-1', '2-0', '0-2', 'other']
                ht_score = df.apply(lambda x: f"{int(x['HTHG'])}-{int(x['HTAG'])}", axis=1)
                ht_score = ht_score.apply(lambda x: x if x in ht_scores[:-1] else 'other')
                targets['ht_score'] = ht_score
                logger.info(f"Half-time score distribution:\n{ht_score.value_counts(normalize=True)}")
            else:
                logger.warning("Half-time goals columns not found")
            
            # Full-time score combinations (most common)
            if 'FTHG' in df.columns and 'FTAG' in df.columns:
                ft_scores = ['1-0', '0-1', '1-1', '2-0', '0-2', '2-1', '1-2', '2-2', 'other']
                ft_score = df.apply(lambda x: f"{int(x['FTHG'])}-{int(x['FTAG'])}", axis=1)
                ft_score = ft_score.apply(lambda x: x if x in ft_scores[:-1] else 'other')
                targets['ft_score'] = ft_score
                logger.info(f"Full-time score distribution:\n{ft_score.value_counts(normalize=True)}")
            else:
                logger.warning("Full-time goals columns not found")
            
            # Total corners
            if 'HC' in df.columns and 'AC' in df.columns:
                targets['corners'] = df['HC'].astype(float) + df['AC'].astype(float)
                logger.info(f"Corners stats: mean={targets['corners'].mean():.1f}, std={targets['corners'].std():.1f}")
            else:
                logger.warning("Corner columns not found")
            
            # Total cards
            if 'HY' in df.columns and 'AY' in df.columns:
                targets['cards'] = df['HY'].astype(float) + df['AY'].astype(float)
                logger.info(f"Cards stats: mean={targets['cards'].mean():.1f}, std={targets['cards'].std():.1f}")
            else:
                logger.warning("Card columns not found")
            
            logger.info(f"Prepared {len(targets)} target variables: {list(targets.keys())}")
            
        except Exception as e:
            logger.error(f"Error preparing targets: {str(e)}")
            logger.error(f"DataFrame columns: {df.columns.tolist()}")
            raise
        
        return targets
    
    def train(self, train_data: pd.DataFrame) -> Dict:
        """Train models for different markets."""
        logger.info("Preparing training data...")
        X = self.prepare_features(train_data)
        targets = self.prepare_targets(train_data)
        
        metrics = {}
        for market, target in targets.items():
            if market in self.models:
                logger.info(f"\nTraining {market} model...")
                
                # Scale features
                X_scaled = self.scalers[market].fit_transform(X)
                
                # Train model
                self.models[market].fit(X_scaled, target)
                
                # Get feature importance
                self.feature_importance[market] = pd.DataFrame({
                    'feature': X.columns,
                    'importance': self.models[market].feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Calculate metrics
                if isinstance(self.models[market], RandomForestClassifier):
                    train_preds = self.models[market].predict(X_scaled)
                    metrics[market] = {
                        'accuracy': (train_preds == target).mean(),
                        'unique_values': target.unique().tolist(),
                        'class_distribution': target.value_counts(normalize=True).to_dict()
                    }
                else:
                    train_preds = self.models[market].predict(X_scaled)
                    metrics[market] = {
                        'mae': np.abs(train_preds - target).mean(),
                        'rmse': np.sqrt(((train_preds - target) ** 2).mean())
                    }
        
        return metrics
    
    def predict(self, match_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Make predictions for all markets."""
        predictions = {}
        
        try:
            logger.info("Preparing features for prediction...")
            X = self.prepare_features(match_data)
            
            for market, model in self.models.items():
                if market in self.scalers:
                    logger.info(f"\nMaking predictions for {market}...")
                    X_scaled = self.scalers[market].transform(X)
                    
                    if isinstance(model, RandomForestClassifier):
                        # Get probabilities for each class
                        probas = model.predict_proba(X_scaled)
                        classes = model.classes_
                        
                        # Create dictionary of outcome probabilities
                        pred_dict = [{cls: float(prob) for cls, prob in zip(classes, row_probas)} 
                                   for row_probas in probas]
                        predictions[market] = pred_dict
                        
                        logger.info(f"Made {len(pred_dict)} predictions for {market}")
                        logger.info(f"Sample prediction: {pred_dict[0]}")
                    else:
                        # For regression models (corners, cards)
                        preds = model.predict(X_scaled)
                        
                        # Get prediction intervals (using tree variance)
                        pred_std = np.std([tree.predict(X_scaled) 
                                         for tree in model.estimators_], axis=0)
                        lower = preds - 1.96 * pred_std
                        upper = preds + 1.96 * pred_std
                        
                        predictions[market] = [{'prediction': float(pred), 
                                              'lower': float(low), 
                                              'upper': float(up)} 
                                             for pred, low, up in zip(preds, lower, upper)]
                        
                        logger.info(f"Made {len(predictions[market])} predictions for {market}")
                        logger.info(f"Sample prediction: {predictions[market][0]}")
            
            logger.info(f"Completed predictions for {len(predictions)} markets")
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            logger.error("Match data sample:")
            logger.error(match_data.head())
            raise
        
        return predictions
    
    def save_model(self, filename_prefix: str = 'multi_market'):
        """Save all models and scalers."""
        for market in self.models.keys():
            model_path = self.model_dir / f"{filename_prefix}_{market}.joblib"
            joblib.dump({
                'model': self.models[market],
                'scaler': self.scalers[market],
                'feature_importance': self.feature_importance.get(market)
            }, model_path)
        logger.info(f"Models saved to {self.model_dir}")
    
    def load_model(self, filename_prefix: str = 'multi_market'):
        """Load all models and scalers."""
        for market in self.models.keys():
            model_path = self.model_dir / f"{filename_prefix}_{market}.joblib"
            if model_path.exists():
                saved_data = joblib.load(model_path)
                self.models[market] = saved_data['model']
                self.scalers[market] = saved_data['scaler']
                self.feature_importance[market] = saved_data['feature_importance']
        logger.info("Models loaded successfully")
    
    def plot_feature_importance(self, output_dir: Path, market: str = None):
        """Plot feature importance for specified market or all markets."""
        markets_to_plot = [market] if market else self.feature_importance.keys()
        
        for mkt in markets_to_plot:
            if mkt in self.feature_importance:
                plt.figure(figsize=(10, 6))
                sns.barplot(
                    data=self.feature_importance[mkt].head(15),
                    x='importance',
                    y='feature'
                )
                plt.title(f'Top 15 Features for {mkt.upper()} Prediction')
                plt.xlabel('Importance')
                plt.ylabel('Feature')
                
                plot_path = output_dir / f'feature_importance_{mkt}.png'
                plt.savefig(plot_path, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Feature importance plot for {mkt} saved to {plot_path}") 