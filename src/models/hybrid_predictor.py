"""Hybrid prediction module combining deep learning with ensemble voting."""
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score

from src.config.config import ModelConfig

class HybridPredictor:
    """Hybrid predictor combining deep learning with ensemble voting."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.models: Dict[str, VotingClassifier] = {}
        self.deep_models: Dict[str, Model] = {}
        self.regression_models: Dict[str, VotingClassifier] = {}
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
    
    def _create_deep_model(self, input_dim: int, output_dim: int, is_regression: bool = False) -> Model:
        """Create a deep learning model architecture.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output classes/values
            is_regression: Whether this is a regression model
            
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            Input(shape=(input_dim,)),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dense(output_dim, activation='linear' if is_regression else 'softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse' if is_regression else 'categorical_crossentropy',
            metrics=['mae'] if is_regression else ['accuracy']
        )
        
        return model
    
    def _create_voting_ensemble(self, is_regression: bool = False) -> VotingClassifier:
        """Create a voting ensemble of traditional ML models.
        
        Args:
            is_regression: Whether this is a regression ensemble
            
        Returns:
            Voting classifier/regressor
        """
        if is_regression:
            estimators = [
                ('rf', RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42
                )),
                ('gb', GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                ))
            ]
        else:
            estimators = [
                ('rf', RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    class_weight='balanced',
                    random_state=42
                )),
                ('gb', GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                ))
            ]
        
        return VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=[1, 1]
        )
    
    def train(self, df: pd.DataFrame, save_path: Optional[Path] = None, test_mode: bool = False) -> Dict[str, Dict[str, float]]:
        """Train models for all enabled markets using time-series validation.
        
        Args:
            df: DataFrame with features and targets
            save_path: Optional path to save trained models
            test_mode: Whether running in test mode
            
        Returns:
            Dictionary with performance metrics for each market
        """
        metrics = {}
        logger = logging.getLogger(__name__)
        
        # Sort by date to ensure chronological order
        df = df.sort_values('Date')
        
        # Get unique seasons for time-series validation
        seasons = df['Season'].unique()
        logger.info(f"Training data spans {len(seasons)} seasons: {sorted(seasons)}")
        
        if not test_mode and len(seasons) < 3:
            raise ValueError("Need at least 3 seasons for proper validation")
        elif test_mode and len(seasons) < 2:
            raise ValueError("Need at least 2 seasons even in test mode")
        
        # Use the last season as final test set
        train_seasons = seasons[:-1]
        test_season = seasons[-1]
        logger.info(f"Using seasons {train_seasons} for training and season {test_season} for testing")
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        for market in self.config.markets:
            if not self.config.markets[market]:
                logger.info(f"Skipping {market} market (disabled in config)")
                continue
                
            logger.info(f"\n{'='*50}")
            logger.info(f"Training {market} model...")
            logger.info(f"{'='*50}")
            
            try:
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
                    logger.warning(f"Insufficient training data for {market} after filtering")
                    continue
                
                # Handle missing values and split data
                X_train = X[df['Season'].isin(train_seasons)]
                y_train = y[df['Season'].isin(train_seasons)]
                X_test = X[df['Season'] == test_season]
                y_test = y[df['Season'] == test_season]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Store the scaler
                self.scalers[market] = scaler
                
                if market in ['corners', 'cards']:
                    # Regression task
                    
                    # Train deep learning model
                    deep_model = self._create_deep_model(
                        input_dim=X_train_scaled.shape[1],
                        output_dim=1,
                        is_regression=True
                    )
                    
                    deep_model.fit(
                        X_train_scaled,
                        y_train,
                        validation_split=0.2,
                        epochs=100,
                        batch_size=32,
                        callbacks=[early_stopping],
                        verbose=0
                    )
                    
                    # Train ensemble model
                    ensemble_model = self._create_voting_ensemble(is_regression=True)
                    ensemble_model.fit(X_train_scaled, y_train)
                    
                    # Make predictions
                    deep_pred = deep_model.predict(X_test_scaled).flatten()
                    ensemble_pred = ensemble_model.predict(X_test_scaled)
                    
                    # Combine predictions (simple average)
                    y_pred = (deep_pred + ensemble_pred) / 2
                    
                    # Calculate regression metrics
                    metrics[market] = {
                        'mse': mean_squared_error(y_test, y_pred),
                        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                        'mae': mean_absolute_error(y_test, y_pred),
                        'r2': r2_score(y_test, y_pred)
                    }
                    
                    # Store models
                    self.deep_models[market] = deep_model
                    self.regression_models[market] = ensemble_model
                    
                else:
                    # Classification task
                    
                    # Convert target to one-hot for deep learning
                    n_classes = len(np.unique(y_train))
                    y_train_onehot = tf.keras.utils.to_categorical(y_train, n_classes)
                    y_test_onehot = tf.keras.utils.to_categorical(y_test, n_classes)
                    
                    # Train deep learning model
                    deep_model = self._create_deep_model(
                        input_dim=X_train_scaled.shape[1],
                        output_dim=n_classes
                    )
                    
                    deep_model.fit(
                        X_train_scaled,
                        y_train_onehot,
                        validation_split=0.2,
                        epochs=100,
                        batch_size=32,
                        callbacks=[early_stopping],
                        verbose=0
                    )
                    
                    # Train ensemble model
                    ensemble_model = self._create_voting_ensemble()
                    ensemble_model.fit(X_train_scaled, y_train)
                    
                    # Make predictions
                    deep_pred_proba = deep_model.predict(X_test_scaled)
                    ensemble_pred_proba = ensemble_model.predict_proba(X_test_scaled)
                    
                    # Combine probabilities (simple average)
                    combined_proba = (deep_pred_proba + ensemble_pred_proba) / 2
                    y_pred = np.argmax(combined_proba, axis=1)
                    
                    # Calculate classification metrics
                    metrics[market] = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred, average='weighted'),
                        'recall': recall_score(y_test, y_pred, average='weighted'),
                        'f1': f1_score(y_test, y_pred, average='weighted')
                    }
                    
                    # Store models
                    self.deep_models[market] = deep_model
                    self.models[market] = ensemble_model
                
                # Save models if path provided
                if save_path:
                    save_path.mkdir(parents=True, exist_ok=True)
                    
                    # Save deep learning model
                    self.deep_models[market].save(save_path / f"{market}_deep_model")
                    
                    # Save ensemble model
                    if market in ['corners', 'cards']:
                        joblib.dump(self.regression_models[market], save_path / f"{market}_ensemble_model.joblib")
                    else:
                        joblib.dump(self.models[market], save_path / f"{market}_ensemble_model.joblib")
                    
                    # Save scaler
                    joblib.dump(scaler, save_path / f"{market}_scaler.joblib")
                
                # Log metrics
                logger.info(f"\n{market} metrics:")
                for metric, value in metrics[market].items():
                    logger.info(f"- {metric}: {value:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {market} model: {str(e)}", exc_info=True)
                continue
        
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
                if market in self.regression_models and market in self.deep_models:
                    deep_pred = self.deep_models[market].predict(X_scaled).flatten()
                    ensemble_pred = self.regression_models[market].predict(X_scaled)
                    
                    # Combine predictions
                    pred_values = (deep_pred + ensemble_pred) / 2
                    
                    predictions[market] = pd.DataFrame({
                        'Date': df['Date'],
                        'HomeTeam': df['HomeTeam'],
                        'AwayTeam': df['AwayTeam'],
                        'prediction': pred_values.round(1),
                        'confidence': self._get_prediction_confidence(market, X_scaled)
                    })
            else:
                # Classification predictions
                if market in self.models and market in self.deep_models:
                    deep_proba = self.deep_models[market].predict(X_scaled)
                    ensemble_proba = self.models[market].predict_proba(X_scaled)
                    
                    # Combine probabilities
                    probs = (deep_proba + ensemble_proba) / 2
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
                
            deep_model_path = load_path / f"{market}_deep_model"
            ensemble_model_path = load_path / f"{market}_ensemble_model.joblib"
            scaler_path = load_path / f"{market}_scaler.joblib"
            
            if deep_model_path.exists() and ensemble_model_path.exists() and scaler_path.exists():
                # Load deep learning model
                self.deep_models[market] = tf.keras.models.load_model(deep_model_path)
                
                # Load ensemble model
                if market in ['corners', 'cards']:
                    self.regression_models[market] = joblib.load(ensemble_model_path)
                else:
                    self.models[market] = joblib.load(ensemble_model_path)
                
                # Load scaler
                self.scalers[market] = joblib.load(scaler_path)
                print(f"Loaded {market} models and scaler")
    
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
        else:
            raise ValueError(f"Unknown market: {market}")
    
    def _get_prediction_confidence(self, market: str, X_scaled: np.ndarray) -> np.ndarray:
        """Calculate prediction confidence using model agreement.
        
        For regression tasks, we look at the agreement between deep learning
        and ensemble predictions. Higher agreement means higher confidence.
        """
        if market not in self.regression_models or market not in self.deep_models:
            return np.zeros(len(X_scaled))
        
        # Get predictions from both models
        deep_pred = self.deep_models[market].predict(X_scaled).flatten()
        ensemble_pred = self.regression_models[market].predict(X_scaled)
        
        # Calculate absolute difference between predictions
        diff = np.abs(deep_pred - ensemble_pred)
        max_diff = np.max(diff)
        
        # Convert to confidence score (1 - normalized difference)
        confidence = 1 - (diff / max_diff)
        return confidence 