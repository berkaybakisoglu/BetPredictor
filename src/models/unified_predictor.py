import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import logging
from pathlib import Path
import joblib
from typing import Dict, List, Optional, Tuple
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedPredictor:
    """Unified predictor for match results and various markets using XGBoost."""
    
    def __init__(self, model_dir: Path = Path('models')):
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models for different predictions
        self.models = {
            'match_result': xgb.Booster(),
            'ht_score': xgb.Booster(),
            'ft_score': xgb.Booster(),
            'corners': xgb.Booster(),
            'cards': xgb.Booster(),
            'over_under': xgb.Booster()
        }
        
        # Define required features including new chronological features
        self.required_features = [
            # Odds features
            'B365H', 'B365D', 'B365A', 'AvgH', 'AvgD', 'AvgA',
            'B365>2.5', 'B365<2.5', 'Avg>2.5', 'Avg<2.5',
            'AHh', 'B365AHH', 'B365AHA', 'AvgAHH', 'AvgAHA',
            
            # Form features
            'home_recent_wins', 'home_recent_draws', 'home_recent_losses',
            'home_recent_goals_scored', 'home_recent_goals_conceded',
            'home_recent_clean_sheets', 'away_recent_wins', 
            'away_recent_draws', 'away_recent_losses',
            'away_recent_goals_scored', 'away_recent_goals_conceded',
            'away_recent_clean_sheets',
            
            # Season stats
            'home_ppg', 'home_goals_per_game', 'home_goals_conceded_per_game',
            'home_clean_sheet_ratio', 'home_win_ratio',
            'away_ppg', 'away_goals_per_game', 'away_goals_conceded_per_game',
            'away_clean_sheet_ratio', 'away_win_ratio',
            
            # H2H features
            'h2h_home_wins', 'h2h_away_wins', 'h2h_draws',
            'h2h_avg_goals', 'h2h_home_team_avg_goals', 'h2h_away_team_avg_goals'
        ]
        
        # Define hyperparameter grids for each market
        self.param_grids = {
            'match_result': {
                'max_depth': [3, 5],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8],
                'colsample_bytree': [0.8],
                'min_child_weight': [1, 3],
                'gamma': [0, 0.1]
            },
            'over_under': {  # Grid for over/under 2.5 goals
                'max_depth': [3, 5],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8],
                'colsample_bytree': [0.8],
                'min_child_weight': [1, 3],
                'gamma': [0, 0.1]
            }
        }
        # Use same grid for other markets
        for market in ['ht_score', 'ft_score', 'corners', 'cards']:
            self.param_grids[market] = self.param_grids['match_result'].copy()
        
        # Base parameters for each market
        base_params = {
            'random_state': 42,
            'n_estimators': 1000,  # Will be controlled by early stopping
            'verbosity': 0
        }
        
        self.params = {
            'match_result': {
                **base_params,
                'objective': 'multi:softprob',
                'num_class': 3,
                'eval_metric': ['mlogloss', 'merror']
            },
            'ht_score': {
                **base_params,
                'objective': 'multi:softprob',
                'num_class': 9,
                'eval_metric': ['mlogloss', 'merror']
            },
            'ft_score': {
                **base_params,
                'objective': 'multi:softprob',
                'num_class': 10,
                'eval_metric': ['mlogloss', 'merror']
            },
            'corners': {
                **base_params,
                'objective': 'reg:squarederror',
                'eval_metric': ['rmse', 'mae']
            },
            'cards': {
                **base_params,
                'objective': 'reg:squarederror',
                'eval_metric': ['rmse', 'mae']
            },
            'over_under': {  # New parameters for over/under 2.5 goals
                **base_params,
                'objective': 'binary:logistic',
                'eval_metric': ['logloss', 'error']
            }
        }
        
        # Initialize scalers for each model
        self.scalers = {market: StandardScaler() for market in self.models.keys()}
        
        # Store best parameters and feature statistics
        self.best_params = {}
        self.feature_stats = {}
        self.feature_importance = {}
        
        # Define class mappings for classification models
        self.class_mappings = {
            'match_result': {0: 'Away', 1: 'Draw', 2: 'Home'},
            'ht_score': {
                '0-0': 0, '1-0': 1, '0-1': 2, '2-0': 3, '1-1': 4,
                '0-2': 5, '2-1': 6, '1-2': 7, 'other': 8
            },
            'ft_score': {
                '0-0': 0, '1-0': 1, '0-1': 2, '2-0': 3, '1-1': 4,
                '0-2': 5, '2-1': 6, '1-2': 7, '2-2': 8, 'other': 9
            },
            'over_under': {0: 'Under', 1: 'Over'}  # New mapping for over/under 2.5
        }
    
    def prepare_features(self, df: pd.DataFrame, market: str) -> pd.DataFrame:
        """Prepare features for specific market prediction."""
        df = df.copy()
        
        # Handle missing features
        for col in self.required_features:
            if col not in df.columns:
                if col in self.feature_stats:
                    df[col] = self.feature_stats[col]
                else:
                    # Use appropriate default values based on feature type
                    if 'ratio' in col or 'avg' in col:
                        df[col] = 0.0
                    elif 'recent' in col or 'wins' in col or 'draws' in col or 'losses' in col:
                        df[col] = 0
                    else:
                        df[col] = 0.0
        
        # Select features
        X = df[self.required_features].copy()
        
        # Additional feature engineering specific to each market
        if market == 'match_result':
            X['form_difference'] = (
                X['home_recent_wins'] - X['away_recent_wins']
            )
            X['goal_difference'] = (
                X['home_goals_per_game'] - X['away_goals_per_game']
            )
        elif market == 'over_under':
            X['total_avg_goals'] = (
                X['home_goals_per_game'] + X['away_goals_per_game']
            )
            X['defensive_strength'] = (
                X['home_clean_sheet_ratio'] + X['away_clean_sheet_ratio']
            ) / 2
        
        logger.info(f"Prepared {len(self.required_features)} features for {market}")
        return X
    
    def find_best_params(self, X: pd.DataFrame, y: pd.Series, market: str) -> Dict:
        """Find best hyperparameters using cross-validation."""
        logger.info(f"Finding best parameters for {market}...")
        
        best_score = float('-inf')
        best_params = {}
        
        # Convert data to DMatrix
        dtrain = xgb.DMatrix(X, label=y)
        
        # Determine evaluation metric based on market
        if market == 'over_under':
            eval_metric = ['logloss']
        elif market in ['match_result', 'ht_score', 'ft_score']:
            eval_metric = ['mlogloss']
        else:
            eval_metric = ['rmse']
        
        # Perform grid search manually
        for max_depth in self.param_grids[market]['max_depth']:
            for learning_rate in self.param_grids[market]['learning_rate']:
                for subsample in self.param_grids[market]['subsample']:
                    for colsample_bytree in self.param_grids[market]['colsample_bytree']:
                        for min_child_weight in self.param_grids[market]['min_child_weight']:
                            for gamma in self.param_grids[market]['gamma']:
                                # Update parameters
                                params = {
                                    **self.params[market],
                                    'max_depth': max_depth,
                                    'learning_rate': learning_rate,
                                    'subsample': subsample,
                                    'colsample_bytree': colsample_bytree,
                                    'min_child_weight': min_child_weight,
                                    'gamma': gamma
                                }
                                
                                # Perform cross-validation
                                cv_results = xgb.cv(
                                    params,
                                    dtrain,
                                    num_boost_round=100,
                                    nfold=5,
                                    early_stopping_rounds=10,
                                    metrics=eval_metric,
                                    seed=42
                                )
                                
                                # Get best score based on market type
                                if market == 'over_under':
                                    score = -cv_results['test-logloss-mean'].min()
                                elif market in ['match_result', 'ht_score', 'ft_score']:
                                    score = -cv_results['test-mlogloss-mean'].min()
                                else:
                                    score = -cv_results['test-rmse-mean'].min()
                                
                                if score > best_score:
                                    best_score = score
                                    best_params = {
                                        'max_depth': max_depth,
                                        'learning_rate': learning_rate,
                                        'subsample': subsample,
                                        'colsample_bytree': colsample_bytree,
                                        'min_child_weight': min_child_weight,
                                        'gamma': gamma
                                    }
                                    logger.info(f"New best parameters found for {market} with score {best_score}:")
                                    logger.info(best_params)
        
        return best_params
    
    def train(self, train_data: pd.DataFrame, labels: Dict[str, pd.Series]) -> Dict[str, Dict]:
        """Train all models with their respective data and return evaluation metrics."""
        # Store feature statistics from training data
        self.feature_stats = {
            col: train_data[col].median()
            for col in self.required_features
            if col in train_data.columns
        }
        
        evaluation_metrics = {}
        
        for market in self.models.keys():
            if market not in labels:
                logger.warning(f"No labels provided for {market}, skipping training")
                continue
                
            logger.info(f"\nTraining {market} model...")
            X = self.prepare_features(train_data, market)
            y = labels[market]
            
            # Scale features
            X_scaled = self.scalers[market].fit_transform(X)
            
            # Find best parameters
            best_params = self.find_best_params(X_scaled, y, market)
            self.best_params[market] = best_params
            
            # Update parameters with best found
            train_params = {**self.params[market], **best_params}
            
            # Split data for early stopping
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Convert to DMatrix
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            
            # Train with early stopping
            self.models[market] = xgb.train(
                train_params,
                dtrain,
                num_boost_round=1000,
                evals=[(dtrain, 'train'), (dval, 'val')],
                early_stopping_rounds=20,
                verbose_eval=100
            )
            
            # Store feature importance
            importance = self.models[market].get_score(importance_type='gain')
            importance_df = pd.DataFrame({
                'feature': list(importance.keys()),
                'importance': list(importance.values())
            }).sort_values('importance', ascending=False)
            self.feature_importance[market] = importance_df
            
            # Log top 5 important features
            logger.info(f"\nTop 5 important features for {market}:")
            for _, row in importance_df.head().iterrows():
                logger.info(f"{row['feature']}: {row['importance']:.3f}")
            
            logger.info(f"Completed training {market} model")
        
        return evaluation_metrics
    
    def predict(self, match_data: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Make predictions for all markets."""
        predictions = {}
        
        try:
            for market in self.models.keys():
                logger.info(f"\nMaking predictions for {market}...")
                X = self.prepare_features(match_data, market)
                X_scaled = self.scalers[market].transform(X)
                
                # Convert to DMatrix
                dtest = xgb.DMatrix(X_scaled)
                
                # Get predictions
                if market in ['match_result', 'ht_score', 'ft_score']:
                    probas = self.models[market].predict(dtest)
                    
                    if market == 'match_result':
                        pred_list = []
                        for row_probas in probas:
                            pred_dict = {
                                'Home': float(row_probas[2]),
                                'Draw': float(row_probas[1]),
                                'Away': float(row_probas[0])
                            }
                            pred_list.append(pred_dict)
                        predictions[market] = pred_list
                    else:
                        # For score predictions
                        pred_list = []
                        for row_probas in probas:
                            pred_dict = {}
                            for score, idx in self.class_mappings[market].items():
                                if idx < len(row_probas):
                                    pred_dict[score] = float(row_probas[idx])
                            pred_list.append(pred_dict)
                        predictions[market] = pred_list
                elif market == 'over_under':
                    probas = self.models[market].predict(dtest)
                    pred_list = []
                    for prob in probas:
                        pred_dict = {
                            'Over': float(prob),
                            'Under': float(1 - prob)
                        }
                        pred_list.append(pred_dict)
                    predictions[market] = pred_list
                else:
                    # For regression models (corners, cards)
                    preds = self.models[market].predict(dtest)
                    
                    # Calculate prediction intervals using standard deviation
                    n_rounds = 10
                    bootstrap_preds = []
                    for i in range(n_rounds):
                        # Use different subsets of trees for bootstrapping
                        ntree_limit = int(self.models[market].num_boosted_rounds() * 0.8)
                        start_index = i * (ntree_limit // n_rounds)
                        end_index = (i + 1) * (ntree_limit // n_rounds)
                        bootstrap_preds.append(
                            self.models[market].predict(dtest, iteration_range=(start_index, end_index))
                        )
                    bootstrap_preds = np.array(bootstrap_preds)
                    pred_std = np.std(bootstrap_preds, axis=0)
                    
                    predictions[market] = [{
                        'prediction': float(pred),
                        'range': f"{float(pred - 1.96 * std):.1f} - {float(pred + 1.96 * std):.1f}"
                    } for pred, std in zip(preds, pred_std)]
                
                logger.info(f"Made {len(predictions[market])} predictions for {market}")
                logger.info(f"Sample prediction: {predictions[market][0]}")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return {}
    
    def save(self) -> None:
        """Save all models, scalers, and feature statistics."""
        try:
            for market in self.models.keys():
                # Save model
                model_path = self.model_dir / f"{market}_model.json"
                self.models[market].save_model(str(model_path))
                
                # Save scaler
                scaler_path = self.model_dir / f"{market}_scaler.joblib"
                joblib.dump(self.scalers[market], scaler_path)
                
                # Save feature importance if available
                if market in self.feature_importance:
                    importance_path = self.model_dir / f"{market}_importance.json"
                    self.feature_importance[market].to_json(importance_path)
            
            # Save feature statistics and best parameters
            stats_path = self.model_dir / "feature_stats.json"
            with open(stats_path, 'w') as f:
                json.dump(self.feature_stats, f)
            
            params_path = self.model_dir / "best_params.json"
            with open(params_path, 'w') as f:
                json.dump(self.best_params, f)
            
            logger.info(f"Models, scalers, and parameters saved to {self.model_dir}")
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
    
    def load(self) -> None:
        """Load all models, scalers, and feature statistics."""
        try:
            for market in self.models.keys():
                # Load model
                model_path = self.model_dir / f"{market}_model.json"
                if model_path.exists():
                    self.models[market].load_model(str(model_path))
                
                # Load scaler
                scaler_path = self.model_dir / f"{market}_scaler.joblib"
                if scaler_path.exists():
                    self.scalers[market] = joblib.load(scaler_path)
                
                # Load feature importance
                importance_path = self.model_dir / f"{market}_importance.json"
                if importance_path.exists():
                    self.feature_importance[market] = pd.read_json(importance_path)
            
            # Load feature statistics and best parameters
            stats_path = self.model_dir / "feature_stats.json"
            if stats_path.exists():
                with open(stats_path, 'r') as f:
                    self.feature_stats = json.load(f)
            
            params_path = self.model_dir / "best_params.json"
            if params_path.exists():
                with open(params_path, 'r') as f:
                    self.best_params = json.load(f)
            
            logger.info(f"Models, scalers, and parameters loaded from {self.model_dir}")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
    
    def format_predictions(self, predictions: Dict[str, List], data: pd.DataFrame) -> List[Dict]:
        """Format predictions into a standardized format with odds."""
        formatted = []
        
        for i in range(len(data)):
            match_pred = {
                'date': data.iloc[i]['Date'],
                'match': f"{data.iloc[i]['HomeTeam']} vs {data.iloc[i]['AwayTeam']}",
                'league': data.iloc[i].get('League', ''),
                'predictions': {}
            }
            
            # Match result predictions with odds
            if 'match_result' in predictions:
                home_prob = predictions['match_result'][i]['Home']
                draw_prob = predictions['match_result'][i]['Draw']
                away_prob = predictions['match_result'][i]['Away']
                
                # Check if B365 odds are available
                if 'B365H' not in data.columns or 'B365D' not in data.columns or 'B365A' not in data.columns:
                    logger.warning(f"B365 match result odds missing for {match_pred['match']}")
                    continue
                
                home_odds = float(data.iloc[i]['B365H'])
                draw_odds = float(data.iloc[i]['B365D'])
                away_odds = float(data.iloc[i]['B365A'])
                
                # Skip if any odds are invalid
                if home_odds <= 1.0 or draw_odds <= 1.0 or away_odds <= 1.0:
                    logger.warning(f"Invalid B365 match result odds for {match_pred['match']}")
                    continue
                
                match_pred['predictions']['match_result'] = {
                    'Home': f"{home_prob:.1%}",
                    'Draw': f"{draw_prob:.1%}",
                    'Away': f"{away_prob:.1%}",
                    'odds': {
                        'Home': home_odds,
                        'Draw': draw_odds,
                        'Away': away_odds
                    }
                }
            
            # Over/Under 2.5 goals predictions with odds
            if 'over_under' in predictions:
                over_prob = predictions['over_under'][i]['Over']
                under_prob = predictions['over_under'][i]['Under']
                
                # Check if B365 over/under odds are available
                if 'B365>2.5' not in data.columns or 'B365<2.5' not in data.columns:
                    logger.warning(f"B365 over/under odds missing for {match_pred['match']}")
                    continue
                
                over_odds = float(data.iloc[i]['B365>2.5'])
                under_odds = float(data.iloc[i]['B365<2.5'])
                
                # Skip if any odds are invalid
                if over_odds <= 1.0 or under_odds <= 1.0:
                    logger.warning(f"Invalid B365 over/under odds for {match_pred['match']}")
                    continue
                
                match_pred['predictions']['over_under'] = {
                    'Over': f"{over_prob:.1%}",
                    'Under': f"{under_prob:.1%}",
                    'odds': {
                        'Over': over_odds,
                        'Under': under_odds
                    }
                }
            
            # Half-time score predictions
            if 'ht_score' in predictions:
                ht_pred = predictions['ht_score'][i]
                match_pred['predictions']['ht_score'] = {
                    **{k: f"{v:.1%}" for k, v in ht_pred.items()},
                    'odds': {
                        score: max(1/prob * 0.9, 1.1)  # Estimated odds with 10% margin, minimum 1.1
                        for score, prob in ht_pred.items()
                        if prob > 0  # Only include non-zero probabilities
                    }
                }
            
            # Full-time score predictions
            if 'ft_score' in predictions:
                ft_pred = predictions['ft_score'][i]
                match_pred['predictions']['ft_score'] = {
                    **{k: f"{v:.1%}" for k, v in ft_pred.items()},
                    'odds': {
                        score: max(1/prob * 0.9, 1.1)  # Estimated odds with 10% margin, minimum 1.1
                        for score, prob in ft_pred.items()
                        if prob > 0  # Only include non-zero probabilities
                    }
                }
            
            # Corners prediction
            if 'corners' in predictions:
                corners_pred = predictions['corners'][i]
                match_pred['predictions']['corners'] = {
                    'prediction': corners_pred['prediction'],
                    'range': corners_pred['range'],
                    'confidence': 0.7,  # Example confidence level
                    'odds': 1.9  # Standard over/under odds
                }
            
            # Cards prediction
            if 'cards' in predictions:
                cards_pred = predictions['cards'][i]
                match_pred['predictions']['cards'] = {
                    'prediction': cards_pred['prediction'],
                    'range': cards_pred['range'],
                    'confidence': 0.7,  # Example confidence level
                    'odds': 1.9  # Standard over/under odds
                }
            
            # Only add the match prediction if it has at least one market with valid odds
            if match_pred['predictions']:
                formatted.append(match_pred)
        
        return formatted