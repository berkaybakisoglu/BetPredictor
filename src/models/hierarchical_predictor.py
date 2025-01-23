import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.preprocessing import StandardScaler
import logging
import os
import joblib
from .base_predictor import BasePredictor
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, log_loss, accuracy_score
from sklearn.metrics import make_scorer

logger = logging.getLogger(__name__)

class HierarchicalPredictor(BasePredictor):
    def __init__(self, config):
        super().__init__(config)
        self.cards_predictor = LGBMRegressor(
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
        
        self.corners_predictor = LGBMRegressor(
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
        
        self.result_predictor = LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.01,
            num_leaves=31,
            max_depth=8,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            random_state=42
        )
        
        self.scalers = {
            'cards': StandardScaler(),
            'corners': StandardScaler(),
            'result': StandardScaler()
        }
        
        self.feature_importances = {
            'cards': None,
            'corners': None,
            'match_result': None
        }
        
        self.cv_results = {
            'cards': None,
            'corners': None,
            'result': None
        }
    
    def _perform_cross_validation(self, X, y, model, market):
        """Perform time-series based cross validation."""
        logger = logging.getLogger(__name__)
        scores = []
        
        # Use appropriate scoring metric based on market
        if market in ['match_result', 'over_under']:
            scorer = make_scorer(log_loss, needs_proba=True)
            accuracy_scores = []
        else:
            scorer = make_scorer(mean_squared_error, squared=False)  # RMSE
        
        for train_idx, val_idx in self.cv_splits:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            
            if market in ['match_result', 'over_under']:
                # Calculate log loss
                y_pred_proba = model.predict_proba(X_val)
                score = log_loss(y_val, y_pred_proba)
                scores.append(score)
                
                # Calculate accuracy
                y_pred = model.predict(X_val)
                accuracy = accuracy_score(y_val, y_pred)
                accuracy_scores.append(accuracy)
            else:
                y_pred = model.predict(X_val)
                score = mean_squared_error(y_val, y_pred, squared=False)
                scores.append(score)
        
        cv_score = np.mean(scores)
        cv_std = np.std(scores)
        
        if market in ['match_result', 'over_under']:
            mean_accuracy = np.mean(accuracy_scores)
            std_accuracy = np.std(accuracy_scores)
            logger.info(f"{market} CV Metrics:")
            logger.info(f"- Log Loss: {cv_score:.4f} (+/- {cv_std:.4f})")
            logger.info(f"- Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")
            return {
                'log_loss_mean': cv_score,
                'log_loss_std': cv_std,
                'accuracy_mean': mean_accuracy,
                'accuracy_std': std_accuracy
            }
        else:
            logger.info(f"{market} CV Score (RMSE): {cv_score:.4f} (+/- {cv_std:.4f})")
            return cv_score, cv_std
    
    def _calculate_feature_importance(self, model, feature_names, market):
        """Calculate and store feature importance."""
        importances = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        })
        importances = importances.sort_values('importance', ascending=False)
        
        # Ensure consistent market naming
        if market == 'result':
            market = 'match_result'
            
        self.feature_importances[market] = importances
        
        # Log top 10 important features
        logger.info(f"\nTop 10 important features for {market}:")
        for idx, row in importances.head(10).iterrows():
            logger.info(f"{row['feature']}: {row['importance']:.4f}")
        
        return importances
    
    def train(self, train_data):
        logger.info("Training hierarchical predictor...")
        
        # Cards prediction
        cards_features = self._get_cards_features(train_data)
        cards_target = self._get_target(train_data, 'cards')
        feature_names = cards_features.columns
        
        X_cards_scaled = self.scalers['cards'].fit_transform(cards_features)
        
        # Cross-validation for cards
        cv_score, cv_std = self._perform_cross_validation(
            X_cards_scaled, cards_target, 
            self.cards_predictor, 'cards'
        )
        self.cv_results['cards'] = {'mean': cv_score, 'std': cv_std}
        logger.info(f"Cards CV Score: {cv_score:.4f} (+/- {cv_std:.4f})")
        
        self.cards_predictor.fit(X_cards_scaled, cards_target)
        self._calculate_feature_importance(
            self.cards_predictor, feature_names, 'cards'
        )
        logger.info("Cards predictor trained")
        
        # Corners prediction
        corners_features = self._get_corners_features(train_data)
        corners_target = self._get_target(train_data, 'corners')
        feature_names = corners_features.columns
        
        X_corners_scaled = self.scalers['corners'].fit_transform(corners_features)
        
        # Cross-validation for corners
        cv_score, cv_std = self._perform_cross_validation(
            X_corners_scaled, corners_target,
            self.corners_predictor, 'corners'
        )
        self.cv_results['corners'] = {'mean': cv_score, 'std': cv_std}
        logger.info(f"Corners CV Score: {cv_score:.4f} (+/- {cv_std:.4f})")
        
        self.corners_predictor.fit(X_corners_scaled, corners_target)
        self._calculate_feature_importance(
            self.corners_predictor, feature_names, 'corners'
        )
        logger.info("Corners predictor trained")
        
        # Get initial predictions for training data
        train_cards_pred = self.cards_predictor.predict(X_cards_scaled)
        train_corners_pred = self.corners_predictor.predict(X_corners_scaled)
        
        # Match result prediction with enhanced features
        base_features = self._create_enhanced_features(train_data)
        base_features['predicted_cards'] = train_cards_pred
        base_features['predicted_corners'] = train_corners_pred
        result_target = self._get_target(train_data, 'match_result')
        feature_names = base_features.columns
        
        X_result_scaled = self.scalers['result'].fit_transform(base_features)
        
        # Cross-validation for match results
        cv_metrics = self._perform_cross_validation(
            X_result_scaled, result_target,
            self.result_predictor, 'match_result'
        )
        self.cv_results['result'] = cv_metrics
        
        self.result_predictor.fit(X_result_scaled, result_target)
        self._calculate_feature_importance(
            self.result_predictor, feature_names, 'match_result'
        )
        logger.info("Result predictor trained")
        
        # Return metrics in the expected format
        return {
            'metrics': {
                'cards': {
                    'rmse_cv_mean': self.cv_results['cards']['mean'],
                    'rmse_cv_std': self.cv_results['cards']['std']
                },
                'corners': {
                    'rmse_cv_mean': self.cv_results['corners']['mean'],
                    'rmse_cv_std': self.cv_results['corners']['std']
                },
                'match_result': {
                    'log_loss_mean': cv_metrics['log_loss_mean'],
                    'log_loss_std': cv_metrics['log_loss_std'],
                    'accuracy_mean': cv_metrics['accuracy_mean'],
                    'accuracy_std': cv_metrics['accuracy_std']
                }
            }
        }
    
    def plot_feature_importance(self, save_dir=None):
        """Plot feature importance for each predictor using only matplotlib."""
        for market, importance_df in self.feature_importances.items():
            if importance_df is None:
                continue
                
            plt.figure(figsize=(12, 8))
            
            # Get top 20 features and their importances
            features = importance_df.head(20)['feature'].values
            importances = importance_df.head(20)['importance'].values
            
            # Create horizontal bar plot
            bars = plt.barh(range(len(features)), importances, 
                          color='lightblue', edgecolor='navy', alpha=0.6)
            
            # Customize the plot
            plt.title(f'Top 20 Important Features for {market.title()} Prediction', 
                     fontsize=14, pad=20)
            plt.xlabel('Feature Importance', fontsize=12)
            
            # Set y-ticks to feature names
            plt.yticks(range(len(features)), features, fontsize=10)
            
            # Add value labels on the bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', 
                        ha='left', va='center', fontsize=9)
            
            # Add grid for better readability
            plt.grid(axis='x', linestyle='--', alpha=0.3)
            
            # Adjust layout
            plt.tight_layout()
            
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(
                    os.path.join(save_dir, f'{market}_feature_importance.png'),
                    dpi=300,
                    bbox_inches='tight'
                )
            plt.close()
    
    def predict(self, match_data):
        # Get cards predictions
        cards_features = self._get_cards_features(match_data)
        X_cards_scaled = self.scalers['cards'].transform(cards_features)
        cards_pred = self.cards_predictor.predict(X_cards_scaled)
        
        # Get corners predictions
        corners_features = self._get_corners_features(match_data)
        X_corners_scaled = self.scalers['corners'].transform(corners_features)
        corners_pred = self.corners_predictor.predict(X_corners_scaled)
        
        # Create enhanced features with predictions
        base_features = self._create_enhanced_features(match_data)
        base_features['predicted_cards'] = cards_pred
        base_features['predicted_corners'] = corners_pred
        
        # Get match result predictions
        X_result_scaled = self.scalers['result'].transform(base_features)
        result_probas = self.result_predictor.predict_proba(X_result_scaled)
        
        # Format predictions
        predictions = self._format_match_predictions(match_data, result_probas)
        predictions['Predicted_Cards'] = cards_pred.round(1)
        predictions['Predicted_Corners'] = corners_pred.round(1)
        
        # Return predictions in a dictionary with market keys
        return {
            'match_result': predictions,
            'cards': self._format_regression_predictions(match_data, cards_pred, 'cards'),
            'corners': self._format_regression_predictions(match_data, corners_pred, 'corners')
        }
    
    def _get_cards_features(self, data):
        return data[self._define_card_features()]
    
    def _get_corners_features(self, data):
        return data[self._define_corner_features()]
    
    def _create_enhanced_features(self, data):
        """Create base features for match result prediction."""
        return data[self._define_base_features()]
    
    def save(self, model_dir):
        super().save(model_dir)
        
        # Save additional metadata
        metadata = {
            'feature_importances': {
                market: df.to_dict('records') if df is not None else None
                for market, df in self.feature_importances.items()
            },
            'cv_results': self.cv_results
        }
        
        with open(os.path.join(model_dir, 'model_metadata.json'), 'w') as f:
            json.dump(metadata, f)
        
        joblib.dump(self.cards_predictor, os.path.join(model_dir, 'cards_predictor.joblib'))
        joblib.dump(self.corners_predictor, os.path.join(model_dir, 'corners_predictor.joblib'))
        joblib.dump(self.result_predictor, os.path.join(model_dir, 'result_predictor.joblib'))
    
    def load(self, model_dir):
        super().load(model_dir)
        
        # Load additional metadata
        try:
            with open(os.path.join(model_dir, 'model_metadata.json'), 'r') as f:
                metadata = json.load(f)
                # Convert feature importances back to DataFrames
                self.feature_importances = {
                    market: pd.DataFrame(data) if data is not None else None
                    for market, data in metadata['feature_importances'].items()
                }
                self.cv_results = metadata['cv_results']
        except FileNotFoundError:
            logger.warning("No metadata file found. Feature importances and CV results will be empty.")
        
        self.cards_predictor = joblib.load(os.path.join(model_dir, 'cards_predictor.joblib'))
        self.corners_predictor = joblib.load(os.path.join(model_dir, 'corners_predictor.joblib'))
        self.result_predictor = joblib.load(os.path.join(model_dir, 'result_predictor.joblib')) 