"""Hierarchical predictor that uses multiple predictions to enhance match result prediction."""
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.preprocessing import StandardScaler
import logging
import os
import joblib

logger = logging.getLogger(__name__)

class HierarchicalPredictor:
    """A predictor that uses multiple predictions to enhance match result prediction."""
    
    def __init__(self, config):
        # Base predictors for different aspects
        self.cards_predictor = LGBMRegressor(
            n_estimators=100,
            learning_rate=0.01
        )
        
        self.corners_predictor = LGBMRegressor(
            n_estimators=100,
            learning_rate=0.01
        )
        
        # Final match result predictor
        self.result_predictor = LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.01,
            num_leaves=31,
            class_weight='balanced'
        )
        
        # Scalers for each model
        self.scalers = {
            'cards': StandardScaler(),
            'corners': StandardScaler(),
            'result': StandardScaler()
        }
        
    def train(self, train_data):
        """Train all models in sequence."""
        logger.info("Training hierarchical predictor...")
        
        # 1. Train cards predictor
        cards_features = self._get_cards_features(train_data)
        cards_target = train_data['HY'] + train_data['AY']  # Total cards
        
        X_cards_scaled = self.scalers['cards'].fit_transform(cards_features)
        self.cards_predictor.fit(X_cards_scaled, cards_target)
        logger.info("Cards predictor trained")
        
        # 2. Train corners predictor
        corners_features = self._get_corners_features(train_data)
        corners_target = train_data['HC'] + train_data['AC']  # Total corners
        
        X_corners_scaled = self.scalers['corners'].fit_transform(corners_features)
        self.corners_predictor.fit(X_corners_scaled, corners_target)
        logger.info("Corners predictor trained")
        
        # 3. Get enhanced features with predictions
        enhanced_features = self._create_enhanced_features(train_data)
        result_target = train_data['FTR']  # Full Time Result
        
        X_result_scaled = self.scalers['result'].fit_transform(enhanced_features)
        self.result_predictor.fit(X_result_scaled, result_target)
        logger.info("Result predictor trained")
        
    def predict(self, match_data):
        """Make hierarchical predictions."""
        # 1. Predict cards
        cards_features = self._get_cards_features(match_data)
        X_cards_scaled = self.scalers['cards'].transform(cards_features)
        cards_pred = self.cards_predictor.predict(X_cards_scaled)
        
        # 2. Predict corners
        corners_features = self._get_corners_features(match_data)
        X_corners_scaled = self.scalers['corners'].transform(corners_features)
        corners_pred = self.corners_predictor.predict(X_corners_scaled)
        
        # 3. Create enhanced features
        enhanced_features = self._create_enhanced_features(
            match_data, 
            predicted_cards=cards_pred,
            predicted_corners=corners_pred
        )
        
        # 4. Final match result prediction
        X_result_scaled = self.scalers['result'].transform(enhanced_features)
        result_probas = self.result_predictor.predict_proba(X_result_scaled)
        
        # Format predictions
        predictions = pd.DataFrame({
            'Home_Prob': result_probas[:, 2],  # Home win probability
            'Draw_Prob': result_probas[:, 1],  # Draw probability
            'Away_Prob': result_probas[:, 0],  # Away win probability
            'Predicted_Cards': cards_pred,
            'Predicted_Corners': corners_pred
        })
        
        # Add confidence scores
        predictions['Confidence'] = predictions[['Home_Prob', 'Draw_Prob', 'Away_Prob']].max(axis=1)
        predictions['Predicted'] = self.result_predictor.predict(X_result_scaled)
        
        return predictions
    
    def _get_cards_features(self, data):
        """Extract features relevant for cards prediction."""
        return data[[
            'home_team_form', 'away_team_form',
            'home_recent_cards', 'away_recent_cards',
            'referee_avg_cards', 'is_derby',
            'home_aggression_index', 'away_aggression_index'
        ]]
    
    def _get_corners_features(self, data):
        """Extract features relevant for corners prediction."""
        return data[[
            'home_recent_corners', 'away_recent_corners',
            'home_attack_strength', 'away_attack_strength',
            'home_possession_avg', 'away_possession_avg'
        ]]
    
    def _create_enhanced_features(self, data, predicted_cards=None, predicted_corners=None):
        """Create enhanced feature set including predictions if available."""
        base_features = data[[
            # Form features
            'home_recent_wins', 'home_recent_draws', 'home_recent_losses',
            'home_recent_goals_scored', 'home_recent_goals_conceded',
            'away_recent_wins', 'away_recent_draws', 'away_recent_losses',
            'away_recent_goals_scored', 'away_recent_goals_conceded',
            
            # Team strength features
            'home_ppg', 'home_goals_per_game', 'home_conceded_per_game',
            'away_ppg', 'away_goals_per_game', 'away_conceded_per_game',
            
            # Market features
            'B365H', 'B365D', 'B365A'
        ]]
        
        if predicted_cards is not None and predicted_corners is not None:
            # Add predictions as new features
            enhanced_features = pd.DataFrame(base_features)
            enhanced_features['predicted_cards'] = predicted_cards
            enhanced_features['predicted_corners'] = predicted_corners
            return enhanced_features
            
        return base_features
    
    def save(self, model_dir):
        """Save the model and scalers."""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Save predictors
        joblib.dump(self.cards_predictor, os.path.join(model_dir, 'cards_predictor.joblib'))
        joblib.dump(self.corners_predictor, os.path.join(model_dir, 'corners_predictor.joblib'))
        joblib.dump(self.result_predictor, os.path.join(model_dir, 'result_predictor.joblib'))
        
        # Save scalers
        for name, scaler in self.scalers.items():
            scaler_path = os.path.join(model_dir, '{0}_scaler.joblib'.format(name))
            joblib.dump(scaler, scaler_path)
            
        logger.info("Model saved to {0}".format(model_dir))
    
    def load(self, model_dir):
        """Load the model and scalers."""
        # Load predictors
        self.cards_predictor = joblib.load(os.path.join(model_dir, 'cards_predictor.joblib'))
        self.corners_predictor = joblib.load(os.path.join(model_dir, 'corners_predictor.joblib'))
        self.result_predictor = joblib.load(os.path.join(model_dir, 'result_predictor.joblib'))
        
        # Load scalers
        for name in self.scalers.keys():
            scaler_path = os.path.join(model_dir, '{0}_scaler.joblib'.format(name))
            self.scalers[name] = joblib.load(scaler_path)
            
        logger.info("Model loaded from {0}".format(model_dir)) 