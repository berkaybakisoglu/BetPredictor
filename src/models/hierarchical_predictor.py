import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.preprocessing import StandardScaler
import logging
import os
import joblib
from .base_predictor import BasePredictor

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
        
    def train(self, train_data):
        logger.info("Training hierarchical predictor...")
        
        cards_features = self._get_cards_features(train_data)
        cards_target = self._get_target(train_data, 'cards')
        
        X_cards_scaled = self.scalers['cards'].fit_transform(cards_features)
        self.cards_predictor.fit(X_cards_scaled, cards_target)
        logger.info("Cards predictor trained")
        
        corners_features = self._get_corners_features(train_data)
        corners_target = self._get_target(train_data, 'corners')
        
        X_corners_scaled = self.scalers['corners'].fit_transform(corners_features)
        self.corners_predictor.fit(X_corners_scaled, corners_target)
        logger.info("Corners predictor trained")
        
        enhanced_features = self._create_enhanced_features(train_data)
        result_target = self._get_target(train_data, 'match_result')
        
        X_result_scaled = self.scalers['result'].fit_transform(enhanced_features)
        self.result_predictor.fit(X_result_scaled, result_target)
        logger.info("Result predictor trained")
        
    def predict(self, match_data):
        cards_features = self._get_cards_features(match_data)
        X_cards_scaled = self.scalers['cards'].transform(cards_features)
        cards_pred = self.cards_predictor.predict(X_cards_scaled)
        
        corners_features = self._get_corners_features(match_data)
        X_corners_scaled = self.scalers['corners'].transform(corners_features)
        corners_pred = self.corners_predictor.predict(X_corners_scaled)
        
        enhanced_features = self._create_enhanced_features(
            match_data, 
            predicted_cards=cards_pred,
            predicted_corners=corners_pred
        )
        
        X_result_scaled = self.scalers['result'].transform(enhanced_features)
        result_probas = self.result_predictor.predict_proba(X_result_scaled)
        
        predictions = self._format_match_predictions(match_data, result_probas)
        predictions['Predicted_Cards'] = cards_pred.round(1)
        predictions['Predicted_Corners'] = corners_pred.round(1)
        
        return predictions
    
    def _get_cards_features(self, data):
        return data[self._define_card_features()]
    
    def _get_corners_features(self, data):
        return data[self._define_corner_features()]
    
    def _create_enhanced_features(self, data, predicted_cards=None, predicted_corners=None):
        base_features = data[self._define_base_features()]
        
        if predicted_cards is not None and predicted_corners is not None:
            enhanced_features = pd.DataFrame(base_features)
            enhanced_features['predicted_cards'] = predicted_cards
            enhanced_features['predicted_corners'] = predicted_corners
            return enhanced_features
            
        return base_features
    
    def save(self, model_dir):
        super().save(model_dir)
        
        joblib.dump(self.cards_predictor, os.path.join(model_dir, 'cards_predictor.joblib'))
        joblib.dump(self.corners_predictor, os.path.join(model_dir, 'corners_predictor.joblib'))
        joblib.dump(self.result_predictor, os.path.join(model_dir, 'result_predictor.joblib'))
    
    def load(self, model_dir):
        super().load(model_dir)
        
        self.cards_predictor = joblib.load(os.path.join(model_dir, 'cards_predictor.joblib'))
        self.corners_predictor = joblib.load(os.path.join(model_dir, 'corners_predictor.joblib'))
        self.result_predictor = joblib.load(os.path.join(model_dir, 'result_predictor.joblib')) 