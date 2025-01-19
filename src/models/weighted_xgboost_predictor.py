import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import StandardScaler
from .base_predictor import BasePredictor
import logging
import os

logger = logging.getLogger(__name__)

class WeightedXGBoostPredictor(BasePredictor):
    def __init__(self, config):
        super().__init__(config)
        self.models = {}
        self.regression_models = {}
        self.feature_sets = self._define_feature_sets()
        
    def _define_feature_sets(self):
        base_features = self._define_base_features()
        
        return {
            'match_result': base_features + [
                'Home_League_Position', 'Away_League_Position',
                'Position_Diff', 'Points_Diff'
            ],
            'over_under': base_features + [
                'Goals_Diff_Home', 'Goals_Diff_Away',
                'Home_Clean_Sheets', 'Away_Clean_Sheets'
            ],
            'corners': base_features + self._define_corner_features(),
            'cards': base_features + self._define_card_features()
        }
    
    def _initialize_model(self, market):
        if market in ['corners', 'cards']:
            return XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_child_weight=1,
                gamma=0,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42
            )
        else:
            return XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_child_weight=1,
                gamma=0,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='multi:softprob' if market == 'match_result' else 'binary:logistic',
                num_class=3 if market == 'match_result' else None,
                eval_metric=['mlogloss', 'auc'],
                use_label_encoder=False,
                random_state=42
            )
    
    def train(self, df, validation_windows):
        logger.info("Training Weighted XGBoost predictor...")
        
        for market in self.config.markets:
            if not self.config.markets[market]:
                logger.info(f"Skipping {market} market (disabled in config)")
                continue
            
            try:
                logger.info(f"\nTraining {market} model...")
                X = df[self.feature_sets[market]].copy()
                y = self._get_target(df, market)
                
                # Calculate sample weights based on odds
                weights = self._calculate_weights(df)
                
                # Initialize and store scaler
                self.scalers[market] = StandardScaler()
                X_scaled = self.scalers[market].fit_transform(X)
                
                # Initialize appropriate model type
                if market in ['corners', 'cards']:
                    self.regression_models[market] = self._initialize_model(market)
                    model = self.regression_models[market]
                else:
                    self.models[market] = self._initialize_model(market)
                    model = self.models[market]
                
                # Train with early stopping
                eval_set = [(X_scaled, y)]
                model.fit(
                    X_scaled, y,
                    sample_weight=weights,
                    eval_set=eval_set,
                    early_stopping_rounds=10,
                    verbose=True
                )
                
                # Store feature importance
                importance = model.feature_importances_
                self.feature_importances[market] = pd.DataFrame({
                    'feature': self.feature_sets[market],
                    'importance': importance
                }).sort_values('importance', ascending=False)
                
                logger.info(f"\nTop 10 important features for {market}:")
                for _, row in self.feature_importances[market].head(10).iterrows():
                    logger.info(f"- {row['feature']}: {row['importance']:.4f}")
                    
            except Exception as e:
                logger.error(f"Error training {market} model: {str(e)}")
                continue
    
    def predict(self, df):
        logger.info("Making predictions with Weighted XGBoost models...")
        predictions = {}
        
        for market in self.config.markets:
            if not self.config.markets[market]:
                continue
                
            try:
                X = df[self.feature_sets[market]]
                X_scaled = self.scalers[market].transform(X)
                
                if market in ['corners', 'cards']:
                    if market in self.regression_models:
                        pred_values = self.regression_models[market].predict(X_scaled)
                        predictions[market] = self._format_regression_predictions(df, pred_values, market)
                else:
                    if market in self.models:
                        probs = self.models[market].predict_proba(X_scaled)
                        predictions[market] = self._format_match_predictions(df, probs)
                        
            except Exception as e:
                logger.error(f"Error predicting {market}: {str(e)}")
                continue
        
        return predictions
    
    def save(self, path):
        for market in self.models:
            self.models[market].save_model(f"{path}_{market}.json")
        
        for market in self.regression_models:
            self.regression_models[market].save_model(f"{path}_{market}.json")
            
        for market in self.scalers:
            np.save(f"{path}_{market}_scaler.npy", {
                'mean_': self.scalers[market].mean_,
                'scale_': self.scalers[market].scale_,
                'n_features_in_': self.scalers[market].n_features_in_,
                'feature_names': self.feature_sets[market]
            })
    
    def load(self, path):
        for market in self.config.markets:
            if not self.config.markets[market]:
                continue
                
            model_path = f"{path}_{market}.json"
            if os.path.exists(model_path):
                if market in ['corners', 'cards']:
                    self.regression_models[market] = self._initialize_model(market)
                    self.regression_models[market].load_model(model_path)
                else:
                    self.models[market] = self._initialize_model(market)
                    self.models[market].load_model(model_path)
                    
            scaler_path = f"{path}_{market}_scaler.npy"
            if os.path.exists(scaler_path):
                scaler_data = np.load(scaler_path, allow_pickle=True).item()
                self.scalers[market] = StandardScaler()
                self.scalers[market].mean_ = scaler_data['mean_']
                self.scalers[market].scale_ = scaler_data['scale_']
                self.scalers[market].n_features_in_ = scaler_data['n_features_in_']
                self.feature_sets[market] = scaler_data['feature_names'] 