import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import logging
from pathlib import Path
import joblib
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MatchPredictor:
    def __init__(self, model_dir: Path = Path('models')):
        """Initialize the match predictor."""
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Reduced complexity in Random Forest
        self.model = RandomForestClassifier(
            n_estimators=50,      # Reduced from 100
            max_depth=5,          # Reduced from 10
            min_samples_split=5,  # Require more samples to split
            min_samples_leaf=3,   # Require more samples in leaves
            max_features='sqrt',  # Use sqrt of features for each tree
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_importance = None
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target variable."""
        df = df.copy()
        
        # List of pre-match features only
        prematch_features = [
            # Betting odds (available before match)
            'B365H', 'B365D', 'B365A',  # Bet365 odds
            'AvgH', 'AvgD', 'AvgA',     # Average market odds
            
            # Team form and standings (if available)
            'home_team_rank',
            'away_team_rank',
            'home_team_form_api',
            'away_team_form_api',
            
            # H2H statistics
            'h2h_total',
            'h2h_home_wins',
            'h2h_away_wins'
        ]
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        available_features = [col for col in prematch_features if col in numeric_cols]
        
        # Select features and handle missing values
        X = df[available_features].copy()
        
        # Fill missing values with appropriate strategies
        for col in X.columns:
            if 'rank' in col or 'form' in col:
                X[col] = X[col].fillna(X[col].mean())
            elif 'h2h' in col:
                X[col] = X[col].fillna(0)
            else:  # odds
                X[col] = X[col].fillna(X[col].median())
        
        # Create target variable (1: Home Win, 0: Draw, -1: Away Win)
        y = df['FTR'].map({'H': 1, 'D': 0, 'A': -1})
        
        # Log information about the features
        logger.info(f"Selected {len(available_features)} features: {available_features}")
        logger.info(f"Sample size: {len(X)}")
        
        return X, y
    
    def perform_cross_validation(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict:
        """Perform cross-validation and return scores."""
        cv_scores = cross_val_score(self.model, X, y, cv=cv)
        
        cv_results = {
            'mean_accuracy': cv_scores.mean(),
            'std_accuracy': cv_scores.std(),
            'cv_scores': cv_scores
        }
        
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Mean CV accuracy: {cv_results['mean_accuracy']:.4f} (+/- {cv_results['std_accuracy']*2:.4f})")
        
        return cv_results
    
    def train(self, train_data: pd.DataFrame, valid_data: pd.DataFrame = None) -> Dict:
        """Train the model and return performance metrics."""
        logger.info("Preparing training data...")
        X, y = self.prepare_features(train_data)
        
        if len(X) == 0:
            raise ValueError("No valid training samples after feature preparation")
        
        # Perform cross-validation first
        logger.info("Performing cross-validation...")
        cv_results = self.perform_cross_validation(X, y)
        
        # Split into train and validation
        if valid_data is None:
            X_train, X_valid, y_train, y_valid = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        else:
            X_train, y_train = X, y
            X_valid, y_valid = self.prepare_features(valid_data)
            
            if len(X_valid) == 0:
                raise ValueError("No valid validation samples after feature preparation")
        
        # Scale features
        logger.info("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_valid_scaled = self.scaler.transform(X_valid)
        
        # Train model
        logger.info("Training model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Get feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Make predictions
        train_preds = self.model.predict(X_train_scaled)
        valid_preds = self.model.predict(X_valid_scaled)
        
        # Calculate metrics
        metrics = {
            'train_report': classification_report(y_train, train_preds, output_dict=True),
            'valid_report': classification_report(y_valid, valid_preds, output_dict=True),
            'feature_importance': self.feature_importance,
            'cv_results': cv_results
        }
        
        # Log detailed metrics
        logger.info("\nTraining Metrics:")
        logger.info(f"Training accuracy: {metrics['train_report']['accuracy']:.4f}")
        logger.info(f"Validation accuracy: {metrics['valid_report']['accuracy']:.4f}")
        
        return metrics
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict match probabilities."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def predict_matches(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions for matches and add prediction probabilities."""
        X, _ = self.prepare_features(matches_df)
        probas = self.predict_proba(X)
        
        # Add prediction probabilities to dataframe
        results_df = matches_df.copy()
        results_df['pred_away_win'] = probas[:, 0]  # -1 class
        results_df['pred_draw'] = probas[:, 1]      # 0 class
        results_df['pred_home_win'] = probas[:, 2]  # 1 class
        
        return results_df
    
    def plot_feature_importance(self, output_dir: Path, top_n: int = 15):
        """Plot feature importance."""
        if self.feature_importance is None:
            logger.error("No feature importance available. Train the model first.")
            return
        
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=self.feature_importance.head(top_n),
            x='importance',
            y='feature'
        )
        plt.title(f'Top {top_n} Most Important Features')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        
        plot_path = output_dir / 'feature_importance.png'
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature importance plot saved to {plot_path}")
    
    def plot_confusion_matrix(self, y_true: pd.Series, y_pred: np.ndarray, output_dir: Path):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plot_path = output_dir / 'confusion_matrix.png'
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Confusion matrix plot saved to {plot_path}")
    
    def save_model(self, filename: str = 'match_predictor.joblib'):
        """Save the trained model and scaler."""
        model_path = self.model_dir / filename
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance
        }, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, filename: str = 'match_predictor.joblib'):
        """Load a trained model and scaler."""
        model_path = self.model_dir / filename
        if model_path.exists():
            saved_data = joblib.load(model_path)
            self.model = saved_data['model']
            self.scaler = saved_data['scaler']
            self.feature_importance = saved_data['feature_importance']
            logger.info(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"No saved model found at {model_path}")