"""Configuration classes for the betting prediction system."""
import os
from dataclasses import dataclass, field

@dataclass
class DataConfig:
    """Configuration for data loading."""
    data_dir: str = 'data/raw'
    test_size: float = 0.2
    random_state: int = 42
    min_matches: int = 10
    min_odds: float = 1.1
    max_odds: float = 30.0

@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    form_matches: int = 5
    h2h_matches: int = 3
    recent_matches: int = 10
    
    # Feature groups
    use_form_features: bool = True
    use_team_stats: bool = True
    use_h2h_features: bool = True
    use_league_features: bool = True
    use_market_features: bool = True
    
    # Feature parameters
    form_weight_decay: float = 0.2
    h2h_weight_decay: float = 0.9
    
    # Derived features
    calculate_goal_stats: bool = True
    calculate_possession_stats: bool = True
    calculate_shot_stats: bool = True

@dataclass
class ModelConfig:
    """Configuration for model training and prediction."""
    # General settings
    random_state: int = 42
    n_jobs: int = -1
    
    # Training parameters
    test_size: float = 0.2
    validation_size: float = 0.2
    
    # Model hyperparameters
    learning_rate: float = 0.01
    n_estimators: int = 1000
    max_depth: int = 7
    num_leaves: int = 31
    min_child_samples: int = 20
    
    # Feature selection
    feature_selection: bool = True
    feature_importance_threshold: float = 0.001
    
    # Prediction settings
    confidence_threshold: float = 0.6
    probability_calibration: bool = True

@dataclass
class BettingConfig:
    """Configuration for betting strategy and evaluation."""
    # Betting parameters
    initial_bankroll: float = 1000.0
    bet_size: float = 100.0
    min_odds: float = 1.5
    max_odds: float = 10.0
    
    # Strategy parameters
    kelly_fraction: float = 0.5
    use_kelly_criterion: bool = True
    min_edge: float = 0.05
    confidence_threshold: float = 0.6
    
    # Risk management
    max_stake_percent: float = 0.1
    stop_loss: float = -0.5
    take_profit: float = 2.0
    
    # Market settings
    markets: list = field(default_factory=lambda: ['match_odds', 'over_under'])
    
    # Evaluation settings
    rolling_window: int = 20
    min_bets_for_analysis: int = 10

@dataclass
class Config:
    """Main configuration class."""
    data: DataConfig
    features: FeatureConfig
    model: ModelConfig
    betting: BettingConfig
    output_dir: str = 'output'
    models_dir: str = 'models'
    
    def __post_init__(self):
        """Create necessary directories."""
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Create models directory if it doesn't exist
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
            
        # Create subdirectories
        for subdir in ['logs', 'models', 'predictions', 'visualizations']:
            path = os.path.join(self.output_dir, subdir)
            if not os.path.exists(path):
                os.makedirs(path) 