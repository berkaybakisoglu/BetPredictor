import os
from dataclasses import dataclass, field

@dataclass
class DataConfig:
    data_dir: str = 'data/raw_csv'
    test_size: float = 0.2
    random_state: int = 42
    min_matches: int = 10
    min_odds: float = 1.1
    max_odds: float = 30.0

@dataclass
class FeatureConfig:
    form_matches: int = 5
    test_size: float = 0.2
    use_team_stats: bool = True
    use_market_features: bool = True
    use_form_features: bool = True
    use_position_features: bool = True
    use_h2h_features: bool = True
    use_corner_features: bool = True
    use_card_features: bool = True
    
    calculate_goal_stats: bool = True
    calculate_possession_stats: bool = True
    calculate_shot_stats: bool = True

@dataclass
class ModelConfig:
    random_state: int = 42
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    num_leaves: int = 31
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    
    feature_selection: bool = True
    feature_importance_threshold: float = 0.001
    
    confidence_threshold: float = 0.6
    probability_calibration: bool = True
    
    min_training_samples: int = 1000
    
    # Market configuration
    markets: dict = field(default_factory=lambda: {
        'match_result': True,
        'over_under': True,
        'corners': True,
        'cards': True
    })

@dataclass
class BettingConfig:
    initial_bankroll: float = 1000.0
    bet_size: float = 100.0
    min_odds: float = 1.5
    max_odds: float = 10.0
    
    kelly_fraction: float = 0.5
    use_kelly_criterion: bool = True
    min_edge: float = 0.05
    confidence_threshold: float = 0.6
    
    max_stake_percent: float = 0.1
    stop_loss: float = -0.5
    take_profit: float = 2.0
    
    markets: list = field(default_factory=lambda: ['match_odds', 'over_under'])
    
    rolling_window: int = 20
    min_bets_for_analysis: int = 10

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    betting: BettingConfig = field(default_factory=BettingConfig)
    output_dir: str = "output"
    models_dir: str = "models"
    
    def __post_init__(self):
        for directory in [self.output_dir, self.models_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory) 