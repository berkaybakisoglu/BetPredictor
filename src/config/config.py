"""Configuration module for the betting predictor."""
from pathlib import Path
from typing import Set, Dict
from dataclasses import dataclass, field

@dataclass
class DataConfig:
    """Data loading and processing configuration."""
    train_cutoff_year: int = 2023
    min_training_samples: int = 1000
    data_dir: Path = Path('data/raw')
    
    # Expected number of matches per season for each league
    expected_matches: Dict[str, int] = field(default_factory=lambda: {
        'E0': 380,  # EPL: 20 teams, each plays 38 matches (19 home + 19 away)
        'SP1': 380, # LaLiga: 20 teams, 38 matches each
        'D1': 306,  # Bundesliga: 18 teams, 34 matches each
        'I1': 380,  # Serie A: 20 teams, 38 matches each
        'F1': 306   # Ligue 1: 18 teams, 34 matches each
    })
    
    # Leagues to include in data loading
    leagues_to_include: Set[str] = field(default_factory=lambda: {
        'E0',  # EPL
        'SP1', # LaLiga
        'D1',  # Bundesliga
        'I1',  # Serie A
        'F1'   # Ligue 1
    })
    
    # Required columns for data validation
    required_columns: Set[str] = field(default_factory=lambda: {
        'Date', 'HomeTeam', 'AwayTeam',
        'B365H', 'B365D', 'B365A',  # Pre-match odds
        'FTHG', 'FTAG', 'FTR',      # Full-time results
        'HTHG', 'HTAG', 'HTR',      # Half-time results
        'HC', 'AC',                  # Historical corners data
        'HY', 'AY',                  # Historical cards data
    })
    
    # Optional columns that enhance prediction if available
    optional_columns: Set[str] = field(default_factory=lambda: {
        'B365>2.5', 'B365<2.5',  # Over/under odds
    })

@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    form_window: int = 5      # Number of recent matches for form calculation
    h2h_window: int = 5       # Number of H2H matches to consider
    min_matches_required: int = 5  # Minimum matches needed for reliable features
    decay_factor: float = 0.2  # Exponential decay factor for recent form

@dataclass
class ModelConfig:
    """Model training and prediction configuration."""
    min_training_samples: int = 1000  # Minimum samples needed for training
    
    markets: Dict[str, bool] = field(default_factory=lambda: {
        'match_result': True,
        'over_under': True,
        'ht_score': False,  # Disabled as not needed
        'corners': True,
        'cards': True,
        'ft_score': False  # Disabled due to poor performance
    })

@dataclass
class BettingConfig:
    """Betting simulation configuration."""
    initial_bankroll: float = 1000.0
    stop_loss_pct: float = 0.3  # Reduced from 0.5 to limit potential losses
    kelly_fraction: float = 0.1  # Reduced from 0.25 for more conservative betting
    
    # Betting thresholds
    value_threshold: float = 1.1    # Minimum value ratio for value bets
    confidence_threshold: float = 0.6  # Minimum confidence for selective betting
    over_under_threshold: float = 0.6  # Minimum probability for over/under bets
    
    # Market-specific thresholds
    thresholds: Dict[str, Dict] = field(default_factory=lambda: {
        'match_result': {
            'confidence': 0.70,  # Increased from 0.65 for higher certainty
            'min_odds': 1.8,     # Increased from 1.5 for better value
            'max_stake_pct': 0.02  # Reduced from 0.03 for risk management
        },
        'over_under': {
            'confidence': 0.65,
            'min_odds': 1.5,
            'max_stake_pct': 0.03
        },
        'ht_score': {
            'confidence': 0.55,  # Reduced from 0.25 to allow more bets
            'min_odds': 2.0,     # Reduced from 3.0 for more opportunities
            'max_stake_pct': 0.01  # Reduced for higher risk market
        },
        'corners': {
            'confidence': 0.55,  # Reduced from 0.60 to allow more bets
            'min_odds': 1.6,     # Reduced from 1.8 for more opportunities
            'max_stake_pct': 0.01
        },
        'cards': {
            'confidence': 0.55,  # Reduced from 0.60 to allow more bets
            'min_odds': 1.6,     # Reduced from 1.8 for more opportunities
            'max_stake_pct': 0.01
        }
    })
    
    # Market-specific loss limits (as percentage of bankroll)
    market_loss_limits: Dict[str, float] = field(default_factory=lambda: {
        'match_result': 0.15,  # 15% max loss on match results
        'over_under': 0.10,    # 10% max loss on over/under
        'ht_score': 0.05,      # 5% max loss on half-time scores
        'corners': 0.05,       # 5% max loss on corners
        'cards': 0.05          # 5% max loss on cards
    })

@dataclass
class Config:
    """Main configuration class combining all config components."""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    betting: BettingConfig = field(default_factory=BettingConfig)
    
    # Output settings
    output_dir: Path = Path('output')
    models_dir: Path = Path('models')
    
    def __post_init__(self):
        """Ensure directories exist after initialization."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True) 