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
    
    expected_matches: Dict[str, int] = field(default_factory=lambda: {
        'E0': 380,  # 20 teams, 38 matches each
        'E1': 552,  # 24 teams, 46 matches each
        'E2': 552,  # 24 teams, 46 matches each
        'E3': 552,  # 24 teams, 46 matches each
        'EC': 552,  # 24 teams, 46 matches each
        'SC0': 330, # 12 teams, 33 matches each 
        'SC1': 360, # 10 teams, 36 matches each
        'SC2': 360, # 10 teams, 36 matches each
        'SC3': 360, # 10 teams, 36 matches each
        'D1': 306,  # 18 teams, 34 matches each
        'D2': 306,  # 18 teams, 34 matches each
        'SP1': 380, # 20 teams, 38 matches each
        'SP2': 462, # 22 teams, 42 matches each
        'I1': 380,  # 20 teams, 38 matches each
        'I2': 380,  # 20 teams, 38 matches each
        'F1': 380,  # 20 teams, 38 matches each 
        'F2': 380,  # 20 teams, 38 matches each
        'B1': 306,  # 18 teams, 34 matches each
        'N1': 306,  # 18 teams, 34 matches each
        'P1': 306,  # 18 teams, 34 matches each
        'T1': 380,  # 20 teams, 38 matches each
        'G1': 306   # 18 teams, 34 matches each
    })
    
    leagues_to_include: Set[str] = field(default_factory=lambda: {
        'E0',   # English Premier League
        'E1',   # English Championship
        'E2',   # English League One
        'E3',   # English League Two
        'EC',   # English Conference
        'SC0',  # Scottish Premiership
        'SC1',  # Scottish Championship
        'SC2',  # Scottish League One
        'SC3',  # Scottish League Two
        'D1',   # German Bundesliga
        'D2',   # German 2. Bundesliga
        'SP1',  # Spanish La Liga
        'SP2',  # Spanish Segunda División
        'I1',   # Italian Serie A
        'I2',   # Italian Serie B
        'F1',   # French Ligue 1
        'F2',   # French Ligue 2
        'B1',   # Belgian First Division
        'N1',   # Dutch Eredivisie
        'P1',   # Portuguese Primeira Liga
        'T1',   # Turkish Super Lig
        'G1'    # Greek Super League
    })
    
    required_columns: Set[str] = field(default_factory=lambda: {
        'Date', 'HomeTeam', 'AwayTeam',
        'B365H', 'B365D', 'B365A',
        'FTHG', 'FTAG', 'FTR',
        'HTHG', 'HTAG', 'HTR',
        'HC', 'AC',
        'HY', 'AY',
    })
    
    optional_columns: Set[str] = field(default_factory=lambda: {
        'B365>2.5', 'B365<2.5',
    })

@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    form_window: int = 5
    h2h_window: int = 5
    min_matches_required: int = 5
    decay_factor: float = 0.2

@dataclass
class ModelConfig:
    """Model training and prediction configuration."""
    min_training_samples: int = 1000
    
    markets: Dict[str, bool] = field(default_factory=lambda: {
        'match_result': True,
        'over_under': True,
        'ht_score': False,
        'corners': True,
        'cards': True,
        'ft_score': False
    })

@dataclass
class BettingConfig:
    """Betting simulation configuration."""
    initial_bankroll: float = 1000.0
    stop_loss_pct: float = 0.3
    kelly_fraction: float = 0.1
    value_threshold: float = 1.1
    confidence_threshold: float = 0.6
    over_under_threshold: float = 0.6
    
    thresholds: Dict[str, Dict] = field(default_factory=lambda: {
        'match_result': {
            'confidence': 0.70,
            'min_odds': 1.8,
            'max_stake_pct': 0.02
        },
        'over_under': {
            'confidence': 0.65,
            'min_odds': 1.5,
            'max_stake_pct': 0.03
        },
        'ht_score': {
            'confidence': 0.55,
            'min_odds': 2.0,
            'max_stake_pct': 0.01
        },
        'corners': {
            'confidence': 0.55,
            'min_odds': 1.6,
            'max_stake_pct': 0.01
        },
        'cards': {
            'confidence': 0.55,
            'min_odds': 1.6,
            'max_stake_pct': 0.01
        }
    })
    
    market_loss_limits: Dict[str, float] = field(default_factory=lambda: {
        'match_result': 0.15,
        'over_under': 0.10,
        'ht_score': 0.05,
        'corners': 0.05,
        'cards': 0.05
    })

@dataclass
class Config:
    """Main configuration class combining all config components."""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    betting: BettingConfig = field(default_factory=BettingConfig)
    
    output_dir: Path = Path('output')
    models_dir: Path = Path('models')
    
    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True) 