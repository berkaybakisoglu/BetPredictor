from pathlib import Path

# Base directories
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# API Configuration
FOOTBALL_DATA_URL = "https://www.football-data.co.uk/new/new_league_data"
API_FOOTBALL_BASE_URL = "https://api-football-v1.p.rapidapi.com/v3"

# Model Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Feature Configuration
FORM_WINDOW = 5  # Number of previous matches to consider for form

# Basic match statistics
MATCH_STATS_FEATURES = [
    'HS',   # Home Shots
    'AS',   # Away Shots
    'HST',  # Home Shots on Target
    'AST',  # Away Shots on Target
    'HC',   # Home Corners
    'AC',   # Away Corners
    'HF',   # Home Fouls
    'AF',   # Away Fouls
    'HY',   # Home Yellows
    'AY'    # Away Yellows
]

# Average odds features
ODDS_FEATURES = [
    'AvgH',     # Average Home Win Odds
    'AvgD',     # Average Draw Odds
    'AvgA',     # Average Away Win Odds
    'Avg>2.5',  # Average Over 2.5 Goals Odds
    'Avg<2.5'   # Average Under 2.5 Goals Odds
]

# Form and goal statistics (these will be calculated)
CALCULATED_FEATURES = [
    'home_team_form',
    'away_team_form',
    'home_goals_scored_avg',
    'home_goals_conceded_avg',
    'away_goals_scored_avg',
    'away_goals_conceded_avg'
]

# Combine all features
FEATURES = MATCH_STATS_FEATURES + ODDS_FEATURES + CALCULATED_FEATURES

# Target variables
TARGETS = ['match_outcome']  # 1: Home Win, 0: Draw, -1: Away Win

# Leagues to track
LEAGUES = {
    'EPL': 'E0',
    'LaLiga': 'SP1',
    'Bundesliga': 'D1',
    'SerieA': 'I1',
    'Ligue1': 'F1'
} 