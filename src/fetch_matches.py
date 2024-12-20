import requests
import json
import os
from datetime import datetime
import logging
from typing import Dict, List, Optional
import time
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API-Football configuration
FOOTBALL_API_KEY = os.getenv('API_FOOTBALL_KEY')
FOOTBALL_API_KEY = '151c84560f9e35940d45837d99f632c9'
if not FOOTBALL_API_KEY:
    raise ValueError("API_FOOTBALL_KEY environment variable is not set")

# Football-Data.org configuration
FOOTBALL_DATA_KEY = os.getenv('FOOTBALL_DATA_KEY')
if not FOOTBALL_DATA_KEY:
    raise ValueError("FOOTBALL_DATA_KEY environment variable is not set")

print(f"API Keys loaded: API-Football={FOOTBALL_API_KEY[:5]}..., Football-Data={FOOTBALL_DATA_KEY[:5]}...")

# API endpoints
FOOTBALL_API_URL = 'https://v3.football.api-sports.io'
FOOTBALL_DATA_URL = 'https://api.football-data.org/v4'

# API headers
FOOTBALL_API_HEADERS = {
    'x-rapidapi-host': 'v3.football.api-sports.io',
    'x-rapidapi-key': FOOTBALL_API_KEY
}

FOOTBALL_DATA_HEADERS = {
    'X-Auth-Token': FOOTBALL_DATA_KEY
}

# League ID mappings between APIs
LEAGUE_MAPPINGS = {
    'Premier League': {'api_football': 39, 'football_data': 'PL'},
    'Championship': {'api_football': 40, 'football_data': 'ELC'},
    'La Liga': {'api_football': 140, 'football_data': 'PD'},
    'Bundesliga': {'api_football': 78, 'football_data': 'BL1'},
    'Serie A': {'api_football': 135, 'football_data': 'SA'},
    'Ligue 1': {'api_football': 61, 'football_data': 'FL1'},
    'Eredivisie': {'api_football': 88, 'football_data': 'DED'},
    'Primeira Liga': {'api_football': 94, 'football_data': 'PPL'},
}

# Team ID mappings between APIs
TEAM_MAPPINGS = {
    # Bundesliga
    'Bayern München': {'api_football': 157, 'football_data': 5},
    'RB Leipzig': {'api_football': 173, 'football_data': 721},
    
    # Serie A
    'AC Milan': {'api_football': 489, 'football_data': 98},
    'Verona': {'api_football': 504, 'football_data': 450},
    
    # La Liga
    'Girona': {'api_football': 547, 'football_data': 298},
    'Valladolid': {'api_football': 720, 'football_data': 250},
    
    # Eredivisie
    'Waalwijk': {'api_football': 417, 'football_data': 673},
    'PEC Zwolle': {'api_football': 193, 'football_data': 671},
    
    # Championship
    'Luton': {'api_football': 1359, 'football_data': 389},
    'Derby': {'api_football': 69, 'football_data': 342},
    
    # Primeira Liga
    'Casa Pia': {'api_football': 4716, 'football_data': 5531},
    'Arouca': {'api_football': 240, 'football_data': 810}
}

def get_football_data_team_id(team_name: str) -> Optional[int]:
    """Get Football-Data.org team ID from team name."""
    team_info = TEAM_MAPPINGS.get(team_name)
    if team_info:
        return team_info['football_data']
    print(f"Warning: No Football-Data.org ID mapping found for team: {team_name}")
    return None

def fetch_fixtures(date_str):
    """Fetch fixtures for a specific date."""
    url = f"{FOOTBALL_API_URL}/fixtures"
    params = {'date': date_str}
    
    try:
        print(f"Making request to {url} with date={date_str}")
        response = requests.get(url, headers=FOOTBALL_API_HEADERS, params=params)
        print(f"Response status code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"API error: {response.text}")
            return None
            
        data = response.json()
        
        if 'errors' in data and data['errors']:
            print(f"API returned errors: {data['errors']}")
            return None
            
        if 'response' not in data:
            print("No 'response' field in API response")
            return None
            
        fixtures = data['response']
        print(f"Successfully fetched {len(fixtures)} fixtures")
        return fixtures
    except Exception as e:
        print(f"Error fetching fixtures: {str(e)}")
        return None

def fetch_odds(fixture_id):
    """Fetch odds for a specific fixture."""
    url = f"{FOOTBALL_API_URL}/odds"
    params = {
        'fixture': fixture_id,
        'bookmaker': 8  # Bet365
    }
    
    try:
        print(f"Fetching odds for fixture {fixture_id}")
        response = requests.get(url, headers=FOOTBALL_API_HEADERS, params=params)
        
        if response.status_code != 200:
            print(f"API error: {response.text}")
            return None
            
        data = response.json()
        
        if 'errors' in data and data['errors']:
            print(f"API returned errors: {data['errors']}")
            return None
            
        if 'response' not in data:
            print("No 'response' field in API response")
            return None
            
        odds = data['response']
        if not odds:
            print(f"No odds available for fixture {fixture_id}")
            return None
            
        return odds[0]
    except Exception as e:
        print(f"Error fetching odds: {str(e)}")
        return None

def save_data(data, filename):
    """Save data to a JSON file."""
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Successfully saved data to {filename}")
    except Exception as e:
        print(f"Error saving data: {str(e)}")

def fetch_team_stats_football_data(team_id: str, competition_id: str) -> Optional[Dict]:
    """Fetch team statistics from Football-Data.org API."""
    url = f"{FOOTBALL_DATA_URL}/teams/{team_id}/matches"
    params = {
        'limit': 10,  # Last 10 matches
        'status': 'FINISHED'
    }
    
    max_retries = 3
    retry_delay = 60  # seconds
    
    for attempt in range(max_retries):
        try:
            print(f"\nFetching Football-Data.org stats for team {team_id}")
            response = requests.get(url, headers=FOOTBALL_DATA_HEADERS, params=params)
            
            if response.status_code == 429:  # Rate limit exceeded
                if attempt < max_retries - 1:  # Don't sleep on last attempt
                    print(f"Rate limit exceeded. Waiting {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                    
            if response.status_code != 200:
                print(f"Football-Data.org API error: {response.text}")
                return None
                
            matches_data = response.json()
            
            if 'matches' not in matches_data:
                print("No matches data in response")
                return None
                
            # Calculate stats from recent matches
            total_goals_scored = 0
            total_goals_conceded = 0
            form_results = []
            
            for match in matches_data['matches']:
                if match['status'] != 'FINISHED':
                    continue
                    
                is_home = match['homeTeam']['id'] == int(team_id)
                team_goals = match['score']['fullTime']['home'] if is_home else match['score']['fullTime']['away']
                opponent_goals = match['score']['fullTime']['away'] if is_home else match['score']['fullTime']['home']
                
                if team_goals is not None and opponent_goals is not None:
                    total_goals_scored += team_goals
                    total_goals_conceded += opponent_goals
                    
                    # Calculate form (W/D/L)
                    if team_goals > opponent_goals:
                        form_results.append('W')
                    elif team_goals < opponent_goals:
                        form_results.append('L')
                    else:
                        form_results.append('D')
            
            matches_played = len([m for m in matches_data['matches'] if m['status'] == 'FINISHED'])
            if matches_played == 0:
                return None
                
            stats = {
                'goals_scored_avg': total_goals_scored / matches_played,
                'goals_conceded_avg': total_goals_conceded / matches_played,
                'form': ''.join(form_results[-5:])  # Last 5 matches
            }
            
            print(f"Successfully calculated stats from Football-Data.org")
            return stats
            
        except Exception as e:
            print(f"Error fetching Football-Data.org stats: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                return None

def calculate_team_features(football_data_stats: Dict = None) -> Dict:
    """Calculate team features using Football-Data.org data."""
    if not football_data_stats:
        # Default values if no stats available
        return {
            'Goals_Scored_Avg': 1.5,
            'Goals_Conceded_Avg': 1.5,
            'Form': 0.5,
            'Corners_For_Avg': 5.0,
            'Corners_Against_Avg': 5.0,
            'Cards_Avg': 2.0
        }
    
    try:
        features = {}
        
        # Goals
        features['Goals_Scored_Avg'] = football_data_stats.get('goals_scored_avg', 1.5)
        features['Goals_Conceded_Avg'] = football_data_stats.get('goals_conceded_avg', 1.5)
        
        # Form
        form_map = {'W': 1, 'D': 0.5, 'L': 0}
        form = football_data_stats.get('form', '')
        if form:
            features['Form'] = sum(form_map.get(result, 0.5) for result in form) / len(form)
        else:
            features['Form'] = 0.5
        
        # Default values for corners and cards (not available in Football-Data.org)
        features['Corners_For_Avg'] = 5.0
        features['Corners_Against_Avg'] = 5.0
        features['Cards_Avg'] = 2.0
        
        print(f"Calculated features: {features}")
        return features
        
    except Exception as e:
        print(f"Error calculating team features: {str(e)}")
        print(f"Football-Data.org stats: {json.dumps(football_data_stats, indent=2)}")
        # Return default values on error
        return {
            'Goals_Scored_Avg': 1.5,
            'Goals_Conceded_Avg': 1.5,
            'Form': 0.5,
            'Corners_For_Avg': 5.0,
            'Corners_Against_Avg': 5.0,
            'Cards_Avg': 2.0
        }

def format_match_data(fixtures: List[Dict], odds: List[Dict]) -> pd.DataFrame:
    """Format API data with additional features from both APIs."""
    matches = []
    current_season = datetime.now().year
    
    for fixture in fixtures:
        # Skip if not in supported leagues
        league_name = next((name for name, ids in LEAGUE_MAPPINGS.items() 
                          if ids['api_football'] == fixture['league']['id']), None)
        if not league_name:
            continue
            
        match_odds = next((o for o in odds if o['fixture']['id'] == fixture['fixture']['id']), None)
        if not match_odds:
            continue
            
        # Get basic match info
        match_date = datetime.fromtimestamp(fixture['fixture']['timestamp'])
        home_team = fixture['teams']['home']['name']
        away_team = fixture['teams']['away']['name']
        
        match = {
            'Date': match_date,
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'League': fixture['league']['name'],
            'Season': match_date.year if match_date.month >= 7 else match_date.year - 1,
            'match_result': None  # Initialize match_result column
        }
        
        # Get Football-Data.org team IDs
        home_football_data_id = get_football_data_team_id(home_team)
        away_football_data_id = get_football_data_team_id(away_team)
        
        if not home_football_data_id or not away_football_data_id:
            print(f"Skipping match {home_team} vs {away_team} due to missing team ID mappings")
            continue
        
        # Get Football-Data.org stats
        league_ids = LEAGUE_MAPPINGS[league_name]
        home_stats = fetch_team_stats_football_data(
            str(home_football_data_id),
            league_ids['football_data']
        )
        
        # Add delay between API calls to respect rate limits
        time.sleep(2)
        
        away_stats = fetch_team_stats_football_data(
            str(away_football_data_id),
            league_ids['football_data']
        )
        
        # Calculate features using Football-Data.org data
        home_features = calculate_team_features(home_stats)
        away_features = calculate_team_features(away_stats)
        
        # Add features to match data
        match['Home_Goals_Scored_Avg'] = home_features['Goals_Scored_Avg']
        match['Home_Goals_Conceded_Avg'] = home_features['Goals_Conceded_Avg']
        match['Away_Goals_Scored_Avg'] = away_features['Goals_Scored_Avg']
        match['Away_Goals_Conceded_Avg'] = away_features['Goals_Conceded_Avg']
        match['Home_Form'] = home_features['Form']
        match['Away_Form'] = away_features['Form']
        
        # Default values for corners and cards
        match['Home_Corners_For_Avg'] = home_features['Corners_For_Avg']
        match['Home_Corners_Against_Avg'] = home_features['Corners_Against_Avg']
        match['Away_Corners_For_Avg'] = away_features['Corners_For_Avg']
        match['Away_Corners_Against_Avg'] = away_features['Corners_Against_Avg']
        match['Home_Cards_Avg'] = home_features['Cards_Avg']
        match['Away_Cards_Avg'] = away_features['Cards_Avg']
        
        # Add H2H features with default values
        match.update({
            'H2H_Home_Wins': 0.33,
            'H2H_Away_Wins': 0.33,
            'H2H_Draws': 0.34,
            'H2H_Avg_Goals': 2.5,
            'H2H_Avg_Corners': 10.0
        })
        
        # Add odds from API-Football
        if match_odds and 'bookmakers' in match_odds:
            bet365 = next((b for b in match_odds['bookmakers'] if b['name'].lower() == 'bet365'), None)
            if bet365:
                for market in bet365['bets']:
                    if market['name'] == 'Match Winner':
                        for value in market['values']:
                            if value['value'] == 'Home':
                                match['B365H'] = float(value['odd'])
                                match['Home_ImpliedProb'] = 1 / float(value['odd'])
                            elif value['value'] == 'Draw':
                                match['B365D'] = float(value['odd'])
                                match['Draw_ImpliedProb'] = 1 / float(value['odd'])
                            elif value['value'] == 'Away':
                                match['B365A'] = float(value['odd'])
                                match['Away_ImpliedProb'] = 1 / float(value['odd'])
                    elif market['name'] == 'Goals Over/Under':
                        for value in market['values']:
                            if value['value'] == 'Over 2.5':
                                match['B365>2.5'] = float(value['odd'])
                            elif value['value'] == 'Under 2.5':
                                match['B365<2.5'] = float(value['odd'])
        
        matches.append(match)
        
        # Add delay to respect API rate limits
        time.sleep(1)
    
    df = pd.DataFrame(matches)
    # Ensure all required columns exist with default values
    required_columns = ['match_result', 'Home_ImpliedProb', 'Draw_ImpliedProb', 'Away_ImpliedProb']
    for col in required_columns:
        if col not in df.columns:
            df[col] = None
    
    return df

def main():
    """Fetch today's matches and save the data."""
    # Only run if this script is run directly
    if __name__ != "__main__":
        return
        
    try:
        # Get tomorrow's date in YYYY-MM-DD format
        date_str = datetime.now().strftime('%Y-%m-%d')
        date_folder = date_str.replace('-', '')
        
        print(f"\nStarting data fetch for date: {date_str}")
        print(f"Using API key: {FOOTBALL_API_KEY[:5]}...")  # Only show first 5 chars for security
        
        # Create data directory if it doesn't exist
        data_dir = f'data/historical_api_data/{date_folder}'
        os.makedirs(data_dir, exist_ok=True)
        
        # Fetch fixtures
        fixtures = fetch_fixtures(date_str)
        if not fixtures:
            print("Failed to fetch fixtures")
            return
            
        # Filter for supported leagues
        supported_fixtures = [f for f in fixtures if f['league']['id'] in LEAGUE_MAPPINGS.values()]
        print(f"\nFound {len(supported_fixtures)} fixtures in supported leagues")
        
        # Fetch odds for each fixture
        all_odds = []
        total_fixtures = len(supported_fixtures)
        
        for i, fixture in enumerate(supported_fixtures, 1):
            fixture_id = fixture['fixture']['id']
            print(f"\nProgress: {i}/{total_fixtures}")
            odds = fetch_odds(fixture_id)
            if odds:
                all_odds.append(odds)
            time.sleep(1)  # Rate limiting
        
        # Format data with all required features
        df = format_match_data(supported_fixtures, all_odds)
        
        # Save processed data
        processed_file = f'{data_dir}/processed_matches.csv'
        df.to_csv(processed_file, index=False)
        print(f"\nSaved processed data to {processed_file}")
        
        # Also save raw data for reference
        fixtures_file = f'{data_dir}/fixtures.json'
        odds_file = f'{data_dir}/odds.json'
        save_data(supported_fixtures, fixtures_file)
        save_data(all_odds, odds_file)
        
        print(f"\nCompleted! Processed {len(df)} matches with features")
        
    except Exception as e:
        print(f"Error in main process: {str(e)}")
        raise  # Re-raise the exception for debugging

if __name__ == "__main__":
    main() 