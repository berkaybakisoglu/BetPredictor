import requests
import json
from datetime import datetime
from pathlib import Path

# API config
FOOTBALL_API_KEY = '4af382b3c844b09444445ef9382a759d'
FOOTBALL_API_URL = 'https://v3.football.api-sports.io'
FOOTBALL_API_HEADERS = {
    'x-rapidapi-host': 'v3.football.api-sports.io',
    'x-rapidapi-key': FOOTBALL_API_KEY
}

# Only the leagues we're interested in
SUPPORTED_LEAGUES = {
    'Premier League': 39,
    'Championship': 40,
    'Serie A': 135,
    'La Liga': 140,
    'Bundesliga': 78,
    'Ligue 1': 61,
    'Eredivisie': 88,
    'Primeira Liga': 94
}

def validate_match(fixture):
    """Validate match data."""
    try:
        # Check if match is in the future
        match_time = datetime.fromtimestamp(fixture['fixture']['timestamp'])
        if match_time <= datetime.now():
            print(f"Skipping past match: {match_time}")
            return False
            
        # Check if match status is scheduled/not started
        if fixture['fixture']['status']['short'] != 'NS':  # NS = Not Started
            print(f"Skipping match with status: {fixture['fixture']['status']['short']}")
            return False
            
        # Check if we have both teams
        if not fixture['teams']['home']['name'] or not fixture['teams']['away']['name']:
            print("Missing team names")
            return False
            
        return True
        
    except KeyError as e:
        print(f"Invalid fixture data structure: {e}")
        return False

def validate_odds(odds):
    """Validate odds data."""
    try:
        required_markets = {'1', 'X', '2', 'O2.5', 'U2.5'}
        
        # Check if we have all required odds
        if not all(market in odds for market in required_markets):
            print(f"Missing required odds markets. Have: {odds.keys()}")
            return False
            
        # Check if odds are in reasonable range (e.g., 1.01 to 50.0)
        for market, value in odds.items():
            if market in required_markets:
                if not isinstance(value, (int, float)):
                    print(f"Invalid odds value for {market}: {value}")
                    return False
                if value < 1.01 or value > 50.0:
                    print(f"Odds out of reasonable range for {market}: {value}")
                    return False
                    
        return True
        
    except Exception as e:
        print(f"Error validating odds: {e}")
        return False

def fetch_and_save_fixtures():
    """Fetch today's fixtures and their odds for supported leagues only."""
    date_str = datetime.now().strftime('%Y-%m-%d')
    
    # First, let's get all fixtures without league filter to see what's available
    url = f"{FOOTBALL_API_URL}/fixtures"
    params = {
        'date': date_str
        # Removed league filter temporarily for debugging
    }
    
    print(f"Making request to {url} with date={date_str}")
    response = requests.get(url, headers=FOOTBALL_API_HEADERS, params=params)
    print(f"Response status code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        all_fixtures = data.get('response', [])
        
        print(f"\nFound {len(all_fixtures)} total fixtures")
        
        # Print all available leagues to debug
        leagues = set()
        for fixture in all_fixtures:
            league_name = fixture['league']['name']
            league_id = fixture['league']['id']
            leagues.add((league_name, league_id))
        
        print("\nAvailable leagues today:")
        for league_name, league_id in sorted(leagues):
            print(f"  - {league_name} (ID: {league_id})")
        
        # Now filter for supported leagues
        fixtures = [f for f in all_fixtures if f['league']['id'] in SUPPORTED_LEAGUES.values()]
        print(f"\nFound {len(fixtures)} fixtures in supported leagues:")
        
        matches_with_odds = []
        for fixture in fixtures:
            # Validate match data first
            if not validate_match(fixture):
                continue
            
            home = fixture['teams']['home']['name']
            away = fixture['teams']['away']['name']
            league = fixture['league']['name']
            fixture_id = fixture['fixture']['id']
            time = datetime.fromtimestamp(fixture['fixture']['timestamp']).strftime('%H:%M')
            
            print(f"\n{time} - {league}: {home} vs {away}")
            
            # Fetch and validate odds
            odds = fetch_odds_for_match(fixture_id)
            if odds and validate_odds(odds):
                print("  Bet365 odds:")
                print(f"    1X2: {odds.get('1', '-')} | {odds.get('X', '-')} | {odds.get('2', '-')}")
                print(f"    O/U 2.5: Over {odds.get('O2.5', '-')} | Under {odds.get('U2.5', '-')}")
                
                match_data = {
                    'fixture_id': fixture_id,
                    'time': time,
                    'league': league,
                    'home': home,
                    'away': away,
                    'match_timestamp': fixture['fixture']['timestamp'],
                    'status': fixture['fixture']['status']['short'],
                    **odds
                }
                matches_with_odds.append(match_data)
            else:
                print("  Skipping match due to invalid odds")
        
        # Save processed data
        debug_dir = Path('debug')
        debug_dir.mkdir(exist_ok=True)
        with open(debug_dir / f"matches_with_odds_{date_str}.json", 'w') as f:
            json.dump(matches_with_odds, f, indent=2)
        
        return matches_with_odds

def fetch_odds_for_match(fixture_id):
    """Fetch Bet365 odds for a specific fixture."""
    url = f"{FOOTBALL_API_URL}/odds"
    params = {
        'fixture': fixture_id,
        'bookmaker': 8  # Bet365 bookmaker ID
    }
    
    print(f"Fetching odds for fixture {fixture_id}")
    response = requests.get(url, headers=FOOTBALL_API_HEADERS, params=params)
    
    if response.status_code == 200:
        data = response.json()
        odds = data.get('response', [])
        
        # Save raw response for debugging
        debug_dir = Path('debug/odds')
        debug_dir.mkdir(exist_ok=True, parents=True)
        debug_file = debug_dir / f"odds_fixture_{fixture_id}.json"
        with open(debug_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        if odds and 'bookmakers' in odds[0]:
            bet365 = next((b for b in odds[0]['bookmakers'] if b['name'].lower() == 'bet365'), None)
            if bet365:
                match_odds = {}
                for market in bet365['bets']:
                    if market['name'] == 'Match Winner':
                        for value in market['values']:
                            if value['value'] == 'Home':
                                match_odds['1'] = float(value['odd'])
                            elif value['value'] == 'Draw':
                                match_odds['X'] = float(value['odd'])
                            elif value['value'] == 'Away':
                                match_odds['2'] = float(value['odd'])
                    elif market['name'] == 'Goals Over/Under':
                        for value in market['values']:
                            if value['value'] == 'Over 2.5':
                                match_odds['O2.5'] = float(value['odd'])
                            elif value['value'] == 'Under 2.5':
                                match_odds['U2.5'] = float(value['odd'])
                return match_odds
            
        print(f"No odds found for fixture {fixture_id}")
        return None
    else:
        print(f"API error: {response.status_code} - {response.text}")
        return None

if __name__ == "__main__":
    fetch_and_save_fixtures() 