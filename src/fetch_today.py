"""Script to fetch today's matches without loading historical data."""
import os
from datetime import datetime
import pandas as pd
from src.fetch_matches import (
    fetch_fixtures,
    fetch_odds,
    format_match_data,
    save_data,
    SUPPORTED_LEAGUES,
    API_KEY
)

def main():
    """Fetch today's matches and save the data."""
    try:
        date_str = datetime.now().strftime('%Y-%m-%d')
        date_folder = date_str.replace('-', '')
        
        print(f"\nStarting data fetch for date: {date_str}")
        print(f"Using API key: {API_KEY[:5]}...")  # Only show first 5 chars for security
        
        # Create data directory if it doesn't exist
        data_dir = f'data/api_data/{date_folder}'
        os.makedirs(data_dir, exist_ok=True)
        
        # Fetch fixtures
        fixtures = fetch_fixtures(date_str)
        if not fixtures:
            print("Failed to fetch fixtures")
            return
            
        # Filter for supported leagues
        supported_fixtures = [f for f in fixtures if f['league']['id'] in SUPPORTED_LEAGUES.values()]
        print(f"\nFound {len(supported_fixtures)} fixtures in supported leagues:")
        
        # Print fixtures details
        for fixture in supported_fixtures:
            print(f"\n{fixture['league']['name']}")
            print(f"{fixture['teams']['home']['name']} vs {fixture['teams']['away']['name']}")
            print(f"Kickoff: {datetime.fromtimestamp(fixture['fixture']['timestamp']).strftime('%H:%M')}")
        
        # Fetch odds for each fixture
        all_odds = []
        total_fixtures = len(supported_fixtures)
        
        for i, fixture in enumerate(supported_fixtures, 1):
            fixture_id = fixture['fixture']['id']
            print(f"\nFetching odds ({i}/{total_fixtures}): {fixture['teams']['home']['name']} vs {fixture['teams']['away']['name']}")
            odds = fetch_odds(fixture_id)
            if odds:
                all_odds.append(odds)
        
        # Format data with all required features
        df = format_match_data(supported_fixtures, all_odds)
        
        # Save processed data
        processed_file = f'{data_dir}/matches.csv'
        df.to_csv(processed_file, index=False)
        print(f"\nSaved processed data to {processed_file}")
        
        # Also save raw data for reference
        fixtures_file = f'{data_dir}/fixtures.json'
        odds_file = f'{data_dir}/odds.json'
        save_data(supported_fixtures, fixtures_file)
        save_data(all_odds, odds_file)
        
        print(f"\nCompleted! Processed {len(df)} matches")
        print("\nFeatures collected for each match:")
        print(df.columns.tolist())
        
    except Exception as e:
        print(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 