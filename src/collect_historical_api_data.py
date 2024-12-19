import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime, timedelta
from src.data_collection.current_matches_collector import CurrentMatchesCollector
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def collect_data_for_date(collector: CurrentMatchesCollector, date: str) -> tuple:
    """Collect both fixtures and odds data for a specific date."""
    logger.info(f"Collecting data for {date}")
    
    # Get fixtures
    fixtures = collector.test_date_request(date)
    if not fixtures:
        logger.warning(f"No fixtures found for {date}")
        return None, None
    
    # Get odds (Bet365 - bookmaker ID 8)
    odds = collector.test_odds_request(date, bookmaker=8)
    if not odds:
        logger.warning(f"No odds found for {date}")
    
    return fixtures, odds

def save_data(fixtures: list, odds: list, date: str, output_dir: Path):
    """Save fixtures and odds data for a specific date."""
    date_dir = output_dir / date
    date_dir.mkdir(parents=True, exist_ok=True)
    
    # Save fixtures
    if fixtures:
        fixtures_file = date_dir / 'fixtures.json'
        with open(fixtures_file, 'w') as f:
            json.dump(fixtures, f, indent=2)
        logger.info(f"Saved {len(fixtures)} fixtures to {fixtures_file}")
    
    # Save odds
    if odds:
        odds_file = date_dir / 'odds.json'
        with open(odds_file, 'w') as f:
            json.dump(odds, f, indent=2)
        logger.info(f"Saved odds for {len(odds)} matches to {odds_file}")

def create_daily_summary(fixtures: list, odds: list) -> dict:
    """Create a summary of collected data for the day."""
    summary = {
        'total_matches': len(fixtures) if fixtures else 0,
        'matches_with_odds': len(odds) if odds else 0,
        'leagues': set(),
        'teams': set()
    }
    
    if fixtures:
        for match in fixtures:
            summary['leagues'].add(match.get('league', {}).get('name'))
            summary['teams'].add(match.get('teams', {}).get('home', {}).get('name'))
            summary['teams'].add(match.get('teams', {}).get('away', {}).get('name'))
    
    summary['leagues'] = list(summary['leagues'])
    summary['teams'] = list(summary['teams'])
    return summary

def main():
    try:
        # Initialize collector
        collector = CurrentMatchesCollector()
        
        # Create output directory
        output_dir = Path('data') / 'historical_api_data'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define date range
        start_date = datetime(2024, 12, 1)
        end_date = datetime.now()
        
        # Collect data for each date
        current_date = start_date
        summaries = {}
        
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            
            # Check if data already exists
            date_dir = output_dir / date_str
            if date_dir.exists():
                logger.info(f"Data already exists for {date_str}, skipping...")
                current_date += timedelta(days=1)
                continue
            
            # Collect data
            fixtures, odds = collect_data_for_date(collector, date_str)
            
            # Save data
            if fixtures or odds:
                save_data(fixtures, odds, date_str, output_dir)
                
                # Create and store summary
                summary = create_daily_summary(fixtures, odds)
                summaries[date_str] = summary
                
                # Save updated summary
                summary_file = output_dir / 'collection_summary.json'
                with open(summary_file, 'w') as f:
                    json.dump(summaries, f, indent=2)
            
            # Sleep to respect API rate limits
            time.sleep(1)
            current_date += timedelta(days=1)
        
        # Create final summary
        total_matches = sum(summary['total_matches'] for summary in summaries.values())
        total_matches_with_odds = sum(summary['matches_with_odds'] for summary in summaries.values())
        all_leagues = set()
        all_teams = set()
        
        for summary in summaries.values():
            all_leagues.update(summary['leagues'])
            all_teams.update(summary['teams'])
        
        final_summary = {
            'date_range': {
                'start': start_date.strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d')
            },
            'total_days': len(summaries),
            'total_matches': total_matches,
            'total_matches_with_odds': total_matches_with_odds,
            'unique_leagues': len(all_leagues),
            'unique_teams': len(all_teams),
            'leagues': list(all_leagues),
            'teams': list(all_teams)
        }
        
        # Save final summary
        final_summary_file = output_dir / 'final_summary.json'
        with open(final_summary_file, 'w') as f:
            json.dump(final_summary, f, indent=2)
        
        logger.info("\nData collection completed!")
        logger.info(f"Total matches collected: {total_matches}")
        logger.info(f"Total matches with odds: {total_matches_with_odds}")
        logger.info(f"Unique leagues: {len(all_leagues)}")
        logger.info(f"Unique teams: {len(all_teams)}")
        logger.info(f"Data saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Error in data collection: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 