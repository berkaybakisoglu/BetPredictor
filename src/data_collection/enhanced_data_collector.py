import pandas as pd
import logging
from typing import Dict, List, Optional
from datetime import datetime
from .football_data_collector import FootballDataCollector
from .api_football_client import APIFootballClient
from src.utils.config import LEAGUES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDataCollector:
    """Combines historical data with real-time API data."""
    
    # League ID mapping for API-Football
    LEAGUE_MAPPING = {
        'EPL': 39,      # Premier League
        'LaLiga': 140,  # La Liga
        'Bundesliga': 78,  # Bundesliga
        'SerieA': 135,  # Serie A
        'Ligue1': 61    # Ligue 1
    }
    
    def __init__(self):
        self.historical_collector = FootballDataCollector()
        self.api_client = APIFootballClient()
    
    def get_historical_data(self, league: str, seasons: List[str]) -> Optional[pd.DataFrame]:
        """Get historical data from football-data.co.uk."""
        logger.info(f"Collecting historical data for {league} seasons {seasons}")
        return self.historical_collector.combine_seasons_data(league, seasons)
    
    def get_team_ids(self, league_id: int, season: int) -> Dict[str, int]:
        """Get mapping of team names to API team IDs."""
        fixtures = self.api_client.get_fixtures(league_id, season)
        if not fixtures:
            return {}
        
        team_ids = {}
        for fixture in fixtures:
            home_team = fixture['teams']['home']
            away_team = fixture['teams']['away']
            team_ids[home_team['name']] = home_team['id']
            team_ids[away_team['name']] = away_team['id']
        
        return team_ids
    
    def enhance_with_api_data(self, df: pd.DataFrame, league: str, season: int) -> pd.DataFrame:
        """Add additional features from API data."""
        logger.info(f"Enhancing data with API features for {league} season {season}")
        
        # Get API league ID
        league_id = self.LEAGUE_MAPPING.get(league)
        if not league_id:
            logger.error(f"No API league ID mapping found for {league}")
            return df
        
        # Get team IDs
        team_ids = self.get_team_ids(league_id, season)
        if not team_ids:
            logger.error("Failed to get team IDs")
            return df
        
        # Initialize new columns
        df['home_team_rank'] = None
        df['away_team_rank'] = None
        df['home_team_form_api'] = None
        df['away_team_form_api'] = None
        df['home_team_injured_players'] = 0
        df['away_team_injured_players'] = 0
        
        # Process each match
        for idx, row in df.iterrows():
            try:
                home_team_id = team_ids.get(row['HomeTeam'])
                away_team_id = team_ids.get(row['AwayTeam'])
                
                if not (home_team_id and away_team_id):
                    continue
                
                # Get team statistics
                home_stats = self.api_client.get_team_statistics(home_team_id, league_id, season)
                away_stats = self.api_client.get_team_statistics(away_team_id, league_id, season)
                
                if home_stats and away_stats:
                    # Add team rankings
                    standings = self.api_client.get_league_standings(league_id, season)
                    if standings:
                        for team in standings:
                            if team['team']['id'] == home_team_id:
                                df.at[idx, 'home_team_rank'] = team['rank']
                            if team['team']['id'] == away_team_id:
                                df.at[idx, 'away_team_rank'] = team['rank']
                    
                    # Add form (last 5 matches)
                    if 'form' in home_stats:
                        df.at[idx, 'home_team_form_api'] = home_stats['form']
                    if 'form' in away_stats:
                        df.at[idx, 'away_team_form_api'] = away_stats['form']
                
                # Get H2H statistics
                h2h = self.api_client.get_h2h(home_team_id, away_team_id)
                if h2h:
                    df.at[idx, 'h2h_total'] = len(h2h)
                    df.at[idx, 'h2h_home_wins'] = sum(1 for match in h2h 
                        if match['teams']['home']['id'] == home_team_id 
                        and match['teams']['home']['winner'])
                    df.at[idx, 'h2h_away_wins'] = sum(1 for match in h2h 
                        if match['teams']['away']['id'] == away_team_id 
                        and match['teams']['away']['winner'])
                
            except Exception as e:
                logger.error(f"Error processing match {idx}: {str(e)}")
                continue
        
        logger.info("API data enhancement completed")
        return df
    
    def collect_enhanced_data(self, league: str, seasons: List[str]) -> Optional[pd.DataFrame]:
        """Collect and combine both historical and API data."""
        try:
            # Get historical data first
            df = self.get_historical_data(league, seasons)
            if df is None:
                return None
            
            # Enhance with API data for the most recent season
            latest_season = max(seasons)
            season_year = int('20' + latest_season[:2])  # Convert '2223' to 2022
            df = self.enhance_with_api_data(df, league, season_year)
            
            return df
            
        except Exception as e:
            logger.error(f"Error collecting enhanced data: {str(e)}")
            return None 