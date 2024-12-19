import requests
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIFootballClient:
    """Client for the API-Football API."""
    
    BASE_URL = "https://api-football-v1.p.rapidapi.com/v3"
    
    def __init__(self):
        load_dotenv()  # Load environment variables
        self.api_key = os.getenv('API_FOOTBALL_KEY')
        
        if not self.api_key:
            raise ValueError("API_FOOTBALL_KEY environment variable not set")
        
        self.headers = {
            'X-RapidAPI-Key': self.api_key,
            'X-RapidAPI-Host': 'api-football-v1.p.rapidapi.com'
        }
        logger.info("API client initialized with key")
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make a request to the API."""
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return None
    
    def get_fixtures(self, league_id: int, season: int) -> Optional[List[Dict]]:
        """Get fixtures for a specific league and season."""
        params = {
            'league': league_id,
            'season': season
        }
        
        response = self._make_request('fixtures', params)
        if response and response.get('response'):
            return response['response']
        return None
    
    def get_fixture_statistics(self, fixture_id: int) -> Optional[Dict]:
        """Get detailed statistics for a specific fixture."""
        params = {'fixture': fixture_id}
        
        response = self._make_request('fixtures/statistics', params)
        if response and response.get('response'):
            return response['response']
        return None
    
    def get_h2h(self, team1_id: int, team2_id: int, limit: int = 5) -> Optional[List[Dict]]:
        """Get head-to-head matches between two teams."""
        params = {
            'h2h': f"{team1_id}-{team2_id}",
            'last': limit
        }
        
        response = self._make_request('fixtures/headtohead', params)
        if response and response.get('response'):
            return response['response']
        return None
    
    def get_team_statistics(self, team_id: int, league_id: int, season: int) -> Optional[Dict]:
        """Get team statistics for a specific season."""
        params = {
            'team': team_id,
            'league': league_id,
            'season': season
        }
        
        response = self._make_request('teams/statistics', params)
        if response and response.get('response'):
            return response['response']
        return None
    
    def get_odds(self, fixture_id: int) -> Optional[List[Dict]]:
        """Get odds for a specific fixture."""
        params = {'fixture': fixture_id}
        
        response = self._make_request('odds', params)
        if response and response.get('response'):
            return response['response']
        return None
    
    def get_players(self, team_id: int, season: int) -> Optional[List[Dict]]:
        """Get player statistics for a team in a specific season."""
        params = {
            'team': team_id,
            'season': season
        }
        
        response = self._make_request('players', params)
        if response and response.get('response'):
            return response['response']
        return None
    
    def get_injuries(self, fixture_id: int) -> Optional[List[Dict]]:
        """Get injuries and suspensions for a fixture."""
        params = {'fixture': fixture_id}
        
        response = self._make_request('injuries', params)
        if response and response.get('response'):
            return response['response']
        return None
    
    def get_league_standings(self, league_id: int, season: int) -> Optional[List[Dict]]:
        """Get current league standings."""
        params = {
            'league': league_id,
            'season': season
        }
        
        response = self._make_request('standings', params)
        if response and response.get('response'):
            return response['response'][0]['league']['standings'][0]
        return None 