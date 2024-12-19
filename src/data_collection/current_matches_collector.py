import pandas as pd
import requests
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Optional, List, Dict
import json
import os
from dotenv import load_dotenv
import time

logger = logging.getLogger(__name__)

class CurrentMatchesCollector:
    """Collects current week's matches using API-Football."""
    
    def __init__(self):
        load_dotenv()
        self.api_key = '7892e9eb18f0c970784ee77562f4f496'  # Using the provided API key
        
        self.base_url = "https://v3.football.api-sports.io"
        self.headers = {
            'x-apisports-key': self.api_key
        }
        
        # Test API connection
        if not self._test_connection():
            raise ValueError("Failed to connect to API-Football. Please check your API key.")
        
        # League IDs and season (using 2022 for free plan compatibility)
        self.league_ids = {
            'E0': {'id': 39, 'season': 2022},  # Premier League
            'E1': {'id': 40, 'season': 2022},  # Championship
            'SP1': {'id': 140, 'season': 2022},  # La Liga
            'I1': {'id': 135, 'season': 2022},  # Serie A
            'D1': {'id': 78, 'season': 2022},  # Bundesliga
            'F1': {'id': 61, 'season': 2022}   # Ligue 1
        }
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1  # Minimum seconds between requests
    
    def _test_connection(self) -> bool:
        """Test the API connection with a simple status request."""
        try:
            url = f"{self.base_url}/status"
            response = requests.get(url, headers=self.headers, verify=False)
            
            if response.status_code == 200:
                data = response.json()
                logger.info("Successfully connected to API-Football")
                logger.info(f"Account: {data.get('response', {}).get('account', {}).get('email', 'Unknown')}")
                logger.info(f"Requests remaining: {data.get('response', {}).get('requests', {}).get('current', 'Unknown')}")
                return True
            else:
                logger.error(f"Failed to connect to API. Status code: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error testing API connection: {str(e)}")
            return False
    
    def _make_request(self, url: str, params: Dict = None) -> Optional[Dict]:
        """Make a rate-limited API request."""
        try:
            # Rate limiting
            current_time = time.time()
            time_since_last_request = current_time - self.last_request_time
            if time_since_last_request < self.min_request_interval:
                time.sleep(self.min_request_interval - time_since_last_request)
            
            # Make request
            logger.info(f"Making API request to: {url}")
            logger.info(f"With params: {params}")
            # Don't log headers as they contain the API key
            
            response = requests.get(url, headers=self.headers, params=params, verify=False)
            self.last_request_time = time.time()
            
            # Log response details for debugging
            logger.info(f"Response status code: {response.status_code}")
            
            if response.status_code != 200:
                logger.error(f"API Error Response: {response.text}")
                if response.status_code == 403:
                    logger.error("API key invalid or expired. Please check your API key.")
                    return None
                elif response.status_code == 429:
                    logger.error("Rate limit exceeded. Waiting before retrying...")
                    time.sleep(60)  # Wait a minute before retrying
                    return self._make_request(url, params)  # Retry the request
                else:
                    response.raise_for_status()
            
            data = response.json()
            
            # Log API response for debugging
            if data.get('errors'):
                logger.error(f"API Errors: {json.dumps(data['errors'], indent=2)}")
            elif data.get('response') is None:
                logger.error(f"Unexpected API Response: {json.dumps(data, indent=2)}")
            else:
                logger.info(f"API Response contains {len(data['response'])} items")
            
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response text: {e.response.text}")
            return None
    
    def _fetch_matches(self, from_date: str, to_date: str, league_ids: List[str] = None) -> List[Dict]:
        """Fetch matches from API-Football."""
        try:
            all_matches = []
            
            # If no league IDs specified, use all major leagues
            if league_ids is None:
                league_ids = list(self.league_ids.keys())
            
            for league_code in league_ids:
                if league_code not in self.league_ids:
                    logger.warning(f"Unknown league code: {league_code}")
                    continue
                
                league_info = self.league_ids[league_code]
                url = f"{self.base_url}/fixtures"
                params = {
                    'league': league_info['id'],
                    'season': league_info['season'],
                    'from': from_date,
                    'to': to_date,
                    'timezone': 'Europe/London'
                }
                
                data = self._make_request(url, params)
                if data and data.get('response'):
                    all_matches.extend(data['response'])
                    logger.info(f"Found {len(data['response'])} matches for league {league_code}")
                else:
                    logger.warning(f"No data returned for league {league_code}")
            
            return all_matches
        except Exception as e:
            logger.error(f"Error fetching matches: {str(e)}")
            return []
    
    def _convert_to_dataframe(self, matches: List[Dict]) -> pd.DataFrame:
        """Convert API response to DataFrame matching our training data format."""
        try:
            formatted_matches = []
            
            for match in matches:
                # Get league code
                league_code = next((code for code, info in self.league_ids.items() 
                                  if info['id'] == match['league']['id']), 'Unknown')
                
                # Get odds if available
                odds = {}
                if 'odds' in match and match['odds'].get('bookmakers'):
                    for bookmaker in match['odds']['bookmakers']:
                        if bookmaker['name'] == 'Bet365':
                            for bet in bookmaker['bets']:
                                if bet['name'] == 'Match Winner':
                                    for value in bet['values']:
                                        if value['value'] == 'Home':
                                            odds['B365H'] = float(value['odd'])
                                        elif value['value'] == 'Draw':
                                            odds['B365D'] = float(value['odd'])
                                        elif value['value'] == 'Away':
                                            odds['B365A'] = float(value['odd'])
                
                match_data = {
                    'Date': pd.to_datetime(match['fixture']['date']).strftime('%Y-%m-%d'),
                    'Time': pd.to_datetime(match['fixture']['date']).strftime('%H:%M'),
                    'HomeTeam': match['teams']['home']['name'],
                    'AwayTeam': match['teams']['away']['name'],
                    'League': league_code,
                    'B365H': odds.get('B365H'),
                    'B365D': odds.get('B365D'),
                    'B365A': odds.get('B365A'),
                    'MatchID': match['fixture']['id'],
                    'Status': match['fixture']['status']['short'],
                    'Venue': match['fixture'].get('venue', {}).get('name', 'Unknown'),
                    'Round': match['league']['round']
                }
                formatted_matches.append(match_data)
            
            df = pd.DataFrame(formatted_matches)
            return df
            
        except Exception as e:
            logger.error(f"Error converting matches to DataFrame: {str(e)}")
            return pd.DataFrame()
    
    def get_current_week_matches(self, leagues: List[str] = None) -> Optional[pd.DataFrame]:
        """Get matches for the current week."""
        try:
            # Use dates from 2022 season (last week of the season)
            start_date = "2022-05-16"
            end_date = "2022-05-22"
            
            # Fetch matches
            matches = self._fetch_matches(start_date, end_date, leagues)
            
            if matches:
                df = self._convert_to_dataframe(matches)
                df = df.sort_values(['Date', 'Time'])
                
                # Add metadata
                df['DataSource'] = 'API-Football'
                df['CollectedAt'] = datetime.now()
                
                logger.info(f"Successfully collected {len(df)} matches for current week")
                return df
            else:
                logger.warning("No matches found for the current week")
                return None
            
        except Exception as e:
            logger.error(f"Error getting current week matches: {str(e)}")
            return None
    
    def get_upcoming_matches(self, days_ahead: int = 7) -> Optional[pd.DataFrame]:
        """Get upcoming matches for the next N days."""
        try:
            # Use dates from 2022 season (last week of the season)
            start_date = "2022-05-16"
            end_date = "2022-05-22"
            
            # Fetch matches
            matches = self._fetch_matches(start_date, end_date)
            
            if matches:
                df = self._convert_to_dataframe(matches)
                df = df.sort_values(['Date', 'Time'])
                
                # Add metadata
                df['DataSource'] = 'API-Football'
                df['CollectedAt'] = datetime.now()
                
                logger.info(f"Successfully collected {len(df)} upcoming matches")
                return df
            else:
                logger.warning("No upcoming matches found")
                return None
            
        except Exception as e:
            logger.error(f"Error getting upcoming matches: {str(e)}")
            return None
    
    def save_matches(self, df: pd.DataFrame, output_path: str):
        """Save matches to CSV file."""
        try:
            df.to_csv(output_path, index=False)
            logger.info(f"Matches saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving matches: {str(e)}")
    
    def test_date_request(self, date: str = "2024-12-17"):
        """Test a simple date-based fixtures request."""
        try:
            url = f"{self.base_url}/fixtures"
            params = {
                'date': date
            }
            
            logger.info(f"Making test request for date: {date}")
            data = self._make_request(url, params)
            
            if data and data.get('response'):
                logger.info(f"Found {len(data['response'])} matches for {date}")
                return data['response']
            else:
                logger.warning(f"No matches found for {date}")
                return None
                
        except Exception as e:
            logger.error(f"Error in test request: {str(e)}")
            return None
    
    def test_odds_request(self, date: str = "2024-12-17", bookmaker: int = 8):
        """Test odds request for a specific date and bookmaker (8 is Bet365)."""
        try:
            url = f"{self.base_url}/odds"
            params = {
                'date': date,
                'bookmaker': bookmaker,
                'timezone': 'Europe/London'
            }
            
            logger.info(f"Making odds request for date: {date}, bookmaker: {bookmaker}")
            logger.info(f"Request URL: {url}")
            logger.info(f"Request params: {params}")
            
            data = self._make_request(url, params)
            
            if data:
                logger.info(f"Response received: {json.dumps(data.get('errors') or {}, indent=2)}")
                
                if data.get('response'):
                    matches_with_odds = len(data['response'])
                    logger.info(f"Found odds for {matches_with_odds} matches")
                    
                    # Log sample of first match odds if available
                    if matches_with_odds > 0:
                        sample_match = data['response'][0]
                        logger.info("\nSample match odds:")
                        logger.info(f"Match: {sample_match.get('fixture', {}).get('home_team')} vs {sample_match.get('fixture', {}).get('away_team')}")
                        for bookmaker in sample_match.get('bookmakers', []):
                            logger.info(f"Bookmaker: {bookmaker.get('name')}")
                            for bet in bookmaker.get('bets', []):
                                logger.info(f"- Market: {bet.get('name')}")
                                for value in bet.get('values', []):
                                    logger.info(f"  {value.get('value')}: {value.get('odd')}")
                    
                    return data['response']
                else:
                    logger.warning(f"No odds found for {date}")
                    return None
            else:
                logger.error("No data received from API")
                return None
                
        except Exception as e:
            logger.error(f"Error in odds request: {str(e)}")
            if hasattr(e, 'response'):
                logger.error(f"Response text: {e.response.text}")
            return None 