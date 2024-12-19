import requests
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

logger = logging.getLogger(__name__)

class NesineDataCollector:
    """Data collector for Nesine API."""
    
    def __init__(self):
        self.base_url = "https://bulten.nesine.com/api/bulten/getprebultenfull"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def _fetch_data(self) -> Optional[Dict]:
        """Fetch data from Nesine API."""
        try:
            response = requests.get(self.base_url, headers=self.headers, verify=False)
            response.raise_for_status()
            data = response.json()
            
            # Debug logging
            logger.info("API Response Status Code: %s", response.status_code)
            logger.info("API Response Headers: %s", dict(response.headers))
            logger.info("API Response Content Type: %s", response.headers.get('content-type', 'unknown'))
            logger.info("API Response Structure:")
            logger.info(json.dumps(data, indent=2, ensure_ascii=False)[:1000] + "...")  # First 1000 chars
            
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data from Nesine API: {str(e)}")
            return None
    
    def _parse_match_data(self, raw_data: Dict) -> List[Dict]:
        """Parse raw API data into structured match data."""
        matches = []
        
        try:
            # Get events array from the correct path
            events_data = raw_data.get('sg', {}).get('EA', [])
            
            if events_data:
                logger.info(f"Found {len(events_data)} events in the response")
                
                # Log sample event structure
                if events_data:
                    logger.info("Sample event structure:")
                    logger.info(json.dumps(events_data[0], indent=2, ensure_ascii=False))
            
            for event in events_data:
                try:
                    # Only process football matches (TYPE = 1)
                    if event.get('TYPE') != 1:
                        continue
                    
                    # Convert timestamp to datetime
                    event_date = datetime.fromtimestamp(event.get('ESD', 0)/1000) if event.get('ESD') else None
                    
                    # Get markets data
                    markets = event.get('MA', [])
                    odds_data = {}
                    
                    # Process markets
                    for market in markets:
                        market_id = market.get('MTID')
                        odds = market.get('OCA', [])
                        
                        # Match Result (MTID = 1)
                        if market_id == 1:
                            for odd in odds:
                                if odd.get('N') == 1:  # Home
                                    odds_data['HomeWin'] = odd.get('O')
                                elif odd.get('N') == 0:  # Draw
                                    odds_data['Draw'] = odd.get('O')
                                elif odd.get('N') == 2:  # Away
                                    odds_data['AwayWin'] = odd.get('O')
                        
                        # Over/Under 2.5 (MTID = 2)
                        elif market_id == 2:
                            for odd in odds:
                                if odd.get('N') == 1:  # Over
                                    odds_data['Over25'] = odd.get('O')
                                elif odd.get('N') == 2:  # Under
                                    odds_data['Under25'] = odd.get('O')
                    
                    match_data = {
                        'Date': event_date.strftime('%Y-%m-%d') if event_date else event.get('D'),
                        'Time': event.get('T'),
                        'HomeTeam': event.get('HN'),
                        'AwayTeam': event.get('AN'),
                        'League': event.get('LC'),  # League code
                        'MatchID': event.get('C'),  # Event code
                        'Status': event.get('NS'),  # Match status
                        'EventID': event.get('EV'),
                        'DrawNo': raw_data.get('sg', {}).get('drawNo'),
                        
                        # Odds
                        'HomeWin': odds_data.get('HomeWin'),
                        'Draw': odds_data.get('Draw'),
                        'AwayWin': odds_data.get('AwayWin'),
                        'Over25': odds_data.get('Over25'),
                        'Under25': odds_data.get('Under25'),
                        
                        # Additional metadata
                        'Day': event.get('DAY'),
                        'LeagueType': event.get('LE'),
                        'BetStatus': event.get('BS'),
                        'Priority': event.get('P')
                    }
                    
                    # Only add matches with valid data
                    if match_data['HomeTeam'] and match_data['AwayTeam']:
                        matches.append(match_data)
                        
                except Exception as e:
                    logger.error(f"Error parsing match data: {str(e)}")
                    logger.error(f"Problematic event data: {json.dumps(event, ensure_ascii=False)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error processing API response: {str(e)}")
            logger.error(f"Raw data: {json.dumps(raw_data, ensure_ascii=False)[:1000]}...")
        
        logger.info(f"Successfully parsed {len(matches)} matches")
        return matches
    
    def get_weekly_matches(self) -> Optional[pd.DataFrame]:
        """Get weekly matches from Nesine."""
        try:
            # Fetch raw data
            raw_data = self._fetch_data()
            if not raw_data:
                return None
            
            # Parse matches
            matches = self._parse_match_data(raw_data)
            if not matches:
                logger.error("No matches found in the API response")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(matches)
            
            # Convert date and time
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Sort by date and time
            df = df.sort_values(['Date', 'Time'])
            
            # Add metadata
            df['DataSource'] = 'Nesine'
            df['CollectedAt'] = datetime.now()
            
            logger.info(f"Successfully collected {len(df)} matches from Nesine")
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting weekly matches: {str(e)}")
            return None
    
    def save_matches(self, df: pd.DataFrame, output_path: str):
        """Save matches to CSV file."""
        try:
            df.to_csv(output_path, index=False)
            logger.info(f"Matches saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving matches: {str(e)}")
    
    def get_match_details(self, match_id: str) -> Optional[Dict]:
        """Get detailed odds and statistics for a specific match."""
        try:
            # Note: Update this if there's a separate endpoint for match details
            raw_data = self._fetch_data()
            if not raw_data:
                return None
            
            # Find the specific match in the data
            matches = self._parse_match_data(raw_data)
            match_details = next((m for m in matches if m['MatchID'] == match_id), None)
            
            if match_details:
                return match_details
            else:
                logger.error(f"Match with ID {match_id} not found")
                return None
                
        except Exception as e:
            logger.error(f"Error getting match details: {str(e)}")
            return None 