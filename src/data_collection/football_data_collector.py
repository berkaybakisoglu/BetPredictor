import pandas as pd
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
import logging
from src.utils.config import FOOTBALL_DATA_URL, RAW_DATA_DIR
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FootballDataCollector:
    def __init__(self):
        self.base_url = FOOTBALL_DATA_URL
        self.raw_data_dir = RAW_DATA_DIR
        self._setup_directories()

    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

    def _get_local_files(self) -> List[str]:
        """Get list of local Excel files."""
        return [f.name for f in self.raw_data_dir.glob("*.xlsx")]

    def _load_local_data(self, league: str, season: str) -> Optional[pd.DataFrame]:
        """Try to load data from local Excel file."""
        # Convert season format (e.g., '2223' to '2022-2023')
        if len(season) == 4:  # Format like '2223'
            start_year = '20' + season[:2]
            end_year = '20' + season[2:]
            season_format = f"{start_year}-{end_year}"
        else:
            season_format = season

        excel_file = self.raw_data_dir / f"all-euro-data-{season_format}.xlsx"
        
        if excel_file.exists():
            logger.info(f"Loading data from Excel file: {excel_file}")
            try:
                # Read all sheets from Excel file
                df = pd.read_excel(excel_file)
                
                # Filter for the specific league
                if 'League' in df.columns:
                    league_name = self._map_league_code_to_name(league)
                    df = df[df['League'] == league_name]
                    
                if not df.empty:
                    logger.info(f"Found {len(df)} matches for {league} in {season_format}")
                    return df
                else:
                    logger.warning(f"No data found for {league} in {season_format}")
                    return None
                    
            except Exception as e:
                logger.error(f"Error reading Excel file {excel_file}: {str(e)}")
                return None
        
        logger.warning(f"Excel file not found: {excel_file}")
        return None

    def _map_league_code_to_name(self, league_code: str) -> str:
        """Map league codes to full names as they appear in the Excel files."""
        mapping = {
            'E0': 'Premier League',
            'SP1': 'La Liga',
            'D1': 'Bundesliga',
            'I1': 'Serie A',
            'F1': 'Ligue 1'
        }
        return mapping.get(league_code, league_code)

    def download_season_data(self, league: str, season: str) -> Optional[pd.DataFrame]:
        """
        Get data for a specific league and season from local Excel files.
        
        Args:
            league: League code (e.g., 'E0' for EPL)
            season: Season (e.g., '2223' for 2022/23)
        """
        return self._load_local_data(league, season)

    def download_multiple_seasons(self, league: str, seasons: List[str]) -> Dict[str, pd.DataFrame]:
        """Get data for multiple seasons of a league."""
        data_dict = {}
        for season in seasons:
            df = self.download_season_data(league, season)
            if df is not None:
                data_dict[season] = df
        return data_dict

    def combine_seasons_data(self, league: str, seasons: List[str]) -> Optional[pd.DataFrame]:
        """Combine data from multiple seasons into a single DataFrame."""
        data_dict = self.download_multiple_seasons(league, seasons)
        if not data_dict:
            return None
        
        combined_df = pd.concat(data_dict.values(), ignore_index=True)
        output_path = self.raw_data_dir / f"{league}_combined.csv"
        combined_df.to_csv(output_path, index=False)
        logger.info(f"Combined data saved to {output_path}")
        logger.info(f"Total matches in combined dataset: {len(combined_df)}")
        
        return combined_df