
import aiohttp
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class OddsEvent:
    event_id: str
    sport_key: str
    sport_title: str
    commence_time: str
    home_team: str
    away_team: str
    bookmakers: List[Dict]

class OddsClient:
    """
    Client for The-Odds-API to fetch real-world betting odds for arbitrage.
    """
    BASE_URL = "https://api.the-odds-api.com/v4/sports"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ODDS_API_KEY")
        if not self.api_key:
            logger.warning("ODDS_API_KEY not found. OddsClient will be disabled.")
        
        self.session = None

    async def _get_session(self):
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=10)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session

    async def get_active_sports(self) -> List[Dict]:
        """Fetch list of active sports/leagues."""
        if not self.api_key: return []
        
        url = f"{self.BASE_URL}/"
        params = {"apiKey": self.api_key}
        
        try:
            session = await self._get_session()
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    logger.error(f"Failed to fetch sports: {resp.status} - {await resp.text()}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching sports: {e}")
            return []

    async def get_odds(self, sport_key: str, regions: str = "us", markets: str = "h2h") -> List[OddsEvent]:
        """
        Fetch odds for a specific sport.
        sport_key: e.g., 'basketball_nba', 'politics_us_presidential_election_winner'
        """
        if not self.api_key: return []
        
        url = f"{self.BASE_URL}/{sport_key}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": "decimal"
        }
        
        try:
            session = await self._get_session()
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    events = []
                    for item in data:
                        events.append(OddsEvent(
                            event_id=item.get("id"),
                            sport_key=item.get("sport_key"),
                            sport_title=item.get("sport_title"),
                            commence_time=item.get("commence_time"),
                            home_team=item.get("home_team"),
                            away_team=item.get("away_team"),
                            bookmakers=item.get("bookmakers", [])
                        ))
                    return events
                else:
                    logger.error(f"Failed to fetch odds for {sport_key}: {resp.status}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching odds: {e}")
            return []

    def get_implied_probability(self, price: float) -> float:
        """Convert Decimal odds to Implied Probability."""
        if price <= 0: return 0.0
        return 1 / price

    async def close(self):
        if self.session:
            await self.session.close()
