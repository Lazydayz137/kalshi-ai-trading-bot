import aiohttp
import asyncio
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import json

@dataclass
class PolymarketEvent:
    title: str
    yes_price: float
    no_price: float
    volume: float
    url: str

class PolymarketClient:
    """
    Client for interacting with the Polymarket Gamma API (Public).
    Used for Arbitrage: Fetching 'True' probabilities for events.
    """
    BASE_URL = "https://gamma-api.polymarket.com"

    def __init__(self):
        self.logger = logging.getLogger("trading_system.polymarket")
        self.session = None

    async def _get_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
                headers={"Accept": "application/json"}
            )
        return self.session

    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None

    async def get_active_events(self, limit: int = 50) -> List[PolymarketEvent]:
        """
        Fetch active markets from Polymarket.
        """
        session = await self._get_session()
        
        # Verified params via test_poly.py: closed=false & sort=volume
        url = f"{self.BASE_URL}/markets?limit={limit}&closed=false&sort=volume"
        
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    self.logger.error(f"Polymarket API Error: {response.status}")
                    return []
                
                data = await response.json()
                events = []
                
                for item in data:
                    try:
                        # Item is now the market itself
                        outcomes = json.loads(item.get('outcomes', '[]')) if isinstance(item.get('outcomes'), str) else item.get('outcomes', [])
                        prices = json.loads(item.get('outcomePrices', '[]')) if isinstance(item.get('outcomePrices'), str) else item.get('outcomePrices', [])
                        
                        if not prices:
                            continue

                        # Validate it's a binary Yes/No (approximate)
                        if len(outcomes) != 2:
                            continue
                            
                        # Normalize Yes/No order
                        yes_index = -1
                        no_index = -1
                        
                        for i, outcome in enumerate(outcomes):
                            if outcome == "Yes": yes_index = i
                            if outcome == "No": no_index = i
                                
                        if yes_index == -1 or no_index == -1:
                            continue

                        try:
                            yes_price = float(prices[yes_index])
                            no_price = float(prices[no_index])
                        except (ValueError, IndexError):
                            continue
                        
                        # Market title is often in 'question'
                        title = item.get('question', item.get('title', 'Unknown'))
                            
                        events.append(PolymarketEvent(
                            title=title,
                            yes_price=yes_price,
                            no_price=no_price,
                            volume=float(item.get('volume', 0)),
                            url=f"https://polymarket.com/event/{item.get('slug', '')}"
                        ))
                    except Exception as e:
                        # self.logger.error(f"Error parsing market: {e}")
                        continue
                        
                self.logger.info(f"Fetched {len(events)} active markets from Polymarket.")
                return events
                
        except asyncio.TimeoutError:
            self.logger.warning("Polymarket API Request Timed Out")
            return []
        except Exception as e:
            self.logger.error(f"Polymarket Client Error: {e}")
            return []

if __name__ == "__main__":
    # fast test
    async def main():
        logging.basicConfig(level=logging.INFO)
        client = PolymarketClient()
        events = await client.get_active_events(limit=10)
        for e in events:
            print(f"{e.title[:50]}... | YES: {e.yes_price:.2f} | NO: {e.no_price:.2f} | Vol: ${e.volume:,.0f}")
        await client.close()
        
    asyncio.run(main())
