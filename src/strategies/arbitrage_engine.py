import asyncio
import logging
import difflib
from typing import List, Optional, Dict
from src.clients.polymarket_client import PolymarketClient, PolymarketEvent

class ArbitrageEngine:
    """
    The 'Political Edge' Engine.
    Responsibilities:
    1. Maintain fresh cache of Polymarket events.
    2. Fuzzy match Kalshi market titles to Polymarket events.
    3. Return the 'Empirical Probability' (Arbitrage Context) for AI analysis.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("trading_system.arbitrage")
        self.client = PolymarketClient()
        self.cached_events: List[PolymarketEvent] = []
        self.last_update = 0
        self.running = False
        
    async def start(self):
        """Start the background ingestion loop."""
        self.running = True
        asyncio.create_task(self._refresh_loop())
        
    async def stop(self):
        self.running = False
        await self.client.close()
        
    async def _refresh_loop(self):
        """Fetch Polymarket data every 60 seconds."""
        while self.running:
            try:
                # Fetch top 100 markets by volume (most likely to have overlap)
                events = await self.client.get_active_events(limit=100)
                if events:
                    self.cached_events = events
                    self.logger.info(f"Updated Arbitrage Cache with {len(events)} Polymarket events.")
                else:
                    self.logger.warning("Arbitrage Cache update failed (0 events).")
            except Exception as e:
                self.logger.error(f"Arbitrage Refresh Error: {e}")
                
            await asyncio.sleep(60) # 1 minute refresh
            
    def get_polymarket_match(self, kalshi_title: str) -> Optional[PolymarketEvent]:
        """
        Find the best matching Polymarket event for a given Kalshi title.
        Uses SequenceMatcher for fuzzy comparison.
        """
        if not self.cached_events:
            return None
            
        best_match = None
        best_ratio = 0.0
        
        # Normalize Kalshi title simplified
        k_norm = self._normalize(kalshi_title)
        
        for event in self.cached_events:
            p_norm = self._normalize(event.title)
            
            # Key optimization: If "Trump" not in both, skip (for political speed)
            # Actually, let's just do full match for now.
            
            ratio = difflib.SequenceMatcher(None, k_norm, p_norm).ratio()
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = event
                
        # Threshold: 0.6 is usually good for "Will X happen?" vs "X?"
        if best_ratio > 0.6:
            self.logger.info(f"Arbitrage Match Found: '{kalshi_title}' ~= '{best_match.title}' ({best_ratio:.2f})")
            return best_match
            
        return None
        
    def _normalize(self, text: str) -> str:
        """Simple normalization for matching."""
        text = text.lower()
        # Remove common filler words
        removals = ["will", "the", "be", "?", " in 2024", " in 2025"]
        for r in removals:
            text = text.replace(r, "")
        return text.strip()

# Global instance
arbitrage_engine = ArbitrageEngine()
