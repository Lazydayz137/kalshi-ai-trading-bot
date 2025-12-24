from typing import List, Optional
import logging
from src.utils.database import Market

logger = logging.getLogger("trading_system.filters")

class MarketFilter:
    """
    Heuristic filters to reject "Junk" or "Trap" markets before expensive AI Analysis.
    Saves API costs and prevents stupid losses.
    """
    
    # Keywords that often indicate random/unpredictable/trap markets
    TRAP_KEYWORDS = [
        "tweet", "mention", "say", "posted", # Social media traps
        "taylor swift", "kardashian", "drake", # Celebrity gossip often unpredictable
        "temperature", "degrees", # Weather (unless using weather API)
        "grammys", "oscars", # Award shows (highly subjective/insider info)
        "billboard", "spotify", # Music charts
    ]
    
    @staticmethod
    def is_junk_market(market: Market) -> bool:
        """
        Returns True if the market should be arguably IGNORED to save money/risk.
        """
        title_lower = market.title.lower()
        
        # 1. Social Media / Gossip Traps
        # "Will Elon Musk tweet..." -> Pure gambling, no edge.
        if any(keyword in title_lower for keyword in MarketFilter.TRAP_KEYWORDS):
            logger.info(f"🗑️ REJECTING Junk Market: '{market.title}' (Trap Keyword)")
            return True
            
        # 2. Extreme Spreads (Definition of "No Liquidity" / "Trap")
        # If Bid 1c / Ask 99c, it's a trap or dead market.
        # Calculated in finding logic, but good to reinforce here.
        # (Assuming we pass market object with prices if available, currently mostly metadata)
        
        # 3. Specific silly categories
        if market.category.lower() in ["weather", "entertainment"]:
            # User wants "Real Data". We likely don't have real-time weather/entertainment feeds yet.
            # Better to skip than guess.
            logger.info(f"🗑️ REJECTING Junk Market: '{market.title}' (Category {market.category} has no data feed)")
            return True
            
        return False

    @staticmethod
    def is_low_quality(market: Market, min_volume: float = 10.0) -> bool:
        """Reject mostly dead markets."""
        if market.volume < min_volume:
            logger.debug(f"Rejecting low volume: {market.volume}")
            return True
        return False
