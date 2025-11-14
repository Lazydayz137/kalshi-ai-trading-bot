"""
Base Repository - Abstract interface for database operations.

This provides a database-agnostic interface that can be implemented
for different databases (SQLite, PostgreSQL, etc.).
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime

from src.utils.database import Market, Position, TradeLog, LLMQuery


class BaseRepository(ABC):
    """
    Abstract base repository for database operations.

    Implementations:
    - SQLiteRepository - For development/testing
    - PostgreSQLRepository - For production
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize database schema and run migrations."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close database connections."""
        pass

    # Market operations
    @abstractmethod
    async def upsert_markets(self, markets: List[Market]) -> None:
        """Upsert a list of markets."""
        pass

    @abstractmethod
    async def get_eligible_markets(self, volume_min: int, max_days_to_expiry: int) -> List[Market]:
        """Get markets eligible for trading."""
        pass

    @abstractmethod
    async def get_markets_with_positions(self) -> set:
        """Get set of market IDs with open positions."""
        pass

    # Position operations
    @abstractmethod
    async def add_position(self, position: Position) -> Optional[int]:
        """Add a new position."""
        pass

    @abstractmethod
    async def get_position_by_market_and_side(self, market_id: str, side: str) -> Optional[Position]:
        """Get position by market ID and side."""
        pass

    @abstractmethod
    async def get_open_positions(self) -> List[Position]:
        """Get all open positions."""
        pass

    @abstractmethod
    async def get_open_live_positions(self) -> List[Position]:
        """Get all open live positions."""
        pass

    @abstractmethod
    async def get_open_non_live_positions(self) -> List[Position]:
        """Get all open non-live positions."""
        pass

    @abstractmethod
    async def update_position_to_live(self, position_id: int, entry_price: float) -> None:
        """Update position to live status."""
        pass

    @abstractmethod
    async def update_position_status(self, position_id: int, status: str) -> None:
        """Update position status."""
        pass

    # Trade log operations
    @abstractmethod
    async def add_trade_log(self, trade_log: TradeLog) -> None:
        """Add a trade log entry."""
        pass

    @abstractmethod
    async def get_all_trade_logs(self) -> List[TradeLog]:
        """Get all trade logs."""
        pass

    @abstractmethod
    async def get_performance_by_strategy(self) -> Dict[str, Dict]:
        """Get performance metrics by strategy."""
        pass

    # LLM query operations
    @abstractmethod
    async def log_llm_query(self, llm_query: LLMQuery) -> None:
        """Log an LLM query."""
        pass

    @abstractmethod
    async def get_llm_queries(
        self,
        strategy: Optional[str] = None,
        hours_back: int = 24,
        limit: int = 100
    ) -> List[LLMQuery]:
        """Get recent LLM queries."""
        pass

    @abstractmethod
    async def get_llm_stats_by_strategy(self) -> Dict[str, Dict]:
        """Get LLM usage statistics by strategy."""
        pass

    # Analysis tracking operations
    @abstractmethod
    async def record_market_analysis(
        self,
        market_id: str,
        decision_action: str,
        confidence: float,
        cost_usd: float,
        analysis_type: str = "standard"
    ) -> None:
        """Record market analysis."""
        pass

    @abstractmethod
    async def was_recently_analyzed(self, market_id: str, hours: int = 6) -> bool:
        """Check if market was recently analyzed."""
        pass

    @abstractmethod
    async def get_daily_ai_cost(self, date: str = None) -> float:
        """Get total AI cost for a date."""
        pass

    @abstractmethod
    async def get_market_analysis_count_today(self, market_id: str) -> int:
        """Get number of analyses for market today."""
        pass
