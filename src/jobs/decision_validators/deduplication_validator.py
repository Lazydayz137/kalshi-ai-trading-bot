"""
Deduplication Validator - Prevents analyzing same market too frequently.
"""

from src.utils.database import DatabaseManager, Market
from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger
from .validation_result import ValidationResult


class DeduplicationValidator:
    """Validates that market hasn't been analyzed recently."""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.logger = get_trading_logger("deduplication_validator")

    async def validate(self, market: Market) -> ValidationResult:
        """
        Check if market was recently analyzed.

        Args:
            market: Market to validate

        Returns:
            ValidationResult indicating if market can be analyzed
        """
        # Check recent analysis
        if await self.db_manager.was_recently_analyzed(
            market.market_id,
            settings.trading.analysis_cooldown_hours
        ):
            self.logger.info(
                f"Market {market.market_id} was recently analyzed, skipping",
                cooldown_hours=settings.trading.analysis_cooldown_hours
            )
            return ValidationResult.fail_validation(
                reason=f"Recently analyzed (< {settings.trading.analysis_cooldown_hours}h ago)",
                metadata={"cooldown_hours": settings.trading.analysis_cooldown_hours}
            )

        # Check daily analysis limit
        analysis_count_today = await self.db_manager.get_market_analysis_count_today(market.market_id)
        if analysis_count_today >= settings.trading.max_analyses_per_market_per_day:
            self.logger.info(
                f"Market {market.market_id} analyzed {analysis_count_today} times today, skipping",
                limit=settings.trading.max_analyses_per_market_per_day
            )
            return ValidationResult.fail_validation(
                reason=f"Daily analysis limit reached: {analysis_count_today}/{settings.trading.max_analyses_per_market_per_day}",
                metadata={"analysis_count": analysis_count_today, "limit": settings.trading.max_analyses_per_market_per_day}
            )

        return ValidationResult.pass_validation(
            reason=f"Deduplication OK: {analysis_count_today}/{settings.trading.max_analyses_per_market_per_day} analyses today",
            metadata={"analysis_count": analysis_count_today}
        )
