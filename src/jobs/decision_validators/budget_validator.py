"""
Budget Validator - Checks daily AI cost limits.
"""

from src.utils.database import DatabaseManager, Market
from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger
from .validation_result import ValidationResult


class BudgetValidator:
    """Validates that daily AI budget has not been exceeded."""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.logger = get_trading_logger("budget_validator")

    async def validate(self, market: Market) -> ValidationResult:
        """
        Check if daily AI budget allows for more analysis.

        Args:
            market: Market to validate

        Returns:
            ValidationResult indicating if budget allows analysis
        """
        daily_cost = await self.db_manager.get_daily_ai_cost()

        if daily_cost >= settings.trading.daily_ai_budget:
            self.logger.warning(
                f"Daily AI budget of ${settings.trading.daily_ai_budget} exceeded",
                current_cost=daily_cost,
                market_id=market.market_id
            )
            return ValidationResult.fail_validation(
                reason=f"Daily budget exceeded: ${daily_cost:.2f} / ${settings.trading.daily_ai_budget}",
                metadata={"daily_cost": daily_cost, "budget": settings.trading.daily_ai_budget}
            )

        self.logger.debug(
            f"Budget check passed for {market.market_id}",
            daily_cost=daily_cost,
            budget=settings.trading.daily_ai_budget,
            remaining=settings.trading.daily_ai_budget - daily_cost
        )

        return ValidationResult.pass_validation(
            reason=f"Budget OK: ${daily_cost:.2f} / ${settings.trading.daily_ai_budget}",
            metadata={"daily_cost": daily_cost, "remaining": settings.trading.daily_ai_budget - daily_cost}
        )
