"""
Category Validator - Filters markets by category.
"""

from src.utils.database import Market
from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger
from .validation_result import ValidationResult


class CategoryValidator:
    """Validates that market category is not excluded."""

    def __init__(self):
        self.logger = get_trading_logger("category_validator")

    async def validate(self, market: Market) -> ValidationResult:
        """
        Check if market category is allowed.

        Args:
            market: Market to validate

        Returns:
            ValidationResult indicating if category is allowed
        """
        excluded_categories = settings.trading.exclude_low_liquidity_categories

        if market.category.lower() in [cat.lower() for cat in excluded_categories]:
            self.logger.info(
                f"Market {market.market_id} in excluded category",
                category=market.category
            )
            return ValidationResult.fail_validation(
                reason=f"Excluded category: {market.category}",
                metadata={"category": market.category, "excluded_categories": excluded_categories}
            )

        return ValidationResult.pass_validation(
            reason=f"Category OK: {market.category}",
            metadata={"category": market.category}
        )
