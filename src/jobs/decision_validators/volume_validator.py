"""
Volume Validator - Checks minimum volume requirements.
"""

from src.utils.database import Market
from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger
from .validation_result import ValidationResult


class VolumeValidator:
    """Validates that market has sufficient volume."""

    def __init__(self):
        self.logger = get_trading_logger("volume_validator")

    async def validate(self, market: Market) -> ValidationResult:
        """
        Check if market has sufficient volume for AI analysis.

        Args:
            market: Market to validate

        Returns:
            ValidationResult indicating if volume is sufficient
        """
        if market.volume < settings.trading.min_volume_for_ai_analysis:
            self.logger.info(
                f"Market {market.market_id} volume too low for analysis",
                volume=market.volume,
                min_volume=settings.trading.min_volume_for_ai_analysis
            )
            return ValidationResult.fail_validation(
                reason=f"Insufficient volume: {market.volume} < {settings.trading.min_volume_for_ai_analysis}",
                metadata={"volume": market.volume, "min_volume": settings.trading.min_volume_for_ai_analysis}
            )

        return ValidationResult.pass_validation(
            reason=f"Volume OK: {market.volume} >= {settings.trading.min_volume_for_ai_analysis}",
            metadata={"volume": market.volume}
        )
