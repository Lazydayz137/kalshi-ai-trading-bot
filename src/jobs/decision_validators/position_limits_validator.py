"""
Position Limits Validator - Checks portfolio position limits.
"""

from src.utils.database import DatabaseManager, Market
from src.clients.kalshi_client import KalshiClient
from src.utils.logging_setup import get_trading_logger
from .validation_result import ValidationResult


class PositionLimitsValidator:
    """Validates that adding a new position won't exceed limits."""

    def __init__(self, db_manager: DatabaseManager, kalshi_client: KalshiClient):
        self.db_manager = db_manager
        self.kalshi_client = kalshi_client
        self.logger = get_trading_logger("position_limits_validator")

    async def validate(self, market: Market, position_value: float) -> ValidationResult:
        """
        Check if new position can be added within limits.

        Args:
            market: Market to validate
            position_value: Proposed position value in dollars

        Returns:
            ValidationResult indicating if position can be added
        """
        from src.utils.position_limits import check_can_add_position

        can_add_position, limit_reason = await check_can_add_position(
            position_value, self.db_manager, self.kalshi_client
        )

        if not can_add_position:
            self.logger.info(
                f"Position limits exceeded for {market.market_id}",
                position_value=position_value,
                reason=limit_reason
            )
            return ValidationResult.fail_validation(
                reason=f"Position limits: {limit_reason}",
                metadata={"position_value": position_value}
            )

        return ValidationResult.pass_validation(
            reason="Position limits OK",
            metadata={"position_value": position_value}
        )

    async def validate_with_adjustment(self, market: Market, initial_value: float, price: float) -> ValidationResult:
        """
        Validate position and attempt to adjust size if needed.

        Args:
            market: Market to validate
            initial_value: Initial proposed position value
            price: Price per contract

        Returns:
            ValidationResult with adjusted value if applicable
        """
        from src.utils.position_limits import check_can_add_position, PositionLimitsManager

        # Try initial size
        can_add, reason = await check_can_add_position(initial_value, self.db_manager, self.kalshi_client)
        if can_add:
            return ValidationResult.pass_validation(
                reason="Position limits OK at initial size",
                metadata={"position_value": initial_value, "adjusted": False}
            )

        # Try progressively smaller sizes
        for reduction_factor in [0.8, 0.6, 0.4, 0.2, 0.1]:
            reduced_value = initial_value * reduction_factor
            reduced_quantity = int(reduced_value / price)

            if reduced_quantity < 1:
                break

            can_add_reduced, _ = await check_can_add_position(reduced_value, self.db_manager, self.kalshi_client)

            if can_add_reduced:
                self.logger.info(
                    f"Position size reduced for {market.market_id}",
                    original_value=initial_value,
                    reduced_value=reduced_value,
                    reduction_factor=reduction_factor
                )
                return ValidationResult.pass_validation(
                    reason=f"Position limits OK with {reduction_factor:.0%} size",
                    metadata={
                        "position_value": reduced_value,
                        "adjusted": True,
                        "original_value": initial_value,
                        "reduction_factor": reduction_factor
                    }
                )

        # Check if issue is position count vs size
        limits_manager = PositionLimitsManager(self.db_manager, self.kalshi_client)
        current_positions = await limits_manager._get_position_count()

        if current_positions >= limits_manager.max_positions:
            return ValidationResult.fail_validation(
                reason=f"Position count limit: {current_positions}/{limits_manager.max_positions}",
                metadata={"position_count": current_positions, "max_positions": limits_manager.max_positions}
            )

        return ValidationResult.fail_validation(
            reason="Even minimum position size exceeds limits",
            metadata={"minimum_attempted": initial_value * 0.1}
        )
