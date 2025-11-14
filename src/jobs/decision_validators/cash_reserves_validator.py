"""
Cash Reserves Validator - Ensures sufficient cash for trade.
"""

from src.utils.database import DatabaseManager, Market
from src.clients.kalshi_client import KalshiClient
from src.utils.logging_setup import get_trading_logger
from .validation_result import ValidationResult


class CashReservesValidator:
    """Validates that sufficient cash reserves exist for trade."""

    def __init__(self, db_manager: DatabaseManager, kalshi_client: KalshiClient):
        self.db_manager = db_manager
        self.kalshi_client = kalshi_client
        self.logger = get_trading_logger("cash_reserves_validator")

    async def validate(self, market: Market, trade_value: float) -> ValidationResult:
        """
        Check if sufficient cash is available for trade.

        Args:
            market: Market to validate
            trade_value: Required cash for trade

        Returns:
            ValidationResult indicating if cash is sufficient
        """
        from src.utils.cash_reserves import check_can_trade_with_cash_reserves

        can_trade, cash_reason = await check_can_trade_with_cash_reserves(
            trade_value, self.db_manager, self.kalshi_client
        )

        if not can_trade:
            self.logger.info(
                f"Insufficient cash reserves for {market.market_id}",
                trade_value=trade_value,
                reason=cash_reason
            )
            return ValidationResult.fail_validation(
                reason=f"Cash reserves: {cash_reason}",
                metadata={"trade_value": trade_value}
            )

        return ValidationResult.pass_validation(
            reason=f"Cash reserves OK: {cash_reason}",
            metadata={"trade_value": trade_value}
        )
