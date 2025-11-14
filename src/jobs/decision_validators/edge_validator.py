"""
Edge Validator - Validates minimum edge requirements for trades.
"""

from src.utils.database import Market
from src.utils.logging_setup import get_trading_logger
from .validation_result import ValidationResult


class EdgeValidator:
    """Validates that trade has sufficient edge."""

    def __init__(self):
        self.logger = get_trading_logger("edge_validator")

    async def validate(
        self,
        market: Market,
        ai_probability: float,
        market_probability: float,
        confidence: float,
        side: str
    ) -> ValidationResult:
        """
        Check if trade has sufficient edge.

        Args:
            market: Market to validate
            ai_probability: AI-estimated probability
            market_probability: Market-implied probability
            confidence: AI confidence level
            side: Trade side (YES/NO)

        Returns:
            ValidationResult indicating if edge is sufficient
        """
        from src.utils.edge_filter import EdgeFilter
        from src.config.settings import settings

        # Calculate time to expiry
        import time
        time_to_expiry_days = (market.expiration_ts - time.time()) / 86400

        should_trade, trade_reason, edge_result = EdgeFilter.should_trade_market(
            ai_probability=ai_probability,
            market_probability=market_probability,
            confidence=confidence,
            additional_filters={
                'volume': market.volume,
                'min_volume': settings.trading.min_volume,
                'time_to_expiry_days': time_to_expiry_days,
                'max_time_to_expiry': settings.trading.max_time_to_expiry_days
            }
        )

        if not should_trade:
            self.logger.info(
                f"Edge filter rejected {market.market_id}",
                reason=trade_reason,
                edge_result=edge_result
            )
            return ValidationResult.fail_validation(
                reason=f"Insufficient edge: {trade_reason}",
                metadata={
                    "ai_probability": ai_probability,
                    "market_probability": market_probability,
                    "confidence": confidence,
                    "edge_result": edge_result
                }
            )

        self.logger.info(
            f"Edge filter approved {market.market_id}",
            reason=trade_reason,
            edge_result=edge_result
        )

        return ValidationResult.pass_validation(
            reason=f"Edge approved: {trade_reason}",
            metadata={
                "ai_probability": ai_probability,
                "market_probability": market_probability,
                "confidence": confidence,
                "edge_result": edge_result
            }
        )
