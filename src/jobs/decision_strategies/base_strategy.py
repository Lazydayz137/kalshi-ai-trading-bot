"""
Base Decision Strategy - Abstract base class for all decision strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime

from src.utils.database import DatabaseManager, Market, Position
from src.clients.kalshi_client import KalshiClient
from src.clients.xai_client import XAIClient
from src.utils.logging_setup import get_trading_logger


@dataclass
class DecisionContext:
    """
    Context for making trading decisions.
    Contains all data needed by decision strategies.
    """
    market: Market
    db_manager: DatabaseManager
    kalshi_client: KalshiClient
    xai_client: XAIClient
    available_balance: float

    # Additional context that may be provided
    market_data: Optional[Dict[str, Any]] = None
    portfolio_data: Optional[Dict[str, Any]] = None
    news_summary: Optional[str] = None

    def __post_init__(self):
        """Initialize portfolio data if not provided."""
        if self.portfolio_data is None:
            self.portfolio_data = {"available_balance": self.available_balance}


@dataclass
class DecisionResult:
    """
    Result of a decision strategy.
    Contains position to be created or None if no action.
    """
    position: Optional[Position]
    reasoning: str
    confidence: float
    cost: float  # Cost incurred making this decision
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_position(self) -> bool:
        """Check if decision resulted in a position."""
        return self.position is not None

    @classmethod
    def no_action(cls, reasoning: str, cost: float = 0.0, **kwargs) -> 'DecisionResult':
        """Create a result with no position."""
        return cls(
            position=None,
            reasoning=reasoning,
            confidence=0.0,
            cost=cost,
            **kwargs
        )

    @classmethod
    def create_position(
        cls,
        position: Position,
        reasoning: str,
        confidence: float,
        cost: float,
        **kwargs
    ) -> 'DecisionResult':
        """Create a result with a position."""
        return cls(
            position=position,
            reasoning=reasoning,
            confidence=confidence,
            cost=cost,
            **kwargs
        )


class BaseDecisionStrategy(ABC):
    """
    Abstract base class for all decision strategies.

    Each strategy implements a specific approach to analyzing markets
    and generating trading decisions.
    """

    def __init__(self, name: str):
        """
        Initialize the strategy.

        Args:
            name: Name of the strategy for logging
        """
        self.name = name
        self.logger = get_trading_logger(f"strategy_{name}")

    @abstractmethod
    async def decide(self, context: DecisionContext) -> DecisionResult:
        """
        Make a trading decision for the given market.

        Args:
            context: Decision context with all necessary data

        Returns:
            DecisionResult containing position or no action
        """
        pass

    @abstractmethod
    def can_handle(self, market: Market) -> bool:
        """
        Check if this strategy can handle the given market.

        Args:
            market: Market to check

        Returns:
            True if strategy can handle this market
        """
        pass

    def _calculate_dynamic_quantity(
        self,
        balance: float,
        market_price: float,
        confidence_delta: float,
    ) -> int:
        """
        Calculate trade quantity based on balance and confidence delta.

        Args:
            balance: Available portfolio balance
            market_price: Price of the contract
            confidence_delta: Difference between AI confidence and market price

        Returns:
            Number of contracts to purchase
        """
        from src.config.settings import settings

        if market_price <= 0:
            return 0

        # Use a percentage of the balance for the trade
        base_investment_pct = settings.trading.position_sizing.max_position_size_pct / 100

        # Scale investment by confidence delta
        investment_scaler = 1 + confidence_delta
        investment_amount = (balance * base_investment_pct) * investment_scaler

        # Apply max position size limit
        max_investment = (balance * settings.trading.position_sizing.max_position_size_pct) / 100
        final_investment = min(investment_amount, max_investment)

        quantity = int(final_investment // market_price)

        self.logger.debug(
            "Calculated position size",
            investment_amount=final_investment,
            quantity=quantity,
            confidence_delta=confidence_delta
        )

        return quantity

    def _calculate_exit_strategy(
        self,
        entry_price: float,
        side: str,
        confidence: float,
        market: Market
    ) -> Dict[str, Any]:
        """
        Calculate exit strategy using stop loss calculator.

        Args:
            entry_price: Entry price for the position
            side: Side of the trade (YES/NO)
            confidence: AI confidence level
            market: Market being traded

        Returns:
            Dictionary with exit strategy parameters
        """
        from src.utils.stop_loss_calculator import StopLossCalculator
        from src.jobs.decide import estimate_market_volatility, get_time_to_expiry_days

        return StopLossCalculator.calculate_stop_loss_levels(
            entry_price=entry_price,
            side=side,
            confidence=confidence,
            market_volatility=estimate_market_volatility(market),
            time_to_expiry_days=get_time_to_expiry_days(market)
        )

    async def _record_analysis(
        self,
        context: DecisionContext,
        decision_action: str,
        confidence: float,
        cost: float,
        analysis_type: str = "standard"
    ) -> None:
        """
        Record the analysis in the database.

        Args:
            context: Decision context
            decision_action: Action taken (BUY, SKIP, etc.)
            confidence: Confidence level
            cost: Cost of analysis
            analysis_type: Type of analysis performed
        """
        await context.db_manager.record_market_analysis(
            context.market.market_id,
            decision_action,
            confidence,
            cost,
            analysis_type
        )
