"""
Strategy Selector - Selects the appropriate decision strategy for a market.
"""

from typing import List
from src.utils.database import Market
from src.utils.logging_setup import get_trading_logger

from .base_strategy import BaseDecisionStrategy
from .high_confidence_strategy import HighConfidenceStrategy
from .standard_strategy import StandardStrategy


class StrategySelector:
    """
    Selects the appropriate decision strategy based on market characteristics.

    Strategies are checked in priority order:
    1. High Confidence Strategy - Near-expiry, high-odds trades
    2. Standard Strategy - Default comprehensive analysis
    """

    def __init__(self):
        self.logger = get_trading_logger("strategy_selector")

        # Initialize strategies in priority order
        self.strategies: List[BaseDecisionStrategy] = [
            HighConfidenceStrategy(),
            StandardStrategy(),  # Catch-all strategy
        ]

    def select_strategy(self, market: Market) -> BaseDecisionStrategy:
        """
        Select the appropriate strategy for the given market.

        Strategies are evaluated in priority order. The first strategy
        that can handle the market is selected.

        Args:
            market: Market to select strategy for

        Returns:
            Selected strategy (guaranteed to return a strategy)
        """
        for strategy in self.strategies:
            if strategy.can_handle(market):
                self.logger.debug(
                    f"Selected {strategy.name} strategy for {market.market_id}",
                    market_title=market.title
                )
                return strategy

        # This should never happen since StandardStrategy is catch-all
        # But as a safety net, return standard strategy
        self.logger.warning(
            f"No strategy found for {market.market_id}, defaulting to standard"
        )
        return self.strategies[-1]  # StandardStrategy

    def get_strategy_by_name(self, name: str) -> BaseDecisionStrategy:
        """
        Get a strategy by name.

        Args:
            name: Strategy name

        Returns:
            Strategy instance

        Raises:
            ValueError: If strategy name not found
        """
        for strategy in self.strategies:
            if strategy.name == name:
                return strategy

        raise ValueError(f"Strategy '{name}' not found")


# Global strategy selector instance
_selector = StrategySelector()


def select_strategy(market: Market) -> BaseDecisionStrategy:
    """
    Convenience function to select strategy for a market.

    Args:
        market: Market to select strategy for

    Returns:
        Selected strategy
    """
    return _selector.select_strategy(market)
