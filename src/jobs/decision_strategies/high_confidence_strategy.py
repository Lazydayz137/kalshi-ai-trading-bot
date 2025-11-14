"""
High Confidence Strategy - Near-expiry high-confidence trades.

This strategy focuses on markets that are:
- Near expiry (< 24 hours typically)
- Have high market odds (> 90% typically)
- Can be analyzed quickly without expensive news searches
"""

import time
from datetime import datetime
from typing import Optional

from src.utils.database import Market, Position
from src.config.settings import settings
from .base_strategy import BaseDecisionStrategy, DecisionContext, DecisionResult


class HighConfidenceStrategy(BaseDecisionStrategy):
    """
    Strategy for high-confidence, near-expiry opportunities.

    This strategy looks for markets that are:
    1. Close to expiration (hours not days)
    2. Have very high market odds (90%+)
    3. Can benefit from quick AI confirmation
    4. Lower analysis cost (skip expensive news searches)
    """

    def __init__(self):
        super().__init__("high_confidence")

    def can_handle(self, market: Market) -> bool:
        """
        Check if this strategy can handle the market.

        Criteria:
        - High confidence strategy enabled
        - Near expiry (within configured hours)
        - High market odds (above threshold)

        Args:
            market: Market to check

        Returns:
            True if this strategy should handle the market
        """
        if not settings.trading.strategy_allocation.enable_high_confidence_strategy:
            return False

        # Check time to expiry
        hours_to_expiry = (market.expiration_ts - time.time()) / 3600
        if hours_to_expiry > settings.trading.dynamic_exit.high_confidence_expiry_hours:
            return False

        # Check if market has high odds on either side
        high_yes_odds = market.yes_price >= settings.trading.ai_model.high_confidence_market_odds
        high_no_odds = market.no_price >= settings.trading.ai_model.high_confidence_market_odds

        return high_yes_odds or high_no_odds

    async def decide(self, context: DecisionContext) -> DecisionResult:
        """
        Make high-confidence trading decision.

        Process:
        1. Skip expensive news search (near expiry, time sensitive)
        2. Get quick AI confirmation
        3. If AI agrees with market odds at high confidence, trade
        4. Use aggressive position sizing (high confidence)

        Args:
            context: Decision context

        Returns:
            DecisionResult with position or no action
        """
        market = context.market
        hours_to_expiry = (market.expiration_ts - time.time()) / 3600

        self.logger.info(
            f"High confidence strategy analyzing {market.market_id}",
            hours_to_expiry=hours_to_expiry,
            yes_price=market.yes_price,
            no_price=market.no_price
        )

        total_cost = 0.0

        # Skip expensive news search for high-confidence near-expiry
        news_summary = f"Near-expiry high-confidence analysis. Market at YES:{market.yes_price:.2f} NO:{market.no_price:.2f}"

        # Quick AI decision
        try:
            decision = await context.xai_client.get_trading_decision(
                market_data={
                    "title": market.title,
                    "yes_price": market.yes_price,
                    "no_price": market.no_price,
                    "volume": market.volume,
                    "expiration_ts": market.expiration_ts
                },
                portfolio_data=context.portfolio_data,
                news_summary=news_summary
            )

            # Estimate cost (typically lower for high-confidence strategy)
            estimated_cost = 0.01
            total_cost += estimated_cost

        except Exception as e:
            self.logger.error(f"Error getting AI decision for {market.market_id}", error=str(e))
            await self._record_analysis(context, "ERROR", 0.0, total_cost, "high_confidence")
            return DecisionResult.no_action(
                reasoning=f"AI decision error: {str(e)}",
                cost=total_cost
            )

        if not decision:
            await self._record_analysis(context, "SKIP", 0.0, total_cost, "high_confidence")
            return DecisionResult.no_action(
                reasoning="No decision from AI",
                cost=total_cost
            )

        # Check if AI confidence is high enough
        if decision.confidence < settings.trading.ai_model.high_confidence_threshold:
            self.logger.info(
                f"AI confidence too low for high-confidence strategy",
                confidence=decision.confidence,
                threshold=settings.trading.ai_model.high_confidence_threshold
            )
            await self._record_analysis(context, "LOW_CONFIDENCE", decision.confidence, total_cost, "high_confidence")
            return DecisionResult.no_action(
                reasoning=f"Confidence {decision.confidence:.2%} below threshold {settings.trading.ai_model.high_confidence_threshold:.2%}",
                cost=total_cost,
                metadata={"confidence": decision.confidence}
            )

        # Check if AI agrees with the trade direction
        if decision.action != "BUY" or decision.side not in ["YES", "NO"]:
            await self._record_analysis(context, decision.action, decision.confidence, total_cost, "high_confidence")
            return DecisionResult.no_action(
                reasoning=f"AI decision: {decision.action} {decision.side}",
                cost=total_cost,
                metadata={"action": decision.action, "side": decision.side}
            )

        # Record successful analysis
        await self._record_analysis(context, "BUY", decision.confidence, total_cost, "high_confidence")

        # Calculate position size
        price = market.yes_price if decision.side == "YES" else market.no_price
        confidence_delta = decision.confidence - price
        quantity = self._calculate_dynamic_quantity(
            context.available_balance,
            price,
            confidence_delta
        )

        if quantity <= 0:
            return DecisionResult.no_action(
                reasoning="Calculated quantity is 0",
                cost=total_cost
            )

        # Calculate exit strategy
        exit_strategy = self._calculate_exit_strategy(
            entry_price=price,
            side=decision.side,
            confidence=decision.confidence,
            market=market
        )

        # Create position
        position = Position(
            market_id=market.market_id,
            side=decision.side,
            entry_price=price,
            quantity=quantity,
            timestamp=datetime.now(),
            rationale=f"High-confidence near-expiry {decision.side} bet. Hours to expiry: {hours_to_expiry:.1f}",
            confidence=decision.confidence,
            live=False,
            strategy="high_confidence",
            stop_loss_price=exit_strategy['stop_loss_price'],
            take_profit_price=exit_strategy['take_profit_price'],
            max_hold_hours=exit_strategy['max_hold_hours'],
            target_confidence_change=exit_strategy['target_confidence_change']
        )

        self.logger.info(
            f"High-confidence position created for {market.market_id}",
            side=decision.side,
            quantity=quantity,
            price=price,
            confidence=decision.confidence
        )

        return DecisionResult.create_position(
            position=position,
            reasoning=f"High-confidence {decision.side} trade near expiry",
            confidence=decision.confidence,
            cost=total_cost,
            metadata={
                "hours_to_expiry": hours_to_expiry,
                "entry_price": price,
                "quantity": quantity
            }
        )
