"""
Standard Decision Strategy - Full AI-powered market analysis.

This strategy performs comprehensive analysis including:
- Market data fetching
- News search (with optimization for low volume)
- Full AI reasoning
- Edge filtering
- Position limits checking
- Cash reserves validation
"""

import asyncio
from datetime import datetime
from typing import Optional

from src.utils.database import Market, Position
from src.config.settings import settings
from .base_strategy import BaseDecisionStrategy, DecisionContext, DecisionResult


class StandardStrategy(BaseDecisionStrategy):
    """
    Standard strategy for comprehensive market analysis.

    This is the default strategy that performs full AI-powered
    analysis with all checks and validations.
    """

    def __init__(self):
        super().__init__("standard")

    def can_handle(self, market: Market) -> bool:
        """
        Standard strategy can handle any market.

        Args:
            market: Market to check

        Returns:
            Always returns True (catch-all strategy)
        """
        return True

    async def decide(self, context: DecisionContext) -> DecisionResult:
        """
        Make standard trading decision with full analysis.

        Process:
        1. Fetch detailed market data
        2. Perform news search (with cost optimization)
        3. Get AI trading decision
        4. Apply edge filter
        5. Check position limits with adjustment
        6. Validate cash reserves
        7. Create position with exit strategy

        Args:
            context: Decision context

        Returns:
            DecisionResult with position or no action
        """
        market = context.market
        self.logger.info(f"Standard strategy analyzing {market.market_id}")

        total_cost = 0.0

        # Step 1: Fetch detailed market data
        try:
            full_market_data_response = await context.kalshi_client.get_market(market.market_id)
            full_market_data = full_market_data_response.get("market", {})
            rules = full_market_data.get("rules", "No rules available.")

            market_data = {
                "ticker": market.market_id,
                "title": market.title,
                "rules": rules,
                "yes_price": market.yes_price,
                "no_price": market.no_price,
                "volume": market.volume,
                "expiration_ts": market.expiration_ts,
            }
        except Exception as e:
            self.logger.error(f"Error fetching market data for {market.market_id}", error=str(e))
            return DecisionResult.no_action(
                reasoning=f"Failed to fetch market data: {str(e)}",
                cost=total_cost
            )

        # Step 2: News search with cost optimization
        news_summary = await self._get_news_summary(context, market)
        if news_summary.startswith("Search timeout") or news_summary.startswith("News search unavailable"):
            total_cost += 0.0  # No cost for failed search
        elif news_summary.startswith("Low volume market"):
            total_cost += 0.0  # No cost for skipped search
        else:
            total_cost += 0.02  # Estimated search cost

        # Step 3: Check if we're approaching cost limits
        if total_cost > settings.trading.cost_control.max_ai_cost_per_decision:
            self.logger.warning(
                f"Analysis cost ${total_cost:.3f} exceeds per-decision limit",
                market_id=market.market_id
            )
            await self._record_analysis(context, "SKIP", 0.0, total_cost, "cost_limited")
            return DecisionResult.no_action(
                reasoning=f"Cost ${total_cost:.3f} exceeds limit",
                cost=total_cost
            )

        # Step 4: Get AI decision
        try:
            decision = await context.xai_client.get_trading_decision(
                market_data=market_data,
                portfolio_data=context.portfolio_data,
                news_summary=news_summary
            )

            # Estimate decision cost
            estimated_decision_cost = 0.015
            total_cost += estimated_decision_cost

        except Exception as e:
            self.logger.error(f"Error getting AI decision for {market.market_id}", error=str(e))
            await self._record_analysis(context, "ERROR", 0.0, total_cost, "standard")
            return DecisionResult.no_action(
                reasoning=f"AI decision error: {str(e)}",
                cost=total_cost
            )

        if not decision:
            await self._record_analysis(context, "SKIP", 0.0, total_cost, "no_decision")
            return DecisionResult.no_action(
                reasoning="No decision from AI",
                cost=total_cost
            )

        self.logger.info(
            f"AI decision for {market.market_id}",
            action=decision.action,
            side=decision.side,
            confidence=decision.confidence,
            cost=total_cost
        )

        # Step 5: Check if decision meets confidence threshold
        if decision.action != "BUY" or decision.confidence < settings.trading.ai_model.min_confidence_to_trade:
            await self._record_analysis(context, decision.action, decision.confidence, total_cost, "standard")
            return DecisionResult.no_action(
                reasoning=f"Decision: {decision.action}, confidence: {decision.confidence:.2%}",
                cost=total_cost,
                metadata={"action": decision.action, "confidence": decision.confidence}
            )

        # Step 6: Apply edge filter
        from src.jobs.decision_validators import EdgeValidator

        price = market.yes_price if decision.side == "YES" else market.no_price
        market_prob = price
        ai_prob = decision.confidence

        edge_validator = EdgeValidator()
        edge_result = await edge_validator.validate(
            market=market,
            ai_probability=ai_prob,
            market_probability=market_prob,
            confidence=decision.confidence,
            side=decision.side
        )

        if edge_result.failed:
            self.logger.info(f"Edge filter rejected {market.market_id}", reason=edge_result.reason)
            await self._record_analysis(context, "EDGE_FILTERED", decision.confidence, total_cost, edge_result.reason)
            return DecisionResult.no_action(
                reasoning=f"Edge filter: {edge_result.reason}",
                cost=total_cost,
                metadata=edge_result.metadata
            )

        # Step 7: Calculate position size and check limits
        confidence_delta = decision.confidence - price
        initial_quantity = self._calculate_dynamic_quantity(
            context.available_balance,
            price,
            confidence_delta
        )
        initial_position_value = initial_quantity * price

        # Step 8: Check position limits with adjustment
        from src.jobs.decision_validators import PositionLimitsValidator

        position_validator = PositionLimitsValidator(context.db_manager, context.kalshi_client)
        position_result = await position_validator.validate_with_adjustment(
            market=market,
            initial_value=initial_position_value,
            price=price
        )

        if position_result.failed:
            self.logger.info(f"Position limits exceeded for {market.market_id}", reason=position_result.reason)
            await self._record_analysis(context, "POSITION_LIMITS", decision.confidence, total_cost, position_result.reason)
            return DecisionResult.no_action(
                reasoning=f"Position limits: {position_result.reason}",
                cost=total_cost,
                metadata=position_result.metadata
            )

        # Use adjusted position value if applicable
        final_position_value = position_result.metadata.get("position_value", initial_position_value)
        final_quantity = int(final_position_value / price)

        # Step 9: Check cash reserves
        from src.jobs.decision_validators import CashReservesValidator

        cash_validator = CashReservesValidator(context.db_manager, context.kalshi_client)
        cash_result = await cash_validator.validate(
            market=market,
            trade_value=final_position_value
        )

        if cash_result.failed:
            self.logger.info(f"Insufficient cash for {market.market_id}", reason=cash_result.reason)
            await self._record_analysis(context, "CASH_RESERVES", decision.confidence, total_cost, cash_result.reason)
            return DecisionResult.no_action(
                reasoning=f"Cash reserves: {cash_result.reason}",
                cost=total_cost,
                metadata=cash_result.metadata
            )

        # Step 10: Record successful analysis
        await self._record_analysis(context, "BUY", decision.confidence, total_cost, "standard")

        # Step 11: Calculate exit strategy
        exit_strategy = self._calculate_exit_strategy(
            entry_price=price,
            side=decision.side,
            confidence=decision.confidence,
            market=market
        )

        # Step 12: Create position
        rationale = getattr(decision, 'reasoning', 'No reasoning provided by LLM.')
        position = Position(
            market_id=market.market_id,
            side=decision.side,
            entry_price=price,
            quantity=final_quantity,
            timestamp=datetime.now(),
            rationale=rationale,
            confidence=decision.confidence,
            live=False,
            strategy="directional_trading",
            stop_loss_price=exit_strategy['stop_loss_price'],
            take_profit_price=exit_strategy['take_profit_price'],
            max_hold_hours=exit_strategy['max_hold_hours'],
            target_confidence_change=exit_strategy['target_confidence_change']
        )

        self.logger.info(
            f"Standard position created for {market.market_id}",
            side=decision.side,
            quantity=final_quantity,
            price=price,
            confidence=decision.confidence,
            adjusted=position_result.metadata.get("adjusted", False)
        )

        return DecisionResult.create_position(
            position=position,
            reasoning=rationale,
            confidence=decision.confidence,
            cost=total_cost,
            metadata={
                "entry_price": price,
                "quantity": final_quantity,
                "position_adjusted": position_result.metadata.get("adjusted", False),
                "edge_approved": True
            }
        )

    async def _get_news_summary(self, context: DecisionContext, market: Market) -> str:
        """
        Get news summary with cost optimization.

        Args:
            context: Decision context
            market: Market to search news for

        Returns:
            News summary string
        """
        # Skip news for low volume markets
        if (settings.trading.cost_control.skip_news_for_low_volume and
            market.volume < settings.trading.cost_control.news_search_volume_threshold):
            self.logger.info(
                f"Skipping news search for low volume market {market.market_id}",
                volume=market.volume
            )
            return f"Low volume market ({market.volume}). Analysis based on market data only."

        # Perform news search with timeout
        try:
            news_summary = await asyncio.wait_for(
                context.xai_client.search(market.title, max_length=200),
                timeout=15.0
            )
            return news_summary
        except asyncio.TimeoutError:
            self.logger.warning(f"Search timeout for market {market.market_id}")
            return f"Search timeout. Analyzing {market.title} based on market data only."
        except Exception as e:
            self.logger.warning(f"Search failed for market {market.market_id}", error=str(e))
            return f"News search unavailable. Analysis based on market data only."
