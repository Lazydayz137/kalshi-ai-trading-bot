"""
Trading Decision Job - Refactored Version

This is the refactored version of the decision engine with:
- Modular validators for all pre-checks
- Strategy pattern for different decision approaches
- Clean separation of concerns
- Improved testability

Original: 417 lines with nested conditionals
Refactored: ~150 lines with clear structure
"""

from typing import Optional
from datetime import datetime

from src.utils.database import DatabaseManager, Market, Position
from src.clients.xai_client import XAIClient
from src.clients.kalshi_client import KalshiClient
from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger

# Import validators
from src.jobs.decision_validators import (
    BudgetValidator,
    DeduplicationValidator,
    VolumeValidator,
    CategoryValidator,
    ValidationResult,
)

# Import strategies
from src.jobs.decision_strategies import (
    DecisionContext,
    select_strategy,
)


async def make_decision_for_market(
    market: Market,
    db_manager: DatabaseManager,
    xai_client: XAIClient,
    kalshi_client: KalshiClient,
) -> Optional[Position]:
    """
    Analyze a market and make a trading decision.

    This refactored version uses:
    1. Validators for pre-checks (budget, deduplication, volume, category)
    2. Strategy pattern for decision logic (high-confidence vs standard)
    3. Clean separation of concerns

    Args:
        market: Market to analyze
        db_manager: Database manager
        xai_client: XAI client for AI decisions
        kalshi_client: Kalshi API client

    Returns:
        Position if trade decision made, None otherwise
    """
    logger = get_trading_logger("decision_engine_v2")
    logger.info(f"Analyzing market: {market.title} ({market.market_id})")

    # Step 1: Run pre-flight validators
    validation_result = await _run_validators(market, db_manager)
    if validation_result.failed:
        logger.info(
            f"Validation failed for {market.market_id}",
            reason=validation_result.reason
        )
        return None

    # Step 2: Get portfolio balance
    try:
        balance_response = await kalshi_client.get_balance()
        available_balance = balance_response.get("balance", 0) / 100  # Convert cents to dollars
        logger.info(f"Current available balance: ${available_balance:.2f}")
    except Exception as e:
        logger.error(f"Failed to get balance for {market.market_id}", error=str(e))
        return None

    # Step 3: Select appropriate strategy
    strategy = select_strategy(market)
    logger.info(
        f"Selected {strategy.name} strategy for {market.market_id}",
        market_title=market.title
    )

    # Step 4: Create decision context
    context = DecisionContext(
        market=market,
        db_manager=db_manager,
        kalshi_client=kalshi_client,
        xai_client=xai_client,
        available_balance=available_balance
    )

    # Step 5: Execute strategy
    try:
        result = await strategy.decide(context)

        if result.has_position:
            logger.info(
                f"Position created for {market.market_id}",
                strategy=strategy.name,
                side=result.position.side,
                quantity=result.position.quantity,
                confidence=result.confidence,
                cost=result.cost
            )
            return result.position
        else:
            logger.info(
                f"No position for {market.market_id}",
                strategy=strategy.name,
                reasoning=result.reasoning,
                cost=result.cost
            )
            return None

    except Exception as e:
        logger.error(
            f"Strategy execution failed for {market.market_id}",
            strategy=strategy.name,
            error=str(e),
            exc_info=True
        )
        # Record failed analysis
        try:
            await db_manager.record_market_analysis(
                market.market_id,
                "ERROR",
                0.0,
                0.01,
                f"error_{strategy.name}"
            )
        except:
            pass
        return None


async def _run_validators(
    market: Market,
    db_manager: DatabaseManager
) -> ValidationResult:
    """
    Run all pre-flight validators.

    Validators are run in order and execution stops at first failure.
    This provides fast rejection for markets that don't meet basic criteria.

    Args:
        market: Market to validate
        db_manager: Database manager

    Returns:
        ValidationResult (passed or failed with reason)
    """
    logger = get_trading_logger("validators")

    # Create validators
    validators = [
        ("Budget", BudgetValidator(db_manager)),
        ("Deduplication", DeduplicationValidator(db_manager)),
        ("Volume", VolumeValidator()),
        ("Category", CategoryValidator()),
    ]

    # Run each validator
    for name, validator in validators:
        result = await validator.validate(market)

        if result.failed:
            logger.debug(
                f"{name} validation failed for {market.market_id}",
                reason=result.reason,
                metadata=result.metadata
            )
            return result

        logger.debug(
            f"{name} validation passed for {market.market_id}",
            reason=result.reason,
            metadata=result.metadata
        )

    # All validators passed
    return ValidationResult.pass_validation(
        reason="All pre-flight checks passed",
        metadata={"validators_run": len(validators)}
    )


# Backward compatibility - export the same function name
__all__ = ['make_decision_for_market']
