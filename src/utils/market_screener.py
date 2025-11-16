"""
Advanced Market Screener

This module provides sophisticated market screening and filtering capabilities
to identify the best trading opportunities based on multiple criteria.

Features:
- Multi-factor screening (volume, volatility, edge, time to expiry)
- Custom scoring algorithms
- Portfolio optimization integration
- Real-time market ranking
- Anomaly detection for unusual opportunities
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import asyncio

from ..models import Market
from ..config.settings import settings
from .logger import get_logger

logger = get_logger(__name__)


class ScreeningCriteria(Enum):
    """Market screening criteria types."""

    VOLUME = "volume"  # High trading volume
    LIQUIDITY = "liquidity"  # Tight bid-ask spreads
    VOLATILITY = "volatility"  # Price movement
    EDGE = "edge"  # Perceived mispricing
    TIME_VALUE = "time_value"  # Time until expiration
    MOMENTUM = "momentum"  # Recent price trends
    CORRELATION = "correlation"  # Correlation with other markets


@dataclass
class ScreeningConfig:
    """Configuration for market screening."""

    # Volume criteria
    min_volume_24h: int = 100
    min_open_interest: int = 1000

    # Liquidity criteria
    max_spread_pct: float = 5.0  # Maximum bid-ask spread percentage

    # Edge criteria
    min_edge: float = 0.05  # Minimum 5% edge
    edge_confidence: float = 0.7  # Minimum confidence in edge calculation

    # Time criteria
    min_hours_to_expiry: int = 24
    max_days_to_expiry: int = 90

    # Category filtering
    allowed_categories: List[str] = field(
        default_factory=lambda: [
            "crypto",
            "finance",
            "economics",
            "politics",
            "sports",
        ]
    )
    excluded_categories: List[str] = field(default_factory=list)

    # Volatility criteria
    min_daily_range_pct: float = 2.0  # Minimum price movement

    # Scoring weights (must sum to 1.0)
    volume_weight: float = 0.20
    liquidity_weight: float = 0.15
    edge_weight: float = 0.30
    time_value_weight: float = 0.15
    momentum_weight: float = 0.10
    volatility_weight: float = 0.10

    # Result limits
    max_results: int = 50
    min_score: float = 0.5  # Minimum composite score


@dataclass
class MarketScore:
    """Composite score for a market."""

    ticker: str
    title: str
    composite_score: float
    volume_score: float
    liquidity_score: float
    edge_score: float
    time_value_score: float
    momentum_score: float
    volatility_score: float
    reasoning: str = ""
    metadata: Dict = field(default_factory=dict)


class MarketScreener:
    """
    Advanced market screener for identifying high-quality trading opportunities.

    This screener evaluates markets across multiple dimensions and ranks them
    by a composite score to prioritize the best opportunities.
    """

    def __init__(self, config: Optional[ScreeningConfig] = None):
        """
        Initialize market screener.

        Args:
            config: Screening configuration (uses default if None)
        """
        self.config = config or ScreeningConfig()
        self._validate_config()

    def _validate_config(self):
        """Validate screening configuration."""
        # Check weights sum to 1.0
        total_weight = (
            self.config.volume_weight
            + self.config.liquidity_weight
            + self.config.edge_weight
            + self.config.time_value_weight
            + self.config.momentum_weight
            + self.config.volatility_weight
        )

        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Screening weights sum to {total_weight}, normalizing to 1.0")
            # Normalize weights
            norm_factor = 1.0 / total_weight
            self.config.volume_weight *= norm_factor
            self.config.liquidity_weight *= norm_factor
            self.config.edge_weight *= norm_factor
            self.config.time_value_weight *= norm_factor
            self.config.momentum_weight *= norm_factor
            self.config.volatility_weight *= norm_factor

    async def screen_markets(
        self,
        markets: List[Market],
        additional_criteria: Optional[Dict] = None,
    ) -> List[MarketScore]:
        """
        Screen and rank markets based on multiple criteria.

        Args:
            markets: List of markets to screen
            additional_criteria: Optional additional screening criteria

        Returns:
            List of MarketScore objects ranked by composite score
        """
        logger.info(f"Screening {len(markets)} markets with {len(self.config.allowed_categories)} allowed categories")

        scored_markets = []

        for market in markets:
            # Apply basic filters first
            if not self._passes_basic_filters(market):
                continue

            # Calculate composite score
            score = await self._calculate_market_score(market)

            if score.composite_score >= self.config.min_score:
                scored_markets.append(score)

        # Sort by composite score descending
        scored_markets.sort(key=lambda x: x.composite_score, reverse=True)

        # Limit results
        scored_markets = scored_markets[: self.config.max_results]

        logger.info(
            f"Screening complete: {len(scored_markets)} markets passed filters "
            f"(avg score: {sum(s.composite_score for s in scored_markets) / len(scored_markets):.3f})"
            if scored_markets
            else "Screening complete: 0 markets passed filters"
        )

        return scored_markets

    def _passes_basic_filters(self, market: Market) -> bool:
        """
        Check if market passes basic filtering criteria.

        Args:
            market: Market to evaluate

        Returns:
            True if market passes all basic filters
        """
        # Category filtering
        if self.config.allowed_categories and market.category not in self.config.allowed_categories:
            return False

        if market.category in self.config.excluded_categories:
            return False

        # Volume filtering
        if market.volume_24h < self.config.min_volume_24h:
            return False

        if market.open_interest < self.config.min_open_interest:
            return False

        # Time filtering
        if market.close_time:
            hours_to_expiry = (market.close_time - datetime.utcnow()).total_seconds() / 3600

            if hours_to_expiry < self.config.min_hours_to_expiry:
                return False

            if hours_to_expiry > (self.config.max_days_to_expiry * 24):
                return False

        # Liquidity filtering (spread check)
        spread_pct = self._calculate_spread_pct(market)
        if spread_pct > self.config.max_spread_pct:
            return False

        return True

    async def _calculate_market_score(self, market: Market) -> MarketScore:
        """
        Calculate comprehensive score for a market.

        Args:
            market: Market to score

        Returns:
            MarketScore with component scores and composite
        """
        # Calculate individual component scores (0-1 scale)
        volume_score = self._score_volume(market)
        liquidity_score = self._score_liquidity(market)
        edge_score = await self._score_edge(market)
        time_value_score = self._score_time_value(market)
        momentum_score = self._score_momentum(market)
        volatility_score = self._score_volatility(market)

        # Calculate weighted composite score
        composite = (
            volume_score * self.config.volume_weight
            + liquidity_score * self.config.liquidity_weight
            + edge_score * self.config.edge_weight
            + time_value_score * self.config.time_value_weight
            + momentum_score * self.config.momentum_weight
            + volatility_score * self.config.volatility_weight
        )

        # Generate reasoning
        reasoning = self._generate_reasoning(
            market,
            volume_score,
            liquidity_score,
            edge_score,
            time_value_score,
            momentum_score,
            volatility_score,
        )

        return MarketScore(
            ticker=market.ticker,
            title=market.title,
            composite_score=composite,
            volume_score=volume_score,
            liquidity_score=liquidity_score,
            edge_score=edge_score,
            time_value_score=time_value_score,
            momentum_score=momentum_score,
            volatility_score=volatility_score,
            reasoning=reasoning,
            metadata={"category": market.category, "volume_24h": market.volume_24h},
        )

    def _score_volume(self, market: Market) -> float:
        """
        Score market based on trading volume.

        Higher volume = better liquidity and less slippage.
        """
        # Logarithmic scaling for volume
        # 100 volume = 0.0, 10,000 volume = 0.5, 1,000,000 volume = 1.0
        import math

        if market.volume_24h <= 0:
            return 0.0

        # Use log scale with floor and ceiling
        min_log = math.log10(max(1, self.config.min_volume_24h))
        max_log = math.log10(1_000_000)  # 1M volume = perfect score

        volume_log = math.log10(market.volume_24h)
        score = (volume_log - min_log) / (max_log - min_log)

        return max(0.0, min(1.0, score))

    def _score_liquidity(self, market: Market) -> float:
        """
        Score market based on bid-ask spread.

        Tighter spreads = better liquidity = higher score.
        """
        spread_pct = self._calculate_spread_pct(market)

        # Perfect score (1.0) at 0.5% spread or less
        # Zero score (0.0) at max_spread_pct or more
        if spread_pct <= 0.5:
            return 1.0

        score = 1.0 - ((spread_pct - 0.5) / (self.config.max_spread_pct - 0.5))
        return max(0.0, min(1.0, score))

    async def _score_edge(self, market: Market) -> float:
        """
        Score market based on perceived edge/mispricing.

        This is a placeholder - in practice, you would integrate your
        edge calculation algorithm here.
        """
        # Placeholder implementation
        # In real implementation, this would call your edge calculation model

        # For now, use a simple heuristic based on volume and spread
        spread_pct = self._calculate_spread_pct(market)

        # Markets with wide spreads and low volume might indicate inefficiency
        if spread_pct > 3.0 and market.volume_24h < 1000:
            return 0.7  # Potential opportunity

        # Markets with tight spreads and high volume are efficient
        if spread_pct < 1.0 and market.volume_24h > 10000:
            return 0.3  # Less opportunity

        return 0.5  # Neutral

    def _score_time_value(self, market: Market) -> float:
        """
        Score market based on time to expiration.

        Sweet spot is typically 7-30 days:
        - Too soon: Not enough time for thesis to play out
        - Too far: Less certainty, more premium decay
        """
        if not market.close_time:
            return 0.5  # Neutral score if no expiry

        hours_to_expiry = (market.close_time - datetime.utcnow()).total_seconds() / 3600
        days_to_expiry = hours_to_expiry / 24

        # Optimal range: 7-30 days
        if 7 <= days_to_expiry <= 30:
            return 1.0

        # Good range: 3-7 days or 30-60 days
        if (3 <= days_to_expiry < 7) or (30 < days_to_expiry <= 60):
            return 0.7

        # Acceptable: 1-3 days or 60-90 days
        if (1 <= days_to_expiry < 3) or (60 < days_to_expiry <= 90):
            return 0.4

        # Poor: Less than 1 day or more than 90 days
        return 0.2

    def _score_momentum(self, market: Market) -> float:
        """
        Score market based on price momentum.

        Strong momentum can indicate trend continuation opportunities.
        """
        # Placeholder implementation
        # In real implementation, this would track price changes over time

        # For now, use market price as proxy
        # Markets trading near extremes (0.05 or 0.95) have strong momentum
        price = market.last_price

        if price < 0.1 or price > 0.9:
            return 0.8  # Strong momentum

        if 0.1 <= price < 0.3 or 0.7 < price <= 0.9:
            return 0.6  # Moderate momentum

        return 0.3  # Low momentum (near 0.5)

    def _score_volatility(self, market: Market) -> float:
        """
        Score market based on price volatility.

        Higher volatility = more trading opportunities.
        """
        # Placeholder implementation
        # In real implementation, this would calculate historical volatility

        # Use spread as proxy for volatility
        spread_pct = self._calculate_spread_pct(market)

        # Higher spreads often correlate with higher volatility
        if spread_pct > 3.0:
            return 0.8

        if spread_pct > 1.5:
            return 0.6

        return 0.4

    def _calculate_spread_pct(self, market: Market) -> float:
        """Calculate bid-ask spread as percentage of mid-price."""
        spread = market.yes_ask - market.yes_bid
        mid_price = (market.yes_bid + market.yes_ask) / 2

        if mid_price <= 0:
            return 100.0  # Invalid market

        return (spread / mid_price) * 100

    def _generate_reasoning(
        self,
        market: Market,
        volume_score: float,
        liquidity_score: float,
        edge_score: float,
        time_value_score: float,
        momentum_score: float,
        volatility_score: float,
    ) -> str:
        """Generate human-readable reasoning for the score."""
        reasons = []

        # Identify top factors
        scores = {
            "volume": volume_score,
            "liquidity": liquidity_score,
            "edge": edge_score,
            "time": time_value_score,
            "momentum": momentum_score,
            "volatility": volatility_score,
        }

        sorted_factors = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Highlight top 2 factors
        if sorted_factors[0][1] >= 0.7:
            factor_name = sorted_factors[0][0].title()
            reasons.append(f"Strong {factor_name.lower()}")

        if sorted_factors[1][1] >= 0.6:
            factor_name = sorted_factors[1][0].title()
            reasons.append(f"Good {factor_name.lower()}")

        # Note any weaknesses
        weak_factors = [name for name, score in sorted_factors if score < 0.3]
        if weak_factors:
            reasons.append(f"Weak {', '.join(weak_factors)}")

        # Add volume and time context
        if market.volume_24h > 10000:
            reasons.append(f"High volume ({market.volume_24h:,})")

        if market.close_time:
            days = (market.close_time - datetime.utcnow()).days
            if days <= 7:
                reasons.append(f"Expires soon ({days}d)")

        return " • ".join(reasons) if reasons else "Market meets basic criteria"

    async def find_best_opportunities(
        self, markets: List[Market], limit: int = 10
    ) -> List[Tuple[Market, MarketScore]]:
        """
        Find the best trading opportunities from a list of markets.

        Args:
            markets: List of markets to evaluate
            limit: Maximum number of opportunities to return

        Returns:
            List of (Market, MarketScore) tuples for top opportunities
        """
        scored_markets = await self.screen_markets(markets)

        # Match scores back to original markets
        ticker_to_market = {m.ticker: m for m in markets}

        opportunities = []
        for score in scored_markets[:limit]:
            if score.ticker in ticker_to_market:
                opportunities.append((ticker_to_market[score.ticker], score))

        return opportunities

    def print_top_markets(self, scored_markets: List[MarketScore], n: int = 10):
        """Print top N markets in a formatted table."""
        print("\n" + "=" * 120)
        print(f"TOP {n} MARKETS BY COMPOSITE SCORE")
        print("=" * 120)
        print(
            f"{'Ticker':<20} {'Score':<8} {'Vol':<6} {'Liq':<6} {'Edge':<6} "
            f"{'Time':<6} {'Mom':<6} {'Reasoning':<40}"
        )
        print("-" * 120)

        for score in scored_markets[:n]:
            print(
                f"{score.ticker:<20} {score.composite_score:>6.3f} "
                f"{score.volume_score:>6.2f} {score.liquidity_score:>6.2f} "
                f"{score.edge_score:>6.2f} {score.time_value_score:>6.2f} "
                f"{score.momentum_score:>6.2f} {score.reasoning:<40}"
            )

        print("=" * 120 + "\n")


class AnomalyDetector:
    """
    Detects unusual market conditions that may present trading opportunities.

    Examples of anomalies:
    - Sudden volume spikes
    - Large price movements
    - Unusual spread widening
    - Markets trading at extreme probabilities (< 5% or > 95%)
    """

    def __init__(self):
        """Initialize anomaly detector."""
        self.baseline_volumes: Dict[str, float] = {}
        self.baseline_spreads: Dict[str, float] = {}

    async def detect_anomalies(self, markets: List[Market]) -> List[Tuple[Market, str, float]]:
        """
        Detect anomalous market conditions.

        Args:
            markets: List of markets to analyze

        Returns:
            List of (Market, anomaly_type, severity) tuples
        """
        anomalies = []

        for market in markets:
            # Volume spike detection
            if market.ticker in self.baseline_volumes:
                baseline = self.baseline_volumes[market.ticker]
                if market.volume_24h > baseline * 3:  # 3x normal volume
                    severity = min(1.0, market.volume_24h / baseline / 10)
                    anomalies.append((market, "volume_spike", severity))

            # Extreme probability detection
            if market.last_price < 0.05:
                anomalies.append((market, "extreme_low_probability", 0.9))
            elif market.last_price > 0.95:
                anomalies.append((market, "extreme_high_probability", 0.9))

            # Wide spread detection
            spread_pct = (market.yes_ask - market.yes_bid) / market.last_price * 100
            if spread_pct > 10:  # More than 10% spread
                anomalies.append((market, "wide_spread", min(1.0, spread_pct / 20)))

            # Update baselines
            self.baseline_volumes[market.ticker] = market.volume_24h
            self.baseline_spreads[market.ticker] = spread_pct

        return anomalies
