"""
Performance Attribution Module

This module provides detailed analysis of trading strategy performance,
breaking down returns by various factors to understand what drives profitability.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict

from .framework import Trade, Position
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CategoryAttribution:
    """Performance attribution by market category."""

    category: str
    num_trades: int = 0
    total_pnl: float = 0.0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    average_pnl: float = 0.0
    sharpe_ratio: float = 0.0


@dataclass
class TimeAttribution:
    """Performance attribution by time period."""

    period: str  # "daily", "weekly", "monthly"
    start_date: datetime
    end_date: datetime
    num_trades: int = 0
    total_pnl: float = 0.0
    winning_trades: int = 0
    losing_trades: int = 0
    best_trade_pnl: float = 0.0
    worst_trade_pnl: float = 0.0
    portfolio_return: float = 0.0


@dataclass
class HoldingPeriodAttribution:
    """Performance attribution by holding period."""

    period_bucket: str  # "< 1 day", "1-7 days", "7-30 days", "> 30 days"
    num_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    average_holding_hours: float = 0.0


@dataclass
class SideAttribution:
    """Performance attribution by trade side (yes/no)."""

    side: str  # "yes" or "no"
    num_trades: int = 0
    total_pnl: float = 0.0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    average_pnl: float = 0.0


@dataclass
class MarketAttribution:
    """Performance attribution for individual markets."""

    ticker: str
    title: str
    category: str
    num_trades: int = 0
    total_pnl: float = 0.0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    best_trade_pnl: float = 0.0
    worst_trade_pnl: float = 0.0
    average_holding_hours: float = 0.0


@dataclass
class AttributionReport:
    """Comprehensive performance attribution report."""

    # Overall metrics
    total_pnl: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0

    # Attribution breakdowns
    category_attribution: List[CategoryAttribution] = field(default_factory=list)
    time_attribution: List[TimeAttribution] = field(default_factory=list)
    holding_period_attribution: List[HoldingPeriodAttribution] = field(default_factory=list)
    side_attribution: List[SideAttribution] = field(default_factory=list)
    market_attribution: List[MarketAttribution] = field(default_factory=list)

    # Risk contributions
    category_risk_contribution: Dict[str, float] = field(default_factory=dict)
    market_risk_contribution: Dict[str, float] = field(default_factory=dict)


class PerformanceAttributionAnalyzer:
    """
    Analyzes trading performance and attributes returns to various factors.

    This class helps answer questions like:
    - Which market categories are most profitable?
    - What time periods generate the best returns?
    - Is the strategy better at Yes or No positions?
    - What is the optimal holding period?
    - Which specific markets drive performance?
    """

    def __init__(self, trades: List[Trade], initial_capital: float):
        """
        Initialize attribution analyzer.

        Args:
            trades: List of all trades from backtest
            initial_capital: Starting capital
        """
        self.trades = trades
        self.initial_capital = initial_capital
        self.closed_trades = [t for t in trades if t.action == "close"]

    def analyze(self) -> AttributionReport:
        """
        Perform comprehensive performance attribution analysis.

        Returns:
            AttributionReport with detailed breakdowns
        """
        logger.info(f"Analyzing performance attribution for {len(self.trades)} trades")

        report = AttributionReport()

        # Calculate matched trade pairs (open + close)
        trade_pairs = self._match_trade_pairs()

        # Overall metrics
        report.total_trades = len(trade_pairs)
        report.total_pnl = sum(pair["pnl"] for pair in trade_pairs)
        winning_pairs = [p for p in trade_pairs if p["pnl"] > 0]
        report.win_rate = len(winning_pairs) / len(trade_pairs) if trade_pairs else 0

        # Category attribution
        report.category_attribution = self._analyze_by_category(trade_pairs)

        # Time attribution
        report.time_attribution = self._analyze_by_time(trade_pairs)

        # Holding period attribution
        report.holding_period_attribution = self._analyze_by_holding_period(trade_pairs)

        # Side attribution (yes vs no)
        report.side_attribution = self._analyze_by_side(trade_pairs)

        # Individual market attribution
        report.market_attribution = self._analyze_by_market(trade_pairs)

        # Risk contributions
        report.category_risk_contribution = self._calculate_category_risk(trade_pairs)
        report.market_risk_contribution = self._calculate_market_risk(trade_pairs)

        logger.info(
            f"Attribution analysis complete: Total PnL=${report.total_pnl:.2f}, "
            f"Win Rate={report.win_rate:.1%}"
        )

        return report

    def _match_trade_pairs(self) -> List[Dict]:
        """
        Match opening and closing trades to calculate PnL for each trade.

        Returns:
            List of trade pair dictionaries with PnL and metadata
        """
        pairs = []

        # Group trades by ticker
        ticker_trades = defaultdict(list)
        for trade in self.trades:
            ticker_trades[trade.market_ticker].append(trade)

        # Match open/close pairs for each ticker
        for ticker, trades_list in ticker_trades.items():
            open_trades = [t for t in trades_list if t.action == "open"]
            close_trades = [t for t in trades_list if t.action == "close"]

            # Match pairs (FIFO)
            for i in range(min(len(open_trades), len(close_trades))):
                open_trade = open_trades[i]
                close_trade = close_trades[i]

                # Calculate PnL
                entry_cost = open_trade.quantity * open_trade.price + open_trade.commission
                exit_proceeds = close_trade.quantity * close_trade.price - close_trade.commission
                pnl = exit_proceeds - entry_cost

                # Holding period
                holding_period = (close_trade.timestamp - open_trade.timestamp).total_seconds() / 3600  # hours

                pair = {
                    "ticker": ticker,
                    "title": open_trade.market_title,
                    "side": open_trade.side,
                    "open_time": open_trade.timestamp,
                    "close_time": close_trade.timestamp,
                    "holding_hours": holding_period,
                    "quantity": open_trade.quantity,
                    "entry_price": open_trade.price,
                    "exit_price": close_trade.price,
                    "pnl": pnl,
                    "return_pct": (pnl / entry_cost) if entry_cost > 0 else 0,
                    "open_reason": open_trade.reason,
                    "close_reason": close_trade.reason,
                    "category": open_trade.metadata.get("category", "unknown"),
                }

                pairs.append(pair)

        return pairs

    def _analyze_by_category(self, trade_pairs: List[Dict]) -> List[CategoryAttribution]:
        """Analyze performance by market category."""
        category_stats = defaultdict(lambda: CategoryAttribution(category=""))

        for pair in trade_pairs:
            category = pair["category"]
            stats = category_stats[category]
            stats.category = category
            stats.num_trades += 1
            stats.total_pnl += pair["pnl"]

            if pair["pnl"] > 0:
                stats.winning_trades += 1
            else:
                stats.losing_trades += 1

        # Calculate derived metrics
        for stats in category_stats.values():
            stats.win_rate = stats.winning_trades / stats.num_trades if stats.num_trades > 0 else 0
            stats.average_pnl = stats.total_pnl / stats.num_trades if stats.num_trades > 0 else 0

        # Sort by total PnL descending
        return sorted(category_stats.values(), key=lambda x: x.total_pnl, reverse=True)

    def _analyze_by_time(self, trade_pairs: List[Dict]) -> List[TimeAttribution]:
        """Analyze performance by time period (daily, weekly, monthly)."""
        # Daily attribution
        daily_stats = defaultdict(lambda: TimeAttribution(period="daily", start_date=None, end_date=None))

        for pair in trade_pairs:
            date_key = pair["close_time"].date()

            stats = daily_stats[date_key]
            if stats.start_date is None:
                stats.start_date = datetime.combine(date_key, datetime.min.time())
                stats.end_date = datetime.combine(date_key, datetime.max.time())

            stats.num_trades += 1
            stats.total_pnl += pair["pnl"]

            if pair["pnl"] > 0:
                stats.winning_trades += 1
                if pair["pnl"] > stats.best_trade_pnl:
                    stats.best_trade_pnl = pair["pnl"]
            else:
                stats.losing_trades += 1
                if pair["pnl"] < stats.worst_trade_pnl:
                    stats.worst_trade_pnl = pair["pnl"]

        # Sort by date
        return sorted(daily_stats.values(), key=lambda x: x.start_date)

    def _analyze_by_holding_period(self, trade_pairs: List[Dict]) -> List[HoldingPeriodAttribution]:
        """Analyze performance by holding period buckets."""
        buckets = {
            "< 1 day": HoldingPeriodAttribution(period_bucket="< 1 day"),
            "1-7 days": HoldingPeriodAttribution(period_bucket="1-7 days"),
            "7-30 days": HoldingPeriodAttribution(period_bucket="7-30 days"),
            "> 30 days": HoldingPeriodAttribution(period_bucket="> 30 days"),
        }

        for pair in trade_pairs:
            hours = pair["holding_hours"]

            # Determine bucket
            if hours < 24:
                bucket_key = "< 1 day"
            elif hours < 24 * 7:
                bucket_key = "1-7 days"
            elif hours < 24 * 30:
                bucket_key = "7-30 days"
            else:
                bucket_key = "> 30 days"

            stats = buckets[bucket_key]
            stats.num_trades += 1
            stats.total_pnl += pair["pnl"]
            stats.average_holding_hours = (
                (stats.average_holding_hours * (stats.num_trades - 1) + hours) / stats.num_trades
            )

            if pair["pnl"] > 0:
                stats.win_rate = ((stats.win_rate * (stats.num_trades - 1)) + 1) / stats.num_trades

        return list(buckets.values())

    def _analyze_by_side(self, trade_pairs: List[Dict]) -> List[SideAttribution]:
        """Analyze performance by trade side (yes vs no)."""
        side_stats = {
            "yes": SideAttribution(side="yes"),
            "no": SideAttribution(side="no"),
        }

        for pair in trade_pairs:
            side = pair["side"]
            stats = side_stats[side]

            stats.num_trades += 1
            stats.total_pnl += pair["pnl"]

            if pair["pnl"] > 0:
                stats.winning_trades += 1
            else:
                stats.losing_trades += 1

        # Calculate derived metrics
        for stats in side_stats.values():
            stats.win_rate = stats.winning_trades / stats.num_trades if stats.num_trades > 0 else 0
            stats.average_pnl = stats.total_pnl / stats.num_trades if stats.num_trades > 0 else 0

        return list(side_stats.values())

    def _analyze_by_market(self, trade_pairs: List[Dict]) -> List[MarketAttribution]:
        """Analyze performance for individual markets."""
        market_stats = defaultdict(lambda: MarketAttribution(ticker="", title="", category=""))

        for pair in trade_pairs:
            ticker = pair["ticker"]
            stats = market_stats[ticker]

            stats.ticker = ticker
            stats.title = pair["title"]
            stats.category = pair["category"]
            stats.num_trades += 1
            stats.total_pnl += pair["pnl"]

            if pair["pnl"] > 0:
                stats.winning_trades += 1
                if pair["pnl"] > stats.best_trade_pnl:
                    stats.best_trade_pnl = pair["pnl"]
            else:
                stats.losing_trades += 1
                if pair["pnl"] < stats.worst_trade_pnl:
                    stats.worst_trade_pnl = pair["pnl"]

            # Update average holding period
            stats.average_holding_hours = (
                (stats.average_holding_hours * (stats.num_trades - 1) + pair["holding_hours"]) / stats.num_trades
            )

        # Calculate derived metrics
        for stats in market_stats.values():
            stats.win_rate = stats.winning_trades / stats.num_trades if stats.num_trades > 0 else 0

        # Sort by total PnL descending
        return sorted(market_stats.values(), key=lambda x: x.total_pnl, reverse=True)

    def _calculate_category_risk(self, trade_pairs: List[Dict]) -> Dict[str, float]:
        """
        Calculate risk contribution by category.

        Risk contribution is measured as the standard deviation of returns
        for each category.
        """
        import statistics

        category_returns = defaultdict(list)

        for pair in trade_pairs:
            category = pair["category"]
            category_returns[category].append(pair["return_pct"])

        risk_contributions = {}
        for category, returns in category_returns.items():
            if len(returns) > 1:
                risk_contributions[category] = statistics.stdev(returns)
            else:
                risk_contributions[category] = 0.0

        return risk_contributions

    def _calculate_market_risk(self, trade_pairs: List[Dict]) -> Dict[str, float]:
        """
        Calculate risk contribution by individual market.

        Returns standard deviation of returns for each market.
        """
        import statistics

        market_returns = defaultdict(list)

        for pair in trade_pairs:
            ticker = pair["ticker"]
            market_returns[ticker].append(pair["return_pct"])

        risk_contributions = {}
        for ticker, returns in market_returns.items():
            if len(returns) > 1:
                risk_contributions[ticker] = statistics.stdev(returns)
            else:
                risk_contributions[ticker] = 0.0

        return risk_contributions

    def get_top_contributors(self, n: int = 10) -> List[MarketAttribution]:
        """
        Get top N markets by total PnL contribution.

        Args:
            n: Number of top markets to return

        Returns:
            List of MarketAttribution sorted by PnL descending
        """
        trade_pairs = self._match_trade_pairs()
        market_attribution = self._analyze_by_market(trade_pairs)
        return market_attribution[:n]

    def get_worst_performers(self, n: int = 10) -> List[MarketAttribution]:
        """
        Get worst N markets by total PnL.

        Args:
            n: Number of worst markets to return

        Returns:
            List of MarketAttribution sorted by PnL ascending
        """
        trade_pairs = self._match_trade_pairs()
        market_attribution = self._analyze_by_market(trade_pairs)
        return sorted(market_attribution, key=lambda x: x.total_pnl)[:n]

    def print_summary(self):
        """Print a formatted attribution summary to console."""
        report = self.analyze()

        print("\n" + "=" * 80)
        print("PERFORMANCE ATTRIBUTION ANALYSIS")
        print("=" * 80)

        print(f"\nOverall Performance:")
        print(f"  Total Trades: {report.total_trades}")
        print(f"  Total PnL: ${report.total_pnl:,.2f}")
        print(f"  Win Rate: {report.win_rate:.1%}")

        print(f"\n{'Category Attribution:':<40}")
        print(f"  {'Category':<20} {'Trades':<8} {'PnL':<12} {'Win Rate':<10}")
        print(f"  {'-'*60}")
        for cat in report.category_attribution[:10]:
            print(f"  {cat.category:<20} {cat.num_trades:<8} ${cat.total_pnl:>10,.2f} {cat.win_rate:>9.1%}")

        print(f"\n{'Side Attribution (Yes vs No):':<40}")
        for side in report.side_attribution:
            print(f"  {side.side.upper():<10} Trades: {side.num_trades:<5} PnL: ${side.total_pnl:>10,.2f} "
                  f"Win Rate: {side.win_rate:.1%}")

        print(f"\n{'Holding Period Attribution:':<40}")
        for hp in report.holding_period_attribution:
            if hp.num_trades > 0:
                print(
                    f"  {hp.period_bucket:<15} Trades: {hp.num_trades:<5} PnL: ${hp.total_pnl:>10,.2f} "
                    f"Avg Hours: {hp.average_holding_hours:>6.1f}"
                )

        print(f"\n{'Top 5 Markets by PnL:':<40}")
        print(f"  {'Ticker':<20} {'Trades':<8} {'PnL':<12} {'Win Rate':<10}")
        print(f"  {'-'*60}")
        for market in report.market_attribution[:5]:
            print(
                f"  {market.ticker:<20} {market.num_trades:<8} ${market.total_pnl:>10,.2f} {market.win_rate:>9.1%}"
            )

        print("\n" + "=" * 80)
