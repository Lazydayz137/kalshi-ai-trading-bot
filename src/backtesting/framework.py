"""
Backtesting Framework for Kalshi AI Trading Bot

This module provides a comprehensive backtesting framework for validating trading strategies
on historical data before deploying them in live trading.

Features:
- Time-travel testing with realistic market conditions
- Historical data replay
- Performance attribution and analytics
- Realistic execution simulation including fees and slippage
- Comprehensive reporting and visualization
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json

from ..utils.logger import get_logger

logger = get_logger(__name__)


class BacktestMode(Enum):
    """Backtesting execution modes."""

    HISTORICAL = "historical"  # Replay historical data
    SYNTHETIC = "synthetic"  # Generate synthetic market data
    HYBRID = "hybrid"  # Mix of historical and synthetic


@dataclass
class BacktestConfig:
    """Configuration for backtesting runs."""

    # Time period
    start_date: datetime
    end_date: datetime

    # Initial conditions
    initial_capital: float = 10000.0
    max_positions: int = 10

    # Execution parameters
    mode: BacktestMode = BacktestMode.HISTORICAL
    commission_per_contract: float = 1.00  # Kalshi fee per contract
    slippage_bps: float = 5.0  # Basis points of slippage

    # Risk parameters
    max_position_size: float = 1000.0
    max_daily_loss: float = 500.0
    position_sizing_method: str = "kelly"  # kelly, fixed, proportional

    # Data parameters
    data_frequency: str = "1h"  # Frequency for data snapshots
    warmup_period_days: int = 30  # Days of data before start_date for warmup

    # Strategy parameters
    strategy_name: str = "default"
    strategy_config: Dict = field(default_factory=dict)

    # Reporting
    generate_reports: bool = True
    report_frequency: str = "daily"  # daily, weekly, monthly
    save_trades: bool = True


@dataclass
class Trade:
    """Represents a single trade execution."""

    timestamp: datetime
    market_ticker: str
    market_title: str
    side: str  # "yes" or "no"
    action: str  # "open" or "close"
    quantity: int
    price: float
    commission: float
    slippage: float
    total_cost: float
    portfolio_value_before: float
    portfolio_value_after: float
    reason: str  # Why the trade was made
    metadata: Dict = field(default_factory=dict)


@dataclass
class Position:
    """Represents an open position during backtest."""

    market_ticker: str
    market_title: str
    side: str  # "yes" or "no"
    quantity: int
    entry_price: float
    entry_timestamp: datetime
    current_price: float
    unrealized_pnl: float = 0.0
    max_profit: float = 0.0
    max_drawdown: float = 0.0
    holding_period_hours: float = 0.0

    def update_current_price(self, price: float, timestamp: datetime):
        """Update current price and derived metrics."""
        self.current_price = price
        cost_basis = self.quantity * self.entry_price
        current_value = self.quantity * price
        self.unrealized_pnl = current_value - cost_basis

        # Track max profit and drawdown
        if self.unrealized_pnl > self.max_profit:
            self.max_profit = self.unrealized_pnl
        if self.unrealized_pnl < self.max_drawdown:
            self.max_drawdown = self.unrealized_pnl

        # Update holding period
        self.holding_period_hours = (timestamp - self.entry_timestamp).total_seconds() / 3600


@dataclass
class PerformanceMetrics:
    """Performance metrics for backtest results."""

    # Returns
    total_return: float = 0.0
    annualized_return: float = 0.0
    daily_returns: List[float] = field(default_factory=list)

    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0

    # Trading metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    profit_factor: float = 0.0

    # Exposure metrics
    average_positions: float = 0.0
    max_positions: int = 0
    average_holding_period_hours: float = 0.0

    # Cost metrics
    total_commissions: float = 0.0
    total_slippage: float = 0.0

    # Time metrics
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    trading_days: int = 0


class BacktestEngine:
    """
    Core backtesting engine for strategy validation.

    This engine simulates trading strategies on historical data,
    tracking performance, risk, and execution costs realistically.
    """

    def __init__(self, config: BacktestConfig):
        """Initialize backtesting engine with configuration."""
        self.config = config
        self.current_time: Optional[datetime] = None
        self.portfolio_value: float = config.initial_capital
        self.cash: float = config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.daily_portfolio_values: List[Tuple[datetime, float]] = []
        self.metrics: Optional[PerformanceMetrics] = None

        logger.info(
            f"Initialized backtest engine: {config.start_date} to {config.end_date}, "
            f"capital=${config.initial_capital:,.2f}"
        )

    async def run(self, strategy, data_loader) -> PerformanceMetrics:
        """
        Run backtest with given strategy and data.

        Args:
            strategy: Trading strategy instance with decide() method
            data_loader: Data loader instance providing historical market data

        Returns:
            PerformanceMetrics: Comprehensive performance metrics
        """
        logger.info(f"Starting backtest run for strategy: {self.config.strategy_name}")

        # Load historical data
        historical_data = await data_loader.load_data(
            start_date=self.config.start_date - timedelta(days=self.config.warmup_period_days),
            end_date=self.config.end_date,
            frequency=self.config.data_frequency,
        )

        logger.info(f"Loaded {len(historical_data)} data points for backtesting")

        # Time-travel through historical data
        for timestamp, market_snapshot in historical_data:
            self.current_time = timestamp

            # Skip warmup period
            if timestamp < self.config.start_date:
                continue

            # Update positions with current prices
            await self._update_positions(market_snapshot)

            # Check risk limits
            if not self._check_risk_limits():
                logger.warning(f"Risk limits breached at {timestamp}, stopping backtest")
                break

            # Get strategy decisions
            decisions = await strategy.decide(market_snapshot, self.positions, self.portfolio_value)

            # Execute decisions
            for decision in decisions:
                await self._execute_trade(decision, market_snapshot)

            # Record daily portfolio value
            if self._is_new_day(timestamp):
                self.daily_portfolio_values.append((timestamp, self.portfolio_value))

        # Close all positions at end of backtest
        await self._close_all_positions()

        # Calculate performance metrics
        self.metrics = self._calculate_metrics()

        # Generate reports if configured
        if self.config.generate_reports:
            await self._generate_reports()

        logger.info(
            f"Backtest complete: Return={self.metrics.total_return:.2%}, "
            f"Sharpe={self.metrics.sharpe_ratio:.2f}, "
            f"Trades={self.metrics.total_trades}"
        )

        return self.metrics

    async def _update_positions(self, market_snapshot: Dict):
        """Update all open positions with current market prices."""
        for ticker, position in self.positions.items():
            if ticker in market_snapshot:
                current_price = market_snapshot[ticker]["last_price"]
                position.update_current_price(current_price, self.current_time)

        # Update portfolio value
        positions_value = sum(pos.quantity * pos.current_price for pos in self.positions.values())
        self.portfolio_value = self.cash + positions_value

    async def _execute_trade(self, decision: Dict, market_snapshot: Dict):
        """
        Execute a trade decision with realistic costs.

        Args:
            decision: Trading decision with ticker, side, action, quantity
            market_snapshot: Current market data
        """
        ticker = decision["ticker"]
        side = decision["side"]
        action = decision["action"]  # "open" or "close"
        quantity = decision["quantity"]

        # Get market price with slippage
        base_price = market_snapshot[ticker]["last_price"]
        slippage = base_price * (self.config.slippage_bps / 10000)
        execution_price = base_price + slippage if action == "open" else base_price - slippage

        # Calculate costs
        commission = quantity * self.config.commission_per_contract
        total_cost = (quantity * execution_price) + commission

        # Check if we have enough cash
        if action == "open" and total_cost > self.cash:
            logger.warning(f"Insufficient cash for trade: need ${total_cost:.2f}, have ${self.cash:.2f}")
            return

        # Record portfolio value before trade
        portfolio_before = self.portfolio_value

        # Execute the trade
        if action == "open":
            await self._open_position(ticker, side, quantity, execution_price, commission, slippage, decision)
        else:
            await self._close_position(ticker, quantity, execution_price, commission, slippage, decision)

        # Record the trade
        trade = Trade(
            timestamp=self.current_time,
            market_ticker=ticker,
            market_title=market_snapshot[ticker].get("title", ticker),
            side=side,
            action=action,
            quantity=quantity,
            price=execution_price,
            commission=commission,
            slippage=slippage * quantity,
            total_cost=total_cost,
            portfolio_value_before=portfolio_before,
            portfolio_value_after=self.portfolio_value,
            reason=decision.get("reason", ""),
            metadata=decision.get("metadata", {}),
        )
        self.trades.append(trade)

    async def _open_position(
        self, ticker: str, side: str, quantity: int, price: float, commission: float, slippage: float, decision: Dict
    ):
        """Open a new position."""
        cost = (quantity * price) + commission
        self.cash -= cost

        position = Position(
            market_ticker=ticker,
            market_title=decision.get("market_title", ticker),
            side=side,
            quantity=quantity,
            entry_price=price,
            entry_timestamp=self.current_time,
            current_price=price,
        )
        self.positions[ticker] = position

        logger.debug(f"Opened position: {ticker} {side} {quantity}@${price:.2f}, cost=${cost:.2f}")

    async def _close_position(
        self, ticker: str, quantity: int, price: float, commission: float, slippage: float, decision: Dict
    ):
        """Close an existing position."""
        if ticker not in self.positions:
            logger.warning(f"Attempted to close non-existent position: {ticker}")
            return

        position = self.positions[ticker]
        proceeds = (quantity * price) - commission
        self.cash += proceeds

        # Remove position
        del self.positions[ticker]

        pnl = proceeds - (position.quantity * position.entry_price)
        logger.debug(f"Closed position: {ticker} {quantity}@${price:.2f}, PnL=${pnl:.2f}")

    async def _close_all_positions(self):
        """Close all open positions at end of backtest."""
        tickers_to_close = list(self.positions.keys())
        for ticker in tickers_to_close:
            position = self.positions[ticker]
            # Use current price for final close
            decision = {"ticker": ticker, "side": position.side, "action": "close", "quantity": position.quantity}
            await self._close_position(
                ticker, position.quantity, position.current_price, 0, 0, {"reason": "Backtest end"}
            )

    def _check_risk_limits(self) -> bool:
        """Check if risk limits are breached."""
        # Check max daily loss
        if len(self.daily_portfolio_values) > 0:
            previous_value = self.daily_portfolio_values[-1][1]
            daily_pnl = self.portfolio_value - previous_value
            if daily_pnl < -self.config.max_daily_loss:
                return False

        # Check max positions
        if len(self.positions) > self.config.max_positions:
            return False

        return True

    def _is_new_day(self, timestamp: datetime) -> bool:
        """Check if timestamp is a new trading day."""
        if not self.daily_portfolio_values:
            return True
        last_day = self.daily_portfolio_values[-1][0].date()
        return timestamp.date() > last_day

    def _calculate_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        metrics = PerformanceMetrics()

        # Time metrics
        metrics.start_date = self.config.start_date
        metrics.end_date = self.config.end_date
        metrics.trading_days = len(self.daily_portfolio_values)

        # Return metrics
        initial_capital = self.config.initial_capital
        final_value = self.portfolio_value
        metrics.total_return = (final_value - initial_capital) / initial_capital

        # Annualized return
        if metrics.trading_days > 0:
            years = metrics.trading_days / 252  # Trading days per year
            metrics.annualized_return = ((1 + metrics.total_return) ** (1 / years)) - 1 if years > 0 else 0

        # Daily returns for risk calculations
        if len(self.daily_portfolio_values) > 1:
            metrics.daily_returns = [
                (self.daily_portfolio_values[i][1] - self.daily_portfolio_values[i - 1][1])
                / self.daily_portfolio_values[i - 1][1]
                for i in range(1, len(self.daily_portfolio_values))
            ]

            # Volatility (annualized)
            import statistics

            if len(metrics.daily_returns) > 1:
                metrics.volatility = statistics.stdev(metrics.daily_returns) * (252**0.5)

            # Sharpe ratio (assuming 0% risk-free rate)
            if metrics.volatility > 0:
                metrics.sharpe_ratio = metrics.annualized_return / metrics.volatility

            # Max drawdown
            peak = initial_capital
            max_dd = 0
            for _, value in self.daily_portfolio_values:
                if value > peak:
                    peak = value
                dd = (value - peak) / peak
                if dd < max_dd:
                    max_dd = dd
            metrics.max_drawdown = max_dd

        # Trading metrics
        metrics.total_trades = len([t for t in self.trades if t.action == "close"])
        winning_trades = []
        losing_trades = []

        for trade in self.trades:
            if trade.action == "close":
                # Find corresponding open trade
                open_trade = next(
                    (t for t in reversed(self.trades) if t.ticker == trade.ticker and t.action == "open"), None
                )
                if open_trade:
                    pnl = trade.quantity * (trade.price - open_trade.price) - trade.commission - open_trade.commission
                    if pnl > 0:
                        winning_trades.append(pnl)
                    else:
                        losing_trades.append(pnl)

        metrics.winning_trades = len(winning_trades)
        metrics.losing_trades = len(losing_trades)
        metrics.win_rate = metrics.winning_trades / metrics.total_trades if metrics.total_trades > 0 else 0

        if winning_trades:
            metrics.average_win = sum(winning_trades) / len(winning_trades)
        if losing_trades:
            metrics.average_loss = sum(losing_trades) / len(losing_trades)

        # Profit factor
        total_wins = sum(winning_trades) if winning_trades else 0
        total_losses = abs(sum(losing_trades)) if losing_trades else 0
        metrics.profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

        # Cost metrics
        metrics.total_commissions = sum(t.commission for t in self.trades)
        metrics.total_slippage = sum(t.slippage for t in self.trades)

        return metrics

    async def _generate_reports(self):
        """Generate backtest reports."""
        # This will be implemented in the report_generator module
        logger.info("Generating backtest reports...")

    def get_trades_summary(self) -> List[Dict]:
        """Get summary of all trades."""
        return [
            {
                "timestamp": t.timestamp.isoformat(),
                "ticker": t.market_ticker,
                "title": t.market_title,
                "side": t.side,
                "action": t.action,
                "quantity": t.quantity,
                "price": t.price,
                "commission": t.commission,
                "total_cost": t.total_cost,
                "reason": t.reason,
            }
            for t in self.trades
        ]

    def get_equity_curve(self) -> List[Tuple[datetime, float]]:
        """Get equity curve data."""
        return self.daily_portfolio_values
