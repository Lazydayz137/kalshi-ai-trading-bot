"""
Paper Trading Mode - Simulated Trading Without Real Money

Simulates all trading operations for testing strategies without risking capital.
Tracks simulated positions, P&L, and performance metrics.
"""

import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import os

from src.utils.database import DatabaseManager, Position, TradeLog
from src.clients.kalshi_client import KalshiClient
from src.utils.logging_setup import get_trading_logger


@dataclass
class PaperAccount:
    """Simulated trading account."""
    starting_balance: float = 10000.0
    current_balance: float = 10000.0
    cash_available: float = 10000.0
    positions_value: float = 0.0
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_fees: float = 0.0
    max_drawdown: float = 0.0
    peak_balance: float = 10000.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'starting_balance': self.starting_balance,
            'current_balance': self.current_balance,
            'cash_available': self.cash_available,
            'positions_value': self.positions_value,
            'total_pnl': self.total_pnl,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'total_fees': self.total_fees,
            'max_drawdown': self.max_drawdown,
            'peak_balance': self.peak_balance
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'PaperAccount':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class PaperPosition:
    """Simulated position."""
    market_id: str
    side: str
    entry_price: float
    quantity: int
    entry_timestamp: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    strategy: Optional[str] = None
    confidence: Optional[float] = None

    def update_price(self, new_price: float):
        """Update current price and recalculate P&L."""
        self.current_price = new_price

        # Calculate unrealized P&L
        if self.side == "YES":
            self.unrealized_pnl = (new_price - self.entry_price) * self.quantity
        else:  # NO
            self.unrealized_pnl = (self.entry_price - new_price) * self.quantity


class PaperTradingEngine:
    """
    Paper trading engine that simulates all trading operations.

    Features:
    - Simulated order execution with realistic fills
    - Position tracking with P&L calculation
    - Slippage and fee simulation
    - Market data from real Kalshi API
    - Performance metrics and reporting
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        kalshi_client: KalshiClient,
        starting_balance: float = 10000.0,
        simulate_slippage: bool = True,
        simulate_fees: bool = True,
        slippage_bps: float = 5.0,  # 5 basis points
        fee_percentage: float = 0.007  # 0.7% Kalshi fee
    ):
        """
        Initialize paper trading engine.

        Args:
            db_manager: Database manager
            kalshi_client: Kalshi client (for real market data)
            starting_balance: Starting account balance
            simulate_slippage: Whether to simulate slippage
            simulate_fees: Whether to simulate fees
            slippage_bps: Slippage in basis points
            fee_percentage: Fee as percentage of trade value
        """
        self.db_manager = db_manager
        self.kalshi_client = kalshi_client
        self.logger = get_trading_logger("paper_trading")

        # Paper account
        self.account = self._load_or_create_account(starting_balance)

        # Simulated positions
        self.positions: Dict[str, PaperPosition] = {}

        # Settings
        self.simulate_slippage = simulate_slippage
        self.simulate_fees = simulate_fees
        self.slippage_bps = slippage_bps
        self.fee_percentage = fee_percentage

        # Storage
        self.account_file = "logs/paper_account.json"
        self.positions_file = "logs/paper_positions.json"

        self._load_positions()

        self.logger.info(
            "Paper trading engine initialized",
            starting_balance=self.account.starting_balance,
            current_balance=self.account.current_balance,
            positions=len(self.positions),
            simulate_slippage=simulate_slippage,
            simulate_fees=simulate_fees
        )

    def _load_or_create_account(self, starting_balance: float) -> PaperAccount:
        """Load existing paper account or create new one."""
        os.makedirs("logs", exist_ok=True)

        if os.path.exists(self.account_file):
            try:
                with open(self.account_file, 'r') as f:
                    data = json.load(f)
                    return PaperAccount.from_dict(data)
            except Exception as e:
                self.logger.warning(f"Failed to load paper account: {e}")

        return PaperAccount(starting_balance=starting_balance)

    def _save_account(self):
        """Save paper account to disk."""
        try:
            os.makedirs("logs", exist_ok=True)
            with open(self.account_file, 'w') as f:
                json.dump(self.account.to_dict(), f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save paper account: {e}")

    def _load_positions(self):
        """Load simulated positions from disk."""
        if os.path.exists(self.positions_file):
            try:
                with open(self.positions_file, 'r') as f:
                    data = json.load(f)
                    for pos_data in data:
                        pos = PaperPosition(
                            market_id=pos_data['market_id'],
                            side=pos_data['side'],
                            entry_price=pos_data['entry_price'],
                            quantity=pos_data['quantity'],
                            entry_timestamp=datetime.fromisoformat(pos_data['entry_timestamp']),
                            current_price=pos_data.get('current_price', pos_data['entry_price']),
                            strategy=pos_data.get('strategy'),
                            confidence=pos_data.get('confidence')
                        )
                        self.positions[pos.market_id] = pos
            except Exception as e:
                self.logger.warning(f"Failed to load paper positions: {e}")

    def _save_positions(self):
        """Save simulated positions to disk."""
        try:
            os.makedirs("logs", exist_ok=True)
            data = []
            for pos in self.positions.values():
                data.append({
                    'market_id': pos.market_id,
                    'side': pos.side,
                    'entry_price': pos.entry_price,
                    'quantity': pos.quantity,
                    'entry_timestamp': pos.entry_timestamp.isoformat(),
                    'current_price': pos.current_price,
                    'strategy': pos.strategy,
                    'confidence': pos.confidence
                })
            with open(self.positions_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save paper positions: {e}")

    def get_balance(self) -> Dict:
        """Get simulated account balance (mimics Kalshi API response)."""
        return {
            "balance": int(self.account.cash_available * 100)  # Convert to cents
        }

    async def place_order(
        self,
        ticker: str,
        side: str,
        action: str,
        count: int,
        type_: str = "market",
        yes_price: Optional[int] = None,
        no_price: Optional[int] = None
    ) -> Dict:
        """
        Simulate order placement.

        Args:
            ticker: Market ticker
            side: "yes" or "no"
            action: "buy" or "sell"
            count: Number of contracts
            type_: Order type
            yes_price: YES price in cents
            no_price: NO price in cents

        Returns:
            Simulated order response
        """
        try:
            # Get current market price from Kalshi
            market_data = await self.kalshi_client.get_market(ticker)
            market = market_data.get("market", {})

            # Determine execution price
            if side.lower() == "yes":
                base_price = yes_price / 100.0 if yes_price else market.get("yes_bid", 50) / 100.0
            else:
                base_price = no_price / 100.0 if no_price else market.get("no_bid", 50) / 100.0

            # Apply slippage for market orders
            if type_ == "market" and self.simulate_slippage:
                if action == "buy":
                    base_price *= (1 + self.slippage_bps / 10000)
                else:
                    base_price *= (1 - self.slippage_bps / 10000)

            # Calculate trade value
            trade_value = base_price * count

            # Calculate fees
            fees = 0.0
            if self.simulate_fees:
                fees = trade_value * self.fee_percentage

            total_cost = trade_value + fees

            # Check if we have enough cash
            if action == "buy" and total_cost > self.account.cash_available:
                self.logger.warning(
                    f"Insufficient cash for paper trade: ${total_cost:.2f} needed, ${self.account.cash_available:.2f} available"
                )
                return {
                    "error": "Insufficient funds",
                    "order": None
                }

            # Execute the simulated trade
            if action == "buy":
                # Open position
                self.account.cash_available -= total_cost
                self.account.total_fees += fees

                # Create or add to position
                if ticker in self.positions:
                    # Average up/down
                    pos = self.positions[ticker]
                    total_quantity = pos.quantity + count
                    total_value = (pos.entry_price * pos.quantity) + (base_price * count)
                    pos.entry_price = total_value / total_quantity
                    pos.quantity = total_quantity
                    pos.current_price = base_price
                else:
                    # New position
                    self.positions[ticker] = PaperPosition(
                        market_id=ticker,
                        side=side.upper(),
                        entry_price=base_price,
                        quantity=count,
                        entry_timestamp=datetime.now(),
                        current_price=base_price
                    )

                self.logger.info(
                    f"Paper BUY executed",
                    ticker=ticker,
                    side=side,
                    quantity=count,
                    price=base_price,
                    total_cost=total_cost,
                    fees=fees
                )

            else:  # sell
                # Close position
                if ticker not in self.positions:
                    self.logger.warning(f"Cannot sell - no position in {ticker}")
                    return {
                        "error": "No position to close",
                        "order": None
                    }

                pos = self.positions[ticker]

                # Calculate P&L
                if pos.side == "YES":
                    pnl = (base_price - pos.entry_price) * count
                else:
                    pnl = (pos.entry_price - base_price) * count

                pnl -= fees  # Subtract fees

                # Update account
                self.account.cash_available += (base_price * count) - fees
                self.account.realized_pnl += pnl
                self.account.total_pnl += pnl
                self.account.total_fees += fees
                self.account.total_trades += 1

                if pnl > 0:
                    self.account.winning_trades += 1
                else:
                    self.account.losing_trades += 1

                # Update position or remove if fully closed
                if count >= pos.quantity:
                    del self.positions[ticker]
                else:
                    pos.quantity -= count

                # Log trade to database
                await self._log_paper_trade(
                    ticker=ticker,
                    side=pos.side,
                    entry_price=pos.entry_price,
                    exit_price=base_price,
                    quantity=count,
                    pnl=pnl
                )

                self.logger.info(
                    f"Paper SELL executed",
                    ticker=ticker,
                    side=side,
                    quantity=count,
                    price=base_price,
                    pnl=pnl,
                    fees=fees
                )

            # Update account balance
            self.account.current_balance = self.account.cash_available + self._calculate_positions_value()

            # Update peak balance and drawdown
            if self.account.current_balance > self.account.peak_balance:
                self.account.peak_balance = self.account.current_balance

            drawdown = (self.account.peak_balance - self.account.current_balance) / self.account.peak_balance
            self.account.max_drawdown = max(self.account.max_drawdown, drawdown)

            # Save state
            self._save_account()
            self._save_positions()

            # Return simulated order response
            return {
                "order": {
                    "order_id": f"paper_{ticker}_{int(datetime.now().timestamp())}",
                    "ticker": ticker,
                    "side": side,
                    "action": action,
                    "count": count,
                    "price": int(base_price * 100),
                    "status": "filled",
                    "created_time": datetime.now().isoformat()
                }
            }

        except Exception as e:
            self.logger.error(f"Paper order execution failed: {e}")
            return {
                "error": str(e),
                "order": None
            }

    def _calculate_positions_value(self) -> float:
        """Calculate total value of open positions."""
        total = 0.0
        for pos in self.positions.values():
            total += pos.current_price * pos.quantity
        return total

    async def update_positions(self):
        """Update all positions with current market prices."""
        for ticker, pos in self.positions.items():
            try:
                # Fetch current market price
                market_data = await self.kalshi_client.get_market(ticker)
                market = market_data.get("market", {})

                if pos.side == "YES":
                    current_price = market.get("yes_bid", pos.entry_price * 100) / 100.0
                else:
                    current_price = market.get("no_bid", pos.entry_price * 100) / 100.0

                pos.update_price(current_price)

            except Exception as e:
                self.logger.warning(f"Failed to update price for {ticker}: {e}")

        # Recalculate unrealized P&L
        self.account.unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        self.account.total_pnl = self.account.realized_pnl + self.account.unrealized_pnl

        # Update positions value
        self.account.positions_value = self._calculate_positions_value()
        self.account.current_balance = self.account.cash_available + self.account.positions_value

        self._save_account()
        self._save_positions()

    async def _log_paper_trade(
        self,
        ticker: str,
        side: str,
        entry_price: float,
        exit_price: float,
        quantity: int,
        pnl: float
    ):
        """Log paper trade to database."""
        try:
            trade_log = TradeLog(
                market_id=ticker,
                side=side,
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=quantity,
                pnl=pnl,
                entry_timestamp=datetime.now() - timedelta(hours=1),  # Approximate
                exit_timestamp=datetime.now(),
                rationale=f"Paper trade: {side} {quantity} contracts",
                strategy="paper_trading"
            )

            import aiosqlite
            async with aiosqlite.connect(self.db_manager.db_path) as db:
                await db.execute("""
                    INSERT INTO trade_logs
                    (market_id, side, entry_price, exit_price, quantity, pnl,
                     entry_timestamp, exit_timestamp, rationale, strategy)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_log.market_id,
                    trade_log.side,
                    trade_log.entry_price,
                    trade_log.exit_price,
                    trade_log.quantity,
                    trade_log.pnl,
                    trade_log.entry_timestamp.isoformat(),
                    trade_log.exit_timestamp.isoformat(),
                    trade_log.rationale,
                    trade_log.strategy
                ))
                await db.commit()

        except Exception as e:
            self.logger.error(f"Failed to log paper trade: {e}")

    def get_performance_summary(self) -> Dict:
        """Get performance summary."""
        win_rate = (self.account.winning_trades / self.account.total_trades * 100) if self.account.total_trades > 0 else 0

        avg_win = (self.account.realized_pnl / self.account.winning_trades) if self.account.winning_trades > 0 else 0
        avg_loss = (self.account.realized_pnl / self.account.losing_trades) if self.account.losing_trades > 0 else 0

        return {
            "starting_balance": self.account.starting_balance,
            "current_balance": self.account.current_balance,
            "total_pnl": self.account.total_pnl,
            "total_pnl_pct": (self.account.total_pnl / self.account.starting_balance * 100),
            "realized_pnl": self.account.realized_pnl,
            "unrealized_pnl": self.account.unrealized_pnl,
            "total_trades": self.account.total_trades,
            "winning_trades": self.account.winning_trades,
            "losing_trades": self.account.losing_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "total_fees": self.account.total_fees,
            "max_drawdown": self.account.max_drawdown * 100,
            "open_positions": len(self.positions),
            "positions_value": self.account.positions_value
        }

    def reset_account(self, starting_balance: float = 10000.0):
        """Reset paper trading account."""
        self.account = PaperAccount(starting_balance=starting_balance)
        self.positions = {}
        self._save_account()
        self._save_positions()

        self.logger.info(f"Paper trading account reset to ${starting_balance}")
