"""
Trading Client Wrapper - Unified Interface for Paper and Live Trading

Provides a single interface that routes to either paper trading or live trading
based on configuration, making it easy to switch between modes.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime

from src.clients.kalshi_client import KalshiClient
from src.utils.paper_trading import PaperTradingEngine
from src.utils.database import DatabaseManager
from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger


class TradingClientWrapper:
    """
    Wrapper that provides unified interface for both paper and live trading.

    Automatically routes to paper trading engine or live Kalshi client
    based on configuration settings.
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        force_paper_trading: Optional[bool] = None
    ):
        """
        Initialize trading client wrapper.

        Args:
            db_manager: Database manager instance
            force_paper_trading: Override settings to force paper trading mode
        """
        self.db_manager = db_manager
        self.logger = get_trading_logger("trading_client_wrapper")

        # Determine trading mode
        if force_paper_trading is not None:
            self.paper_trading = force_paper_trading
        else:
            # Check settings
            self.paper_trading = getattr(
                settings.trading,
                'paper_trading_mode',
                not getattr(settings.trading, 'live_trading_enabled', False)
            )

        # Initialize appropriate client
        self.kalshi_client = KalshiClient()

        if self.paper_trading:
            self.paper_engine = PaperTradingEngine(
                db_manager=db_manager,
                kalshi_client=self.kalshi_client,
                starting_balance=getattr(settings, 'paper_trading_balance', 10000.0)
            )
            self.logger.info(
                "Trading client initialized in PAPER TRADING mode",
                starting_balance=self.paper_engine.account.starting_balance
            )
        else:
            self.paper_engine = None
            self.logger.warning(
                "⚠️  Trading client initialized in LIVE TRADING mode - REAL MONEY AT RISK"
            )

    @property
    def is_paper_trading(self) -> bool:
        """Check if in paper trading mode."""
        return self.paper_trading

    async def get_balance(self) -> Dict[str, Any]:
        """
        Get account balance.

        Returns:
            Balance dict with 'balance' in cents
        """
        if self.paper_trading:
            return self.paper_engine.get_balance()
        else:
            return await self.kalshi_client.get_balance()

    async def get_positions(self, ticker: Optional[str] = None) -> Dict[str, Any]:
        """
        Get portfolio positions.

        Args:
            ticker: Optional ticker to filter

        Returns:
            Positions dict
        """
        if self.paper_trading:
            # Convert paper positions to Kalshi API format
            positions = []
            for market_id, pos in self.paper_engine.positions.items():
                if ticker is None or market_id == ticker:
                    positions.append({
                        'ticker': pos.market_id,
                        'position': pos.quantity if pos.side == "YES" else -pos.quantity,
                        'market_price': int(pos.current_price * 100),
                        'total_traded': int(pos.entry_price * pos.quantity * 100)
                    })

            return {
                'market_positions': positions,
                'cursor': None
            }
        else:
            return await self.kalshi_client.get_positions(ticker=ticker)

    async def place_order(
        self,
        ticker: str,
        client_order_id: str,
        side: str,
        action: str,
        count: int,
        type_: str = "market",
        yes_price: Optional[int] = None,
        no_price: Optional[int] = None,
        expiration_ts: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Place a trading order.

        Args:
            ticker: Market ticker
            client_order_id: Unique client order ID
            side: "yes" or "no"
            action: "buy" or "sell"
            count: Number of contracts
            type_: Order type ("market" or "limit")
            yes_price: Yes price in cents (for limit orders)
            no_price: No price in cents (for limit orders)
            expiration_ts: Order expiration timestamp

        Returns:
            Order response dict
        """
        if self.paper_trading:
            result = await self.paper_engine.place_order(
                ticker=ticker,
                side=side,
                action=action,
                count=count,
                type_=type_,
                yes_price=yes_price,
                no_price=no_price
            )

            # Log to console for visibility
            if result.get("order"):
                order = result["order"]
                self.logger.info(
                    f"📝 PAPER TRADE: {action.upper()} {count} {ticker} {side.upper()} @ {order['price']/100:.2f}",
                    order_id=order["order_id"]
                )
            else:
                self.logger.warning(
                    f"❌ PAPER TRADE FAILED: {result.get('error', 'Unknown error')}"
                )

            return result
        else:
            self.logger.warning(
                f"🚨 LIVE TRADE: {action.upper()} {count} {ticker} {side.upper()}"
            )
            return await self.kalshi_client.place_order(
                ticker=ticker,
                client_order_id=client_order_id,
                side=side,
                action=action,
                count=count,
                type_=type_,
                yes_price=yes_price,
                no_price=no_price,
                expiration_ts=expiration_ts
            )

    async def get_markets(
        self,
        limit: int = 100,
        cursor: Optional[str] = None,
        event_ticker: Optional[str] = None,
        series_ticker: Optional[str] = None,
        status: Optional[str] = None,
        tickers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get markets data (same for both modes - real market data).

        Returns:
            Markets data dict
        """
        return await self.kalshi_client.get_markets(
            limit=limit,
            cursor=cursor,
            event_ticker=event_ticker,
            series_ticker=series_ticker,
            status=status,
            tickers=tickers
        )

    async def get_market(self, ticker: str) -> Dict[str, Any]:
        """
        Get specific market data (same for both modes - real market data).

        Returns:
            Market data dict
        """
        return await self.kalshi_client.get_market(ticker)

    async def get_orderbook(self, ticker: str, depth: int = 100) -> Dict[str, Any]:
        """
        Get market orderbook (same for both modes - real market data).

        Returns:
            Orderbook data dict
        """
        return await self.kalshi_client.get_orderbook(ticker, depth=depth)

    async def update_paper_positions(self):
        """Update paper trading positions with current prices (no-op for live)."""
        if self.paper_trading:
            await self.paper_engine.update_positions()

    def get_paper_performance_summary(self) -> Optional[Dict]:
        """Get paper trading performance summary (None for live trading)."""
        if self.paper_trading:
            return self.paper_engine.get_performance_summary()
        return None

    def reset_paper_account(self, starting_balance: float = 10000.0):
        """Reset paper trading account (only works in paper mode)."""
        if self.paper_trading:
            self.paper_engine.reset_account(starting_balance)
            self.logger.info(f"Paper trading account reset to ${starting_balance}")
        else:
            self.logger.warning("Cannot reset account in live trading mode")

    async def close(self):
        """Close the client."""
        await self.kalshi_client.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Helper function to create trading client based on settings
def create_trading_client(
    db_manager: DatabaseManager,
    force_paper_trading: Optional[bool] = None
) -> TradingClientWrapper:
    """
    Create a trading client with appropriate mode.

    Args:
        db_manager: Database manager instance
        force_paper_trading: Force paper trading mode

    Returns:
        TradingClientWrapper instance
    """
    return TradingClientWrapper(
        db_manager=db_manager,
        force_paper_trading=force_paper_trading
    )
