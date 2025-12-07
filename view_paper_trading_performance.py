#!/usr/bin/env python3
"""
View Paper Trading Performance

Display detailed performance metrics from paper trading mode.
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from src.utils.database import DatabaseManager
from src.clients.kalshi_client import KalshiClient
from src.utils.paper_trading import PaperTradingEngine


def format_currency(value: float) -> str:
    """Format value as currency."""
    return f"${value:,.2f}"


def format_percentage(value: float) -> str:
    """Format value as percentage."""
    return f"{value:+.2f}%"


async def main():
    """Display paper trading performance."""
    print("=" * 80)
    print("PAPER TRADING PERFORMANCE REPORT".center(80))
    print("=" * 80)
    print()

    # Check if paper trading data exists
    account_file = "logs/paper_account.json"
    positions_file = "logs/paper_positions.json"

    if not os.path.exists(account_file):
        print("❌ No paper trading data found!")
        print("   Start the bot in paper trading mode to generate performance data.")
        print()
        print("   Run: python beast_mode_bot.py")
        return

    # Initialize components
    db_manager = DatabaseManager()
    await db_manager.initialize()

    kalshi_client = KalshiClient()
    paper_engine = PaperTradingEngine(db_manager, kalshi_client)

    # Update positions with current prices
    print("📊 Updating positions with current market prices...")
    await paper_engine.update_positions()
    print()

    # Get performance summary
    summary = paper_engine.get_performance_summary()

    # Display Account Summary
    print("💰 ACCOUNT SUMMARY")
    print("-" * 80)
    print(f"  Starting Balance:        {format_currency(summary['starting_balance'])}")
    print(f"  Current Balance:         {format_currency(summary['current_balance'])}")
    print(f"  Total P&L:              {format_currency(summary['total_pnl'])} ({format_percentage(summary['total_pnl_pct'])})")
    print(f"  Realized P&L:           {format_currency(summary['realized_pnl'])}")
    print(f"  Unrealized P&L:         {format_currency(summary['unrealized_pnl'])}")
    print(f"  Total Fees Paid:        {format_currency(summary['total_fees'])}")
    print()

    # Display Trading Statistics
    print("📈 TRADING STATISTICS")
    print("-" * 80)
    print(f"  Total Trades:           {summary['total_trades']}")
    print(f"  Winning Trades:         {summary['winning_trades']}")
    print(f"  Losing Trades:          {summary['losing_trades']}")
    print(f"  Win Rate:               {summary['win_rate']:.1f}%")

    if summary['winning_trades'] > 0:
        print(f"  Average Win:            {format_currency(summary['avg_win'])}")
    if summary['losing_trades'] > 0:
        print(f"  Average Loss:           {format_currency(summary['avg_loss'])}")

    if summary['winning_trades'] > 0 and summary['losing_trades'] > 0:
        profit_factor = abs(summary['avg_win'] * summary['winning_trades']) / abs(summary['avg_loss'] * summary['losing_trades'])
        print(f"  Profit Factor:          {profit_factor:.2f}")

    print()

    # Display Risk Metrics
    print("⚠️  RISK METRICS")
    print("-" * 80)
    print(f"  Maximum Drawdown:       {summary['max_drawdown']:.2f}%")
    print()

    # Display Open Positions
    print("📍 OPEN POSITIONS")
    print("-" * 80)

    if summary['open_positions'] > 0:
        print(f"  Total Open Positions:   {summary['open_positions']}")
        print(f"  Positions Value:        {format_currency(summary['positions_value'])}")
        print()

        for ticker, pos in paper_engine.positions.items():
            pnl_pct = (pos.unrealized_pnl / (pos.entry_price * pos.quantity) * 100) if pos.quantity > 0 else 0
            print(f"  {ticker}")
            print(f"    Side:            {pos.side}")
            print(f"    Quantity:        {pos.quantity}")
            print(f"    Entry Price:     {format_currency(pos.entry_price)}")
            print(f"    Current Price:   {format_currency(pos.current_price)}")
            print(f"    Unrealized P&L:  {format_currency(pos.unrealized_pnl)} ({format_percentage(pnl_pct)})")
            if pos.strategy:
                print(f"    Strategy:        {pos.strategy}")
            print()
    else:
        print("  No open positions")
        print()

    # Display Recent Trades from Database
    print("📜 RECENT CLOSED TRADES (Last 10)")
    print("-" * 80)

    import aiosqlite
    async with aiosqlite.connect(db_manager.db_path) as db:
        cursor = await db.execute("""
            SELECT market_id, side, entry_price, exit_price, quantity, pnl,
                   entry_timestamp, exit_timestamp, strategy
            FROM trade_logs
            WHERE strategy = 'paper_trading'
            ORDER BY exit_timestamp DESC
            LIMIT 10
        """)

        rows = await cursor.fetchall()

        if rows:
            for row in rows:
                market_id, side, entry_price, exit_price, quantity, pnl, entry_ts, exit_ts, strategy = row

                pnl_pct = (pnl / (entry_price * quantity) * 100) if quantity > 0 else 0
                duration = datetime.fromisoformat(exit_ts) - datetime.fromisoformat(entry_ts)

                print(f"  {market_id}")
                print(f"    Side:         {side}")
                print(f"    Quantity:     {quantity}")
                print(f"    Entry:        {format_currency(entry_price)} @ {entry_ts}")
                print(f"    Exit:         {format_currency(exit_price)} @ {exit_ts}")
                print(f"    P&L:          {format_currency(pnl)} ({format_percentage(pnl_pct)})")
                print(f"    Duration:     {duration}")
                print()
        else:
            print("  No closed trades yet")
            print()

    # Display Recommendations
    print("💡 RECOMMENDATIONS")
    print("-" * 80)

    if summary['total_trades'] < 10:
        print("  ⚠️  Not enough trades for meaningful analysis (< 10 trades)")
        print("     Continue paper trading to build more data")
    elif summary['win_rate'] < 40:
        print("  ❌ Low win rate - Review strategy and AI model selection")
    elif summary['total_pnl_pct'] < -5:
        print("  ❌ Negative P&L - Do NOT use live trading yet")
        print("     Review and optimize strategy in paper mode")
    elif summary['total_pnl_pct'] > 5 and summary['win_rate'] > 50:
        print("  ✅ Positive performance! Consider:")
        print("     - Continue paper trading for 30+ days")
        print("     - Analyze consistency over time")
        print("     - Test with different market conditions")
        print("     - Start live trading with small capital ($100-500)")
    else:
        print("  📊 Performance is neutral - Continue monitoring")

    print()

    # Display Trading Mode Info
    print("⚙️  TRADING MODE")
    print("-" * 80)
    print("  Mode:                   PAPER TRADING")
    print("  Slippage Simulation:    Enabled (5 bps)")
    print("  Fee Simulation:         Enabled (0.7%)")
    print()
    print("  To switch to live trading:")
    print("  1. Ensure consistent profitability for 30+ days")
    print("  2. Update .env: LIVE_TRADING_ENABLED=true")
    print("  3. Update settings.py: paper_trading_mode = False")
    print("  4. Start with small capital ($100-500)")
    print()

    print("=" * 80)

    await kalshi_client.close()


if __name__ == "__main__":
    asyncio.run(main())
