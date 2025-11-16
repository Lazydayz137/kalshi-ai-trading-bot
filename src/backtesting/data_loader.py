"""
Historical Data Loader for Backtesting

This module provides functionality to load and manage historical market data
for backtesting trading strategies.
"""

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import aiosqlite

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MarketSnapshot:
    """Snapshot of market state at a point in time."""

    timestamp: datetime
    ticker: str
    title: str
    last_price: float
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    volume_24h: int
    open_interest: int
    close_time: datetime
    category: str
    metadata: Dict


class HistoricalDataLoader:
    """
    Loader for historical market data from database or CSV files.

    This class provides efficient loading and caching of historical market data
    for backtesting purposes.
    """

    def __init__(self, db_path: Optional[str] = None, csv_dir: Optional[Path] = None):
        """
        Initialize data loader.

        Args:
            db_path: Path to SQLite database with historical data
            csv_dir: Directory containing CSV files with historical data
        """
        self.db_path = db_path
        self.csv_dir = csv_dir
        self.cache: Dict[str, List[MarketSnapshot]] = {}

        if not db_path and not csv_dir:
            logger.warning("No data source specified, will attempt to load from default locations")
            self.db_path = "data/kalshi_historical.db"
            self.csv_dir = Path("data/historical_csv")

    async def load_data(
        self, start_date: datetime, end_date: datetime, frequency: str = "1h", tickers: Optional[List[str]] = None
    ) -> List[Tuple[datetime, Dict[str, Dict]]]:
        """
        Load historical market data for specified time period.

        Args:
            start_date: Start of backtest period
            end_date: End of backtest period
            frequency: Data frequency (1h, 1d, etc.)
            tickers: Optional list of specific tickers to load

        Returns:
            List of (timestamp, market_data) tuples sorted by timestamp
        """
        logger.info(f"Loading historical data: {start_date} to {end_date}, frequency={frequency}")

        # Try loading from database first
        if self.db_path:
            try:
                data = await self._load_from_database(start_date, end_date, frequency, tickers)
                if data:
                    logger.info(f"Loaded {len(data)} data points from database")
                    return data
            except Exception as e:
                logger.warning(f"Failed to load from database: {e}")

        # Fallback to CSV
        if self.csv_dir:
            try:
                data = await self._load_from_csv(start_date, end_date, frequency, tickers)
                if data:
                    logger.info(f"Loaded {len(data)} data points from CSV")
                    return data
            except Exception as e:
                logger.warning(f"Failed to load from CSV: {e}")

        # If both fail, generate synthetic data for testing
        logger.warning("No historical data available, generating synthetic data for testing")
        return await self._generate_synthetic_data(start_date, end_date, frequency, tickers)

    async def _load_from_database(
        self, start_date: datetime, end_date: datetime, frequency: str, tickers: Optional[List[str]]
    ) -> List[Tuple[datetime, Dict[str, Dict]]]:
        """Load data from SQLite database."""
        if not Path(self.db_path).exists():
            logger.debug(f"Database not found: {self.db_path}")
            return []

        data_points = []

        async with aiosqlite.connect(self.db_path) as db:
            # Query historical snapshots
            query = """
                SELECT
                    timestamp,
                    ticker,
                    title,
                    last_price,
                    yes_bid,
                    yes_ask,
                    no_bid,
                    no_ask,
                    volume_24h,
                    open_interest,
                    close_time,
                    category,
                    metadata
                FROM market_snapshots
                WHERE timestamp BETWEEN ? AND ?
            """
            params = [start_date.isoformat(), end_date.isoformat()]

            if tickers:
                query += " AND ticker IN ({})".format(",".join("?" * len(tickers)))
                params.extend(tickers)

            query += " ORDER BY timestamp ASC"

            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()

            # Group by timestamp
            timestamp_groups: Dict[datetime, Dict[str, Dict]] = {}

            for row in rows:
                timestamp = datetime.fromisoformat(row[0])
                ticker = row[1]

                if timestamp not in timestamp_groups:
                    timestamp_groups[timestamp] = {}

                timestamp_groups[ticker] = {
                    "ticker": ticker,
                    "title": row[2],
                    "last_price": row[3],
                    "yes_bid": row[4],
                    "yes_ask": row[5],
                    "no_bid": row[6],
                    "no_ask": row[7],
                    "volume_24h": row[8],
                    "open_interest": row[9],
                    "close_time": datetime.fromisoformat(row[10]) if row[10] else None,
                    "category": row[11],
                    "metadata": json.loads(row[12]) if row[12] else {},
                }

            # Convert to list of tuples
            data_points = sorted(timestamp_groups.items(), key=lambda x: x[0])

        return data_points

    async def _load_from_csv(
        self, start_date: datetime, end_date: datetime, frequency: str, tickers: Optional[List[str]]
    ) -> List[Tuple[datetime, Dict[str, Dict]]]:
        """Load data from CSV files."""
        if not self.csv_dir or not self.csv_dir.exists():
            logger.debug(f"CSV directory not found: {self.csv_dir}")
            return []

        import csv

        data_points = []
        timestamp_groups: Dict[datetime, Dict[str, Dict]] = {}

        # Iterate through CSV files in directory
        for csv_file in self.csv_dir.glob("*.csv"):
            # Parse ticker from filename (e.g., "KXBTC-24DEC-50K.csv")
            ticker = csv_file.stem

            # Skip if tickers filter specified and this isn't in it
            if tickers and ticker not in tickers:
                continue

            # Read CSV file
            with open(csv_file, "r") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    try:
                        timestamp = datetime.fromisoformat(row["timestamp"])

                        # Skip if outside date range
                        if timestamp < start_date or timestamp > end_date:
                            continue

                        if timestamp not in timestamp_groups:
                            timestamp_groups[timestamp] = {}

                        timestamp_groups[timestamp][ticker] = {
                            "ticker": ticker,
                            "title": row.get("title", ticker),
                            "last_price": float(row["last_price"]),
                            "yes_bid": float(row.get("yes_bid", row["last_price"])),
                            "yes_ask": float(row.get("yes_ask", row["last_price"])),
                            "no_bid": float(row.get("no_bid", 1 - float(row["last_price"]))),
                            "no_ask": float(row.get("no_ask", 1 - float(row["last_price"]))),
                            "volume_24h": int(row.get("volume_24h", 0)),
                            "open_interest": int(row.get("open_interest", 0)),
                            "close_time": datetime.fromisoformat(row["close_time"]) if row.get("close_time") else None,
                            "category": row.get("category", "unknown"),
                            "metadata": {},
                        }
                    except (ValueError, KeyError) as e:
                        logger.warning(f"Error parsing CSV row in {csv_file}: {e}")
                        continue

        # Convert to sorted list
        data_points = sorted(timestamp_groups.items(), key=lambda x: x[0])

        return data_points

    async def _generate_synthetic_data(
        self, start_date: datetime, end_date: datetime, frequency: str, tickers: Optional[List[str]]
    ) -> List[Tuple[datetime, Dict[str, Dict]]]:
        """
        Generate synthetic market data for testing.

        This creates realistic-looking market data with random walk price movements.
        """
        import random

        logger.info("Generating synthetic market data for testing")

        # Parse frequency to timedelta
        freq_map = {"1h": timedelta(hours=1), "4h": timedelta(hours=4), "1d": timedelta(days=1)}
        freq_delta = freq_map.get(frequency, timedelta(hours=1))

        # Generate sample tickers if none specified
        if not tickers:
            tickers = [
                "KXBTC-24DEC-50K",
                "HIGHNY-24-JAN15",
                "INXD-24Q1-UP",
                "FED-24MAR-HOLD",
                "SPX-24-5000",
            ]

        data_points = []
        current_time = start_date

        # Initialize random walk for each ticker
        ticker_prices = {ticker: random.uniform(0.3, 0.7) for ticker in tickers}

        while current_time <= end_date:
            market_data = {}

            for ticker in tickers:
                # Random walk with drift
                price_change = random.gauss(0, 0.02)  # 2% volatility
                ticker_prices[ticker] = max(0.01, min(0.99, ticker_prices[ticker] + price_change))

                price = ticker_prices[ticker]
                spread = 0.01  # 1 cent spread

                market_data[ticker] = {
                    "ticker": ticker,
                    "title": f"Synthetic Market: {ticker}",
                    "last_price": round(price, 2),
                    "yes_bid": round(max(0.01, price - spread / 2), 2),
                    "yes_ask": round(min(0.99, price + spread / 2), 2),
                    "no_bid": round(max(0.01, (1 - price) - spread / 2), 2),
                    "no_ask": round(min(0.99, (1 - price) + spread / 2), 2),
                    "volume_24h": random.randint(100, 10000),
                    "open_interest": random.randint(1000, 50000),
                    "close_time": current_time + timedelta(days=random.randint(1, 30)),
                    "category": random.choice(["crypto", "finance", "politics", "economics"]),
                    "metadata": {"synthetic": True},
                }

            data_points.append((current_time, market_data))
            current_time += freq_delta

        logger.info(f"Generated {len(data_points)} synthetic data points")
        return data_points

    async def save_snapshot_to_db(self, snapshot: MarketSnapshot):
        """
        Save a market snapshot to the database.

        This is useful for recording live data for future backtesting.
        """
        if not self.db_path:
            logger.warning("No database path configured, cannot save snapshot")
            return

        # Create database if it doesn't exist
        db_path = Path(self.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(self.db_path) as db:
            # Create table if it doesn't exist
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS market_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    title TEXT,
                    last_price REAL,
                    yes_bid REAL,
                    yes_ask REAL,
                    no_bid REAL,
                    no_ask REAL,
                    volume_24h INTEGER,
                    open_interest INTEGER,
                    close_time TEXT,
                    category TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create index
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp_ticker ON market_snapshots(timestamp, ticker)"
            )

            # Insert snapshot
            await db.execute(
                """
                INSERT INTO market_snapshots (
                    timestamp, ticker, title, last_price, yes_bid, yes_ask,
                    no_bid, no_ask, volume_24h, open_interest, close_time,
                    category, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    snapshot.timestamp.isoformat(),
                    snapshot.ticker,
                    snapshot.title,
                    snapshot.last_price,
                    snapshot.yes_bid,
                    snapshot.yes_ask,
                    snapshot.no_bid,
                    snapshot.no_ask,
                    snapshot.volume_24h,
                    snapshot.open_interest,
                    snapshot.close_time.isoformat() if snapshot.close_time else None,
                    snapshot.category,
                    json.dumps(snapshot.metadata),
                ),
            )

            await db.commit()

    async def export_to_csv(self, start_date: datetime, end_date: datetime, output_dir: Path):
        """
        Export historical data to CSV files (one per ticker).

        Args:
            start_date: Start date for export
            end_date: End date for export
            output_dir: Directory to write CSV files
        """
        logger.info(f"Exporting historical data to CSV: {output_dir}")

        data = await self.load_data(start_date, end_date)

        # Group by ticker
        ticker_data: Dict[str, List] = {}

        for timestamp, market_data in data:
            for ticker, market_info in market_data.items():
                if ticker not in ticker_data:
                    ticker_data[ticker] = []

                ticker_data[ticker].append(
                    {
                        "timestamp": timestamp.isoformat(),
                        "title": market_info["title"],
                        "last_price": market_info["last_price"],
                        "yes_bid": market_info["yes_bid"],
                        "yes_ask": market_info["yes_ask"],
                        "no_bid": market_info["no_bid"],
                        "no_ask": market_info["no_ask"],
                        "volume_24h": market_info["volume_24h"],
                        "open_interest": market_info["open_interest"],
                        "close_time": market_info["close_time"].isoformat() if market_info.get("close_time") else "",
                        "category": market_info.get("category", ""),
                    }
                )

        # Write CSV files
        output_dir.mkdir(parents=True, exist_ok=True)

        import csv

        for ticker, rows in ticker_data.items():
            csv_path = output_dir / f"{ticker}.csv"

            with open(csv_path, "w", newline="") as f:
                if rows:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)

            logger.info(f"Exported {len(rows)} rows to {csv_path}")

        logger.info(f"Export complete: {len(ticker_data)} tickers")


class LiveDataRecorder:
    """
    Records live market data for future backtesting.

    This class can be integrated into the live trading bot to continuously
    record market snapshots for building historical datasets.
    """

    def __init__(self, data_loader: HistoricalDataLoader, record_interval_seconds: int = 3600):
        """
        Initialize live data recorder.

        Args:
            data_loader: HistoricalDataLoader instance for saving data
            record_interval_seconds: How often to record snapshots (default: 1 hour)
        """
        self.data_loader = data_loader
        self.record_interval = record_interval_seconds
        self.is_recording = False

    async def start_recording(self, kalshi_client):
        """
        Start recording live market data.

        Args:
            kalshi_client: Kalshi API client instance
        """
        self.is_recording = True
        logger.info(f"Started live data recording (interval: {self.record_interval}s)")

        while self.is_recording:
            try:
                # Fetch current markets
                markets = await kalshi_client.get_markets()

                # Record snapshot for each market
                for market in markets:
                    snapshot = MarketSnapshot(
                        timestamp=datetime.utcnow(),
                        ticker=market.ticker,
                        title=market.title,
                        last_price=market.last_price,
                        yes_bid=market.yes_bid,
                        yes_ask=market.yes_ask,
                        no_bid=market.no_bid,
                        no_ask=market.no_ask,
                        volume_24h=market.volume_24h,
                        open_interest=market.open_interest,
                        close_time=market.close_time,
                        category=market.category,
                        metadata={},
                    )

                    await self.data_loader.save_snapshot_to_db(snapshot)

                logger.debug(f"Recorded {len(markets)} market snapshots")

            except Exception as e:
                logger.error(f"Error recording market data: {e}")

            # Wait for next interval
            await asyncio.sleep(self.record_interval)

    def stop_recording(self):
        """Stop recording market data."""
        self.is_recording = False
        logger.info("Stopped live data recording")
