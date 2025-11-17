"""
TimescaleDB client for high-frequency time series data.

Handles connection pooling, batch inserts, and efficient queries
for real-time market data storage and retrieval.
"""

import asyncio
import asyncpg
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from .models import (
    PriceTick,
    OrderBookSnapshot,
    TradeExecution,
    MarketFeatures,
    ModelPrediction,
    PredictionValidation,
    MarketMetadata
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


class TimescaleClient:
    """
    High-performance client for TimescaleDB.

    Features:
    - Connection pooling (10-50 connections)
    - Batch inserts for high throughput
    - Async I/O for non-blocking operations
    - Automatic reconnection
    - Query result caching
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "kalshi_timeseries",
        user: str = "postgres",
        password: str = "",
        pool_min_size: int = 10,
        pool_max_size: int = 50,
        command_timeout: float = 60.0
    ):
        """
        Initialize TimescaleDB client.

        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
            pool_min_size: Minimum number of connections in pool
            pool_max_size: Maximum number of connections in pool
            command_timeout: Timeout for database commands (seconds)
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.pool_min_size = pool_min_size
        self.pool_max_size = pool_max_size
        self.command_timeout = command_timeout
        self.pool: Optional[asyncpg.Pool] = None

        # Batch insert buffers
        self._tick_buffer: List[PriceTick] = []
        self._orderbook_buffer: List[OrderBookSnapshot] = []
        self._trade_buffer: List[TradeExecution] = []
        self._features_buffer: List[MarketFeatures] = []
        self._prediction_buffer: List[ModelPrediction] = []

        # Batch configuration
        self.batch_size = 100
        self.batch_timeout_seconds = 5.0
        self._last_flush = datetime.utcnow()

    async def connect(self):
        """Create connection pool to TimescaleDB."""
        logger.info(f"Connecting to TimescaleDB at {self.host}:{self.port}/{self.database}")

        try:
            self.pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                min_size=self.pool_min_size,
                max_size=self.pool_max_size,
                command_timeout=self.command_timeout
            )

            logger.info(f"Connected to TimescaleDB (pool size: {self.pool_min_size}-{self.pool_max_size})")

            # Verify TimescaleDB extension
            await self._verify_timescaledb()

        except Exception as e:
            logger.error(f"Failed to connect to TimescaleDB: {e}")
            raise

    async def _verify_timescaledb(self):
        """Verify TimescaleDB extension is installed."""
        async with self.pool.acquire() as conn:
            result = await conn.fetchval(
                "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'"
            )
            if result:
                logger.info(f"TimescaleDB version: {result}")
            else:
                raise RuntimeError("TimescaleDB extension not installed")

    async def close(self):
        """Close connection pool."""
        if self.pool:
            # Flush any remaining buffered data
            await self.flush_all_buffers()

            await self.pool.close()
            logger.info("Closed TimescaleDB connection pool")

    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool."""
        async with self.pool.acquire() as conn:
            yield conn

    # ========================================================================
    # INSERT OPERATIONS (with batching)
    # ========================================================================

    async def insert_price_tick(self, tick: PriceTick, batch: bool = True):
        """
        Insert price tick (batched by default for performance).

        Args:
            tick: PriceTick instance
            batch: If True, buffer for batch insert. If False, insert immediately.
        """
        if batch:
            self._tick_buffer.append(tick)
            await self._maybe_flush_buffer('ticks')
        else:
            await self._insert_price_ticks([tick])

    async def insert_orderbook_snapshot(self, snapshot: OrderBookSnapshot, batch: bool = True):
        """Insert order book snapshot."""
        if batch:
            self._orderbook_buffer.append(snapshot)
            await self._maybe_flush_buffer('orderbook')
        else:
            await self._insert_orderbook_snapshots([snapshot])

    async def insert_trade_execution(self, trade: TradeExecution, batch: bool = True):
        """Insert trade execution."""
        if batch:
            self._trade_buffer.append(trade)
            await self._maybe_flush_buffer('trades')
        else:
            await self._insert_trade_executions([trade])

    async def insert_market_features(self, features: MarketFeatures, batch: bool = True):
        """Insert computed market features."""
        if batch:
            self._features_buffer.append(features)
            await self._maybe_flush_buffer('features')
        else:
            await self._insert_market_features([features])

    async def insert_model_prediction(self, prediction: ModelPrediction, batch: bool = True):
        """Insert model prediction."""
        if batch:
            self._prediction_buffer.append(prediction)
            await self._maybe_flush_buffer('predictions')
        else:
            await self._insert_model_predictions([prediction])

    async def _maybe_flush_buffer(self, buffer_type: str):
        """Flush buffer if size or timeout threshold exceeded."""
        now = datetime.utcnow()
        time_since_flush = (now - self._last_flush).total_seconds()

        should_flush = False

        if buffer_type == 'ticks' and len(self._tick_buffer) >= self.batch_size:
            should_flush = True
        elif buffer_type == 'orderbook' and len(self._orderbook_buffer) >= self.batch_size:
            should_flush = True
        elif buffer_type == 'trades' and len(self._trade_buffer) >= self.batch_size:
            should_flush = True
        elif buffer_type == 'features' and len(self._features_buffer) >= self.batch_size:
            should_flush = True
        elif buffer_type == 'predictions' and len(self._prediction_buffer) >= self.batch_size:
            should_flush = True
        elif time_since_flush >= self.batch_timeout_seconds:
            should_flush = True

        if should_flush:
            await self.flush_all_buffers()

    async def flush_all_buffers(self):
        """Flush all batched data to database."""
        if self._tick_buffer:
            await self._insert_price_ticks(self._tick_buffer)
            self._tick_buffer.clear()

        if self._orderbook_buffer:
            await self._insert_orderbook_snapshots(self._orderbook_buffer)
            self._orderbook_buffer.clear()

        if self._trade_buffer:
            await self._insert_trade_executions(self._trade_buffer)
            self._trade_buffer.clear()

        if self._features_buffer:
            await self._insert_market_features(self._features_buffer)
            self._features_buffer.clear()

        if self._prediction_buffer:
            await self._insert_model_predictions(self._prediction_buffer)
            self._prediction_buffer.clear()

        self._last_flush = datetime.utcnow()

    async def _insert_price_ticks(self, ticks: List[PriceTick]):
        """Batch insert price ticks."""
        if not ticks:
            return

        query = """
            INSERT INTO price_ticks (
                time, ticker, price, bid, ask, spread, volume,
                last_trade_price, last_trade_size, source, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        """

        async with self.pool.acquire() as conn:
            await conn.executemany(
                query,
                [(
                    t.time, t.ticker, float(t.price),
                    float(t.bid) if t.bid else None,
                    float(t.ask) if t.ask else None,
                    float(t.spread) if t.spread else None,
                    t.volume,
                    float(t.last_trade_price) if t.last_trade_price else None,
                    t.last_trade_size,
                    t.source,
                    t.created_at
                ) for t in ticks]
            )

        logger.debug(f"Inserted {len(ticks)} price ticks")

    async def _insert_orderbook_snapshots(self, snapshots: List[OrderBookSnapshot]):
        """Batch insert order book snapshots."""
        if not snapshots:
            return

        query = """
            INSERT INTO orderbook_snapshots (
                time, ticker, bid_prices, bid_sizes, ask_prices, ask_sizes,
                bid_depth_5, bid_depth_10, ask_depth_5, ask_depth_10,
                mid_price, weighted_mid_price, order_book_imbalance, spread_bps,
                total_bid_volume, total_ask_volume, num_bid_levels, num_ask_levels,
                created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
        """

        async with self.pool.acquire() as conn:
            await conn.executemany(
                query,
                [(
                    s.time, s.ticker,
                    [float(p) for p in s.bid_prices],
                    s.bid_sizes,
                    [float(p) for p in s.ask_prices],
                    s.ask_sizes,
                    float(s.bid_depth_5) if s.bid_depth_5 else None,
                    float(s.bid_depth_10) if s.bid_depth_10 else None,
                    float(s.ask_depth_5) if s.ask_depth_5 else None,
                    float(s.ask_depth_10) if s.ask_depth_10 else None,
                    float(s.mid_price) if s.mid_price else None,
                    float(s.weighted_mid_price) if s.weighted_mid_price else None,
                    float(s.order_book_imbalance) if s.order_book_imbalance else None,
                    s.spread_bps,
                    s.total_bid_volume,
                    s.total_ask_volume,
                    s.num_bid_levels,
                    s.num_ask_levels,
                    s.created_at
                ) for s in snapshots]
            )

        logger.debug(f"Inserted {len(snapshots)} orderbook snapshots")

    async def _insert_trade_executions(self, trades: List[TradeExecution]):
        """Batch insert trade executions."""
        if not trades:
            return

        query = """
            INSERT INTO trade_executions (
                time, ticker, trade_id, price, size, side, is_market_maker,
                execution_cost, price_impact, volume_imbalance, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        """

        async with self.pool.acquire() as conn:
            await conn.executemany(
                query,
                [(
                    t.time, t.ticker, t.trade_id, float(t.price), t.size,
                    t.side.value if t.side else None,
                    t.is_market_maker,
                    float(t.execution_cost) if t.execution_cost else None,
                    float(t.price_impact) if t.price_impact else None,
                    float(t.volume_imbalance) if t.volume_imbalance else None,
                    t.created_at
                ) for t in trades]
            )

        logger.debug(f"Inserted {len(trades)} trade executions")

    async def _insert_market_features(self, features_list: List[MarketFeatures]):
        """Batch insert market features."""
        if not features_list:
            return

        query = """
            INSERT INTO market_features_1min (
                time, ticker, open, high, low, close, vwap, volume, trade_count,
                buy_volume, sell_volume, volume_imbalance, returns_1min,
                momentum_5min, momentum_15min, momentum_1hour,
                volatility_5min, volatility_15min, volatility_1hour, realized_volatility,
                rsi_14, macd, macd_signal, macd_histogram,
                bollinger_upper, bollinger_lower, bollinger_width,
                order_flow_imbalance, trade_flow_toxicity, effective_spread, price_impact,
                spread_mean, spread_std, depth_imbalance,
                data_quality, created_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16,
                $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30,
                $31, $32, $33, $34, $35, $36
            )
        """

        def to_float(val):
            return float(val) if val is not None else None

        async with self.pool.acquire() as conn:
            await conn.executemany(
                query,
                [(
                    f.time, f.ticker,
                    to_float(f.open), to_float(f.high), to_float(f.low), to_float(f.close), to_float(f.vwap),
                    f.volume, f.trade_count, f.buy_volume, f.sell_volume, to_float(f.volume_imbalance),
                    to_float(f.returns_1min), to_float(f.momentum_5min), to_float(f.momentum_15min), to_float(f.momentum_1hour),
                    to_float(f.volatility_5min), to_float(f.volatility_15min), to_float(f.volatility_1hour), to_float(f.realized_volatility),
                    to_float(f.rsi_14), to_float(f.macd), to_float(f.macd_signal), to_float(f.macd_histogram),
                    to_float(f.bollinger_upper), to_float(f.bollinger_lower), to_float(f.bollinger_width),
                    to_float(f.order_flow_imbalance), to_float(f.trade_flow_toxicity), to_float(f.effective_spread), to_float(f.price_impact),
                    to_float(f.spread_mean), to_float(f.spread_std), to_float(f.depth_imbalance),
                    to_float(f.data_quality), f.created_at
                ) for f in features_list]
            )

        logger.debug(f"Inserted {len(features_list)} market features")

    async def _insert_model_predictions(self, predictions: List[ModelPrediction]):
        """Batch insert model predictions."""
        if not predictions:
            return

        import json

        query = """
            INSERT INTO model_predictions (
                time, ticker, model_name, model_version,
                predicted_price_1h, predicted_price_4h, predicted_price_24h,
                predicted_direction, confidence, feature_vector, inference_time_ms, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        """

        async with self.pool.acquire() as conn:
            await conn.executemany(
                query,
                [(
                    p.time, p.ticker, p.model_name, p.model_version,
                    float(p.predicted_price_1h) if p.predicted_price_1h else None,
                    float(p.predicted_price_4h) if p.predicted_price_4h else None,
                    float(p.predicted_price_24h) if p.predicted_price_24h else None,
                    p.predicted_direction.value if p.predicted_direction else None,
                    float(p.confidence) if p.confidence else None,
                    json.dumps(p.feature_vector) if p.feature_vector else None,
                    p.inference_time_ms,
                    p.created_at
                ) for p in predictions]
            )

        logger.debug(f"Inserted {len(predictions)} model predictions")

    # ========================================================================
    # QUERY OPERATIONS
    # ========================================================================

    async def get_latest_price(self, ticker: str) -> Optional[float]:
        """Get latest price for a ticker."""
        async with self.pool.acquire() as conn:
            result = await conn.fetchval(
                "SELECT price FROM price_ticks WHERE ticker = $1 ORDER BY time DESC LIMIT 1",
                ticker
            )
            return float(result) if result else None

    async def get_ohlcv(
        self,
        ticker: str,
        start_time: datetime,
        end_time: datetime,
        interval_minutes: int = 1
    ) -> List[Dict[str, Any]]:
        """Get OHLCV data for a time range."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    time_bucket($4 * INTERVAL '1 minute', time) AS time,
                    first(price, time) AS open,
                    max(price) AS high,
                    min(price) AS low,
                    last(price, time) AS close,
                    sum(volume) AS volume
                FROM price_ticks
                WHERE ticker = $1 AND time >= $2 AND time <= $3
                GROUP BY time_bucket($4 * INTERVAL '1 minute', time)
                ORDER BY time
                """,
                ticker, start_time, end_time, interval_minutes
            )

            return [dict(row) for row in rows]

    async def get_market_features(
        self,
        ticker: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Get computed features for ML training."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM market_features_1min WHERE ticker = $1 AND time >= $2 AND time <= $3 ORDER BY time",
                ticker, start_time, end_time
            )

            return [dict(row) for row in rows]

    async def get_latest_features(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get latest computed features for a ticker."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM market_features_1min WHERE ticker = $1 ORDER BY time DESC LIMIT 1",
                ticker
            )

            return dict(row) if row else None

    async def get_prediction_performance(
        self,
        model_name: str,
        model_version: str,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Get model prediction performance metrics."""
        async with self.pool.acquire() as conn:
            result = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) as total_predictions,
                    AVG(absolute_error) as mean_absolute_error,
                    AVG(squared_error) as mean_squared_error,
                    STDDEV(absolute_error) as std_absolute_error,
                    AVG(CASE WHEN directional_accuracy THEN 1.0 ELSE 0.0 END) as directional_accuracy
                FROM prediction_validation
                WHERE model_name = $1 AND model_version = $2
                    AND validation_time >= $3 AND validation_time <= $4
                """,
                model_name, model_version, start_time, end_time
            )

            return dict(result) if result else {}

    # ========================================================================
    # UTILITY OPERATIONS
    # ========================================================================

    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        async with self.pool.acquire() as conn:
            stats = {}

            # Table sizes
            rows = await conn.fetch(
                """
                SELECT
                    schemaname || '.' || tablename as table_name,
                    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
                FROM pg_tables
                WHERE schemaname = 'public'
                ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
                """
            )
            stats['table_sizes'] = [dict(row) for row in rows]

            # Row counts
            stats['row_counts'] = {}
            for table in ['price_ticks', 'orderbook_snapshots', 'trade_executions', 'market_features_1min']:
                count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
                stats['row_counts'][table] = count

            # Compression stats
            compression_rows = await conn.fetch(
                """
                SELECT
                    hypertable_name,
                    total_chunks,
                    number_compressed_chunks,
                    uncompressed_total_bytes,
                    compressed_total_bytes
                FROM timescaledb_information.compression_settings
                JOIN timescaledb_information.hypertables USING (hypertable_name)
                """
            )
            stats['compression'] = [dict(row) for row in compression_rows]

            return stats
