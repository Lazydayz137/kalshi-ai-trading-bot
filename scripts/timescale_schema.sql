-- TimescaleDB Schema for Kalshi AI Trading Bot
-- Optimized for long-term time series storage with automatic partitioning,
-- compression, and retention policies

-- ============================================================================
-- EXTENSIONS
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ============================================================================
-- RAW TICK DATA (High Frequency - 1 second to 1 minute intervals)
-- ============================================================================

-- Price ticks: Every price update from WebSocket
CREATE TABLE IF NOT EXISTS price_ticks (
    time TIMESTAMPTZ NOT NULL,
    ticker TEXT NOT NULL,
    price NUMERIC(10, 4) NOT NULL,  -- 0.0001 precision for prediction markets
    bid NUMERIC(10, 4),
    ask NUMERIC(10, 4),
    spread NUMERIC(10, 4),
    volume INTEGER DEFAULT 0,
    last_trade_price NUMERIC(10, 4),
    last_trade_size INTEGER,
    -- Metadata
    source TEXT DEFAULT 'websocket',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert to hypertable (automatic time-based partitioning)
SELECT create_hypertable('price_ticks', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_price_ticks_ticker_time
    ON price_ticks (ticker, time DESC);
CREATE INDEX IF NOT EXISTS idx_price_ticks_time
    ON price_ticks (time DESC);

-- Compression policy: Compress data older than 7 days
ALTER TABLE price_ticks SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'ticker',
    timescaledb.compress_orderby = 'time DESC'
);

SELECT add_compression_policy('price_ticks', INTERVAL '7 days', if_not_exists => TRUE);

-- Retention policy: Keep raw ticks for 90 days
SELECT add_retention_policy('price_ticks', INTERVAL '90 days', if_not_exists => TRUE);

-- ============================================================================
-- ORDER BOOK SNAPSHOTS (Medium Frequency - 1 second intervals)
-- ============================================================================

CREATE TABLE IF NOT EXISTS orderbook_snapshots (
    time TIMESTAMPTZ NOT NULL,
    ticker TEXT NOT NULL,
    -- Bid side (buy orders)
    bid_prices NUMERIC(10, 4)[] NOT NULL,  -- Array of prices
    bid_sizes INTEGER[] NOT NULL,          -- Array of sizes at each price
    bid_depth_5 NUMERIC(10, 4),            -- Total volume in top 5 levels
    bid_depth_10 NUMERIC(10, 4),
    -- Ask side (sell orders)
    ask_prices NUMERIC(10, 4)[] NOT NULL,
    ask_sizes INTEGER[] NOT NULL,
    ask_depth_5 NUMERIC(10, 4),
    ask_depth_10 NUMERIC(10, 4),
    -- Computed metrics
    mid_price NUMERIC(10, 4),
    weighted_mid_price NUMERIC(10, 4),     -- Volume-weighted
    order_book_imbalance NUMERIC(6, 4),    -- Buy pressure / Sell pressure
    spread_bps INTEGER,                     -- Spread in basis points
    -- Metadata
    total_bid_volume INTEGER,
    total_ask_volume INTEGER,
    num_bid_levels INTEGER,
    num_ask_levels INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('orderbook_snapshots', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_orderbook_ticker_time
    ON orderbook_snapshots (ticker, time DESC);

-- Compression: Compress after 7 days
ALTER TABLE orderbook_snapshots SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'ticker',
    timescaledb.compress_orderby = 'time DESC'
);

SELECT add_compression_policy('orderbook_snapshots', INTERVAL '7 days', if_not_exists => TRUE);

-- Retention: Keep for 90 days
SELECT add_retention_policy('orderbook_snapshots', INTERVAL '90 days', if_not_exists => TRUE);

-- ============================================================================
-- TRADE EXECUTIONS (Event-based)
-- ============================================================================

CREATE TABLE IF NOT EXISTS trade_executions (
    time TIMESTAMPTZ NOT NULL,
    ticker TEXT NOT NULL,
    trade_id TEXT,
    price NUMERIC(10, 4) NOT NULL,
    size INTEGER NOT NULL,
    side TEXT CHECK (side IN ('buy', 'sell')),  -- Aggressor side
    -- Trade characteristics
    is_market_maker BOOLEAN DEFAULT FALSE,
    execution_cost NUMERIC(10, 4),  -- Effective spread
    -- Flow toxicity indicators
    price_impact NUMERIC(10, 4),    -- How much price moved
    volume_imbalance NUMERIC(6, 4), -- Net buying pressure
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('trade_executions', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_trades_ticker_time
    ON trade_executions (ticker, time DESC);
CREATE INDEX IF NOT EXISTS idx_trades_side
    ON trade_executions (ticker, side, time DESC);

-- Compression after 7 days
ALTER TABLE trade_executions SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'ticker',
    timescaledb.compress_orderby = 'time DESC'
);

SELECT add_compression_policy('trade_executions', INTERVAL '7 days', if_not_exists => TRUE);

-- Retention: Keep for 1 year (trades are valuable for analysis)
SELECT add_retention_policy('trade_executions', INTERVAL '365 days', if_not_exists => TRUE);

-- ============================================================================
-- COMPUTED FEATURES (1 minute frequency)
-- ============================================================================

CREATE TABLE IF NOT EXISTS market_features_1min (
    time TIMESTAMPTZ NOT NULL,
    ticker TEXT NOT NULL,
    -- Price features
    open NUMERIC(10, 4),
    high NUMERIC(10, 4),
    low NUMERIC(10, 4),
    close NUMERIC(10, 4),
    vwap NUMERIC(10, 4),  -- Volume-weighted average price
    -- Volume features
    volume INTEGER,
    trade_count INTEGER,
    buy_volume INTEGER,
    sell_volume INTEGER,
    volume_imbalance NUMERIC(6, 4),  -- (buy - sell) / (buy + sell)
    -- Momentum features
    returns_1min NUMERIC(8, 6),   -- 1-minute return
    momentum_5min NUMERIC(8, 6),  -- 5-minute momentum
    momentum_15min NUMERIC(8, 6),
    momentum_1hour NUMERIC(8, 6),
    -- Volatility features
    volatility_5min NUMERIC(8, 6),   -- Rolling std dev
    volatility_15min NUMERIC(8, 6),
    volatility_1hour NUMERIC(8, 6),
    realized_volatility NUMERIC(8, 6),
    -- Technical indicators
    rsi_14 NUMERIC(6, 4),             -- RSI with 14-period
    macd NUMERIC(8, 6),               -- MACD line
    macd_signal NUMERIC(8, 6),        -- Signal line
    macd_histogram NUMERIC(8, 6),     -- Histogram
    bollinger_upper NUMERIC(10, 4),
    bollinger_lower NUMERIC(10, 4),
    bollinger_width NUMERIC(6, 4),    -- Band width
    -- Order flow features
    order_flow_imbalance NUMERIC(6, 4),
    trade_flow_toxicity NUMERIC(6, 4),  -- Kyle's lambda
    effective_spread NUMERIC(10, 4),
    price_impact NUMERIC(8, 6),
    -- Microstructure
    spread_mean NUMERIC(10, 4),
    spread_std NUMERIC(10, 4),
    depth_imbalance NUMERIC(6, 4),
    -- Metadata
    data_quality NUMERIC(3, 2) DEFAULT 1.0,  -- 1.0 = perfect, 0.0 = missing
    created_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('market_features_1min', 'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_features_ticker_time
    ON market_features_1min (ticker, time DESC);

-- Compression after 30 days (features are frequently accessed)
ALTER TABLE market_features_1min SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'ticker',
    timescaledb.compress_orderby = 'time DESC'
);

SELECT add_compression_policy('market_features_1min', INTERVAL '30 days', if_not_exists => TRUE);

-- Retention: Keep for 2 years (ML training data)
SELECT add_retention_policy('market_features_1min', INTERVAL '730 days', if_not_exists => TRUE);

-- ============================================================================
-- AGGREGATED FEATURES (15 minute frequency for faster queries)
-- ============================================================================

-- Continuous aggregate: Pre-compute 15-min aggregates
CREATE MATERIALIZED VIEW IF NOT EXISTS market_features_15min
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('15 minutes', time) AS time,
    ticker,
    -- OHLCV
    first(open, time) as open,
    max(high) as high,
    min(low) as low,
    last(close, time) as close,
    avg(vwap) as vwap,
    sum(volume) as volume,
    -- Averages
    avg(rsi_14) as rsi_14_avg,
    avg(volatility_15min) as volatility_avg,
    avg(order_flow_imbalance) as order_flow_imbalance_avg,
    avg(spread_mean) as spread_avg,
    -- Min/Max
    max(momentum_15min) as momentum_max,
    min(momentum_15min) as momentum_min,
    -- Metadata
    count(*) as num_samples,
    avg(data_quality) as data_quality
FROM market_features_1min
GROUP BY time_bucket('15 minutes', time), ticker;

-- Refresh policy: Update every 15 minutes
SELECT add_continuous_aggregate_policy('market_features_15min',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '15 minutes',
    schedule_interval => INTERVAL '15 minutes',
    if_not_exists => TRUE
);

-- ============================================================================
-- MARKET METADATA (Slowly changing dimensions)
-- ============================================================================

CREATE TABLE IF NOT EXISTS market_metadata (
    ticker TEXT PRIMARY KEY,
    title TEXT,
    category TEXT,
    market_type TEXT,
    strike_type TEXT,
    floor_strike NUMERIC(20, 2),
    cap_strike NUMERIC(20, 2),
    open_time TIMESTAMPTZ,
    close_time TIMESTAMPTZ,
    settlement_value NUMERIC(10, 4),
    status TEXT,
    -- Metadata
    first_seen TIMESTAMPTZ DEFAULT NOW(),
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_market_metadata_category
    ON market_metadata (category);
CREATE INDEX IF NOT EXISTS idx_market_metadata_close_time
    ON market_metadata (close_time) WHERE is_active = TRUE;

-- ============================================================================
-- ML MODEL PREDICTIONS (For tracking and validation)
-- ============================================================================

CREATE TABLE IF NOT EXISTS model_predictions (
    time TIMESTAMPTZ NOT NULL,
    ticker TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    -- Predictions
    predicted_price_1h NUMERIC(10, 4),
    predicted_price_4h NUMERIC(10, 4),
    predicted_price_24h NUMERIC(10, 4),
    predicted_direction TEXT CHECK (predicted_direction IN ('up', 'down', 'neutral')),
    confidence NUMERIC(4, 3),  -- 0.000 to 1.000
    -- Features used (for debugging)
    feature_vector JSONB,
    -- Metadata
    inference_time_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('model_predictions', 'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_predictions_ticker_time
    ON model_predictions (ticker, time DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_model
    ON model_predictions (model_name, model_version, time DESC);

-- Compression after 30 days
ALTER TABLE model_predictions SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'ticker, model_name',
    timescaledb.compress_orderby = 'time DESC'
);

SELECT add_compression_policy('model_predictions', INTERVAL '30 days', if_not_exists => TRUE);

-- Retention: Keep for 1 year (model performance tracking)
SELECT add_retention_policy('model_predictions', INTERVAL '365 days', if_not_exists => TRUE);

-- ============================================================================
-- PREDICTION VALIDATION (Compare predictions vs actuals)
-- ============================================================================

CREATE TABLE IF NOT EXISTS prediction_validation (
    prediction_time TIMESTAMPTZ NOT NULL,
    validation_time TIMESTAMPTZ NOT NULL,
    ticker TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    -- Prediction
    predicted_price NUMERIC(10, 4),
    predicted_direction TEXT,
    confidence NUMERIC(4, 3),
    -- Actual
    actual_price NUMERIC(10, 4),
    actual_direction TEXT,
    -- Error metrics
    absolute_error NUMERIC(10, 4),
    squared_error NUMERIC(12, 6),
    directional_accuracy BOOLEAN,  -- Did we get direction right?
    -- Metadata
    horizon_hours INTEGER,  -- How far ahead was the prediction?
    created_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('prediction_validation', 'validation_time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_validation_model
    ON prediction_validation (model_name, model_version, validation_time DESC);

-- ============================================================================
-- DATA QUALITY MONITORING
-- ============================================================================

CREATE TABLE IF NOT EXISTS data_quality_metrics (
    time TIMESTAMPTZ NOT NULL,
    table_name TEXT NOT NULL,
    -- Metrics
    row_count BIGINT,
    missing_data_pct NUMERIC(5, 2),
    duplicate_count INTEGER,
    out_of_order_count INTEGER,
    -- Latency metrics
    avg_ingestion_delay_ms INTEGER,
    max_ingestion_delay_ms INTEGER,
    -- Coverage
    tickers_tracked INTEGER,
    tickers_with_data INTEGER,
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('data_quality_metrics', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- ============================================================================
-- UTILITY FUNCTIONS
-- ============================================================================

-- Function to get latest price for a ticker
CREATE OR REPLACE FUNCTION get_latest_price(ticker_name TEXT)
RETURNS NUMERIC AS $$
    SELECT price
    FROM price_ticks
    WHERE ticker = ticker_name
    ORDER BY time DESC
    LIMIT 1;
$$ LANGUAGE SQL STABLE;

-- Function to get OHLCV for a time range
CREATE OR REPLACE FUNCTION get_ohlcv(
    ticker_name TEXT,
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    interval_minutes INTEGER DEFAULT 1
)
RETURNS TABLE (
    time TIMESTAMPTZ,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume INTEGER
) AS $$
    SELECT
        time_bucket(interval_minutes * INTERVAL '1 minute', time) AS time,
        first(price, time) AS open,
        max(price) AS high,
        min(price) AS low,
        last(price, time) AS close,
        sum(volume) AS volume
    FROM price_ticks
    WHERE ticker = ticker_name
        AND time >= start_time
        AND time <= end_time
    GROUP BY time_bucket(interval_minutes * INTERVAL '1 minute', time)
    ORDER BY time;
$$ LANGUAGE SQL STABLE;

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- Latest market state
CREATE OR REPLACE VIEW latest_market_state AS
SELECT DISTINCT ON (ticker)
    ticker,
    time,
    price,
    bid,
    ask,
    spread,
    volume
FROM price_ticks
ORDER BY ticker, time DESC;

-- Market summary (last 24 hours)
CREATE OR REPLACE VIEW market_summary_24h AS
SELECT
    ticker,
    count(*) as tick_count,
    avg(price) as avg_price,
    min(price) as min_price,
    max(price) as max_price,
    stddev(price) as price_volatility,
    sum(volume) as total_volume
FROM price_ticks
WHERE time >= NOW() - INTERVAL '24 hours'
GROUP BY ticker;

-- ============================================================================
-- GRANTS (Adjust based on your user setup)
-- ============================================================================

-- Grant permissions to trading bot user
-- GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA public TO kalshi_trading_bot;
-- GRANT USAGE ON SCHEMA public TO kalshi_trading_bot;

-- ============================================================================
-- COMMENTS FOR DOCUMENTATION
-- ============================================================================

COMMENT ON TABLE price_ticks IS
    'Raw price tick data from WebSocket. Partitioned by day, compressed after 7 days, retained for 90 days.';

COMMENT ON TABLE orderbook_snapshots IS
    'Order book snapshots at 1-second intervals. Stores bid/ask depth and computed metrics.';

COMMENT ON TABLE trade_executions IS
    'All trade executions with flow toxicity indicators. Retained for 1 year for ML training.';

COMMENT ON TABLE market_features_1min IS
    'Computed features at 1-minute frequency. Primary table for ML model training. Retained for 2 years.';

COMMENT ON TABLE model_predictions IS
    'ML model predictions for validation and tracking. Retained for 1 year.';

COMMENT ON TABLE prediction_validation IS
    'Comparison of predictions vs actual outcomes for model performance tracking.';
