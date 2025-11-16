# Database Schema Documentation

Complete database schema reference for the Kalshi AI Trading Bot.

## Overview

The trading bot uses a relational database to store:
- Trading positions (open and closed)
- Market analyses and AI decisions
- Performance metrics and logs
- Historical market snapshots (for backtesting)

**Supported Databases**:
- SQLite (development, single-instance)
- PostgreSQL (production, scalable)

---

## Schema Diagram

```
┌─────────────────────┐         ┌──────────────────────┐
│    positions        │         │  market_analyses     │
├─────────────────────┤         ├──────────────────────┤
│ id (PK)             │         │ id (PK)              │
│ ticker              │◄────────│ ticker               │
│ side                │         │ analysis_text        │
│ quantity            │         │ ai_provider          │
│ entry_price         │         │ cost                 │
│ entry_time          │         │ timestamp            │
│ exit_price          │         │ decision             │
│ exit_time           │         │ confidence           │
│ pnl                 │         └──────────────────────┘
│ status              │
│ reason              │         ┌──────────────────────┐
│ close_reason        │         │  performance_logs    │
└─────────────────────┘         ├──────────────────────┤
                                │ id (PK)              │
┌─────────────────────┐         │ date                 │
│  market_snapshots   │         │ total_pnl            │
├─────────────────────┤         │ num_trades           │
│ id (PK)             │         │ win_rate             │
│ timestamp           │         │ portfolio_value      │
│ ticker              │         │ daily_return         │
│ title               │         └──────────────────────┘
│ last_price          │
│ yes_bid             │         ┌──────────────────────┐
│ yes_ask             │         │  ai_cost_tracking    │
│ no_bid              │         ├──────────────────────┤
│ no_ask              │         │ id (PK)              │
│ volume_24h          │         │ date                 │
│ open_interest       │         │ total_cost           │
│ close_time          │         │ num_calls            │
│ category            │         │ provider             │
│ metadata            │         └──────────────────────┘
│ created_at          │
└─────────────────────┘
```

---

## Table Definitions

### `positions`

Stores all trading positions (open and closed).

**Schema**:
```sql
CREATE TABLE positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,  -- PostgreSQL: SERIAL PRIMARY KEY
    ticker TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('yes', 'no')),
    quantity INTEGER NOT NULL,
    entry_price REAL NOT NULL,
    entry_time TEXT NOT NULL,              -- ISO 8601 timestamp
    exit_price REAL,
    exit_time TEXT,
    pnl REAL,
    status TEXT NOT NULL CHECK (status IN ('open', 'closed')),
    reason TEXT,
    close_reason TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_positions_ticker ON positions(ticker);
CREATE INDEX idx_positions_status ON positions(status);
CREATE INDEX idx_positions_entry_time ON positions(entry_time);
```

**Columns**:
- `id`: Unique position identifier
- `ticker`: Market ticker (e.g., "KXBTC-24DEC-50K")
- `side`: Position side ("yes" or "no")
- `quantity`: Number of contracts
- `entry_price`: Entry price per contract (0.0-1.0)
- `entry_time`: When position was opened (ISO 8601)
- `exit_price`: Exit price per contract (NULL if open)
- `exit_time`: When position was closed (NULL if open)
- `pnl`: Realized profit/loss (NULL if open)
- `status`: "open" or "closed"
- `reason`: Why position was opened
- `close_reason`: Why position was closed (NULL if open)

**Example Data**:
```sql
INSERT INTO positions VALUES (
    1,
    'KXBTC-24DEC-50K',
    'yes',
    100,
    0.65,
    '2024-11-15T10:30:00Z',
    0.72,
    '2024-11-16T15:45:00Z',
    7.00,
    'closed',
    'High confidence AI signal',
    'Profit target reached',
    '2024-11-15T10:30:00Z',
    '2024-11-16T15:45:00Z'
);
```

---

### `market_analyses`

Stores AI analyses and decisions for markets.

**Schema**:
```sql
CREATE TABLE market_analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    analysis_text TEXT,
    ai_provider TEXT NOT NULL,
    cost REAL NOT NULL,
    timestamp TEXT NOT NULL,
    decision TEXT,  -- 'OPEN_YES', 'OPEN_NO', 'CLOSE', 'SKIP'
    confidence REAL,
    metadata TEXT,  -- JSON
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_analyses_ticker ON market_analyses(ticker);
CREATE INDEX idx_analyses_timestamp ON market_analyses(timestamp);
CREATE INDEX idx_analyses_decision ON market_analyses(decision);
```

**Columns**:
- `id`: Unique analysis identifier
- `ticker`: Market ticker analyzed
- `analysis_text`: Full AI analysis output
- `ai_provider`: AI provider used ("xai", "openai", "anthropic")
- `cost`: Cost of AI API call in USD
- `timestamp`: When analysis was performed
- `decision`: Trading decision made
- `confidence`: Confidence score (0.0-1.0)
- `metadata`: Additional JSON metadata

**Example Data**:
```sql
INSERT INTO market_analyses VALUES (
    1,
    'KXBTC-24DEC-50K',
    'Bitcoin is showing strong momentum...',
    'xai',
    0.05,
    '2024-11-15T10:25:00Z',
    'OPEN_YES',
    0.85,
    '{"model": "grok-beta", "tokens": 1250}',
    '2024-11-15T10:25:00Z'
);
```

---

### `market_snapshots`

Historical market data for backtesting.

**Schema**:
```sql
CREATE TABLE market_snapshots (
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
    metadata TEXT,  -- JSON
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_snapshots_timestamp_ticker ON market_snapshots(timestamp, ticker);
CREATE INDEX idx_snapshots_category ON market_snapshots(category);
```

**Purpose**: Records historical market states for backtesting strategies.

**Usage**: Populated by `LiveDataRecorder` during live trading.

---

### `performance_logs`

Daily performance metrics.

**Schema**:
```sql
CREATE TABLE performance_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL UNIQUE,
    total_pnl REAL NOT NULL,
    num_trades INTEGER NOT NULL,
    win_rate REAL,
    portfolio_value REAL NOT NULL,
    daily_return REAL,
    max_drawdown REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_performance_date ON performance_logs(date);
```

**Purpose**: Track daily performance for monitoring and reporting.

---

### `ai_cost_tracking`

AI API cost tracking by date and provider.

**Schema**:
```sql
CREATE TABLE ai_cost_tracking (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    total_cost REAL NOT NULL,
    num_calls INTEGER NOT NULL,
    provider TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_cost_date_provider ON ai_cost_tracking(date, provider);
```

**Purpose**: Monitor and control AI API spending.

---

## Queries

### Common Queries

#### Get all open positions
```sql
SELECT * FROM positions
WHERE status = 'open'
ORDER BY entry_time DESC;
```

#### Calculate daily PnL
```sql
SELECT
    DATE(exit_time) as trade_date,
    SUM(pnl) as total_pnl,
    COUNT(*) as num_trades,
    AVG(pnl) as avg_pnl
FROM positions
WHERE status = 'closed'
GROUP BY DATE(exit_time)
ORDER BY trade_date DESC;
```

#### Get AI cost for today
```sql
SELECT SUM(cost) as total_cost
FROM market_analyses
WHERE DATE(timestamp) = DATE('now');
```

#### Find best performing markets
```sql
SELECT
    ticker,
    COUNT(*) as num_trades,
    SUM(pnl) as total_pnl,
    AVG(pnl) as avg_pnl,
    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate
FROM positions
WHERE status = 'closed'
GROUP BY ticker
HAVING num_trades >= 3
ORDER BY total_pnl DESC
LIMIT 10;
```

#### Get position history for a market
```sql
SELECT
    id,
    side,
    quantity,
    entry_price,
    exit_price,
    pnl,
    reason,
    entry_time,
    exit_time
FROM positions
WHERE ticker = 'KXBTC-24DEC-50K'
ORDER BY entry_time DESC;
```

---

## Database Migrations

### SQLite to PostgreSQL Migration

**Export from SQLite**:
```bash
sqlite3 data/kalshi_trading.db .dump > dump.sql
```

**Import to PostgreSQL**:
```bash
# Create database
createdb kalshi_trading

# Convert SQL types (INTEGER AUTOINCREMENT → SERIAL, etc.)
sed 's/INTEGER PRIMARY KEY AUTOINCREMENT/SERIAL PRIMARY KEY/g' dump.sql > postgres_dump.sql
sed 's/TEXT DEFAULT CURRENT_TIMESTAMP/TIMESTAMP DEFAULT CURRENT_TIMESTAMP/g' postgres_dump.sql > postgres_final.sql

# Import
psql kalshi_trading < postgres_final.sql
```

### Adding New Columns

**Example: Add `strategy_name` to positions**:
```sql
-- SQLite
ALTER TABLE positions ADD COLUMN strategy_name TEXT;

-- PostgreSQL
ALTER TABLE positions ADD COLUMN strategy_name VARCHAR(50);
```

---

## Backup and Restore

### SQLite

**Backup**:
```bash
# Full backup
sqlite3 data/kalshi_trading.db ".backup data/backup_$(date +%Y%m%d).db"

# Export to SQL
sqlite3 data/kalshi_trading.db .dump > backup.sql
```

**Restore**:
```bash
# From backup file
cp data/backup_20241116.db data/kalshi_trading.db

# From SQL dump
sqlite3 data/kalshi_trading.db < backup.sql
```

### PostgreSQL

**Backup**:
```bash
# Full backup
pg_dump kalshi_trading > backup_$(date +%Y%m%d).sql

# Compressed backup
pg_dump kalshi_trading | gzip > backup_$(date +%Y%m%d).sql.gz

# Custom format (parallel restore)
pg_dump -Fc kalshi_trading > backup_$(date +%Y%m%d).dump
```

**Restore**:
```bash
# From SQL backup
psql kalshi_trading < backup_20241116.sql

# From custom format
pg_restore -d kalshi_trading backup_20241116.dump
```

---

## Performance Optimization

### Indexing Strategy

**Current Indexes**:
- `positions(ticker)` - Lookup positions by market
- `positions(status)` - Filter open/closed positions
- `positions(entry_time)` - Time-based queries
- `market_analyses(ticker, timestamp)` - Analysis lookup
- `market_snapshots(timestamp, ticker)` - Backtesting data access

**Add Custom Indexes**:
```sql
-- For category-based analysis
CREATE INDEX idx_snapshots_category_timestamp
ON market_snapshots(category, timestamp);

-- For provider cost analysis
CREATE INDEX idx_analyses_provider_timestamp
ON market_analyses(ai_provider, timestamp);
```

### Query Optimization

**Use EXPLAIN to analyze queries**:
```sql
EXPLAIN QUERY PLAN
SELECT * FROM positions
WHERE ticker = 'KXBTC-24DEC-50K'
AND status = 'open';
```

### Connection Pooling

**PostgreSQL** (via asyncpg):
```python
repo = PostgreSQLRepository(
    host="localhost",
    database="kalshi_trading",
    pool_size=10,       # Minimum connections
    max_overflow=20     # Maximum connections
)
```

---

## Data Retention

### Cleanup Old Data

**Delete old snapshots**:
```sql
-- Keep only last 90 days of snapshots
DELETE FROM market_snapshots
WHERE timestamp < datetime('now', '-90 days');
```

**Archive closed positions**:
```sql
-- Move to archive table
INSERT INTO positions_archive
SELECT * FROM positions
WHERE status = 'closed'
AND exit_time < datetime('now', '-1 year');

DELETE FROM positions
WHERE status = 'closed'
AND exit_time < datetime('now', '-1 year');
```

---

## Monitoring

### Database Size

**SQLite**:
```bash
ls -lh data/kalshi_trading.db
```

**PostgreSQL**:
```sql
SELECT
    pg_size_pretty(pg_database_size('kalshi_trading')) as size;
```

### Table Sizes

**PostgreSQL**:
```sql
SELECT
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

---

**Last Updated**: 2025-11-16
**Version**: 1.0.0
