

# API Reference

Complete API documentation for the Kalshi AI Trading Bot.

## Table of Contents

- [Core Models](#core-models)
- [Configuration](#configuration)
- [Decision Strategies](#decision-strategies)
- [Backtesting](#backtesting)
- [Database](#database)
- [Market Screener](#market-screener)
- [Utilities](#utilities)

---

## Core Models

### `Market`

Represents a Kalshi prediction market.

**Location**: `src/models.py`

**Attributes**:
```python
class Market:
    ticker: str                  # Market identifier (e.g., "KXBTC-24DEC-50K")
    title: str                   # Human-readable market description
    category: str                # Market category (crypto, finance, politics, etc.)
    last_price: float            # Most recent trade price (0.0-1.0)
    yes_bid: float               # Highest bid for YES outcome
    yes_ask: float               # Lowest ask for YES outcome
    no_bid: float                # Highest bid for NO outcome
    no_ask: float                # Lowest ask for NO outcome
    volume_24h: int              # Trading volume in last 24 hours
    open_interest: int           # Total open contracts
    close_time: datetime         # Market expiration time
    market_status: str           # Status: "open", "closed", "settled"
    strike_type: Optional[str]   # Strike type (if applicable)
    floor_strike: Optional[float] # Lower bound (if applicable)
    cap_strike: Optional[float]  # Upper bound (if applicable)
```

**Methods**:
```python
def get_spread() -> float:
    """Calculate bid-ask spread for YES side."""

def get_mid_price() -> float:
    """Calculate mid-price between bid and ask."""

def time_to_expiry() -> timedelta:
    """Calculate time remaining until market expires."""

def is_liquid() -> bool:
    """Check if market has sufficient liquidity."""
```

---

### `Position`

Represents an open or closed trading position.

**Location**: `src/models.py`

**Attributes**:
```python
class Position:
    id: Optional[int]            # Database ID
    ticker: str                  # Market ticker
    side: str                    # "yes" or "no"
    quantity: int                # Number of contracts
    entry_price: float           # Entry price per contract
    entry_time: datetime         # When position was opened
    exit_price: Optional[float]  # Exit price (if closed)
    exit_time: Optional[datetime] # When position was closed
    pnl: Optional[float]         # Realized profit/loss
    status: str                  # "open" or "closed"
    reason: str                  # Why position was opened
    close_reason: Optional[str]  # Why position was closed
```

**Methods**:
```python
def calculate_unrealized_pnl(current_price: float) -> float:
    """Calculate unrealized PnL based on current price."""

def calculate_realized_pnl() -> float:
    """Calculate realized PnL for closed position."""

def get_holding_period() -> timedelta:
    """Calculate how long position has been/was held."""
```

---

## Configuration

### `TradingConfig`

Main configuration object for the trading bot.

**Location**: `src/config/settings_v2.py`

**Sub-Configurations**:

#### Position Sizing
```python
class PositionSizingConfig:
    max_position_size: float = 1000.0      # Max $ per position
    risk_per_trade_pct: float = 2.0        # Max % of capital at risk
    position_sizing_method: str = "kelly"   # "kelly", "fixed", "proportional"
    kelly_fraction: float = 0.25            # Fraction of Kelly to use
```

#### Market Filtering
```python
class MarketFilteringConfig:
    min_volume_24h: int = 100               # Minimum daily volume
    min_open_interest: int = 1000           # Minimum open contracts
    max_spread_pct: float = 5.0             # Maximum spread percentage
    allowed_categories: List[str] = [...]   # Whitelisted categories
    excluded_tickers: List[str] = []        # Blacklisted tickers
```

#### AI Model
```python
class AIModelConfig:
    provider: str = "xai"                   # "xai", "openai", or "anthropic"
    model: str = "grok-beta"                # Model identifier
    temperature: float = 0.1                # Sampling temperature
    max_tokens: int = 4096                  # Max response tokens
    timeout_seconds: int = 30               # Request timeout
```

**Loading Configuration**:
```python
# From YAML file
from src.config.settings_v2 import TradingConfig

config = TradingConfig.from_yaml("config/environments/production.yaml")

# From environment variables
config = TradingConfig.from_env()

# Programmatically
config = TradingConfig(
    position_sizing=PositionSizingConfig(max_position_size=500.0),
    market_filtering=MarketFilteringConfig(min_volume_24h=500)
)
```

---

## Decision Strategies

### `BaseDecisionStrategy`

Abstract base class for all decision strategies.

**Location**: `src/jobs/decision_strategies/base_strategy.py`

**Methods to Implement**:
```python
@abstractmethod
async def decide(self, context: DecisionContext) -> DecisionResult:
    """
    Make a trading decision for the given market.

    Args:
        context: DecisionContext with market data, portfolio state, etc.

    Returns:
        DecisionResult with decision (OPEN_YES, OPEN_NO, CLOSE, SKIP)
    """

@abstractmethod
def can_handle(self, market: Market) -> bool:
    """
    Check if this strategy can handle the given market.

    Args:
        market: Market to evaluate

    Returns:
        True if strategy is suitable for this market
    """
```

**Usage Example**:
```python
from src.jobs.decision_strategies import HighConfidenceStrategy, StandardStrategy

# Create strategy instances
high_conf = HighConfidenceStrategy(db_manager)
standard = StandardStrategy(db_manager)

# Select strategy for market
if high_conf.can_handle(market):
    result = await high_conf.decide(context)
else:
    result = await standard.decide(context)
```

---

### `HighConfidenceStrategy`

Fast strategy for high-confidence, near-expiry trades.

**Location**: `src/jobs/decision_strategies/high_confidence_strategy.py`

**When to Use**:
- Markets expiring within 7 days
- High confidence signals (> 80%)
- Clear catalysts or events
- Established trends

**Configuration**:
```python
high_conf = HighConfidenceStrategy(
    db_manager=db,
    min_confidence=0.80,
    max_days_to_expiry=7,
    skip_ai_analysis=True  # Use fast heuristics only
)
```

---

### `StandardStrategy`

Comprehensive AI-powered analysis for all markets.

**Location**: `src/jobs/decision_strategies/standard_strategy.py`

**When to Use**:
- Complex markets requiring deep analysis
- Longer time horizons (> 7 days)
- Uncertain outcomes
- Multi-factor evaluation needed

**Configuration**:
```python
standard = StandardStrategy(
    db_manager=db,
    ai_provider="xai",
    use_ensemble=True,  # Use multiple AI models
    enable_caching=True
)
```

---

## Backtesting

### `BacktestEngine`

Core backtesting engine for strategy validation.

**Location**: `src/backtesting/framework.py`

**Usage**:
```python
from datetime import datetime, timedelta
from src.backtesting import BacktestConfig, BacktestEngine, BacktestMode

# Configure backtest
config = BacktestConfig(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    initial_capital=10000.0,
    max_positions=10,
    commission_per_contract=1.00,
    slippage_bps=5.0,
    strategy_name="high_confidence"
)

# Create engine
engine = BacktestEngine(config)

# Run backtest
metrics = await engine.run(strategy, data_loader)

# Access results
print(f"Return: {metrics.total_return:.2%}")
print(f"Sharpe: {metrics.sharpe_ratio:.2f}")
print(f"Trades: {metrics.total_trades}")
```

**Key Methods**:
```python
async def run(strategy, data_loader) -> PerformanceMetrics:
    """Run backtest with given strategy and data."""

def get_trades_summary() -> List[Dict]:
    """Get summary of all trades executed."""

def get_equity_curve() -> List[Tuple[datetime, float]]:
    """Get portfolio value over time."""
```

---

### `HistoricalDataLoader`

Loads historical market data for backtesting.

**Location**: `src/backtesting/data_loader.py`

**Usage**:
```python
from src.backtesting import HistoricalDataLoader

# From database
loader = HistoricalDataLoader(db_path="data/historical.db")

# From CSV files
loader = HistoricalDataLoader(csv_dir=Path("data/historical_csv"))

# Load data
data = await loader.load_data(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    frequency="1h",
    tickers=["KXBTC-24DEC-50K", "INXD-24Q1-UP"]
)
```

**Key Methods**:
```python
async def load_data(
    start_date: datetime,
    end_date: datetime,
    frequency: str = "1h",
    tickers: Optional[List[str]] = None
) -> List[Tuple[datetime, Dict[str, Dict]]]:
    """Load historical market data."""

async def save_snapshot_to_db(snapshot: MarketSnapshot):
    """Save market snapshot for future backtesting."""

async def export_to_csv(start_date: datetime, end_date: datetime, output_dir: Path):
    """Export historical data to CSV files."""
```

---

### `PerformanceAttributionAnalyzer`

Analyzes performance and attributes returns to various factors.

**Location**: `src/backtesting/performance_attribution.py`

**Usage**:
```python
from src.backtesting import PerformanceAttributionAnalyzer

# Create analyzer
analyzer = PerformanceAttributionAnalyzer(
    trades=engine.trades,
    initial_capital=10000.0
)

# Generate attribution report
report = analyzer.analyze()

# Access attribution data
for cat in report.category_attribution:
    print(f"{cat.category}: ${cat.total_pnl:.2f} ({cat.win_rate:.1%} win rate)")

# Print summary
analyzer.print_summary()
```

---

### `ReportGenerator`

Generates comprehensive backtest reports.

**Location**: `src/backtesting/report_generator.py`

**Usage**:
```python
from pathlib import Path
from src.backtesting import ReportGenerator

# Create generator
report_gen = ReportGenerator(backtest_engine)

# Generate all report formats
await report_gen.generate_all_reports(Path("reports/backtest_2024"))

# Or generate specific formats
await report_gen.generate_html_report(Path("reports/report.html"))
await report_gen.generate_json_report(Path("reports/report.json"))
await report_gen.generate_markdown_report(Path("reports/report.md"))
```

**Output Files**:
- `backtest_report.html` - Interactive HTML report with charts
- `backtest_report.json` - Machine-readable JSON data
- `backtest_report.md` - Markdown documentation
- `trades.csv` - All trades as CSV
- `equity_curve.csv` - Portfolio value over time

---

## Database

### `BaseRepository`

Abstract database interface.

**Location**: `src/database/base_repository.py`

**Key Methods**:
```python
@abstractmethod
async def add_position(position: Position) -> Optional[int]:
    """Add a new position to database."""

@abstractmethod
async def get_open_positions() -> List[Position]:
    """Get all open positions."""

@abstractmethod
async def update_position(position: Position):
    """Update existing position."""

@abstractmethod
async def get_daily_ai_cost(date: str = None) -> float:
    """Get total AI API costs for a date."""

@abstractmethod
async def add_market_analysis(ticker: str, analysis: str, cost: float):
    """Record an AI market analysis."""
```

---

### `PostgreSQLRepository`

PostgreSQL implementation with connection pooling.

**Location**: `src/database/postgres_repository.py`

**Usage**:
```python
from src.database import PostgreSQLRepository

# Create repository
repo = PostgreSQLRepository(
    host="localhost",
    port=5432,
    database="kalshi_trading",
    user="kalshi_user",
    password="password",
    pool_size=10,
    max_overflow=20
)

# Initialize connection pool
await repo.initialize()

# Use repository
position_id = await repo.add_position(position)
open_positions = await repo.get_open_positions()

# Close connection pool
await repo.close()
```

---

## Market Screener

### `MarketScreener`

Advanced market screening and ranking.

**Location**: `src/utils/market_screener.py`

**Usage**:
```python
from src.utils.market_screener import MarketScreener, ScreeningConfig

# Create screener with custom config
config = ScreeningConfig(
    min_volume_24h=500,
    min_open_interest=2000,
    max_spread_pct=3.0,
    min_edge=0.05,
    volume_weight=0.25,
    edge_weight=0.35
)

screener = MarketScreener(config)

# Screen markets
scored_markets = await screener.screen_markets(all_markets)

# Get top opportunities
top_10 = await screener.find_best_opportunities(all_markets, limit=10)

# Print results
screener.print_top_markets(scored_markets, n=20)
```

**Screening Factors**:
- **Volume**: Trading activity and liquidity
- **Liquidity**: Bid-ask spread tightness
- **Edge**: Perceived mispricing opportunity
- **Time Value**: Optimal time to expiration
- **Momentum**: Recent price trends
- **Volatility**: Price movement magnitude

---

### `AnomalyDetector`

Detects unusual market conditions.

**Location**: `src/utils/market_screener.py`

**Usage**:
```python
from src.utils.market_screener import AnomalyDetector

detector = AnomalyDetector()

# Detect anomalies
anomalies = await detector.detect_anomalies(markets)

for market, anomaly_type, severity in anomalies:
    print(f"{market.ticker}: {anomaly_type} (severity: {severity:.2f})")
```

**Detected Anomalies**:
- `volume_spike`: 3x+ normal volume
- `extreme_low_probability`: Price < 5%
- `extreme_high_probability`: Price > 95%
- `wide_spread`: Spread > 10%

---

## Utilities

### Logger

Structured logging with context.

**Location**: `src/utils/logger.py`

**Usage**:
```python
from src.utils.logger import get_logger

logger = get_logger(__name__)

logger.info("Starting backtest", extra={"strategy": "high_confidence"})
logger.warning("Low volume market", extra={"ticker": "KXBTC-24DEC-50K"})
logger.error("API call failed", exc_info=True)
```

---

### Health Check

System health monitoring.

**Location**: `src/utils/health_check.py`

**Usage**:
```python
from src.utils.health_check import perform_health_check

result = await perform_health_check()

print(f"Healthy: {result.healthy}")
print(f"Database: {result.checks['database']['status']}")
print(f"Redis: {result.checks['redis']['status']}")
```

---

## Complete Example

Here's a complete example showing how to use the main APIs together:

```python
import asyncio
from datetime import datetime, timedelta
from pathlib import Path

from src.config.settings_v2 import TradingConfig
from src.backtesting import (
    BacktestConfig,
    BacktestEngine,
    BacktestMode,
    HistoricalDataLoader,
    ReportGenerator,
    PerformanceAttributionAnalyzer
)
from src.jobs.decision_strategies import HighConfidenceStrategy
from src.database import create_database_manager
from src.utils.market_screener import MarketScreener


async def main():
    # Load configuration
    config = TradingConfig.from_yaml("config/environments/production.yaml")

    # Create database manager
    db = await create_database_manager()

    # Create trading strategy
    strategy = HighConfidenceStrategy(db)

    # Configure backtest
    backtest_config = BacktestConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
        initial_capital=10000.0,
        strategy_name="high_confidence"
    )

    # Create backtest engine and data loader
    engine = BacktestEngine(backtest_config)
    data_loader = HistoricalDataLoader(db_path="data/historical.db")

    # Run backtest
    print("Running backtest...")
    metrics = await engine.run(strategy, data_loader)

    # Print results
    print(f"\nBacktest Results:")
    print(f"  Total Return: {metrics.total_return:.2%}")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"  Win Rate: {metrics.win_rate:.1%}")
    print(f"  Total Trades: {metrics.total_trades}")

    # Performance attribution
    analyzer = PerformanceAttributionAnalyzer(engine.trades, backtest_config.initial_capital)
    analyzer.print_summary()

    # Generate reports
    report_gen = ReportGenerator(engine)
    await report_gen.generate_all_reports(Path("reports/backtest_2024"))

    print("\nReports generated in reports/backtest_2024/")

    # Clean up
    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Error Handling

All async methods may raise the following exceptions:

- `DatabaseError`: Database operations failed
- `APIError`: External API call failed
- `ValidationError`: Invalid input parameters
- `TimeoutError`: Operation exceeded timeout
- `ConfigurationError`: Invalid configuration

**Example**:
```python
from src.exceptions import DatabaseError, APIError

try:
    result = await strategy.decide(context)
except DatabaseError as e:
    logger.error(f"Database error: {e}")
    # Handle database failure
except APIError as e:
    logger.error(f"API error: {e}")
    # Handle API failure
except Exception as e:
    logger.error(f"Unexpected error: {e}", exc_info=True)
    # Handle unknown error
```

---

## Rate Limiting

API clients implement automatic rate limiting:

```python
from src.clients.kalshi_client import KalshiClient

client = KalshiClient(
    api_key=config.kalshi_api_key,
    rate_limit_per_second=2  # Max 2 requests/second
)

# Automatic rate limiting
markets = await client.get_markets()  # Respects rate limits
```

---

## Caching

Enable caching for better performance:

```python
from src.cache import RedisCache

# Initialize cache
cache = RedisCache(
    host=config.redis_host,
    port=config.redis_port,
    ttl_seconds=3600  # 1 hour TTL
)

# Use cache with decorator
from src.cache import cached

@cached(ttl=1800)  # 30 minutes
async def get_market_data(ticker: str):
    return await api.get_market(ticker)
```

---

**Last Updated**: 2025-11-16
**Version**: 1.0.0
