# Tutorials and Examples

Step-by-step guides for common tasks with the Kalshi AI Trading Bot.

## Table of Contents

- [Getting Started](#getting-started)
- [Running Your First Backtest](#running-your-first-backtest)
- [Creating a Custom Strategy](#creating-a-custom-strategy)
- [Analyzing Performance](#analyzing-performance)
- [Deploying to Production](#deploying-to-production)

---

## Getting Started

### Tutorial 1: Initial Setup (10 minutes)

**Goal**: Get the trading bot running in demo mode.

#### Step 1: Clone and Install

```bash
# Clone repository
git clone https://github.com/Lazydayz137/kalshi-ai-trading-bot.git
cd kalshi-ai-trading-bot

# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Step 2: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your API keys
nano .env
```

**Add your API keys**:
```bash
KALSHI_API_KEY=your_kalshi_key_here
XAI_API_KEY=your_xai_key_here

# Optional: OpenAI instead of XAI
OPENAI_API_KEY=your_openai_key_here

# Set to demo mode for testing
LIVE_TRADING_ENABLED=false
```

#### Step 3: Initialize Database

```bash
# Create data directory
mkdir -p data

# Initialize SQLite database
python -c "
import asyncio
from src.database import create_database_manager

async def init():
    db = await create_database_manager()
    await db.initialize()
    print('Database initialized!')

asyncio.run(init())
"
```

#### Step 4: Run Demo

```bash
# Run in demo mode (no real trades)
python -m src.main
```

**Expected Output**:
```
2024-11-16 10:00:00 - INFO - Starting Kalshi AI Trading Bot
2024-11-16 10:00:00 - INFO - DEMO MODE: Live trading disabled
2024-11-16 10:00:00 - INFO - Fetching markets...
2024-11-16 10:00:01 - INFO - Found 157 markets
2024-11-16 10:00:01 - INFO - Screening markets...
2024-11-16 10:00:02 - INFO - Top 10 opportunities identified
```

---

## Running Your First Backtest

### Tutorial 2: Backtest a Strategy (30 minutes)

**Goal**: Test the high-confidence strategy on historical data.

#### Step 1: Prepare Historical Data

**Option A: Generate Synthetic Data** (for testing):

```python
# scripts/generate_synthetic_data.py
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from src.backtesting import HistoricalDataLoader

async def main():
    loader = HistoricalDataLoader(db_path="data/backtest.db")

    # Generate 90 days of synthetic data
    start = datetime(2024, 1, 1)
    end = datetime(2024, 3, 31)

    data = await loader._generate_synthetic_data(
        start_date=start,
        end_date=end,
        frequency="1h",
        tickers=None  # Auto-generate tickers
    )

    print(f"Generated {len(data)} data points")

    # Save to database
    for timestamp, markets in data:
        for ticker, market_data in markets.items():
            snapshot = loader.MarketSnapshot(
                timestamp=timestamp,
                ticker=ticker,
                **market_data
            )
            await loader.save_snapshot_to_db(snapshot)

    print("Synthetic data saved to data/backtest.db")

asyncio.run(main())
```

Run it:
```bash
python scripts/generate_synthetic_data.py
```

**Option B: Use Real Historical Data** (if available):
```bash
# Export from production database
python -c "
from src.backtesting import HistoricalDataLoader
import asyncio
from datetime import datetime
from pathlib import Path

async def export():
    loader = HistoricalDataLoader(db_path='data/kalshi_trading.db')
    await loader.export_to_csv(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 3, 31),
        output_dir=Path('data/historical_csv')
    )

asyncio.run(export())
"
```

#### Step 2: Run Backtest

```python
# scripts/run_backtest.py
import asyncio
from datetime import datetime
from pathlib import Path

from src.backtesting import (
    BacktestConfig,
    BacktestEngine,
    BacktestMode,
    HistoricalDataLoader,
    ReportGenerator
)
from src.jobs.decision_strategies import HighConfidenceStrategy
from src.database import create_database_manager


async def main():
    print("🚀 Starting Backtest\n")

    # Step 1: Configuration
    config = BacktestConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 3, 31),
        initial_capital=10000.0,
        max_positions=10,
        commission_per_contract=1.00,
        slippage_bps=5.0,
        strategy_name="high_confidence",
        generate_reports=True
    )

    print(f"Period: {config.start_date.date()} to {config.end_date.date()}")
    print(f"Capital: ${config.initial_capital:,.2f}\n")

    # Step 2: Create components
    db = await create_database_manager()
    strategy = HighConfidenceStrategy(db)
    data_loader = HistoricalDataLoader(db_path="data/backtest.db")
    engine = BacktestEngine(config)

    # Step 3: Run backtest
    print("Running backtest...")
    metrics = await engine.run(strategy, data_loader)

    # Step 4: Print results
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f"Total Return:      {metrics.total_return:>10.2%}")
    print(f"Annualized Return: {metrics.annualized_return:>10.2%}")
    print(f"Sharpe Ratio:      {metrics.sharpe_ratio:>10.2f}")
    print(f"Max Drawdown:      {metrics.max_drawdown:>10.2%}")
    print(f"Total Trades:      {metrics.total_trades:>10}")
    print(f"Win Rate:          {metrics.win_rate:>10.1%}")
    print(f"Profit Factor:     {metrics.profit_factor:>10.2f}")
    print("="*60 + "\n")

    # Step 5: Generate reports
    print("Generating reports...")
    report_gen = ReportGenerator(engine)
    await report_gen.generate_all_reports(Path("reports/q1_2024"))

    print("✅ Reports saved to reports/q1_2024/")
    print("   - backtest_report.html (open in browser)")
    print("   - backtest_report.json (machine-readable)")
    print("   - trades.csv (Excel compatible)\n")

    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
```

Run it:
```bash
python scripts/run_backtest.py
```

#### Step 3: View Results

Open the HTML report:
```bash
open reports/q1_2024/backtest_report.html
```

**Analyze the results**:
- **Equity Curve**: Visual of portfolio growth
- **Category Attribution**: Which market types performed best
- **Trade Log**: Individual trade details

---

## Creating a Custom Strategy

### Tutorial 3: Build a Momentum Strategy (45 minutes)

**Goal**: Create a custom trading strategy based on price momentum.

#### Step 1: Create Strategy File

```python
# src/jobs/decision_strategies/momentum_strategy.py
from typing import Optional
from datetime import datetime, timedelta

from .base_strategy import BaseDecisionStrategy, DecisionContext, DecisionResult, Decision
from ...models import Market
from ...utils.logger import get_logger

logger = get_logger(__name__)


class MomentumStrategy(BaseDecisionStrategy):
    """
    Momentum-based trading strategy.

    Logic:
    - Buys markets showing strong upward price trends
    - Sells when momentum reverses
    - Uses volume confirmation
    """

    def __init__(self, db_manager, lookback_hours: int = 24):
        super().__init__(db_manager)
        self.lookback_hours = lookback_hours
        self.price_history = {}  # ticker -> [(timestamp, price)]

    def can_handle(self, market: Market) -> bool:
        """This strategy handles all markets."""
        return True

    async def decide(self, context: DecisionContext) -> DecisionResult:
        """Make trading decision based on momentum."""
        market = context.market

        # Update price history
        self._update_price_history(market)

        # Calculate momentum
        momentum_score = self._calculate_momentum(market.ticker)

        if momentum_score is None:
            return DecisionResult(decision=Decision.SKIP, reason="Insufficient price history")

        # Decision logic
        if momentum_score > 0.7 and market.volume_24h > 1000:
            # Strong upward momentum + volume confirmation
            return DecisionResult(
                decision=Decision.OPEN_YES,
                reason=f"Strong momentum ({momentum_score:.2f}) with volume confirmation",
                confidence=momentum_score,
                suggested_quantity=self._calculate_position_size(context, momentum_score)
            )

        elif momentum_score < -0.7:
            # Strong downward momentum - consider shorting (NO position)
            return DecisionResult(
                decision=Decision.OPEN_NO,
                reason=f"Negative momentum ({momentum_score:.2f})",
                confidence=abs(momentum_score),
                suggested_quantity=self._calculate_position_size(context, abs(momentum_score))
            )

        else:
            # Weak momentum - skip
            return DecisionResult(decision=Decision.SKIP, reason=f"Weak momentum ({momentum_score:.2f})")

    def _update_price_history(self, market: Market):
        """Track price history for momentum calculation."""
        ticker = market.ticker
        now = datetime.utcnow()

        if ticker not in self.price_history:
            self.price_history[ticker] = []

        # Add current price
        self.price_history[ticker].append((now, market.last_price))

        # Remove old data
        cutoff = now - timedelta(hours=self.lookback_hours)
        self.price_history[ticker] = [
            (ts, price) for ts, price in self.price_history[ticker]
            if ts > cutoff
        ]

    def _calculate_momentum(self, ticker: str) -> Optional[float]:
        """
        Calculate momentum score (-1 to +1).

        Positive = upward trend
        Negative = downward trend
        """
        if ticker not in self.price_history:
            return None

        prices = self.price_history[ticker]

        if len(prices) < 5:  # Need minimum data points
            return None

        # Simple momentum: compare recent average to older average
        mid_point = len(prices) // 2
        old_prices = [p for _, p in prices[:mid_point]]
        recent_prices = [p for _, p in prices[mid_point:]]

        old_avg = sum(old_prices) / len(old_prices)
        recent_avg = sum(recent_prices) / len(recent_prices)

        # Calculate percent change
        if old_avg == 0:
            return 0

        pct_change = (recent_avg - old_avg) / old_avg

        # Normalize to -1 to +1 range
        # +/- 10% change = max score
        momentum = max(-1.0, min(1.0, pct_change / 0.10))

        return momentum

    def _calculate_position_size(self, context: DecisionContext, confidence: float) -> int:
        """Calculate position size based on confidence."""
        # Simple Kelly-like sizing
        max_contracts = 100
        base_size = int(max_contracts * confidence)

        # Adjust for available capital
        available_capital = context.portfolio_value * 0.1  # Max 10% per trade
        market = context.market
        cost_per_contract = market.last_price

        max_affordable = int(available_capital / cost_per_contract)

        return min(base_size, max_affordable)
```

#### Step 2: Register Strategy

```python
# src/jobs/decision_strategies/__init__.py
from .momentum_strategy import MomentumStrategy

# Add to __all__
__all__ = [
    # ... existing strategies
    "MomentumStrategy",
]
```

#### Step 3: Use in Backtest

```python
# scripts/test_momentum_strategy.py
from src.jobs.decision_strategies import MomentumStrategy

# ... (same setup as previous backtest)

# Use momentum strategy instead
strategy = MomentumStrategy(db, lookback_hours=24)

metrics = await engine.run(strategy, data_loader)
```

#### Step 4: Compare to Baseline

Run both strategies and compare:
```bash
python scripts/run_backtest.py  # High confidence
python scripts/test_momentum_strategy.py  # Momentum

# Compare reports
diff reports/high_confidence/backtest_report.json \
     reports/momentum/backtest_report.json
```

---

## Analyzing Performance

### Tutorial 4: Performance Attribution Analysis (20 minutes)

**Goal**: Understand what drives your strategy's performance.

```python
# scripts/analyze_performance.py
import asyncio
from src.backtesting import PerformanceAttributionAnalyzer
from src.database import create_database_manager


async def main():
    # Load historical trades from database
    db = await create_database_manager()

    # Get all closed positions
    query = "SELECT * FROM positions WHERE status = 'closed'"
    # ... load trades

    # Create analyzer
    analyzer = PerformanceAttributionAnalyzer(
        trades=all_trades,
        initial_capital=10000.0
    )

    # Generate full report
    report = analyzer.analyze()

    # Print summary
    analyzer.print_summary()

    # Detailed analysis
    print("\n📊 Category Performance:")
    for cat in report.category_attribution:
        print(f"  {cat.category:20s}: ${cat.total_pnl:>8,.2f} "
              f"({cat.win_rate:>5.1%} win rate, {cat.num_trades} trades)")

    print("\n⏱️ Holding Period Analysis:")
    for hp in report.holding_period_attribution:
        if hp.num_trades > 0:
            print(f"  {hp.period_bucket:15s}: ${hp.total_pnl:>8,.2f} "
                  f"(avg {hp.average_holding_hours:.1f}h holding)")

    print("\n🎯 Side Performance:")
    for side in report.side_attribution:
        print(f"  {side.side.upper():5s}: ${side.total_pnl:>8,.2f} "
              f"({side.win_rate:>5.1%} win rate)")

    # Find best/worst markets
    print("\n🏆 Top 5 Markets:")
    for market in analyzer.get_top_contributors(n=5):
        print(f"  {market.ticker:20s}: ${market.total_pnl:>8,.2f}")

    print("\n📉 Worst 5 Markets:")
    for market in analyzer.get_worst_performers(n=5):
        print(f"  {market.ticker:20s}: ${market.total_pnl:>8,.2f}")

    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
```

**Key Insights to Look For**:
- **Category Attribution**: Which market categories are most profitable?
- **Holding Period**: What's the optimal time to hold positions?
- **Side Performance**: Better at YES or NO positions?
- **Top/Worst Markets**: Which specific markets to avoid/seek?

---

## Deploying to Production

### Tutorial 5: Deploy with Docker (60 minutes)

**Goal**: Deploy the bot to production using Docker Compose.

#### Step 1: Prepare Configuration

```bash
# Copy Docker environment template
cp .env.docker.template .env.production

# Edit production settings
nano .env.production
```

**Key settings**:
```bash
# CRITICAL: Enable live trading
LIVE_TRADING_ENABLED=true

# Production database
DATABASE_TYPE=postgresql
POSTGRES_HOST=postgres
POSTGRES_DATABASE=kalshi_trading
POSTGRES_USER=kalshi_user
POSTGRES_PASSWORD=your_secure_password_here

# Production API keys
KALSHI_API_KEY=your_real_api_key
XAI_API_KEY=your_xai_key

# Risk limits
MAX_POSITION_SIZE=500.0
DAILY_AI_BUDGET=50.0
MAX_DAILY_LOSS=200.0
```

#### Step 2: Build Docker Images

```bash
# Build production image
docker build -t kalshi-trading-bot:production .

# Verify build
docker images | grep kalshi
```

#### Step 3: Start Services

```bash
# Start full stack
docker compose --env-file .env.production up -d

# Check status
docker compose ps
```

**Expected output**:
```
NAME                STATUS              PORTS
postgres            running (healthy)   5432/tcp
redis               running (healthy)   6379/tcp
trading-bot         running (healthy)   -
```

#### Step 4: Monitor Logs

```bash
# Follow trading bot logs
docker compose logs -f trading-bot

# Check specific component
docker compose logs postgres
docker compose logs redis
```

#### Step 5: Verify Health

```bash
# Run health check
docker compose exec trading-bot python -m src.utils.health_check

# Expected output:
# ✅ Database: healthy
# ✅ Redis: healthy
# ✅ API Credentials: valid
# ✅ System Resources: adequate
```

#### Step 6: Monitor Performance

```bash
# Access PostgreSQL
docker compose exec postgres psql -U kalshi_user kalshi_trading

# Check positions
SELECT COUNT(*) as open_positions FROM positions WHERE status = 'open';

# Daily PnL
SELECT
    DATE(exit_time) as date,
    SUM(pnl) as daily_pnl,
    COUNT(*) as trades
FROM positions
WHERE status = 'closed'
GROUP BY DATE(exit_time)
ORDER BY date DESC
LIMIT 7;
```

#### Step 7: Setup Monitoring (Optional)

**Add Prometheus metrics**:
```bash
# Expose metrics endpoint
docker compose exec trading-bot python -m src.monitoring.prometheus_exporter

# Access metrics
curl http://localhost:9090/metrics
```

**Configure alerts**:
```yaml
# alerting/rules.yml
groups:
  - name: trading_alerts
    rules:
      - alert: HighDailyLoss
        expr: daily_pnl < -200
        annotations:
          summary: Daily loss limit exceeded
```

---

## Next Steps

After completing these tutorials, you can:

1. **Optimize Strategies**: Tune parameters based on backtest results
2. **Add Custom Validators**: Create domain-specific filtering logic
3. **Implement Risk Management**: Advanced position sizing and hedging
4. **Build Dashboard**: Create web UI for monitoring
5. **Automate Reporting**: Schedule daily performance emails

**Additional Resources**:
- [API Reference](API_REFERENCE.md)
- [Troubleshooting Guide](TROUBLESHOOTING.md)
- [Database Schema](DATABASE_SCHEMA.md)

---

**Last Updated**: 2025-11-16
