# Troubleshooting Guide

Common issues and solutions for the Kalshi AI Trading Bot.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Configuration Problems](#configuration-problems)
- [API and Authentication](#api-and-authentication)
- [Database Issues](#database-issues)
- [Trading Execution](#trading-execution)
- [Performance Problems](#performance-problems)
- [Docker and Deployment](#docker-and-deployment)

---

## Installation Issues

### Problem: `pip install` fails with dependency conflicts

**Symptoms**:
```
ERROR: pip's dependency resolver does not currently take into account all the packages
```

**Solutions**:
1. **Use Python 3.12** (required):
   ```bash
   python --version  # Must be 3.12.x
   ```

2. **Install in clean virtual environment**:
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Install dependencies one by one** to identify conflict:
   ```bash
   pip install aiohttp
   pip install asyncpg
   # ... continue
   ```

---

### Problem: `ModuleNotFoundError` when running bot

**Symptoms**:
```
ModuleNotFoundError: No module named 'src'
```

**Solutions**:
1. **Run from project root**:
   ```bash
   cd /path/to/kalshi-ai-trading-bot
   python -m src.main
   ```

2. **Add project to PYTHONPATH**:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   python -m src.main
   ```

3. **Install in editable mode**:
   ```bash
   pip install -e .
   ```

---

## Configuration Problems

### Problem: Bot can't find `.env` file

**Symptoms**:
```
WARNING: .env file not found
KeyError: 'KALSHI_API_KEY'
```

**Solutions**:
1. **Create `.env` file** in project root:
   ```bash
   cp .env.example .env
   ```

2. **Set environment variables** manually:
   ```bash
   export KALSHI_API_KEY="your_key_here"
   export XAI_API_KEY="your_key_here"
   ```

3. **Verify `.env` location**:
   ```bash
   ls -la .env  # Should be in project root
   ```

---

### Problem: YAML configuration not loading

**Symptoms**:
```
FileNotFoundError: config/environments/production.yaml
```

**Solutions**:
1. **Create config directory**:
   ```bash
   mkdir -p config/environments
   ```

2. **Use `.env.docker.template` as reference**:
   ```bash
   cp .env.docker.template config/environments/production.yaml
   ```

3. **Specify full path**:
   ```python
   config = TradingConfig.from_yaml("/full/path/to/config.yaml")
   ```

---

## API and Authentication

### Problem: Kalshi API authentication fails

**Symptoms**:
```
APIError: 401 Unauthorized
```

**Solutions**:
1. **Verify API key** is correct:
   ```bash
   echo $KALSHI_API_KEY
   ```

2. **Check API key permissions** in Kalshi dashboard

3. **Test API key** with curl:
   ```bash
   curl -H "Authorization: Bearer $KALSHI_API_KEY" \
     https://api.kalshi.com/v2/markets
   ```

4. **Regenerate API key** if compromised

---

### Problem: AI API rate limiting

**Symptoms**:
```
APIError: 429 Too Many Requests
Retry-After: 60
```

**Solutions**:
1. **Reduce request frequency**:
   ```python
   # settings.yaml
   ai_model:
     rate_limit_per_minute: 10  # Lower this
   ```

2. **Enable caching**:
   ```python
   cache_config:
     enabled: true
     ttl_seconds: 3600
   ```

3. **Use high-confidence strategy** (fewer AI calls):
   ```python
   strategy_selector:
     default_strategy: "high_confidence"
   ```

---

### Problem: XAI/OpenAI API timeouts

**Symptoms**:
```
TimeoutError: Request exceeded 30s timeout
```

**Solutions**:
1. **Increase timeout**:
   ```python
   ai_model:
     timeout_seconds: 60  # Increase from 30
   ```

2. **Reduce max_tokens**:
   ```python
   ai_model:
     max_tokens: 2048  # Reduce from 4096
   ```

3. **Check network connectivity**:
   ```bash
   ping api.x.ai
   curl -I https://api.x.ai/v1/chat/completions
   ```

---

## Database Issues

### Problem: SQLite database locked

**Symptoms**:
```
OperationalError: database is locked
```

**Solutions**:
1. **Close all connections**:
   ```python
   await db_manager.close()
   ```

2. **Use WAL mode** (better concurrency):
   ```sql
   PRAGMA journal_mode=WAL;
   ```

3. **Switch to PostgreSQL** for production:
   ```python
   database_config:
     type: "postgresql"
   ```

---

### Problem: PostgreSQL connection refused

**Symptoms**:
```
ConnectionRefusedError: [Errno 61] Connection refused
```

**Solutions**:
1. **Check PostgreSQL is running**:
   ```bash
   # macOS
   brew services list | grep postgresql

   # Linux
   systemctl status postgresql

   # Docker
   docker compose ps postgres
   ```

2. **Verify connection parameters**:
   ```bash
   psql -h localhost -p 5432 -U kalshi_user -d kalshi_trading
   ```

3. **Check firewall rules**:
   ```bash
   # Allow PostgreSQL port
   sudo ufw allow 5432/tcp
   ```

---

### Problem: Database migration fails

**Symptoms**:
```
ProgrammingError: relation "positions" does not exist
```

**Solutions**:
1. **Run database initialization**:
   ```python
   from src.database import create_database_manager

   db = await create_database_manager()
   await db.initialize()  # Creates tables
   ```

2. **Check table creation**:
   ```sql
   \dt  -- PostgreSQL
   .tables  -- SQLite
   ```

3. **Manually create tables**:
   ```bash
   psql kalshi_trading < scripts/init-db.sql
   ```

---

## Trading Execution

### Problem: No trades being executed

**Symptoms**:
- Bot runs but never opens positions
- All markets filtered out

**Solutions**:
1. **Check filtering criteria**:
   ```python
   market_filtering:
     min_volume_24h: 100  # Lower this
     min_open_interest: 500  # Lower this
   ```

2. **Enable debug logging**:
   ```python
   logging:
     level: "DEBUG"
   ```

3. **Review validator logs**:
   ```bash
   grep "ValidationResult" logs/trading.log
   ```

4. **Check daily budget** not exceeded:
   ```sql
   SELECT SUM(cost) FROM market_analyses
   WHERE DATE(timestamp) = DATE('now');
   ```

---

### Problem: Positions not closing

**Symptoms**:
- Positions remain open past expiry
- Exit logic not triggering

**Solutions**:
1. **Check close time detection**:
   ```python
   # Verify market close_time is set correctly
   if market.close_time < datetime.utcnow():
       await close_position(position)
   ```

2. **Manual position close**:
   ```python
   from src.jobs.execute import close_all_positions

   await close_all_positions(db_manager, kalshi_client)
   ```

3. **Review close_reason** in database:
   ```sql
   SELECT ticker, close_reason FROM positions
   WHERE status = 'open'
   AND entry_time < datetime('now', '-7 days');
   ```

---

### Problem: Incorrect PnL calculations

**Symptoms**:
- PnL doesn't match expected value
- Negative PnL on winning trades

**Solutions**:
1. **Check commission calculation**:
   ```python
   # Verify commission is subtracted correctly
   pnl = (exit_price - entry_price) * quantity - commission
   ```

2. **Verify price direction** for YES/NO:
   ```python
   # YES position: profit when price increases
   # NO position: profit when price decreases
   ```

3. **Review trade log**:
   ```python
   position = await db.get_position(position_id)
   print(f"Entry: ${position.entry_price}")
   print(f"Exit: ${position.exit_price}")
   print(f"Qty: {position.quantity}")
   print(f"PnL: ${position.pnl}")
   ```

---

## Performance Problems

### Problem: Slow decision-making

**Symptoms**:
- Each market takes > 30 seconds to analyze
- High AI API latency

**Solutions**:
1. **Use high-confidence strategy**:
   ```python
   strategy_selector:
     default_strategy: "high_confidence"
     high_confidence_threshold: 0.8
   ```

2. **Enable response caching**:
   ```python
   cache_config:
     enabled: true
     cache_type: "redis"
   ```

3. **Reduce AI model size**:
   ```python
   ai_model:
     model: "gpt-3.5-turbo"  # Faster than gpt-4
   ```

4. **Parallel processing**:
   ```python
   # Process markets in batches
   batch_size = 5
   for batch in chunk_markets(markets, batch_size):
       await asyncio.gather(*[decide(m) for m in batch])
   ```

---

### Problem: High memory usage

**Symptoms**:
```
MemoryError: Unable to allocate memory
```

**Solutions**:
1. **Limit market snapshots in memory**:
   ```python
   backtest_config:
     data_frequency: "4h"  # Instead of "1h"
     warmup_period_days: 7  # Instead of 30
   ```

2. **Clear cache periodically**:
   ```python
   await cache.clear_expired()
   ```

3. **Use database for large datasets**:
   ```python
   # Instead of loading all in memory
   async for snapshot in data_loader.stream_snapshots():
       process(snapshot)
   ```

---

## Docker and Deployment

### Problem: Docker build fails

**Symptoms**:
```
ERROR: failed to solve: process "/bin/sh -c pip install -r requirements.txt" did not complete
```

**Solutions**:
1. **Check Dockerfile** is in project root:
   ```bash
   ls -la Dockerfile
   ```

2. **Build with verbose output**:
   ```bash
   docker build --progress=plain -t kalshi-bot .
   ```

3. **Verify requirements.txt** syntax:
   ```bash
   cat requirements.txt | grep "^[^#]"
   ```

4. **Use specific Python version**:
   ```dockerfile
   FROM python:3.12-slim
   ```

---

### Problem: Docker container exits immediately

**Symptoms**:
```
docker compose ps
# Shows container as "exited (1)"
```

**Solutions**:
1. **Check container logs**:
   ```bash
   docker compose logs trading-bot
   ```

2. **Verify environment variables**:
   ```bash
   docker compose config
   ```

3. **Run interactive shell**:
   ```bash
   docker compose run --rm trading-bot /bin/bash
   ```

4. **Check health status**:
   ```bash
   docker inspect --format='{{.State.Health}}' trading-bot
   ```

---

### Problem: Cannot connect to PostgreSQL in Docker

**Symptoms**:
```
ConnectionRefusedError: Connection to postgres:5432 refused
```

**Solutions**:
1. **Check service order** in docker-compose.yml:
   ```yaml
   trading-bot:
     depends_on:
       postgres:
         condition: service_healthy  # Wait for healthy
   ```

2. **Verify network connectivity**:
   ```bash
   docker compose exec trading-bot ping postgres
   ```

3. **Check PostgreSQL logs**:
   ```bash
   docker compose logs postgres
   ```

4. **Use correct hostname**:
   ```python
   # In Docker, use service name
   POSTGRES_HOST=postgres  # Not "localhost"
   ```

---

## Debugging Tips

### Enable Debug Logging

**In `.env`**:
```bash
LOG_LEVEL=DEBUG
```

**In code**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Inspect Database State

**SQLite**:
```bash
sqlite3 data/kalshi_trading.db
.mode column
.headers on
SELECT * FROM positions WHERE status = 'open';
```

**PostgreSQL**:
```bash
psql kalshi_trading
\x on  -- Expanded display
SELECT * FROM positions WHERE status = 'open';
```

### Monitor API Calls

**Add request logging**:
```python
import aiohttp
import logging

logging.getLogger("aiohttp.client").setLevel(logging.DEBUG)
```

### Profile Performance

**Use cProfile**:
```bash
python -m cProfile -o profile.stats -m src.main
python -m pstats profile.stats
# In pstats: sort cumtime, stats 20
```

---

## Getting Help

If you're still stuck after trying these solutions:

1. **Check existing issues**: https://github.com/Lazydayz137/kalshi-ai-trading-bot/issues
2. **Review logs** in `logs/` directory
3. **Create detailed issue** with:
   - Error message (full traceback)
   - Steps to reproduce
   - Environment details (Python version, OS, etc.)
   - Relevant configuration
   - Log output

---

**Last Updated**: 2025-11-16
