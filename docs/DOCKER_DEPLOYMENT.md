# Docker Deployment Guide

This guide explains how to deploy the Kalshi AI Trading Bot using Docker and Docker Compose.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Production Deployment](#production-deployment)
- [Development Deployment](#development-deployment)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)
- [Maintenance](#maintenance)

---

## Prerequisites

### Required

- **Docker** >= 20.10
- **Docker Compose** >= 2.0
- **Kalshi API Key** (from Kalshi platform)
- **xAI API Key** (for Grok-4 model access)

### Installation

```bash
# macOS
brew install docker docker-compose

# Ubuntu/Debian
sudo apt-get update
sudo apt-get install docker.io docker-compose-plugin

# Verify installation
docker --version
docker compose version
```

---

## Quick Start

### 1. Clone and Configure

```bash
# Clone repository
git clone <repository-url>
cd kalshi-ai-trading-bot

# Create environment file
cp .env.docker.template .env

# Edit .env and add your API keys
nano .env
```

### 2. Start Services

```bash
# Start all services (PostgreSQL + Redis + Trading Bot)
docker compose up -d

# View logs
docker compose logs -f trading-bot

# Check service status
docker compose ps
```

### 3. Access Dashboard

Open your browser to `http://localhost:8501` to view the trading dashboard.

### 4. Stop Services

```bash
# Stop all services
docker compose down

# Stop and remove volumes (WARNING: deletes all data)
docker compose down -v
```

---

## Production Deployment

### 1. Environment Configuration

```bash
# Copy template
cp .env.docker.template .env

# Edit with production values
nano .env
```

**Production .env Example:**

```bash
# API Keys
KALSHI_API_KEY=your_actual_kalshi_key
XAI_API_KEY=your_actual_xai_key

# PostgreSQL (use strong passwords!)
POSTGRES_PASSWORD=use_a_strong_random_password_here
POSTGRES_DATABASE=kalshi_trading_prod
POSTGRES_USER=kalshi_prod

# Redis (use strong password!)
REDIS_PASSWORD=use_a_strong_random_password_here

# Trading Settings
LIVE_TRADING_ENABLED=false  # Set to true for real trading
LOG_LEVEL=INFO
DATABASE_TYPE=postgresql
CACHE_TYPE=redis
```

### 2. Start Production Stack

```bash
# Pull latest images
docker compose pull

# Start services in detached mode
docker compose up -d

# Verify all services are healthy
docker compose ps

# Should show:
# NAME                STATUS
# kalshi-postgres     Up (healthy)
# kalshi-redis        Up (healthy)
# kalshi-trading-bot  Up (healthy)
# kalshi-dashboard    Up
```

### 3. Initialize Database

The database is automatically initialized on first start. Verify:

```bash
# Check PostgreSQL logs
docker compose logs postgres | grep "initialized"

# Should see: "Kalshi Trading Database initialized successfully"
```

### 4. Monitor Logs

```bash
# All services
docker compose logs -f

# Trading bot only
docker compose logs -f trading-bot

# Last 100 lines
docker compose logs --tail=100 trading-bot
```

### 5. Health Checks

```bash
# Check health status
docker compose ps

# Manual health check
docker compose exec trading-bot python -m src.utils.health_check
```

---

## Development Deployment

Development mode includes additional tools (pgAdmin, Redis Commander) and enables hot reload.

### 1. Start Development Stack

```bash
# Use both compose files
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# View all services
docker compose -f docker-compose.yml -f docker-compose.dev.yml ps
```

### 2. Access Development Tools

- **Trading Dashboard**: http://localhost:8501
- **pgAdmin** (PostgreSQL GUI): http://localhost:5050
  - Email: `admin@kalshi.local`
  - Password: `devpassword`
- **Redis Commander**: http://localhost:8081

### 3. Hot Reload

In development mode, source code changes are automatically detected:

```bash
# Edit source files
nano src/jobs/decide.py

# Changes are reflected immediately (no rebuild needed)
```

### 4. Run Tests

```bash
# Run tests inside container
docker compose exec trading-bot pytest tests/unit/ -v

# Run with coverage
docker compose exec trading-bot pytest --cov=src --cov-report=html
```

---

## Configuration

### Docker Compose Services

| Service | Description | Port |
|---------|-------------|------|
| `postgres` | PostgreSQL 16 database | 5432 |
| `redis` | Redis 7 cache | 6379 |
| `trading-bot` | Main trading application | - |
| `dashboard` | Streamlit dashboard | 8501 |
| `pgadmin` (dev) | PostgreSQL admin UI | 5050 |
| `redis-commander` (dev) | Redis admin UI | 8081 |

### Environment Variables

See `.env.docker.template` for all available environment variables.

**Key Variables:**

- `LIVE_TRADING_ENABLED` - Enable real trading (default: false)
- `DATABASE_TYPE` - Database type: sqlite or postgresql
- `CACHE_TYPE` - Cache type: memory or redis
- `LOG_LEVEL` - Logging level: DEBUG, INFO, WARNING, ERROR

### Volumes

Persistent data is stored in Docker volumes:

```bash
# List volumes
docker volume ls | grep kalshi

# Backup database volume
docker run --rm -v kalshi-ai-trading-bot_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres-backup.tar.gz /data

# Restore database volume
docker run --rm -v kalshi-ai-trading-bot_postgres_data:/data -v $(pwd):/backup alpine tar xzf /backup/postgres-backup.tar.gz -C /
```

### Resource Limits

Configured in `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '2'      # Maximum CPU cores
      memory: 2G     # Maximum RAM
    reservations:
      cpus: '0.5'    # Minimum CPU guarantee
      memory: 512M   # Minimum RAM guarantee
```

---

## Monitoring

### View Logs

```bash
# Real-time logs
docker compose logs -f

# Logs for specific service
docker compose logs -f trading-bot

# Last N lines
docker compose logs --tail=50 trading-bot

# Logs since timestamp
docker compose logs --since 2024-01-01T00:00:00 trading-bot
```

### Service Health

```bash
# Check all services
docker compose ps

# Inspect specific service
docker compose exec trading-bot python -m src.utils.health_check

# Check database
docker compose exec postgres pg_isready

# Check Redis
docker compose exec redis redis-cli ping
```

### Resource Usage

```bash
# Container stats
docker stats

# Specific container
docker stats kalshi-trading-bot
```

### Database Queries

```bash
# Connect to PostgreSQL
docker compose exec postgres psql -U kalshi -d kalshi_trading

# Example queries
SELECT COUNT(*) FROM positions WHERE status = 'open';
SELECT * FROM trade_logs ORDER BY exit_timestamp DESC LIMIT 10;
SELECT SUM(pnl) FROM trade_logs WHERE DATE(exit_timestamp) = CURRENT_DATE;
```

---

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker compose logs trading-bot

# Check health status
docker compose ps

# Restart specific service
docker compose restart trading-bot

# Rebuild and restart
docker compose up -d --build trading-bot
```

### Database Connection Issues

```bash
# Verify PostgreSQL is running
docker compose ps postgres

# Check PostgreSQL logs
docker compose logs postgres

# Test connection
docker compose exec postgres pg_isready

# Verify credentials
docker compose exec postgres psql -U kalshi -d kalshi_trading -c "SELECT version();"
```

### Redis Connection Issues

```bash
# Verify Redis is running
docker compose ps redis

# Test connection
docker compose exec redis redis-cli -a changeme ping

# Check Redis info
docker compose exec redis redis-cli -a changeme info
```

### Out of Memory

```bash
# Check memory usage
docker stats

# Increase memory limit in docker-compose.yml
# Then restart:
docker compose down
docker compose up -d
```

### Permission Issues

```bash
# Fix log directory permissions
chmod -R 777 logs/

# Fix data directory permissions
chmod -R 777 data/
```

### Reset Everything

```bash
# WARNING: This deletes ALL data!
docker compose down -v
docker volume prune -f
docker compose up -d
```

---

## Maintenance

### Backup

#### Database Backup

```bash
# Backup PostgreSQL
docker compose exec postgres pg_dump -U kalshi kalshi_trading > backup_$(date +%Y%m%d).sql

# Or use pg_dumpall for all databases
docker compose exec postgres pg_dumpall -U kalshi > backup_all_$(date +%Y%m%d).sql
```

#### Volume Backup

```bash
# Backup all volumes
docker run --rm \
  -v kalshi-ai-trading-bot_postgres_data:/data/postgres \
  -v kalshi-ai-trading-bot_redis_data:/data/redis \
  -v $(pwd):/backup \
  alpine tar czf /backup/volumes-backup-$(date +%Y%m%d).tar.gz /data
```

### Restore

```bash
# Restore PostgreSQL from backup
cat backup_20240101.sql | docker compose exec -T postgres psql -U kalshi kalshi_trading
```

### Updates

```bash
# Pull latest code
git pull

# Rebuild containers
docker compose build

# Restart with new build
docker compose up -d
```

### Cleanup

```bash
# Remove unused containers
docker container prune -f

# Remove unused images
docker image prune -a -f

# Remove unused volumes (WARNING!)
docker volume prune -f

# Remove everything unused
docker system prune -a --volumes -f
```

### Scaling

```bash
# Scale trading bot (if stateless)
docker compose up -d --scale trading-bot=3

# Note: Current implementation is single-instance
# For true scaling, implement:
# 1. Distributed locking (Redis)
# 2. Message queue (RabbitMQ/Kafka)
# 3. Shared state management
```

---

## Security Best Practices

### 1. Use Strong Passwords

```bash
# Generate secure password
openssl rand -base64 32

# Use in .env file
POSTGRES_PASSWORD=<generated-password>
REDIS_PASSWORD=<generated-password>
```

### 2. Limit Network Exposure

```yaml
# In docker-compose.yml, don't expose sensitive ports to host
# Remove these lines in production:
ports:
  - "5432:5432"  # Don't expose PostgreSQL
  - "6379:6379"  # Don't expose Redis
```

### 3. Use Secrets Management

For production, use Docker secrets instead of .env:

```bash
# Create secrets
echo "my_postgres_password" | docker secret create postgres_password -

# Use in docker-compose.yml
secrets:
  postgres_password:
    external: true
```

### 4. Regular Updates

```bash
# Update base images regularly
docker compose pull
docker compose up -d
```

### 5. Monitor Logs

```bash
# Watch for suspicious activity
docker compose logs -f | grep -i "error\|failed\|unauthorized"
```

---

## Performance Tuning

### PostgreSQL

Edit `docker-compose.yml` to add PostgreSQL tuning:

```yaml
postgres:
  command: >
    postgres
    -c shared_buffers=256MB
    -c effective_cache_size=1GB
    -c maintenance_work_mem=64MB
    -c checkpoint_completion_target=0.9
    -c wal_buffers=16MB
    -c default_statistics_target=100
    -c random_page_cost=1.1
    -c effective_io_concurrency=200
```

### Redis

Tune Redis for your workload:

```yaml
redis:
  command: >
    redis-server
    --maxmemory 512mb
    --maxmemory-policy allkeys-lru
    --save 900 1
    --save 300 10
    --save 60 10000
```

---

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [PostgreSQL Docker Image](https://hub.docker.com/_/postgres)
- [Redis Docker Image](https://hub.docker.com/_/redis)
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)

---

## Support

For issues and questions:
1. Check logs: `docker compose logs`
2. Review this documentation
3. Check GitHub issues
4. Contact support

**Last Updated:** 2025-11-14
