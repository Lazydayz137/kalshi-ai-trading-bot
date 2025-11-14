"""
Health Check Module

Provides health check functionality for Docker containers and monitoring systems.
"""

import sys
import asyncio
from typing import Dict, Any, List
from datetime import datetime

from src.utils.logging_setup import get_trading_logger


class HealthCheckResult:
    """Result of a health check."""

    def __init__(self, healthy: bool, checks: Dict[str, Dict[str, Any]]):
        self.healthy = healthy
        self.checks = checks
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "healthy": self.healthy,
            "timestamp": self.timestamp.isoformat(),
            "checks": self.checks
        }


async def check_database_health() -> Dict[str, Any]:
    """
    Check database connectivity and health.

    Returns:
        Dictionary with health status
    """
    try:
        from src.utils.database import DatabaseManager

        db = DatabaseManager()
        await db.initialize()

        # Try a simple query
        import aiosqlite
        async with aiosqlite.connect(db.db_path) as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM positions LIMIT 1")
            await cursor.fetchone()

        return {
            "status": "healthy",
            "message": "Database is accessible",
            "type": "sqlite"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Database error: {str(e)}",
            "type": "sqlite"
        }


async def check_redis_health() -> Dict[str, Any]:
    """
    Check Redis connectivity and health.

    Returns:
        Dictionary with health status
    """
    try:
        from src.config.settings import settings

        if not settings.cache.enable_caching or settings.cache.cache_type != "redis":
            return {
                "status": "disabled",
                "message": "Redis caching is disabled",
                "type": "redis"
            }

        # Try to connect to Redis
        import redis.asyncio as redis

        client = redis.Redis(
            host=settings.cache.redis_host,
            port=settings.cache.redis_port,
            password=settings.cache.redis_password,
            db=settings.cache.redis_db,
            socket_connect_timeout=5
        )

        await client.ping()
        await client.close()

        return {
            "status": "healthy",
            "message": "Redis is accessible",
            "type": "redis"
        }
    except ImportError:
        return {
            "status": "disabled",
            "message": "Redis client not installed",
            "type": "redis"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Redis error: {str(e)}",
            "type": "redis"
        }


async def check_api_credentials() -> Dict[str, Any]:
    """
    Check that required API credentials are configured.

    Returns:
        Dictionary with health status
    """
    try:
        from src.config.settings import settings

        missing_keys = []

        if not settings.api.kalshi_api_key:
            missing_keys.append("KALSHI_API_KEY")

        if not settings.api.xai_api_key:
            missing_keys.append("XAI_API_KEY")

        if missing_keys:
            return {
                "status": "unhealthy",
                "message": f"Missing API keys: {', '.join(missing_keys)}",
                "type": "credentials"
            }

        return {
            "status": "healthy",
            "message": "All required API keys are configured",
            "type": "credentials"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Credential check error: {str(e)}",
            "type": "credentials"
        }


async def check_system_health() -> Dict[str, Any]:
    """
    Check system-level health (disk space, memory, etc.).

    Returns:
        Dictionary with health status
    """
    try:
        import psutil

        # Check disk space
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent

        # Check memory
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        issues = []
        if disk_percent > 90:
            issues.append(f"Disk space critical: {disk_percent}%")
        if memory_percent > 90:
            issues.append(f"Memory usage critical: {memory_percent}%")

        if issues:
            return {
                "status": "warning",
                "message": "; ".join(issues),
                "type": "system",
                "disk_percent": disk_percent,
                "memory_percent": memory_percent
            }

        return {
            "status": "healthy",
            "message": "System resources OK",
            "type": "system",
            "disk_percent": disk_percent,
            "memory_percent": memory_percent
        }
    except ImportError:
        return {
            "status": "disabled",
            "message": "psutil not installed",
            "type": "system"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"System check error: {str(e)}",
            "type": "system"
        }


async def perform_health_check() -> HealthCheckResult:
    """
    Perform comprehensive health check.

    Returns:
        HealthCheckResult with overall health status
    """
    logger = get_trading_logger("health_check")

    # Run all health checks
    checks = {
        "database": await check_database_health(),
        "redis": await check_redis_health(),
        "credentials": await check_api_credentials(),
        "system": await check_system_health(),
    }

    # Determine overall health
    # Healthy if all checks are healthy or disabled
    # Unhealthy if any check is unhealthy
    # Warning if any check is warning
    unhealthy_checks = [
        name for name, result in checks.items()
        if result["status"] == "unhealthy"
    ]

    warning_checks = [
        name for name, result in checks.items()
        if result["status"] == "warning"
    ]

    if unhealthy_checks:
        healthy = False
        logger.warning(f"Health check failed: {', '.join(unhealthy_checks)} unhealthy")
    elif warning_checks:
        healthy = True  # Still healthy, but with warnings
        logger.warning(f"Health check warnings: {', '.join(warning_checks)}")
    else:
        healthy = True
        logger.debug("Health check passed")

    return HealthCheckResult(healthy=healthy, checks=checks)


def check_health() -> bool:
    """
    Synchronous wrapper for health check (for Docker HEALTHCHECK).

    Returns:
        True if healthy, False otherwise
    """
    try:
        result = asyncio.run(perform_health_check())
        return result.healthy
    except Exception as e:
        print(f"Health check error: {e}", file=sys.stderr)
        return False


async def health_check_endpoint() -> Dict[str, Any]:
    """
    Health check endpoint for HTTP servers.

    Returns:
        Dictionary suitable for JSON response
    """
    result = await perform_health_check()
    return result.to_dict()


if __name__ == "__main__":
    # Allow running as: python -m src.utils.health_check
    result = asyncio.run(perform_health_check())
    print(f"Health Check: {'✅ HEALTHY' if result.healthy else '❌ UNHEALTHY'}")
    print(f"Timestamp: {result.timestamp}")
    print("\nChecks:")
    for name, check in result.checks.items():
        status_emoji = {
            "healthy": "✅",
            "unhealthy": "❌",
            "warning": "⚠️",
            "disabled": "⏸️"
        }.get(check["status"], "❓")
        print(f"  {status_emoji} {name}: {check['status']} - {check['message']}")

    sys.exit(0 if result.healthy else 1)
