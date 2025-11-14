"""
Database Factory - Creates appropriate database manager based on configuration.
"""

from typing import Union
from src.utils.logging_setup import get_trading_logger


def get_database_manager() -> 'Union[DatabaseManager, PostgreSQLRepository]':
    """
    Get the appropriate database manager based on configuration.

    Returns:
        DatabaseManager instance (SQLite or PostgreSQL)
    """
    from src.config.settings import settings
    logger = get_trading_logger("database_factory")

    database_type = settings.database.database_type.lower()

    if database_type == "postgresql":
        try:
            from src.database.postgres_repository import PostgreSQLRepository
            logger.info("Using PostgreSQL database")
            return PostgreSQLRepository(
                host=settings.database.postgres_host,
                port=settings.database.postgres_port,
                database=settings.database.postgres_database,
                user=settings.database.postgres_user,
                password=settings.database.postgres_password,
                pool_size=settings.database.pool_size,
                max_overflow=settings.database.max_overflow
            )
        except ImportError as e:
            logger.error(f"PostgreSQL driver not installed: {e}")
            logger.warning("Falling back to SQLite")
            database_type = "sqlite"

    if database_type == "sqlite":
        from src.utils.database import DatabaseManager
        logger.info(f"Using SQLite database: {settings.database.sqlite_path}")
        return DatabaseManager(db_path=settings.database.sqlite_path)

    raise ValueError(f"Unsupported database type: {database_type}")
