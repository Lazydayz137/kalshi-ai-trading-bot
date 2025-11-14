"""
Database abstraction layer for Kalshi Trading Bot.

Supports both SQLite (development) and PostgreSQL (production).
"""

from .base_repository import BaseRepository
from .factory import get_database_manager

__all__ = ['BaseRepository', 'get_database_manager']
