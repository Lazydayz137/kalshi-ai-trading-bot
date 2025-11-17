"""
Real-time data infrastructure for Kalshi AI Trading Bot.

This package provides:
- TimescaleDB client for high-frequency time series storage
- Data models for ticks, order books, trades, and features
- WebSocket client for real-time Kalshi market data
- Event stream processing for feature computation
- Feature store for ML model inference
"""

from .models import (
    PriceTick,
    OrderBookSnapshot,
    TradeExecution,
    MarketFeatures,
    ModelPrediction,
    PredictionValidation,
    MarketMetadata,
    TradeSide,
    PredictedDirection
)

from .timescale_client import TimescaleClient

__all__ = [
    # Data models
    "PriceTick",
    "OrderBookSnapshot",
    "TradeExecution",
    "MarketFeatures",
    "ModelPrediction",
    "PredictionValidation",
    "MarketMetadata",
    "TradeSide",
    "PredictedDirection",
    # Database client
    "TimescaleClient",
]

__version__ = "1.0.0"
