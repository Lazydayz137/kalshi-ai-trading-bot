"""
Data models for real-time market data.

These models map to TimescaleDB tables and represent different granularities
of market data from raw ticks to computed features.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from decimal import Decimal
from enum import Enum


class TradeSide(Enum):
    """Side of trade execution."""
    BUY = "buy"
    SELL = "sell"


class PredictedDirection(Enum):
    """Predicted price movement direction."""
    UP = "up"
    DOWN = "down"
    NEUTRAL = "neutral"


@dataclass
class PriceTick:
    """
    Raw price tick from WebSocket.

    Maps to: price_ticks table
    Frequency: Real-time (sub-second to seconds)
    Retention: 90 days
    """
    time: datetime
    ticker: str
    price: Decimal
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    spread: Optional[Decimal] = None
    volume: int = 0
    last_trade_price: Optional[Decimal] = None
    last_trade_size: Optional[int] = None
    source: str = "websocket"
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion."""
        return {
            'time': self.time,
            'ticker': self.ticker,
            'price': float(self.price),
            'bid': float(self.bid) if self.bid else None,
            'ask': float(self.ask) if self.ask else None,
            'spread': float(self.spread) if self.spread else None,
            'volume': self.volume,
            'last_trade_price': float(self.last_trade_price) if self.last_trade_price else None,
            'last_trade_size': self.last_trade_size,
            'source': self.source,
            'created_at': self.created_at
        }


@dataclass
class OrderBookSnapshot:
    """
    Order book snapshot at a point in time.

    Maps to: orderbook_snapshots table
    Frequency: 1 second
    Retention: 90 days
    """
    time: datetime
    ticker: str
    bid_prices: List[Decimal]
    bid_sizes: List[int]
    ask_prices: List[Decimal]
    ask_sizes: List[int]
    bid_depth_5: Optional[Decimal] = None
    bid_depth_10: Optional[Decimal] = None
    ask_depth_5: Optional[Decimal] = None
    ask_depth_10: Optional[Decimal] = None
    mid_price: Optional[Decimal] = None
    weighted_mid_price: Optional[Decimal] = None
    order_book_imbalance: Optional[Decimal] = None
    spread_bps: Optional[int] = None
    total_bid_volume: Optional[int] = None
    total_ask_volume: Optional[int] = None
    num_bid_levels: Optional[int] = None
    num_ask_levels: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def compute_metrics(self):
        """Compute derived metrics from order book data."""
        # Mid price
        if self.bid_prices and self.ask_prices:
            self.mid_price = (self.bid_prices[0] + self.ask_prices[0]) / 2

        # Weighted mid price (volume-weighted)
        if self.bid_prices and self.ask_prices and self.bid_sizes and self.ask_sizes:
            total_bid = sum(self.bid_sizes)
            total_ask = sum(self.ask_sizes)
            if total_bid + total_ask > 0:
                self.weighted_mid_price = (
                    (self.bid_prices[0] * total_bid + self.ask_prices[0] * total_ask) /
                    (total_bid + total_ask)
                )

        # Order book imbalance (buy pressure)
        if self.total_bid_volume and self.total_ask_volume:
            total = self.total_bid_volume + self.total_ask_volume
            if total > 0:
                self.order_book_imbalance = Decimal(
                    (self.total_bid_volume - self.total_ask_volume) / total
                )

        # Spread in basis points
        if self.bid_prices and self.ask_prices and self.mid_price:
            spread = self.ask_prices[0] - self.bid_prices[0]
            self.spread_bps = int((spread / self.mid_price) * 10000)

        # Depth calculations
        if len(self.bid_sizes) >= 5:
            self.bid_depth_5 = Decimal(sum(self.bid_sizes[:5]))
        if len(self.bid_sizes) >= 10:
            self.bid_depth_10 = Decimal(sum(self.bid_sizes[:10]))
        if len(self.ask_sizes) >= 5:
            self.ask_depth_5 = Decimal(sum(self.ask_sizes[:5]))
        if len(self.ask_sizes) >= 10:
            self.ask_depth_10 = Decimal(sum(self.ask_sizes[:10]))

        # Counts
        self.num_bid_levels = len(self.bid_prices)
        self.num_ask_levels = len(self.ask_prices)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion."""
        return {
            'time': self.time,
            'ticker': self.ticker,
            'bid_prices': [float(p) for p in self.bid_prices],
            'bid_sizes': self.bid_sizes,
            'ask_prices': [float(p) for p in self.ask_prices],
            'ask_sizes': self.ask_sizes,
            'bid_depth_5': float(self.bid_depth_5) if self.bid_depth_5 else None,
            'bid_depth_10': float(self.bid_depth_10) if self.bid_depth_10 else None,
            'ask_depth_5': float(self.ask_depth_5) if self.ask_depth_5 else None,
            'ask_depth_10': float(self.ask_depth_10) if self.ask_depth_10 else None,
            'mid_price': float(self.mid_price) if self.mid_price else None,
            'weighted_mid_price': float(self.weighted_mid_price) if self.weighted_mid_price else None,
            'order_book_imbalance': float(self.order_book_imbalance) if self.order_book_imbalance else None,
            'spread_bps': self.spread_bps,
            'total_bid_volume': self.total_bid_volume,
            'total_ask_volume': self.total_ask_volume,
            'num_bid_levels': self.num_bid_levels,
            'num_ask_levels': self.num_ask_levels,
            'created_at': self.created_at
        }


@dataclass
class TradeExecution:
    """
    Individual trade execution event.

    Maps to: trade_executions table
    Frequency: Event-based (every trade)
    Retention: 1 year
    """
    time: datetime
    ticker: str
    price: Decimal
    size: int
    trade_id: Optional[str] = None
    side: Optional[TradeSide] = None
    is_market_maker: bool = False
    execution_cost: Optional[Decimal] = None
    price_impact: Optional[Decimal] = None
    volume_imbalance: Optional[Decimal] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion."""
        return {
            'time': self.time,
            'ticker': self.ticker,
            'trade_id': self.trade_id,
            'price': float(self.price),
            'size': self.size,
            'side': self.side.value if self.side else None,
            'is_market_maker': self.is_market_maker,
            'execution_cost': float(self.execution_cost) if self.execution_cost else None,
            'price_impact': float(self.price_impact) if self.price_impact else None,
            'volume_imbalance': float(self.volume_imbalance) if self.volume_imbalance else None,
            'created_at': self.created_at
        }


@dataclass
class MarketFeatures:
    """
    Computed features at 1-minute frequency.

    Maps to: market_features_1min table
    Frequency: 1 minute
    Retention: 2 years (primary ML training data)
    """
    time: datetime
    ticker: str

    # OHLCV
    open: Optional[Decimal] = None
    high: Optional[Decimal] = None
    low: Optional[Decimal] = None
    close: Optional[Decimal] = None
    vwap: Optional[Decimal] = None

    # Volume
    volume: Optional[int] = None
    trade_count: Optional[int] = None
    buy_volume: Optional[int] = None
    sell_volume: Optional[int] = None
    volume_imbalance: Optional[Decimal] = None

    # Momentum
    returns_1min: Optional[Decimal] = None
    momentum_5min: Optional[Decimal] = None
    momentum_15min: Optional[Decimal] = None
    momentum_1hour: Optional[Decimal] = None

    # Volatility
    volatility_5min: Optional[Decimal] = None
    volatility_15min: Optional[Decimal] = None
    volatility_1hour: Optional[Decimal] = None
    realized_volatility: Optional[Decimal] = None

    # Technical indicators
    rsi_14: Optional[Decimal] = None
    macd: Optional[Decimal] = None
    macd_signal: Optional[Decimal] = None
    macd_histogram: Optional[Decimal] = None
    bollinger_upper: Optional[Decimal] = None
    bollinger_lower: Optional[Decimal] = None
    bollinger_width: Optional[Decimal] = None

    # Order flow
    order_flow_imbalance: Optional[Decimal] = None
    trade_flow_toxicity: Optional[Decimal] = None
    effective_spread: Optional[Decimal] = None
    price_impact: Optional[Decimal] = None

    # Microstructure
    spread_mean: Optional[Decimal] = None
    spread_std: Optional[Decimal] = None
    depth_imbalance: Optional[Decimal] = None

    # Metadata
    data_quality: Decimal = Decimal('1.0')
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion."""
        def to_float(val):
            return float(val) if val is not None else None

        return {
            'time': self.time,
            'ticker': self.ticker,
            # OHLCV
            'open': to_float(self.open),
            'high': to_float(self.high),
            'low': to_float(self.low),
            'close': to_float(self.close),
            'vwap': to_float(self.vwap),
            # Volume
            'volume': self.volume,
            'trade_count': self.trade_count,
            'buy_volume': self.buy_volume,
            'sell_volume': self.sell_volume,
            'volume_imbalance': to_float(self.volume_imbalance),
            # Momentum
            'returns_1min': to_float(self.returns_1min),
            'momentum_5min': to_float(self.momentum_5min),
            'momentum_15min': to_float(self.momentum_15min),
            'momentum_1hour': to_float(self.momentum_1hour),
            # Volatility
            'volatility_5min': to_float(self.volatility_5min),
            'volatility_15min': to_float(self.volatility_15min),
            'volatility_1hour': to_float(self.volatility_1hour),
            'realized_volatility': to_float(self.realized_volatility),
            # Technical indicators
            'rsi_14': to_float(self.rsi_14),
            'macd': to_float(self.macd),
            'macd_signal': to_float(self.macd_signal),
            'macd_histogram': to_float(self.macd_histogram),
            'bollinger_upper': to_float(self.bollinger_upper),
            'bollinger_lower': to_float(self.bollinger_lower),
            'bollinger_width': to_float(self.bollinger_width),
            # Order flow
            'order_flow_imbalance': to_float(self.order_flow_imbalance),
            'trade_flow_toxicity': to_float(self.trade_flow_toxicity),
            'effective_spread': to_float(self.effective_spread),
            'price_impact': to_float(self.price_impact),
            # Microstructure
            'spread_mean': to_float(self.spread_mean),
            'spread_std': to_float(self.spread_std),
            'depth_imbalance': to_float(self.depth_imbalance),
            # Metadata
            'data_quality': to_float(self.data_quality),
            'created_at': self.created_at
        }


@dataclass
class ModelPrediction:
    """
    ML model prediction for validation and tracking.

    Maps to: model_predictions table
    Frequency: Real-time (on each prediction)
    Retention: 1 year
    """
    time: datetime
    ticker: str
    model_name: str
    model_version: str
    predicted_price_1h: Optional[Decimal] = None
    predicted_price_4h: Optional[Decimal] = None
    predicted_price_24h: Optional[Decimal] = None
    predicted_direction: Optional[PredictedDirection] = None
    confidence: Optional[Decimal] = None
    feature_vector: Optional[Dict[str, float]] = None
    inference_time_ms: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion."""
        import json

        return {
            'time': self.time,
            'ticker': self.ticker,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'predicted_price_1h': float(self.predicted_price_1h) if self.predicted_price_1h else None,
            'predicted_price_4h': float(self.predicted_price_4h) if self.predicted_price_4h else None,
            'predicted_price_24h': float(self.predicted_price_24h) if self.predicted_price_24h else None,
            'predicted_direction': self.predicted_direction.value if self.predicted_direction else None,
            'confidence': float(self.confidence) if self.confidence else None,
            'feature_vector': json.dumps(self.feature_vector) if self.feature_vector else None,
            'inference_time_ms': self.inference_time_ms,
            'created_at': self.created_at
        }


@dataclass
class PredictionValidation:
    """
    Validation of model predictions against actual outcomes.

    Maps to: prediction_validation table
    Used for tracking model performance over time.
    """
    prediction_time: datetime
    validation_time: datetime
    ticker: str
    model_name: str
    model_version: str
    predicted_price: Decimal
    predicted_direction: PredictedDirection
    confidence: Decimal
    actual_price: Decimal
    actual_direction: PredictedDirection
    absolute_error: Optional[Decimal] = None
    squared_error: Optional[Decimal] = None
    directional_accuracy: Optional[bool] = None
    horizon_hours: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def compute_metrics(self):
        """Compute error metrics."""
        self.absolute_error = abs(self.actual_price - self.predicted_price)
        self.squared_error = (self.actual_price - self.predicted_price) ** 2
        self.directional_accuracy = (self.predicted_direction == self.actual_direction)

        # Calculate horizon in hours
        time_diff = self.validation_time - self.prediction_time
        self.horizon_hours = int(time_diff.total_seconds() / 3600)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion."""
        return {
            'prediction_time': self.prediction_time,
            'validation_time': self.validation_time,
            'ticker': self.ticker,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'predicted_price': float(self.predicted_price),
            'predicted_direction': self.predicted_direction.value,
            'confidence': float(self.confidence),
            'actual_price': float(self.actual_price),
            'actual_direction': self.actual_direction.value,
            'absolute_error': float(self.absolute_error) if self.absolute_error else None,
            'squared_error': float(self.squared_error) if self.squared_error else None,
            'directional_accuracy': self.directional_accuracy,
            'horizon_hours': self.horizon_hours,
            'created_at': self.created_at
        }


@dataclass
class MarketMetadata:
    """
    Market metadata (slowly changing dimensions).

    Maps to: market_metadata table
    Updated when market information changes.
    """
    ticker: str
    title: str
    category: str
    market_type: Optional[str] = None
    strike_type: Optional[str] = None
    floor_strike: Optional[Decimal] = None
    cap_strike: Optional[Decimal] = None
    open_time: Optional[datetime] = None
    close_time: Optional[datetime] = None
    settlement_value: Optional[Decimal] = None
    status: str = "open"
    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion."""
        return {
            'ticker': self.ticker,
            'title': self.title,
            'category': self.category,
            'market_type': self.market_type,
            'strike_type': self.strike_type,
            'floor_strike': float(self.floor_strike) if self.floor_strike else None,
            'cap_strike': float(self.cap_strike) if self.cap_strike else None,
            'open_time': self.open_time,
            'close_time': self.close_time,
            'settlement_value': float(self.settlement_value) if self.settlement_value else None,
            'status': self.status,
            'first_seen': self.first_seen,
            'last_updated': self.last_updated,
            'is_active': self.is_active
        }
