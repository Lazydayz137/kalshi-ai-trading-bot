# Phase 7: True AI/ML & Real-Time Data Infrastructure

**Status**: 🚧 Planned
**Priority**: HIGH
**Goal**: Transform from "LLM-assisted bot" to genuine AI/ML trading system

---

## Problem Statement

**Current Reality**: Despite being called an "AI Trading Bot," the system primarily uses:
- LLMs for text interpretation (not predictive modeling)
- Simple rule-based filters (volume, spread thresholds)
- Minimal time series analysis
- No real-time data streaming
- No trained ML models on market data

**Target State**: A true AI trading system with:
- Multiple ML models trained on historical market data
- Real-time data streaming and feature computation
- Quantitative feature engineering
- Online learning with continuous model updates
- Time series forecasting and pattern recognition

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     REAL-TIME DATA LAYER                        │
├─────────────────────────────────────────────────────────────────┤
│  Kalshi WebSocket → Event Stream → Feature Extractor → Models  │
│      ↓                  ↓               ↓                ↓      │
│  Order Book        Time Series    Feature Store      ML Models  │
│  Snapshots         Database       (Redis/Feast)      Inference  │
└─────────────────────────────────────────────────────────────────┘
           ↓                    ↓                    ↓
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  STORAGE LAYER   │  │  ML TRAINING     │  │  SERVING LAYER   │
│                  │  │  PIPELINE        │  │                  │
│ • TimescaleDB    │  │ • Feature Eng    │  │ • Online Models  │
│ • InfluxDB       │  │ • Model Training │  │ • Ensemble       │
│ • Parquet Files  │  │ • Backtesting    │  │ • A/B Testing    │
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

---

## Phase 7 Components

### 1. Real-Time Data Streaming (HIGH PRIORITY)

**Goal**: Capture all market data in real-time for feature computation and model training.

#### 1.1 Kalshi WebSocket Client
**File**: `src/data/streaming/kalshi_websocket.py`

```python
class KalshiWebSocketClient:
    """
    Real-time WebSocket connection to Kalshi for market data.

    Captures:
    - Tick-by-tick price updates
    - Order book snapshots (bid/ask depth)
    - Trade executions
    - Market status changes
    """

    async def subscribe_to_markets(self, tickers: List[str]):
        """Subscribe to real-time updates for specific markets."""

    async def on_price_update(self, data: Dict):
        """Handle price tick updates."""

    async def on_trade(self, data: Dict):
        """Handle trade execution events."""

    async def on_orderbook_update(self, data: Dict):
        """Handle order book depth changes."""
```

**Features**:
- WebSocket connection with auto-reconnect
- Message queue for buffering (Redis Streams)
- Heartbeat monitoring
- Data validation and normalization

#### 1.2 Event Stream Processor
**File**: `src/data/streaming/event_processor.py`

```python
class EventStreamProcessor:
    """
    Process incoming market events and compute features.

    Implements streaming feature computation:
    - Rolling statistics (mean, std, min, max)
    - Technical indicators (RSI, MACD, Bollinger Bands)
    - Order flow toxicity
    - Price momentum
    """

    async def process_price_tick(self, tick: PriceTick):
        """Compute features from price tick."""

    async def process_trade(self, trade: Trade):
        """Analyze trade for flow toxicity."""

    async def process_orderbook(self, orderbook: OrderBook):
        """Extract order book features."""
```

**Computed Features**:
- Price momentum (1min, 5min, 15min, 1hr)
- Volume profile
- Bid-ask spread dynamics
- Order book imbalance
- Trade flow toxicity
- Volatility (realized, implied)

#### 1.3 Time Series Database
**File**: `docker-compose.timescale.yml`

```yaml
services:
  timescaledb:
    image: timescale/timescaledb:latest-pg16
    environment:
      POSTGRES_DB: kalshi_timeseries
    volumes:
      - timescale_data:/var/lib/postgresql/data
```

**Schema**:
```sql
-- Price ticks (high frequency)
CREATE TABLE price_ticks (
    time TIMESTAMPTZ NOT NULL,
    ticker TEXT NOT NULL,
    price REAL NOT NULL,
    volume INTEGER,
    bid REAL,
    ask REAL
);
SELECT create_hypertable('price_ticks', 'time');

-- Order book snapshots (1 second frequency)
CREATE TABLE orderbook_snapshots (
    time TIMESTAMPTZ NOT NULL,
    ticker TEXT NOT NULL,
    bid_prices REAL[],
    bid_sizes INTEGER[],
    ask_prices REAL[],
    ask_sizes INTEGER[],
    depth_5_bid REAL,
    depth_5_ask REAL,
    imbalance REAL
);
SELECT create_hypertable('orderbook_snapshots', 'time');

-- Aggregated features (1 minute frequency)
CREATE TABLE market_features (
    time TIMESTAMPTZ NOT NULL,
    ticker TEXT NOT NULL,
    price_mean REAL,
    price_std REAL,
    volume_total INTEGER,
    momentum_5m REAL,
    momentum_15m REAL,
    rsi_14 REAL,
    bollinger_upper REAL,
    bollinger_lower REAL,
    order_flow_toxicity REAL,
    spread_mean REAL,
    volatility_5m REAL
);
SELECT create_hypertable('market_features', 'time');
```

**Benefits**:
- Optimized for time series queries
- Automatic data compression
- Fast aggregations
- Continuous aggregates (materialized views)

---

### 2. Feature Engineering (HIGH PRIORITY)

**Goal**: Extract predictive features from raw market data.

#### 2.1 Technical Indicators
**File**: `src/ml/features/technical_indicators.py`

```python
class TechnicalIndicators:
    """
    Compute technical indicators adapted for prediction markets.

    Indicators:
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - Bollinger Bands
    - Volume-Weighted Average Price (VWAP)
    - On-Balance Volume (OBV)
    """

    def compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI indicator."""

    def compute_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Compute MACD, signal line, and histogram."""

    def compute_bollinger_bands(self, prices: pd.Series, window: int = 20) -> Tuple[pd.Series, pd.Series]:
        """Compute upper and lower Bollinger Bands."""
```

#### 2.2 Market Microstructure Features
**File**: `src/ml/features/microstructure.py`

```python
class MicrostructureFeatures:
    """
    Extract market microstructure features.

    Features:
    - Order flow imbalance
    - Spread dynamics
    - Order book depth
    - Trade flow toxicity (Kyle's Lambda)
    - Amihud illiquidity measure
    """

    def order_flow_imbalance(self, orderbook: OrderBook) -> float:
        """Compute buy/sell pressure from order book."""

    def trade_flow_toxicity(self, trades: List[Trade], window: timedelta) -> float:
        """Estimate adverse selection cost."""

    def effective_spread(self, trades: List[Trade]) -> float:
        """Compute effective spread from trades."""
```

#### 2.3 Feature Store
**File**: `src/ml/features/feature_store.py`

```python
class FeatureStore:
    """
    Centralized feature storage and retrieval.

    Uses Redis for low-latency online features.
    Uses TimescaleDB for historical features.
    """

    async def get_online_features(self, ticker: str, feature_names: List[str]) -> Dict[str, float]:
        """Get latest features for real-time inference."""

    async def get_historical_features(self, ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Get historical features for training."""

    async def update_features(self, ticker: str, features: Dict[str, float]):
        """Update feature values."""
```

---

### 3. ML Models (HIGH PRIORITY)

**Goal**: Train predictive models on historical data.

#### 3.1 Price Prediction Models
**File**: `src/ml/models/price_predictor.py`

```python
class PricePredictionModel(ABC):
    """Base class for price prediction models."""

    @abstractmethod
    async def train(self, X: pd.DataFrame, y: pd.Series):
        """Train model on features and targets."""

    @abstractmethod
    async def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict future prices."""

    @abstractmethod
    async def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict price movement probabilities."""


class LSTMPricePredictor(PricePredictionModel):
    """
    LSTM model for time series price prediction.

    Architecture:
    - Input: Sequence of features (lookback=60 timesteps)
    - LSTM layers (128, 64 units)
    - Dropout (0.2)
    - Dense output layer

    Predicts: Price in next 1h, 4h, 24h
    """

    def __init__(self, lookback: int = 60, forecast_horizon: int = 24):
        self.model = self._build_model()

    def _build_model(self) -> keras.Model:
        """Build LSTM architecture."""


class TransformerPredictor(PricePredictionModel):
    """
    Transformer model for multi-horizon forecasting.

    Uses attention mechanism to learn temporal patterns.
    Better at long-range dependencies than LSTM.
    """


class XGBoostPredictor(PricePredictionModel):
    """
    XGBoost for feature-based prediction.

    Fast training and inference.
    Good for structured/tabular features.
    Interpretable with SHAP values.
    """
```

#### 3.2 Ensemble Model
**File**: `src/ml/models/ensemble.py`

```python
class EnsembleModel:
    """
    Combine multiple models for robust predictions.

    Models:
    - LSTM (temporal patterns)
    - XGBoost (feature interactions)
    - LightGBM (fast tree-based)
    - Linear model (baseline)

    Combination strategies:
    - Simple average
    - Weighted average (based on validation performance)
    - Stacking (meta-model)
    """

    async def train_models(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train all base models."""

    async def predict(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate ensemble prediction.

        Returns:
        {
            'prediction': float,
            'confidence': float,
            'model_predictions': {...},
            'feature_importance': {...}
        }
        """
```

#### 3.3 Online Learning
**File**: `src/ml/models/online_learning.py`

```python
class OnlineLearningModel:
    """
    Continuously update model with new data.

    Uses:
    - River library for online ML
    - Incremental updates on new data
    - Adaptive to market regime changes
    """

    async def partial_fit(self, X: pd.DataFrame, y: pd.Series):
        """Update model with new batch."""

    async def predict_and_update(self, X: pd.DataFrame, y_true: Optional[pd.Series] = None):
        """Make prediction, then update model."""
```

---

### 4. Model Training Pipeline (MEDIUM PRIORITY)

**Goal**: Automated model training, validation, and deployment.

#### 4.1 Training Pipeline
**File**: `src/ml/training/pipeline.py`

```python
class ModelTrainingPipeline:
    """
    End-to-end ML training pipeline.

    Steps:
    1. Load historical data from TimescaleDB
    2. Feature engineering
    3. Train/validation/test split (time-based)
    4. Model training with hyperparameter tuning
    5. Validation with walk-forward analysis
    6. Model evaluation (metrics, backtesting)
    7. Model versioning and deployment
    """

    async def run_training(self, config: TrainingConfig) -> TrainedModel:
        """Execute full training pipeline."""

    async def hyperparameter_optimization(self, search_space: Dict) -> Dict:
        """Optimize hyperparameters with Optuna."""

    async def walk_forward_validation(self, model, data: pd.DataFrame) -> Metrics:
        """Time series cross-validation."""
```

#### 4.2 Model Registry
**File**: `src/ml/training/model_registry.py`

```python
class ModelRegistry:
    """
    Version and track trained models.

    Uses MLflow for:
    - Model versioning
    - Experiment tracking
    - Performance metrics
    - Model lineage
    """

    def register_model(self, model, metrics: Dict, metadata: Dict) -> str:
        """Register new model version."""

    def load_production_model(self, model_name: str) -> Model:
        """Load current production model."""

    def compare_models(self, model_v1: str, model_v2: str) -> ComparisonReport:
        """A/B test two model versions."""
```

---

### 5. Quantitative Strategies (MEDIUM PRIORITY)

**Goal**: Implement quant strategies beyond simple signals.

#### 5.1 Statistical Arbitrage
**File**: `src/ml/strategies/stat_arb.py`

```python
class StatisticalArbitrage:
    """
    Detect and exploit statistical mispricings.

    Strategies:
    - Pairs trading (correlated markets)
    - Mean reversion
    - Cointegration-based trades
    """

    async def find_cointegrated_pairs(self, markets: List[Market]) -> List[Tuple[Market, Market]]:
        """Find statistically cointegrated market pairs."""

    async def detect_mispricing(self, market1: Market, market2: Market) -> Optional[Trade]:
        """Check if pair has diverged from equilibrium."""
```

#### 5.2 Market Regime Detection
**File**: `src/ml/strategies/regime_detection.py`

```python
class MarketRegimeDetector:
    """
    Classify current market regime.

    Regimes:
    - Trending (strong directional move)
    - Mean-reverting (oscillating around mean)
    - Volatile (high uncertainty)
    - Quiet (low activity)

    Uses Hidden Markov Models (HMM) or clustering.
    """

    async def detect_regime(self, market_data: pd.DataFrame) -> Regime:
        """Classify current market regime."""

    async def get_regime_strategy(self, regime: Regime) -> TradingStrategy:
        """Select optimal strategy for current regime."""
```

---

## Implementation Roadmap

### Week 1-2: Real-Time Data Infrastructure
- [ ] Implement Kalshi WebSocket client
- [ ] Set up TimescaleDB
- [ ] Create event stream processor
- [ ] Build feature computation pipeline
- [ ] Test with live data for 1 week

**Deliverables**:
- Real-time price ticks stored in TimescaleDB
- Computed features (RSI, MACD, momentum, etc.)
- Feature store with Redis cache
- 7 days of historical tick data

### Week 3-4: Feature Engineering
- [ ] Implement technical indicators
- [ ] Build microstructure features
- [ ] Create feature store
- [ ] Backfill historical features
- [ ] Feature validation and testing

**Deliverables**:
- 30+ engineered features
- Historical feature dataset (3+ months)
- Feature documentation
- Feature importance analysis

### Week 5-6: ML Model Development
- [ ] Train LSTM price predictor
- [ ] Train XGBoost model
- [ ] Implement ensemble model
- [ ] Walk-forward validation
- [ ] Model performance benchmarking

**Deliverables**:
- Trained LSTM model (Sharpe > 1.0 in backtest)
- XGBoost model with feature importance
- Ensemble model combining 3+ models
- Model performance report

### Week 7-8: Integration & Testing
- [ ] Integrate models into decision engine
- [ ] Build online inference pipeline
- [ ] Implement online learning
- [ ] A/B testing framework
- [ ] Live paper trading for 2 weeks

**Deliverables**:
- Models integrated in production
- Online learning pipeline
- Performance monitoring dashboard
- 2 weeks of paper trading results

### Week 9-10: Quantitative Strategies
- [ ] Implement statistical arbitrage
- [ ] Build market regime detector
- [ ] Create strategy selector based on regime
- [ ] Backtest quant strategies
- [ ] Optimize portfolio allocation

**Deliverables**:
- Stat arb strategy (pairs trading)
- Regime detection model
- Multi-strategy portfolio

---

## Success Metrics

### Data Infrastructure
- ✅ 99.9% uptime for WebSocket connection
- ✅ < 100ms latency for feature computation
- ✅ > 1 million price ticks captured per day
- ✅ 30+ features computed in real-time

### Model Performance
- ✅ Sharpe ratio > 1.5 (backtested)
- ✅ Win rate > 55%
- ✅ Max drawdown < 20%
- ✅ Prediction accuracy > 60% (directional)
- ✅ Model explains > 30% of variance (R²)

### System Performance
- ✅ Online inference < 50ms
- ✅ Feature store latency < 10ms
- ✅ Model retraining daily
- ✅ Automatic model deployment

---

## Technology Stack

### Data Infrastructure
- **WebSocket**: `websockets` library
- **Message Queue**: Redis Streams
- **Time Series DB**: TimescaleDB (PostgreSQL extension)
- **Feature Store**: Redis + TimescaleDB

### ML/Data Science
- **Deep Learning**: TensorFlow/Keras, PyTorch
- **Tree Models**: XGBoost, LightGBM, CatBoost
- **Online Learning**: River
- **Feature Engineering**: pandas, numpy, ta-lib
- **Model Tracking**: MLflow, Weights & Biases
- **Hyperparameter Tuning**: Optuna

### Quant Analysis
- **Statistical Analysis**: scipy, statsmodels
- **Time Series**: statsmodels, pmdarima
- **Backtesting**: vectorbt, backtrader

---

## Files to Create

### Real-Time Data (8 files)
```
src/data/
├── streaming/
│   ├── __init__.py
│   ├── kalshi_websocket.py      # WebSocket client
│   ├── event_processor.py       # Stream processing
│   └── message_queue.py          # Redis Streams integration
├── storage/
│   ├── __init__.py
│   ├── timescale_client.py      # TimescaleDB interface
│   └── schema.sql                # Database schema
```

### ML/Features (10 files)
```
src/ml/
├── features/
│   ├── __init__.py
│   ├── technical_indicators.py  # RSI, MACD, etc.
│   ├── microstructure.py        # Order flow, toxicity
│   ├── feature_store.py         # Feature management
│   └── feature_definitions.yaml # Feature catalog
├── models/
│   ├── __init__.py
│   ├── price_predictor.py       # LSTM, Transformer
│   ├── ensemble.py              # Model ensembling
│   └── online_learning.py       # Incremental updates
```

### Training Pipeline (6 files)
```
src/ml/training/
├── __init__.py
├── pipeline.py                  # Training pipeline
├── model_registry.py            # MLflow integration
├── hyperparameter_optimization.py
└── validation.py                # Walk-forward CV
```

### Quantitative Strategies (4 files)
```
src/ml/strategies/
├── __init__.py
├── stat_arb.py                  # Statistical arbitrage
├── regime_detection.py          # Market regime HMM
└── portfolio_optimizer.py       # Multi-strategy allocation
```

**Total**: ~30 new files, estimated 8,000+ lines

---

## Risks and Mitigations

### Risk 1: Data Quality
**Risk**: WebSocket drops, data gaps, invalid ticks
**Mitigation**:
- Heartbeat monitoring with auto-reconnect
- Gap detection and backfilling
- Data validation pipeline
- Alerts for data quality issues

### Risk 2: Model Overfitting
**Risk**: Models perform well in backtest but fail live
**Mitigation**:
- Walk-forward validation (not simple train/test split)
- Out-of-sample testing
- Paper trading before live deployment
- Regular model retraining
- Monitor prediction vs actual performance

### Risk 3: Latency
**Risk**: Slow feature computation or model inference
**Mitigation**:
- Pre-compute features where possible
- Model optimization (quantization, pruning)
- Redis for low-latency feature access
- Async I/O throughout
- Benchmark all critical paths (< 100ms target)

### Risk 4: Market Regime Changes
**Risk**: Models trained on one regime fail in another
**Mitigation**:
- Regime detection and strategy switching
- Online learning (continuous model updates)
- Ensemble of models (diverse strategies)
- Conservative position sizing
- Circuit breakers for unusual conditions

---

## Cost Estimate

### Infrastructure
- **TimescaleDB**: Self-hosted (included in existing Docker)
- **Redis**: Self-hosted (already have)
- **Additional Storage**: ~500GB/year ($10/month on cloud)
- **Compute** (model training): ~$100-200/month (GPU instances)

### Third-Party Services (Optional)
- **MLflow**: Self-hosted (free)
- **Weights & Biases**: Free tier → $50/month (team plan)
- **Feature Store** (Feast): Self-hosted (free)

**Total Monthly Cost**: $60-260/month (depending on cloud usage)

---

## Expected Impact

### From Current State:
- Uses LLM for text analysis (~$50/day in API costs)
- Rule-based filtering
- No predictive modeling
- Win rate: Unknown (not tracked)

### To Future State:
- ML models trained on historical data
- Real-time feature computation
- Ensemble predictions
- Expected win rate: 55-60%
- Expected Sharpe ratio: 1.5-2.0
- Reduced LLM costs (use ML predictions primarily, LLM for edge cases)

### ROI Estimate:
If we're trading with $10,000 capital:
- Current system: ~5-10% monthly return (mostly from edge finding)
- ML-enhanced system: ~15-20% monthly return (from predictive models + better risk management)
- Additional monthly profit: ~$500-1,000
- Infrastructure cost: ~$100-200/month
- **Net gain**: $300-900/month

---

**Phase 7 Status**: 🚧 Ready to implement
**Estimated Timeline**: 10 weeks for full implementation
**Expected Outcome**: True AI/ML trading system with quantitative edge

---

**Last Updated**: 2025-11-16
