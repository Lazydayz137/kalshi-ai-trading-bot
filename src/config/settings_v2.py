"""
Enhanced Configuration System for Kalshi Trading Bot

This module provides a unified, environment-aware configuration system with:
- YAML/TOML support for easy configuration management
- Environment-specific configs (dev, staging, prod)
- No duplication between module-level and dataclass configs
- Comprehensive validation
- Backward compatibility with existing code
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Try to import YAML support (optional dependency)
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Try to import TOML support (optional dependency)
try:
    import tomli as toml  # Python 3.11+ has tomllib built-in
    HAS_TOML = True
except ImportError:
    try:
        import tomllib as toml
        HAS_TOML = True
    except ImportError:
        HAS_TOML = False


@dataclass
class APIConfig:
    """API configuration settings."""
    kalshi_api_key: str = field(default_factory=lambda: os.getenv("KALSHI_API_KEY", ""))
    kalshi_base_url: str = "https://api.elections.kalshi.com"
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    xai_api_key: str = field(default_factory=lambda: os.getenv("XAI_API_KEY", ""))
    openai_base_url: str = "https://api.openai.com/v1"


@dataclass
class PositionSizingConfig:
    """Position sizing and risk management."""
    max_position_size_pct: float = 5.0  # Maximum % of portfolio per position
    max_daily_loss_pct: float = 15.0     # Maximum daily loss tolerance
    max_positions: int = 15              # Maximum concurrent positions
    min_balance: float = 50.0            # Minimum balance to continue trading

    # Kelly Criterion settings (primary position sizing method)
    use_kelly_criterion: bool = True
    kelly_fraction: float = 0.75         # Kelly multiplier (0.75 = 3/4 Kelly)
    max_single_position: float = 0.05    # Maximum 5% per position


@dataclass
class MarketFilteringConfig:
    """Market filtering and selection criteria."""
    min_volume: float = 200.0                    # Minimum trading volume
    max_time_to_expiry_days: int = 30            # Maximum days to expiry
    min_volume_for_ai_analysis: float = 200.0    # Volume threshold for AI
    min_volume_for_market_making: float = 500.0  # Volume threshold for market making
    min_price_movement: float = 0.02             # Minimum price range (2¢)
    max_bid_ask_spread: float = 0.15             # Maximum spread (15¢)

    # Category filtering
    preferred_categories: List[str] = field(default_factory=list)
    excluded_categories: List[str] = field(default_factory=list)
    exclude_low_liquidity_categories: List[str] = field(default_factory=list)


@dataclass
class AIModelConfig:
    """AI model configuration."""
    primary_model: str = "grok-4"         # DO NOT CHANGE - primary model
    fallback_model: str = "grok-3"        # Fallback model
    ai_temperature: float = 0.0           # Temperature for consistency
    ai_max_tokens: int = 8000             # Token limit

    # AI decision thresholds
    min_confidence_to_trade: float = 0.50
    min_confidence_threshold: float = 0.45
    min_confidence_for_large_size: float = 0.50
    high_confidence_threshold: float = 0.95
    high_confidence_market_odds: float = 0.90


@dataclass
class CostControlConfig:
    """AI cost control and budget management."""
    # Daily limits
    daily_ai_budget: float = 50.0                 # Maximum daily AI spend
    daily_ai_cost_limit: float = 50.0             # Hard daily limit
    enable_daily_cost_limiting: bool = True       # Enable cost limits
    sleep_when_limit_reached: bool = True         # Sleep until next day

    # Per-decision limits
    max_analysis_cost_per_decision: float = 0.15  # Max cost per analysis
    max_ai_cost_per_decision: float = 0.12        # Max AI cost per decision

    # Analysis frequency controls
    analysis_cooldown_hours: int = 2              # Cooldown between analyses
    max_analyses_per_market_per_day: int = 6      # Max analyses per market/day

    # News search optimization
    skip_news_for_low_volume: bool = True
    news_search_volume_threshold: float = 1000.0


@dataclass
class TradingFrequencyConfig:
    """Trading frequency and timing settings."""
    market_scan_interval: int = 30           # Scan every 30 seconds
    position_check_interval: int = 15        # Check positions every 15 seconds
    max_trades_per_hour: int = 20            # Maximum trades per hour
    run_interval_minutes: int = 10           # Main loop interval
    scan_interval_seconds: int = 30          # Market scan interval
    num_processor_workers: int = 5           # Concurrent workers


@dataclass
class StrategyAllocationConfig:
    """Capital allocation across strategies."""
    market_making_allocation: float = 0.40      # 40% for market making
    directional_allocation: float = 0.50        # 50% for directional trading
    arbitrage_allocation: float = 0.10          # 10% for arbitrage

    # Strategy enablement
    enable_market_making: bool = True
    enable_high_confidence_strategy: bool = True


@dataclass
class PortfolioOptimizationConfig:
    """Portfolio optimization settings."""
    use_risk_parity: bool = True                # Risk parity allocation
    rebalance_hours: int = 6                    # Rebalance frequency
    min_position_size: float = 5.0              # Minimum position size ($)
    max_opportunities_per_batch: int = 50       # Max opportunities to process

    # Risk constraints
    max_volatility: float = 0.80                # Maximum portfolio volatility
    max_correlation: float = 0.95               # Maximum correlation
    max_drawdown: float = 0.50                  # Maximum drawdown tolerance
    max_sector_exposure: float = 0.90           # Maximum sector concentration


@dataclass
class PerformanceTargetsConfig:
    """Performance targets and thresholds."""
    target_sharpe: float = 0.3                  # Target Sharpe ratio
    target_return: float = 0.15                 # Target annual return
    min_trade_edge: float = 0.08                # Minimum edge requirement


@dataclass
class DynamicExitConfig:
    """Dynamic exit strategy configuration."""
    use_dynamic_exits: bool = True
    profit_threshold: float = 0.20              # Take profit at 20%
    loss_threshold: float = 0.15                # Stop loss at -15%
    confidence_decay_threshold: float = 0.25    # Exit if confidence drops
    max_hold_time_hours: int = 240              # Max hold time (10 days)
    volatility_adjustment: bool = True          # Adjust for volatility
    high_confidence_expiry_hours: int = 24      # High confidence expiry


@dataclass
class MarketMakingConfig:
    """Market making strategy configuration."""
    min_spread_for_making: float = 0.01         # Minimum spread (1¢)
    max_inventory_risk: float = 0.15            # Maximum inventory risk
    order_refresh_minutes: int = 15             # Order refresh frequency
    max_orders_per_market: int = 4              # Max orders per market


@dataclass
class SystemBehaviorConfig:
    """Overall system behavior settings."""
    beast_mode_enabled: bool = True             # Enable unified system
    fallback_to_legacy: bool = True             # Fallback to legacy
    live_trading_enabled: bool = False          # Live trading mode
    paper_trading_mode: bool = True             # Paper trading mode
    performance_monitoring: bool = True         # Performance monitoring

    # Advanced features (future)
    cross_market_arbitrage: bool = False
    multi_model_ensemble: bool = False
    sentiment_analysis: bool = False
    options_strategies: bool = False
    algorithmic_execution: bool = False


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = "logs/trading_system.log"
    enable_file_logging: bool = True
    enable_console_logging: bool = True
    max_log_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class DatabaseConfig:
    """Database configuration."""
    database_type: str = field(default_factory=lambda: os.getenv("DATABASE_TYPE", "sqlite"))
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///trading_system.db"))

    # SQLite specific
    sqlite_path: str = "trading_system.db"

    # PostgreSQL specific
    postgres_host: str = field(default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"))
    postgres_port: int = field(default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432")))
    postgres_database: str = field(default_factory=lambda: os.getenv("POSTGRES_DATABASE", "kalshi_trading"))
    postgres_user: str = field(default_factory=lambda: os.getenv("POSTGRES_USER", ""))
    postgres_password: str = field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", ""))

    # Connection pooling
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600


@dataclass
class CacheConfig:
    """Caching configuration."""
    enable_caching: bool = field(default_factory=lambda: os.getenv("ENABLE_CACHING", "true").lower() == "true")
    cache_type: str = field(default_factory=lambda: os.getenv("CACHE_TYPE", "memory"))  # memory or redis

    # Redis configuration
    redis_host: str = field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    redis_port: int = field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))
    redis_db: int = 0
    redis_password: str = field(default_factory=lambda: os.getenv("REDIS_PASSWORD", ""))

    # Cache TTLs (in seconds)
    market_data_ttl: int = 30      # Market data cache TTL
    balance_ttl: int = 60          # Balance cache TTL
    position_ttl: int = 30         # Position cache TTL


@dataclass
class TradingConfig:
    """Main trading configuration combining all sub-configs."""
    # Sub-configurations
    position_sizing: PositionSizingConfig = field(default_factory=PositionSizingConfig)
    market_filtering: MarketFilteringConfig = field(default_factory=MarketFilteringConfig)
    ai_model: AIModelConfig = field(default_factory=AIModelConfig)
    cost_control: CostControlConfig = field(default_factory=CostControlConfig)
    trading_frequency: TradingFrequencyConfig = field(default_factory=TradingFrequencyConfig)
    strategy_allocation: StrategyAllocationConfig = field(default_factory=StrategyAllocationConfig)
    portfolio_optimization: PortfolioOptimizationConfig = field(default_factory=PortfolioOptimizationConfig)
    performance_targets: PerformanceTargetsConfig = field(default_factory=PerformanceTargetsConfig)
    dynamic_exit: DynamicExitConfig = field(default_factory=DynamicExitConfig)
    market_making: MarketMakingConfig = field(default_factory=MarketMakingConfig)
    system_behavior: SystemBehaviorConfig = field(default_factory=SystemBehaviorConfig)

    # Backward compatibility - delegate to sub-configs
    @property
    def max_position_size_pct(self) -> float:
        return self.position_sizing.max_position_size_pct

    @property
    def max_daily_loss_pct(self) -> float:
        return self.position_sizing.max_daily_loss_pct

    @property
    def max_positions(self) -> int:
        return self.position_sizing.max_positions

    @property
    def min_balance(self) -> float:
        return self.position_sizing.min_balance

    @property
    def min_volume(self) -> float:
        return self.market_filtering.min_volume

    @property
    def max_time_to_expiry_days(self) -> int:
        return self.market_filtering.max_time_to_expiry_days

    @property
    def min_confidence_to_trade(self) -> float:
        return self.ai_model.min_confidence_to_trade

    @property
    def scan_interval_seconds(self) -> int:
        return self.trading_frequency.scan_interval_seconds

    @property
    def primary_model(self) -> str:
        return self.ai_model.primary_model

    @property
    def fallback_model(self) -> str:
        return self.ai_model.fallback_model

    @property
    def ai_temperature(self) -> float:
        return self.ai_model.ai_temperature

    @property
    def ai_max_tokens(self) -> int:
        return self.ai_model.ai_max_tokens

    @property
    def daily_ai_budget(self) -> float:
        return self.cost_control.daily_ai_budget

    @property
    def max_ai_cost_per_decision(self) -> float:
        return self.cost_control.max_ai_cost_per_decision

    @property
    def analysis_cooldown_hours(self) -> int:
        return self.cost_control.analysis_cooldown_hours

    @property
    def max_analyses_per_market_per_day(self) -> int:
        return self.cost_control.max_analyses_per_market_per_day

    @property
    def min_volume_for_ai_analysis(self) -> float:
        return self.market_filtering.min_volume_for_ai_analysis

    @property
    def exclude_low_liquidity_categories(self) -> List[str]:
        return self.market_filtering.exclude_low_liquidity_categories

    @property
    def skip_news_for_low_volume(self) -> bool:
        return self.cost_control.skip_news_for_low_volume

    @property
    def news_search_volume_threshold(self) -> float:
        return self.cost_control.news_search_volume_threshold

    @property
    def use_kelly_criterion(self) -> bool:
        return self.position_sizing.use_kelly_criterion

    @property
    def kelly_fraction(self) -> float:
        return self.position_sizing.kelly_fraction

    @property
    def max_single_position(self) -> float:
        return self.position_sizing.max_single_position

    @property
    def market_scan_interval(self) -> int:
        return self.trading_frequency.market_scan_interval

    @property
    def position_check_interval(self) -> int:
        return self.trading_frequency.position_check_interval

    @property
    def max_trades_per_hour(self) -> int:
        return self.trading_frequency.max_trades_per_hour

    @property
    def run_interval_minutes(self) -> int:
        return self.trading_frequency.run_interval_minutes

    @property
    def num_processor_workers(self) -> int:
        return self.trading_frequency.num_processor_workers

    @property
    def preferred_categories(self) -> List[str]:
        return self.market_filtering.preferred_categories

    @property
    def excluded_categories(self) -> List[str]:
        return self.market_filtering.excluded_categories

    @property
    def enable_high_confidence_strategy(self) -> bool:
        return self.strategy_allocation.enable_high_confidence_strategy

    @property
    def high_confidence_threshold(self) -> float:
        return self.ai_model.high_confidence_threshold

    @property
    def high_confidence_market_odds(self) -> float:
        return self.ai_model.high_confidence_market_odds

    @property
    def high_confidence_expiry_hours(self) -> int:
        return self.dynamic_exit.high_confidence_expiry_hours

    @property
    def max_analysis_cost_per_decision(self) -> float:
        return self.cost_control.max_analysis_cost_per_decision

    @property
    def min_confidence_threshold(self) -> float:
        return self.ai_model.min_confidence_threshold

    @property
    def daily_ai_cost_limit(self) -> float:
        return self.cost_control.daily_ai_cost_limit

    @property
    def enable_daily_cost_limiting(self) -> bool:
        return self.cost_control.enable_daily_cost_limiting

    @property
    def sleep_when_limit_reached(self) -> bool:
        return self.cost_control.sleep_when_limit_reached

    @property
    def live_trading_enabled(self) -> bool:
        return self.system_behavior.live_trading_enabled

    @live_trading_enabled.setter
    def live_trading_enabled(self, value: bool):
        self.system_behavior.live_trading_enabled = value

    @property
    def paper_trading_mode(self) -> bool:
        return self.system_behavior.paper_trading_mode

    @paper_trading_mode.setter
    def paper_trading_mode(self, value: bool):
        self.system_behavior.paper_trading_mode = value


@dataclass
class Settings:
    """Main settings class combining all configuration."""
    api: APIConfig = field(default_factory=APIConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

    def validate(self) -> bool:
        """Validate configuration settings."""
        if not self.api.kalshi_api_key:
            raise ValueError("KALSHI_API_KEY environment variable is required")

        if not self.api.xai_api_key:
            raise ValueError("XAI_API_KEY environment variable is required")

        if self.trading.position_sizing.max_position_size_pct <= 0 or self.trading.position_sizing.max_position_size_pct > 100:
            raise ValueError("max_position_size_pct must be between 0 and 100")

        if self.trading.ai_model.min_confidence_to_trade <= 0 or self.trading.ai_model.min_confidence_to_trade > 1:
            raise ValueError("min_confidence_to_trade must be between 0 and 1")

        return True

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Settings':
        """Load settings from YAML file."""
        if not HAS_YAML:
            raise ImportError("PyYAML is required for YAML support. Install with: pip install pyyaml")

        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls._from_dict(config_dict)

    @classmethod
    def from_toml(cls, toml_path: str) -> 'Settings':
        """Load settings from TOML file."""
        if not HAS_TOML:
            raise ImportError("tomli is required for TOML support. Install with: pip install tomli")

        with open(toml_path, 'rb') as f:
            config_dict = toml.load(f)

        return cls._from_dict(config_dict)

    @classmethod
    def _from_dict(cls, config_dict: Dict[str, Any]) -> 'Settings':
        """Create Settings from dictionary."""
        # This is a simplified implementation - in production you'd want more robust parsing
        return cls(
            api=APIConfig(**config_dict.get('api', {})),
            trading=TradingConfig(**config_dict.get('trading', {})),
            logging=LoggingConfig(**config_dict.get('logging', {})),
            database=DatabaseConfig(**config_dict.get('database', {})),
            cache=CacheConfig(**config_dict.get('cache', {}))
        )

    def to_dict(self) -> Dict[str, Any]:
        """Export settings to dictionary."""
        from dataclasses import asdict
        return asdict(self)


# Global settings instance
settings = Settings()

# Module-level aliases for backward compatibility
# These delegate to the new structured config
market_making_allocation = settings.trading.strategy_allocation.market_making_allocation
directional_allocation = settings.trading.strategy_allocation.directional_allocation
arbitrage_allocation = settings.trading.strategy_allocation.arbitrage_allocation

use_risk_parity = settings.trading.portfolio_optimization.use_risk_parity
rebalance_hours = settings.trading.portfolio_optimization.rebalance_hours
min_position_size = settings.trading.portfolio_optimization.min_position_size
max_opportunities_per_batch = settings.trading.portfolio_optimization.max_opportunities_per_batch

max_volatility = settings.trading.portfolio_optimization.max_volatility
max_correlation = settings.trading.portfolio_optimization.max_correlation
max_drawdown = settings.trading.portfolio_optimization.max_drawdown
max_sector_exposure = settings.trading.portfolio_optimization.max_sector_exposure

target_sharpe = settings.trading.performance_targets.target_sharpe
target_return = settings.trading.performance_targets.target_return
min_trade_edge = settings.trading.performance_targets.min_trade_edge
min_confidence_for_large_size = settings.trading.ai_model.min_confidence_for_large_size

use_dynamic_exits = settings.trading.dynamic_exit.use_dynamic_exits
profit_threshold = settings.trading.dynamic_exit.profit_threshold
loss_threshold = settings.trading.dynamic_exit.loss_threshold
confidence_decay_threshold = settings.trading.dynamic_exit.confidence_decay_threshold
max_hold_time_hours = settings.trading.dynamic_exit.max_hold_time_hours
volatility_adjustment = settings.trading.dynamic_exit.volatility_adjustment

enable_market_making = settings.trading.strategy_allocation.enable_market_making
min_spread_for_making = settings.trading.market_making.min_spread_for_making
max_inventory_risk = settings.trading.market_making.max_inventory_risk
order_refresh_minutes = settings.trading.market_making.order_refresh_minutes
max_orders_per_market = settings.trading.market_making.max_orders_per_market

min_volume_for_analysis = settings.trading.market_filtering.min_volume_for_ai_analysis
min_volume_for_market_making = settings.trading.market_filtering.min_volume_for_market_making
min_price_movement = settings.trading.market_filtering.min_price_movement
max_bid_ask_spread = settings.trading.market_filtering.max_bid_ask_spread
min_confidence_long_term = settings.trading.ai_model.min_confidence_threshold

daily_ai_budget = settings.trading.cost_control.daily_ai_budget
max_ai_cost_per_decision = settings.trading.cost_control.max_ai_cost_per_decision
analysis_cooldown_hours = settings.trading.cost_control.analysis_cooldown_hours
max_analyses_per_market_per_day = settings.trading.cost_control.max_analyses_per_market_per_day
skip_news_for_low_volume = settings.trading.cost_control.skip_news_for_low_volume
news_search_volume_threshold = settings.trading.cost_control.news_search_volume_threshold

beast_mode_enabled = settings.trading.system_behavior.beast_mode_enabled
fallback_to_legacy = settings.trading.system_behavior.fallback_to_legacy
live_trading_enabled = settings.trading.system_behavior.live_trading_enabled
paper_trading_mode = settings.trading.system_behavior.paper_trading_mode
log_level = settings.logging.log_level
performance_monitoring = settings.trading.system_behavior.performance_monitoring

cross_market_arbitrage = settings.trading.system_behavior.cross_market_arbitrage
multi_model_ensemble = settings.trading.system_behavior.multi_model_ensemble
sentiment_analysis = settings.trading.system_behavior.sentiment_analysis
options_strategies = settings.trading.system_behavior.options_strategies
algorithmic_execution = settings.trading.system_behavior.algorithmic_execution

# Validate settings on import
try:
    settings.validate()
except ValueError as e:
    print(f"Configuration validation error: {e}")
    print("Please check your environment variables and configuration.")
