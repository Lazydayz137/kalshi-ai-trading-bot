"""
Configuration settings for the Kalshi trading system.
Manages trading parameters, API configurations, and risk management settings.
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class APIConfig:
    """API configuration settings."""
    kalshi_api_key: str = field(default_factory=lambda: os.getenv("KALSHI_API_KEY", ""))
    kalshi_base_url: str = "https://api.elections.kalshi.com"  # Updated to new API endpoint
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    xai_api_key: str = field(default_factory=lambda: os.getenv("XAI_API_KEY", ""))
    openai_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    odds_api_key: str = field(default_factory=lambda: os.getenv("ODDS_API_KEY", ""))


# Trading strategy configuration - INCREASED AGGRESSIVENESS
@dataclass
class TradingConfig:
    """Trading strategy configuration."""
    # Position sizing and risk management - MADE MORE AGGRESSIVE  
    max_position_size_pct: float = 5.0  # INCREASED: Back to 5% per position (was 3%)
    max_daily_loss_pct: float = 15.0    # INCREASED: Allow 15% daily loss (was 10%) 
    max_positions: int = 15              # INCREASED: Allow 15 concurrent positions (was 10)
    min_balance: float = 50.0           # REDUCED: Lower minimum to trade more (was 100)
    
    # Market filtering criteria - MUCH MORE PERMISSIVE
    min_volume: float = 200.0            # DECREASED: Much lower volume requirement (was 500, now 200)
    max_time_to_expiry_days: int = 400    # INCREASED: Allow long-term markets (was 30, now 400 for Political/Yearly events)
    
    # AI decision making - MORE AGGRESSIVE THRESHOLDS
    min_confidence_to_trade: float = 0.50   # DECREASED: Lower confidence barrier (was 0.65, now 0.50)
    scan_interval_seconds: int = 30      # DECREASED: Scan more frequently (was 60, now 30)
    
    # AI model configuration
    primary_model: str = field(default_factory=lambda: os.getenv("PRIMARY_MODEL", "grok-4"))
    fallback_model: str = "grok-3"  # Fallback to available model
    ai_temperature: float = 0.3  # Increased to allow more nuanced confidence scores
    ai_max_tokens: int = 8000    # Reasonable limit for reasoning models (grok-4 works better with 8000)
    
    # Position sizing (LEGACY - now using Kelly-primary approach)
    default_position_size: float = 3.0  # REDUCED: Now using Kelly Criterion as primary method (was 5%, now 3%)
    position_size_multiplier: float = 1.0  # Multiplier for AI confidence
    
    # Kelly Criterion settings (PRIMARY position sizing method) - MORE AGGRESSIVE
    use_kelly_criterion: bool = True        # Use Kelly Criterion for position sizing (PRIMARY METHOD)
    kelly_fraction: float = 0.75            # INCREASED: More aggressive Kelly multiplier (was 0.5, now 0.75)
    max_single_position: float = 0.05       # INCREASED: Higher position cap (was 0.03, now 5%)
    
    # Trading frequency - SNIPER MODE 🔫
    market_scan_interval: int = 15          # DECREASED: Scan every 15 seconds (was 30)
    position_check_interval: int = 5        # DECREASED: Check positions every 5 seconds (was 15)
    max_trades_per_hour: int = 100          # INCREASED: Allow 100 trades/hr (High Frequency)
    run_interval_minutes: float = 0.5       # DECREASED: Run every 30 seconds (was 10 min)
    num_processor_workers: int = 5      # Number of concurrent market processor workers
    
    # Market selection preferences
    preferred_categories: List[str] = field(default_factory=lambda: [])
    excluded_categories: List[str] = field(default_factory=lambda: [])
    
    # High-confidence, near-expiry strategy
    enable_high_confidence_strategy: bool = True
    high_confidence_threshold: float = 0.95  # LLM confidence needed
    high_confidence_market_odds: float = 0.90 # Market price to look for
    high_confidence_expiry_hours: int = 24   # Max hours until expiry

    # AI trading criteria - MORE PERMISSIVE
    exclude_low_liquidity_categories: List[str] = field(default_factory=lambda: [
        # REMOVED weather and entertainment - trade all categories
    ])

    # === CAPITAL ALLOCATION ACROSS STRATEGIES ===
    market_making_allocation: float = 0.40
    directional_allocation: float = 0.50
    arbitrage_allocation: float = 0.10

    # === PORTFOLIO OPTIMIZATION SETTINGS ===
    use_risk_parity: bool = True
    rebalance_hours: int = 6
    min_position_size: float = 0.01       # DECREASED: Allow micro-trades ($0.01)
    max_opportunities_per_batch: int = 50

    # === RISK MANAGEMENT LIMITS ===
    max_volatility: float = 0.80
    max_correlation: float = 0.95
    max_drawdown: float = 0.50
    max_sector_exposure: float = 0.90

    # === PERFORMANCE TARGETS ===
    target_sharpe: float = 0.3
    target_sharpe: float = 0.3
    target_return: float = 0.15
    min_trade_edge: float = 0.01        # DECREASED: Match EdgeFilter (1%)
    min_confidence_for_large_size: float = 0.50

    # === DYNAMIC EXIT STRATEGIES ===
    use_dynamic_exits: bool = True
    profit_threshold: float = 0.20
    loss_threshold: float = 0.15
    confidence_decay_threshold: float = 0.25
    max_hold_time_hours: int = 240
    volatility_adjustment: bool = True

    # === MARKET MAKING STRATEGY ===
    enable_market_making: bool = True
    min_spread_for_making: float = 0.01
    max_inventory_risk: float = 0.15
    order_refresh_minutes: int = 15
    max_orders_per_market: int = 4

    # === MARKET SELECTION ===
    min_volume_for_analysis: float = 50.0   # DECREASED: Match EdgeFilter
    min_volume_for_market_making: float = 500.0
    min_price_movement: float = 0.02
    max_bid_ask_spread: float = 0.50        # INCREASED: Allow wider spreads (50c)
    min_confidence_long_term: float = 0.35  # DECREASED: Match EdgeFilter

    # === COST OPTIMIZATION ===
    daily_ai_budget: float = 250.0  # Increased to $250
    daily_ai_cost_limit: float = 250.0  # limit
    paper_balance: float = 250.0    # Simulated balance for paper trading
    enable_daily_cost_limiting: bool = True
    sleep_when_limit_reached: bool = True
    max_ai_cost_per_decision: float = 0.50  # Increased for deeper reasoning
    analysis_cooldown_hours: int = 2
    max_analyses_per_market_per_day: int = 6
    skip_news_for_low_volume: bool = False  # Enable news for ALL markets
    news_search_volume_threshold: float = 0.0  # Search everything

    # === SYSTEM BEHAVIOR ===
    beast_mode_enabled: bool = True
    fallback_to_legacy: bool = True
    live_trading_enabled: bool = False   # SAFETY: Default to False
    paper_trading_mode: bool = True      # SAFETY: Default to True
    log_level: str = "INFO"
    performance_monitoring: bool = True

    # === ADVANCED FEATURES ===
    cross_market_arbitrage: bool = False
    multi_model_ensemble: bool = False
    sentiment_analysis: bool = False
    options_strategies: bool = False
    algorithmic_execution: bool = False


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_level: str = "DEBUG"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = "logs/trading_system.log"
    enable_file_logging: bool = True
    enable_console_logging: bool = True
    max_log_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5





@dataclass
class Settings:
    """Main settings class combining all configuration."""
    api: APIConfig = field(default_factory=APIConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        if not self.api.kalshi_api_key:
            raise ValueError("KALSHI_API_KEY environment variable is required")
        
        # Check for either xAI or OpenAI/OpenRouter key
        if not self.api.xai_api_key and not self.api.openai_api_key:
            raise ValueError("Either XAI_API_KEY or OPENAI_API_KEY environment variable is required")
        
        if self.trading.max_position_size_pct <= 0 or self.trading.max_position_size_pct > 100:
            raise ValueError("max_position_size_pct must be between 0 and 100")
        
        if self.trading.min_confidence_to_trade <= 0 or self.trading.min_confidence_to_trade > 1:
            raise ValueError("min_confidence_to_trade must be between 0 and 1")
        
        return True


# Global settings instance
settings = Settings()

# Validate settings on import
try:
    settings.validate()
except ValueError as e:
    print(f"Configuration validation error: {e}")
    print("Please check your environment variables and configuration.") 