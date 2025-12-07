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
    openai_base_url: str = "https://api.openai.com/v1"


# Trading strategy configuration - INCREASED AGGRESSIVENESS
@dataclass
class TradingConfig:
    """Trading strategy configuration."""
    # Position sizing and risk management - CONSERVATIVE SETTINGS FOR SAFETY
    max_position_size_pct: float = 3.0  # CONSERVATIVE: 3% max per position for risk control
    max_daily_loss_pct: float = 10.0    # CONSERVATIVE: 10% daily loss limit
    max_positions: int = 10              # CONSERVATIVE: Max 10 concurrent positions
    min_balance: float = 100.0          # CONSERVATIVE: Minimum $100 balance required
    
    # Market filtering criteria - CONSERVATIVE FOR QUALITY
    min_volume: float = 500.0            # CONSERVATIVE: Higher volume for better liquidity
    max_time_to_expiry_days: int = 30    # Reasonable timeframe for predictions

    # AI decision making - CONSERVATIVE CONFIDENCE THRESHOLDS
    min_confidence_to_trade: float = 0.65   # CONSERVATIVE: 65% minimum confidence required
    scan_interval_seconds: int = 60      # CONSERVATIVE: Scan every 60 seconds to reduce API load
    
    # AI model configuration
    primary_model: str = "grok-4" # DO NOT CHANGE THIS UNDER ANY CIRCUMSTANCES
    fallback_model: str = "grok-3"  # Fallback to available model
    ai_temperature: float = 0  # Lower temperature for more consistent JSON output
    ai_max_tokens: int = 8000    # Reasonable limit for reasoning models (grok-4 works better with 8000)
    
    # Position sizing (LEGACY - now using Kelly-primary approach)
    default_position_size: float = 3.0  # REDUCED: Now using Kelly Criterion as primary method (was 5%, now 3%)
    position_size_multiplier: float = 1.0  # Multiplier for AI confidence
    
    # Kelly Criterion settings (PRIMARY position sizing method) - CONSERVATIVE
    use_kelly_criterion: bool = True        # Use Kelly Criterion for position sizing (PRIMARY METHOD)
    kelly_fraction: float = 0.25            # CONSERVATIVE: Quarter Kelly for safety (industry standard)
    max_single_position: float = 0.03       # CONSERVATIVE: 3% max single position cap
    
    # Trading frequency - CONSERVATIVE TO REDUCE COSTS
    market_scan_interval: int = 60          # CONSERVATIVE: Scan every 60 seconds
    position_check_interval: int = 30       # CONSERVATIVE: Check positions every 30 seconds
    max_trades_per_hour: int = 10           # CONSERVATIVE: Limit to 10 trades per hour
    run_interval_minutes: int = 15          # CONSERVATIVE: Run every 15 minutes
    num_processor_workers: int = 5      # Number of concurrent market processor workers
    
    # Market selection preferences
    preferred_categories: List[str] = field(default_factory=lambda: [])
    excluded_categories: List[str] = field(default_factory=lambda: [])
    
    # High-confidence, near-expiry strategy
    enable_high_confidence_strategy: bool = True
    high_confidence_threshold: float = 0.95  # LLM confidence needed
    high_confidence_market_odds: float = 0.90 # Market price to look for
    high_confidence_expiry_hours: int = 24   # Max hours until expiry

    # AI trading criteria - CONSERVATIVE COST CONTROLS
    max_analysis_cost_per_decision: float = 0.10  # CONSERVATIVE: $0.10 max per decision
    min_confidence_threshold: float = 0.60  # CONSERVATIVE: 60% minimum confidence

    # Cost control and market analysis frequency - CONSERVATIVE BUDGET
    daily_ai_budget: float = 5.0  # CONSERVATIVE: $5 daily AI budget
    max_ai_cost_per_decision: float = 0.05  # CONSERVATIVE: $0.05 max per decision
    analysis_cooldown_hours: int = 6  # CONSERVATIVE: 6 hour cooldown between analyses
    max_analyses_per_market_per_day: int = 2  # CONSERVATIVE: Max 2 analyses per market per day
    
    # Daily AI spending limits - SAFETY CONTROLS
    daily_ai_cost_limit: float = 50.0  # Maximum daily spending on AI API calls (USD)
    enable_daily_cost_limiting: bool = True  # Enable daily cost limits
    sleep_when_limit_reached: bool = True  # Sleep until next day when limit reached

    # Paper Trading Configuration
    paper_trading_mode: bool = True  # SAFE DEFAULT: Paper trading enabled
    paper_trading_balance: float = 10000.0  # Starting balance for paper trading
    paper_simulate_slippage: bool = True  # Simulate realistic slippage
    paper_simulate_fees: bool = True  # Simulate Kalshi fees (0.7%)
    paper_slippage_bps: float = 5.0  # Slippage in basis points (0.05%)

    # Enhanced market filtering to reduce analyses - CONSERVATIVE FILTERS
    min_volume_for_ai_analysis: float = 500.0  # CONSERVATIVE: $500 minimum volume for AI analysis
    exclude_low_liquidity_categories: List[str] = field(default_factory=lambda: [
        "weather",  # Exclude low-liquidity weather markets
        "entertainment"  # Exclude entertainment markets
    ])


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


# BEAST MODE UNIFIED TRADING SYSTEM CONFIGURATION 🚀
# These settings control the advanced multi-strategy trading system

# === CAPITAL ALLOCATION ACROSS STRATEGIES ===
# Allocate capital across different trading approaches
market_making_allocation: float = 0.40  # 40% for market making (spread profits)
directional_allocation: float = 0.50    # 50% for directional trading (AI predictions) 
arbitrage_allocation: float = 0.10      # 10% for arbitrage opportunities

  # === PORTFOLIO OPTIMIZATION SETTINGS ===
# Kelly Criterion is now the PRIMARY position sizing method (moved to TradingConfig)
# total_capital: DYNAMICALLY FETCHED from Kalshi balance - never hardcoded!
use_risk_parity: bool = True            # Equal risk allocation vs equal capital
rebalance_hours: int = 6                # Rebalance portfolio every 6 hours
min_position_size: float = 5.0          # Minimum position size ($5 vs $10)
max_opportunities_per_batch: int = 50   # Limit opportunities to prevent optimization issues

# === RISK MANAGEMENT LIMITS ===
# Portfolio-level risk constraints (EXTREMELY RELAXED FOR TESTING)
max_volatility: float = 0.80            # Very high volatility allowed (80%)
max_correlation: float = 0.95           # Very high correlation allowed (95%)
max_drawdown: float = 0.50              # High drawdown tolerance (50%)
max_sector_exposure: float = 0.90       # Very high sector concentration (90%)

# === PERFORMANCE TARGETS ===
# System performance objectives - CONSERVATIVE FOR QUALITY TRADES
target_sharpe: float = 1.5              # CONSERVATIVE: Target 1.5 Sharpe ratio
target_return: float = 0.20             # Reasonable 20% annual return target
min_trade_edge: float = 0.10            # CONSERVATIVE: 10% minimum edge required
min_confidence_for_large_size: float = 0.70  # CONSERVATIVE: 70% confidence for large positions

# === DYNAMIC EXIT STRATEGIES ===
# Enhanced exit strategy settings - CONSERVATIVE RISK MANAGEMENT
use_dynamic_exits: bool = True
profit_threshold: float = 0.25          # CONSERVATIVE: Take 25% profits
loss_threshold: float = 0.10            # CONSERVATIVE: Cut losses at 10%
confidence_decay_threshold: float = 0.20  # CONSERVATIVE: Exit if confidence drops 20%
max_hold_time_hours: int = 168          # CONSERVATIVE: Max 7 days hold time
volatility_adjustment: bool = True      # Adjust exits based on volatility

# === MARKET MAKING STRATEGY ===
# Settings for limit order market making - MORE AGGRESSIVE
enable_market_making: bool = True       # Enable market making strategy
min_spread_for_making: float = 0.01     # DECREASED: Accept smaller spreads (was 0.02, now 1¢)
max_inventory_risk: float = 0.15        # INCREASED: Allow higher inventory risk (was 0.10, now 15%)
order_refresh_minutes: int = 15         # Refresh orders every 15 minutes
max_orders_per_market: int = 4          # Maximum orders per market (2 each side)

# === MARKET SELECTION (ENHANCED FOR MORE OPPORTUNITIES) ===
# Removed time restrictions - trade ANY deadline with dynamic exits!
# max_time_to_expiry_days: REMOVED      # No longer used - trade any timeline!
min_volume_for_analysis: float = 200.0  # DECREASED: Much lower minimum volume (was 1000, now 200)
min_volume_for_market_making: float = 500.0  # DECREASED: Lower volume for market making (was 2000, now 500)
min_price_movement: float = 0.02        # DECREASED: Lower minimum range (was 0.05, now 2¢)
max_bid_ask_spread: float = 0.15        # INCREASED: Allow wider spreads (was 0.10, now 15¢)
min_confidence_long_term: float = 0.45  # DECREASED: Lower confidence for distant expiries (was 0.65, now 45%)

# === COST OPTIMIZATION (CONSERVATIVE BUDGET) ===
# Enhanced cost controls for the beast mode system
daily_ai_budget: float = 5.0            # CONSERVATIVE: $5 daily AI budget
max_ai_cost_per_decision: float = 0.08  # CONSERVATIVE: $0.08 max per decision
analysis_cooldown_hours: int = 6        # CONSERVATIVE: 6 hour cooldown
max_analyses_per_market_per_day: int = 2  # CONSERVATIVE: Max 2 analyses per market per day
skip_news_for_low_volume: bool = True   # Skip expensive searches for low volume
news_search_volume_threshold: float = 1000.0  # News threshold

# === SYSTEM BEHAVIOR ===
# Overall system behavior settings
beast_mode_enabled: bool = True         # Enable the unified advanced system
fallback_to_legacy: bool = True         # Fallback to legacy system if needed
live_trading_enabled: bool = True       # Set to True for live trading
paper_trading_mode: bool = False        # Paper trading for testing
log_level: str = "INFO"                 # Logging level
performance_monitoring: bool = True     # Enable performance monitoring

# === ADVANCED FEATURES ===
# Cutting-edge features for maximum performance
cross_market_arbitrage: bool = False    # Enable when arbitrage module ready
multi_model_ensemble: bool = False      # Use multiple AI models (future)
sentiment_analysis: bool = False        # News sentiment analysis (future)
options_strategies: bool = False        # Complex options strategies (future)
algorithmic_execution: bool = False     # Smart order execution (future)


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
        
        if not self.api.xai_api_key:
            raise ValueError("XAI_API_KEY environment variable is required")
        
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