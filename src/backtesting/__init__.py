"""
Backtesting Framework for Kalshi AI Trading Bot

This package provides comprehensive backtesting capabilities for validating
trading strategies on historical data before deploying them in live trading.

Features:
- Time-travel testing with realistic market conditions
- Historical data replay from database or CSV
- Performance attribution and analytics
- Realistic execution simulation including fees and slippage
- Comprehensive reporting and visualization

Example Usage:
    ```python
    from datetime import datetime, timedelta
    from src.backtesting import BacktestConfig, BacktestEngine, BacktestMode
    from src.backtesting import HistoricalDataLoader, ReportGenerator

    # Configure backtest
    config = BacktestConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
        initial_capital=10000.0,
        strategy_name="high_confidence",
        mode=BacktestMode.HISTORICAL
    )

    # Create engine and data loader
    engine = BacktestEngine(config)
    data_loader = HistoricalDataLoader(db_path="data/historical.db")

    # Run backtest
    metrics = await engine.run(strategy, data_loader)

    # Generate reports
    report_gen = ReportGenerator(engine)
    await report_gen.generate_all_reports(Path("reports/backtest_2024"))

    # Print results
    print(f"Return: {metrics.total_return:.2%}")
    print(f"Sharpe: {metrics.sharpe_ratio:.2f}")
    print(f"Max DD: {metrics.max_drawdown:.2%}")
    ```
"""

from .framework import (
    BacktestConfig,
    BacktestEngine,
    BacktestMode,
    PerformanceMetrics,
    Position,
    Trade,
)
from .data_loader import (
    HistoricalDataLoader,
    LiveDataRecorder,
    MarketSnapshot,
)
from .performance_attribution import (
    AttributionReport,
    CategoryAttribution,
    HoldingPeriodAttribution,
    MarketAttribution,
    PerformanceAttributionAnalyzer,
    SideAttribution,
    TimeAttribution,
)
from .report_generator import ReportGenerator

__all__ = [
    # Framework
    "BacktestConfig",
    "BacktestEngine",
    "BacktestMode",
    "PerformanceMetrics",
    "Position",
    "Trade",
    # Data Loading
    "HistoricalDataLoader",
    "LiveDataRecorder",
    "MarketSnapshot",
    # Attribution
    "AttributionReport",
    "CategoryAttribution",
    "HoldingPeriodAttribution",
    "MarketAttribution",
    "PerformanceAttributionAnalyzer",
    "SideAttribution",
    "TimeAttribution",
    # Reporting
    "ReportGenerator",
]

__version__ = "1.0.0"
