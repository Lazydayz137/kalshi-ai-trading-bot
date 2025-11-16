"""
Backtest Report Generator

This module generates comprehensive backtest reports with visualizations,
performance metrics, and trade analysis.
"""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from .framework import BacktestEngine, PerformanceMetrics
from .performance_attribution import PerformanceAttributionAnalyzer
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """
    Generates comprehensive backtest reports in multiple formats.

    Supports:
    - HTML reports with interactive charts
    - JSON reports for programmatic analysis
    - Markdown reports for documentation
    - CSV exports for Excel analysis
    """

    def __init__(self, backtest_engine: BacktestEngine):
        """
        Initialize report generator.

        Args:
            backtest_engine: Completed backtest engine with results
        """
        self.engine = backtest_engine
        self.config = backtest_engine.config
        self.metrics = backtest_engine.metrics
        self.trades = backtest_engine.trades

        # Create attribution analyzer
        self.attribution = PerformanceAttributionAnalyzer(
            trades=self.trades,
            initial_capital=self.config.initial_capital
        )

    async def generate_all_reports(self, output_dir: Path):
        """
        Generate all report formats.

        Args:
            output_dir: Directory to save reports
        """
        logger.info(f"Generating backtest reports in {output_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate each report type
        await self.generate_json_report(output_dir / "backtest_report.json")
        await self.generate_markdown_report(output_dir / "backtest_report.md")
        await self.generate_html_report(output_dir / "backtest_report.html")
        await self.generate_csv_exports(output_dir)

        logger.info(f"All reports generated successfully in {output_dir}")

    async def generate_json_report(self, output_path: Path):
        """Generate JSON report for programmatic analysis."""
        logger.info(f"Generating JSON report: {output_path}")

        # Get attribution analysis
        attribution_report = self.attribution.analyze()

        report_data = {
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "strategy_name": self.config.strategy_name,
                "start_date": self.config.start_date.isoformat(),
                "end_date": self.config.end_date.isoformat(),
                "initial_capital": self.config.initial_capital,
            },
            "performance_metrics": {
                "total_return": self.metrics.total_return,
                "annualized_return": self.metrics.annualized_return,
                "sharpe_ratio": self.metrics.sharpe_ratio,
                "sortino_ratio": self.metrics.sortino_ratio,
                "max_drawdown": self.metrics.max_drawdown,
                "volatility": self.metrics.volatility,
                "total_trades": self.metrics.total_trades,
                "winning_trades": self.metrics.winning_trades,
                "losing_trades": self.metrics.losing_trades,
                "win_rate": self.metrics.win_rate,
                "average_win": self.metrics.average_win,
                "average_loss": self.metrics.average_loss,
                "profit_factor": self.metrics.profit_factor,
                "total_commissions": self.metrics.total_commissions,
                "total_slippage": self.metrics.total_slippage,
            },
            "trades": self.engine.get_trades_summary(),
            "equity_curve": [
                {"date": dt.isoformat(), "value": val}
                for dt, val in self.engine.get_equity_curve()
            ],
            "attribution": {
                "category": [asdict(cat) for cat in attribution_report.category_attribution],
                "side": [asdict(side) for side in attribution_report.side_attribution],
                "holding_period": [asdict(hp) for hp in attribution_report.holding_period_attribution],
                "top_markets": [asdict(m) for m in attribution_report.market_attribution[:10]],
            },
        }

        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        logger.info(f"JSON report saved: {output_path}")

    async def generate_markdown_report(self, output_path: Path):
        """Generate Markdown report for documentation."""
        logger.info(f"Generating Markdown report: {output_path}")

        attribution_report = self.attribution.analyze()

        md_lines = [
            "# Backtest Report",
            "",
            f"**Strategy:** {self.config.strategy_name}",
            f"**Period:** {self.config.start_date.date()} to {self.config.end_date.date()}",
            f"**Initial Capital:** ${self.config.initial_capital:,.2f}",
            f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "---",
            "",
            "## Performance Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| **Total Return** | {self.metrics.total_return:.2%} |",
            f"| **Annualized Return** | {self.metrics.annualized_return:.2%} |",
            f"| **Sharpe Ratio** | {self.metrics.sharpe_ratio:.2f} |",
            f"| **Max Drawdown** | {self.metrics.max_drawdown:.2%} |",
            f"| **Volatility** | {self.metrics.volatility:.2%} |",
            f"| **Total Trades** | {self.metrics.total_trades} |",
            f"| **Win Rate** | {self.metrics.win_rate:.2%} |",
            f"| **Profit Factor** | {self.metrics.profit_factor:.2f} |",
            "",
            "## Trade Statistics",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| **Winning Trades** | {self.metrics.winning_trades} |",
            f"| **Losing Trades** | {self.metrics.losing_trades} |",
            f"| **Average Win** | ${self.metrics.average_win:.2f} |",
            f"| **Average Loss** | ${self.metrics.average_loss:.2f} |",
            f"| **Total Commissions** | ${self.metrics.total_commissions:.2f} |",
            f"| **Total Slippage** | ${self.metrics.total_slippage:.2f} |",
            "",
            "## Attribution Analysis",
            "",
            "### Performance by Category",
            "",
            f"| Category | Trades | Total PnL | Win Rate |",
            f"|----------|--------|-----------|----------|",
        ]

        for cat in attribution_report.category_attribution[:10]:
            md_lines.append(
                f"| {cat.category} | {cat.num_trades} | ${cat.total_pnl:,.2f} | {cat.win_rate:.1%} |"
            )

        md_lines.extend([
            "",
            "### Performance by Side",
            "",
            f"| Side | Trades | Total PnL | Win Rate |",
            f"|------|--------|-----------|----------|",
        ])

        for side in attribution_report.side_attribution:
            md_lines.append(
                f"| {side.side.upper()} | {side.num_trades} | ${side.total_pnl:,.2f} | {side.win_rate:.1%} |"
            )

        md_lines.extend([
            "",
            "### Performance by Holding Period",
            "",
            f"| Period | Trades | Total PnL | Avg Hours |",
            f"|--------|--------|-----------|-----------|",
        ])

        for hp in attribution_report.holding_period_attribution:
            if hp.num_trades > 0:
                md_lines.append(
                    f"| {hp.period_bucket} | {hp.num_trades} | ${hp.total_pnl:,.2f} | {hp.average_holding_hours:.1f} |"
                )

        md_lines.extend([
            "",
            "### Top 10 Markets by PnL",
            "",
            f"| Ticker | Trades | Total PnL | Win Rate |",
            f"|--------|--------|-----------|----------|",
        ])

        for market in attribution_report.market_attribution[:10]:
            md_lines.append(
                f"| {market.ticker} | {market.num_trades} | ${market.total_pnl:,.2f} | {market.win_rate:.1%} |"
            )

        md_lines.extend([
            "",
            "## Configuration",
            "",
            "```json",
            json.dumps({
                "initial_capital": self.config.initial_capital,
                "max_positions": self.config.max_positions,
                "commission_per_contract": self.config.commission_per_contract,
                "slippage_bps": self.config.slippage_bps,
                "max_position_size": self.config.max_position_size,
                "max_daily_loss": self.config.max_daily_loss,
            }, indent=2),
            "```",
            "",
            "---",
            "",
            f"*Report generated on {datetime.utcnow().strftime('%Y-%m-%d at %H:%M UTC')}*",
        ])

        with open(output_path, "w") as f:
            f.write("\n".join(md_lines))

        logger.info(f"Markdown report saved: {output_path}")

    async def generate_html_report(self, output_path: Path):
        """Generate HTML report with interactive visualizations."""
        logger.info(f"Generating HTML report: {output_path}")

        attribution_report = self.attribution.analyze()

        # Prepare equity curve data for chart
        equity_data = [
            {"x": dt.strftime('%Y-%m-%d'), "y": val}
            for dt, val in self.engine.get_equity_curve()
        ]

        # Prepare category data for chart
        category_data = [
            {"category": cat.category, "pnl": cat.total_pnl}
            for cat in attribution_report.category_attribution[:10]
        ]

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtest Report - {self.config.strategy_name}</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{ margin: 0 0 10px 0; color: #333; }}
        .subtitle {{ color: #666; margin: 5px 0; }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-label {{ color: #666; font-size: 14px; margin-bottom: 5px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #333; }}
        .metric-value.positive {{ color: #10b981; }}
        .metric-value.negative {{ color: #ef4444; }}
        .section {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e5e7eb;
        }}
        th {{
            background: #f9fafb;
            font-weight: 600;
            color: #374151;
        }}
        .chart {{ margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Backtest Report</h1>
        <div class="subtitle">Strategy: {self.config.strategy_name}</div>
        <div class="subtitle">Period: {self.config.start_date.date()} to {self.config.end_date.date()}</div>
        <div class="subtitle">Initial Capital: ${self.config.initial_capital:,.2f}</div>
    </div>

    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">Total Return</div>
            <div class="metric-value {'positive' if self.metrics.total_return > 0 else 'negative'}">
                {self.metrics.total_return:.2%}
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Sharpe Ratio</div>
            <div class="metric-value">{self.metrics.sharpe_ratio:.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Max Drawdown</div>
            <div class="metric-value negative">{self.metrics.max_drawdown:.2%}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Win Rate</div>
            <div class="metric-value">{self.metrics.win_rate:.1%}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Total Trades</div>
            <div class="metric-value">{self.metrics.total_trades}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Profit Factor</div>
            <div class="metric-value">{self.metrics.profit_factor:.2f}</div>
        </div>
    </div>

    <div class="section">
        <h2>Equity Curve</h2>
        <div id="equity-chart" class="chart"></div>
    </div>

    <div class="section">
        <h2>Performance by Category</h2>
        <div id="category-chart" class="chart"></div>
        <table>
            <thead>
                <tr>
                    <th>Category</th>
                    <th>Trades</th>
                    <th>Total PnL</th>
                    <th>Win Rate</th>
                    <th>Avg PnL</th>
                </tr>
            </thead>
            <tbody>
"""

        for cat in attribution_report.category_attribution[:10]:
            html_content += f"""
                <tr>
                    <td>{cat.category}</td>
                    <td>{cat.num_trades}</td>
                    <td>${cat.total_pnl:,.2f}</td>
                    <td>{cat.win_rate:.1%}</td>
                    <td>${cat.average_pnl:.2f}</td>
                </tr>
"""

        html_content += """
            </tbody>
        </table>
    </div>

    <div class="section">
        <h2>Top Markets</h2>
        <table>
            <thead>
                <tr>
                    <th>Ticker</th>
                    <th>Title</th>
                    <th>Trades</th>
                    <th>Total PnL</th>
                    <th>Win Rate</th>
                </tr>
            </thead>
            <tbody>
"""

        for market in attribution_report.market_attribution[:10]:
            html_content += f"""
                <tr>
                    <td>{market.ticker}</td>
                    <td>{market.title[:50]}...</td>
                    <td>{market.num_trades}</td>
                    <td>${market.total_pnl:,.2f}</td>
                    <td>{market.win_rate:.1%}</td>
                </tr>
"""

        html_content += f"""
            </tbody>
        </table>
    </div>

    <script>
        // Equity curve chart
        var equityData = [{json.dumps(equity_data)}];
        var equityTrace = {{
            x: equityData.map(d => d.x),
            y: equityData.map(d => d.y),
            type: 'scatter',
            mode: 'lines',
            name: 'Portfolio Value',
            line: {{ color: '#3b82f6', width: 2 }}
        }};
        var equityLayout = {{
            title: '',
            xaxis: {{ title: 'Date' }},
            yaxis: {{ title: 'Portfolio Value ($)' }},
            hovermode: 'x unified'
        }};
        Plotly.newPlot('equity-chart', [equityTrace], equityLayout);

        // Category chart
        var categoryData = {json.dumps(category_data)};
        var categoryTrace = {{
            x: categoryData.map(d => d.category),
            y: categoryData.map(d => d.pnl),
            type: 'bar',
            marker: {{
                color: categoryData.map(d => d.pnl > 0 ? '#10b981' : '#ef4444')
            }}
        }};
        var categoryLayout = {{
            title: '',
            xaxis: {{ title: 'Category' }},
            yaxis: {{ title: 'Total PnL ($)' }}
        }};
        Plotly.newPlot('category-chart', [categoryTrace], categoryLayout);
    </script>

    <div class="section" style="margin-top: 40px; padding: 15px; background: #f9fafb;">
        <small style="color: #6b7280;">
            Report generated on {datetime.utcnow().strftime('%Y-%m-%d at %H:%M UTC')}
        </small>
    </div>
</body>
</html>
"""

        with open(output_path, "w") as f:
            f.write(html_content)

        logger.info(f"HTML report saved: {output_path}")

    async def generate_csv_exports(self, output_dir: Path):
        """Generate CSV exports for Excel analysis."""
        import csv

        logger.info(f"Generating CSV exports: {output_dir}")

        # Export trades
        trades_path = output_dir / "trades.csv"
        with open(trades_path, "w", newline="") as f:
            if self.trades:
                writer = csv.DictWriter(f, fieldnames=self.engine.get_trades_summary()[0].keys())
                writer.writeheader()
                writer.writerows(self.engine.get_trades_summary())

        # Export equity curve
        equity_path = output_dir / "equity_curve.csv"
        with open(equity_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Date", "Portfolio Value"])
            for dt, val in self.engine.get_equity_curve():
                writer.writerow([dt.strftime('%Y-%m-%d'), val])

        logger.info(f"CSV exports saved: {trades_path}, {equity_path}")
