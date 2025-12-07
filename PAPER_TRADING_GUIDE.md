# Paper Trading Guide 📝

Complete guide to using paper trading mode to test strategies without risking real money.

## What is Paper Trading?

Paper trading simulates all trading operations without placing real orders or using real money. It's **essential** for:
- Testing strategies before risking capital
- Validating AI model performance
- Understanding bot behavior
- Building confidence in the system
- Identifying bugs and issues

## Features

### ✅ Realistic Simulation
- **Real market data** from Kalshi API
- **Simulated slippage** (5 basis points / 0.05%)
- **Simulated fees** (0.7% Kalshi fee)
- **Order fill simulation** with market prices
- **Position tracking** with real-time P&L

### 📊 Complete Analytics
- Realized and unrealized P&L
- Win rate and trade statistics
- Maximum drawdown tracking
- Fee tracking
- Performance metrics

### 💾 Persistent State
- Account state saved to `logs/paper_account.json`
- Positions saved to `logs/paper_positions.json`
- Trades logged to database
- Resume trading across sessions

---

## Setup

### 1. Enable Paper Trading (Default)

Paper trading is **enabled by default** for safety. Verify in `.env`:

```bash
PAPER_TRADING_MODE=true
PAPER_TRADING_BALANCE=10000.00
PAPER_SIMULATE_SLIPPAGE=true
PAPER_SIMULATE_FEES=true
```

### 2. Configure in settings.py

The system defaults to paper trading:

```python
# In src/config/settings.py
paper_trading_mode: bool = True  # SAFE DEFAULT
paper_trading_balance: float = 10000.0
paper_simulate_slippage: bool = True
paper_simulate_fees: bool = True
paper_slippage_bps: float = 5.0  # 0.05% slippage
```

### 3. Verify Mode

```bash
python -c "
from src.config.settings import settings
print(f'Paper Trading: {settings.trading.paper_trading_mode}')
print(f'Balance: \${settings.trading.paper_trading_balance:,.2f}')
"
```

Expected output:
```
Paper Trading: True
Balance: $10,000.00
```

---

## Usage

### Run the Bot

```bash
# Start bot in paper trading mode
python beast_mode_bot.py

# You'll see:
# "🚀 Beast Mode Bot initialized - Mode: PAPER TRADING"
```

The bot will:
1. Fetch real market data from Kalshi
2. Use AI to make trading decisions
3. Simulate order execution
4. Track positions and P&L
5. Log all trades to database

### View Performance

```bash
# View detailed performance report
python view_paper_trading_performance.py
```

Example output:
```
================================================================================
                     PAPER TRADING PERFORMANCE REPORT
================================================================================

💰 ACCOUNT SUMMARY
--------------------------------------------------------------------------------
  Starting Balance:        $10,000.00
  Current Balance:         $10,350.25
  Total P&L:              $350.25 (+3.50%)
  Realized P&L:           $275.00
  Unrealized P&L:         $75.25
  Total Fees Paid:        $42.50

📈 TRADING STATISTICS
--------------------------------------------------------------------------------
  Total Trades:           15
  Winning Trades:         9
  Losing Trades:          6
  Win Rate:               60.0%
  Average Win:            $45.50
  Average Loss:           -$22.00
  Profit Factor:          2.07

⚠️  RISK METRICS
--------------------------------------------------------------------------------
  Maximum Drawdown:       2.15%

📍 OPEN POSITIONS
--------------------------------------------------------------------------------
  Total Open Positions:   3
  Positions Value:        $450.00
  ...
```

### Monitor in Real-Time

The bot logs paper trades to console:
```
📝 PAPER TRADE: BUY 10 PREZ-2024 YES @ 0.65
📝 PAPER TRADE: SELL 10 PREZ-2024 YES @ 0.72 (P&L: +$70.00)
```

---

## Understanding Paper Trading Results

### Key Metrics

1. **Total P&L**: Overall profit/loss (realized + unrealized)
2. **Realized P&L**: Profit from closed positions
3. **Unrealized P&L**: Profit from open positions
4. **Win Rate**: Percentage of profitable trades
5. **Profit Factor**: (Total Wins) / (Total Losses) - aim for > 1.5
6. **Max Drawdown**: Largest peak-to-trough decline

### Performance Thresholds

| Metric | Poor | Acceptable | Good | Excellent |
|--------|------|------------|------|-----------|
| Win Rate | < 40% | 40-50% | 50-60% | > 60% |
| Profit Factor | < 1.0 | 1.0-1.5 | 1.5-2.5 | > 2.5 |
| Max Drawdown | > 20% | 10-20% | 5-10% | < 5% |
| Total P&L | < 0% | 0-5% | 5-15% | > 15% |

---

## Testing Strategy

### Phase 1: Initial Testing (1-2 weeks)
**Goal**: Validate basic functionality

```bash
# Run for 1-2 weeks
python beast_mode_bot.py

# Check daily
python view_paper_trading_performance.py
```

**Success Criteria**:
- Bot runs without crashes
- Trades execute correctly
- No obvious bugs
- Positive or neutral P&L

### Phase 2: Strategy Validation (2-4 weeks)
**Goal**: Prove consistent profitability

```bash
# Continue running
python beast_mode_bot.py

# Review weekly
python view_paper_trading_performance.py
```

**Success Criteria**:
- Win rate > 50%
- Profit factor > 1.5
- Max drawdown < 10%
- Positive P&L over 30 days

### Phase 3: Stress Testing (1-2 weeks)
**Goal**: Test in various market conditions

**Success Criteria**:
- Performance holds in volatile markets
- No catastrophic losses
- Risk limits respected
- Consistent with Phase 2

---

## Transitioning to Live Trading

### ⚠️ DO NOT GO LIVE UNTIL:

✅ **30+ days** of paper trading
✅ **Consistent profitability** (> 3% monthly)
✅ **Win rate > 50%**
✅ **Max drawdown < 10%**
✅ **You understand** how the bot makes decisions
✅ **You can afford to lose** the capital

### Transition Checklist

1. **Review All Performance**
   ```bash
   python view_paper_trading_performance.py
   python view_strategy_performance.py
   ```

2. **Analyze Edge Cases**
   - How does bot handle losing streaks?
   - Performance in different market conditions?
   - Any unexpected behaviors?

3. **Start SMALL**
   - Use $100-500 maximum initially
   - Never risk more than you can afford to lose
   - Monitor closely for first week

4. **Enable Live Trading**

   Update `.env`:
   ```bash
   PAPER_TRADING_MODE=false
   LIVE_TRADING_ENABLED=true
   ```

   Update `src/config/settings.py`:
   ```python
   paper_trading_mode: bool = False
   live_trading_enabled: bool = True
   ```

5. **Monitor Closely**
   - Check every few hours initially
   - Compare live vs paper performance
   - Be ready to shut down if needed

---

## Resetting Paper Account

```bash
# Reset to starting balance
python -c "
import asyncio
from src.utils.database import DatabaseManager
from src.clients.kalshi_client import KalshiClient
from src.utils.paper_trading import PaperTradingEngine

async def reset():
    db = DatabaseManager()
    await db.initialize()
    kalshi = KalshiClient()
    paper = PaperTradingEngine(db, kalshi, starting_balance=10000.0)
    paper.reset_account(10000.0)
    print('Paper account reset to \$10,000')
    await kalshi.close()

asyncio.run(reset())
"
```

---

## Common Issues

### Paper trades not executing
**Solution**: Check that `PAPER_TRADING_MODE=true` in `.env`

### Performance seems unrealistic
**Solution**: Ensure slippage and fees are enabled:
```bash
PAPER_SIMULATE_SLIPPAGE=true
PAPER_SIMULATE_FEES=true
```

### Lost paper trading data
**Solution**: Data is in `logs/paper_account.json` and `logs/paper_positions.json`. Backup regularly.

### Want to test different strategies
**Solution**: Reset paper account between tests:
```bash
# Backup current data
cp logs/paper_account.json logs/paper_account_backup.json
cp logs/paper_positions.json logs/paper_positions_backup.json

# Reset and test new strategy
python -c "from src.utils.paper_trading import PaperTradingEngine; ..."
```

---

## Best Practices

1. **Always start with paper trading**
2. **Run for minimum 30 days** before considering live trading
3. **Track performance daily** with `view_paper_trading_performance.py`
4. **Test different market conditions** (volatile, trending, ranging)
5. **Don't rush to live trading** - paper trading is free!
6. **Backup data regularly** - save `logs/` directory
7. **Document observations** - keep notes on bot behavior
8. **Test changes in paper mode** before deploying to live

---

## FAQ

**Q: How realistic is paper trading?**
A: Very realistic - uses real market data, simulates slippage (0.05%) and fees (0.7%).

**Q: Can I run paper and live trading simultaneously?**
A: Not recommended. Run one instance at a time to avoid confusion.

**Q: How long should I paper trade?**
A: Minimum 30 days, ideally 60-90 days to see various market conditions.

**Q: What's a good starting balance for paper trading?**
A: $10,000 is realistic and allows proper position sizing testing.

**Q: Should I trust paper trading results?**
A: Yes, but expect live trading to be slightly worse due to additional slippage, emotional factors, and execution challenges.

**Q: Can I speed up paper trading?**
A: No - it runs in real-time using actual market data. This is intentional to test realistic conditions.

---

## Summary

Paper trading is **essential** before risking real money. Take it seriously:
- Treat it like real trading
- Monitor performance closely
- Be patient and thorough
- Don't rush to live trading

**Remember**: The goal isn't to prove the bot works, it's to prove it works **consistently** over time in **various market conditions**.

Good luck! 🚀
