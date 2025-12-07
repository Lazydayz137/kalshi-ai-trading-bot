# Upgrade Guide - Conservative & Cost-Optimized Trading System

This guide covers the major improvements made to the Kalshi AI Trading Bot for enhanced safety, cost control, and multi-model AI support.

## 🔥 Major Changes Summary

### 1. **Conservative Risk Parameters** ✅
All trading parameters have been adjusted to industry-standard conservative levels:

| Parameter | Old (Aggressive) | New (Conservative) | Reason |
|-----------|-----------------|-------------------|---------|
| Kelly Fraction | 0.75 | 0.25 | Quarter-Kelly is industry standard |
| Min Confidence | 50% | 65% | Avoid coin-flip trades |
| Max Position Size | 5% | 3% | Reduce single position risk |
| Max Positions | 15 | 10 | Lower concurrent exposure |
| Daily AI Budget | $15 | $5 | Cost control |
| Min Volume | $200 | $500 | Better liquidity |
| Max Trades/Hour | 20 | 10 | Reduce overtrading |

### 2. **Multi-Model AI Support** 🤖
New OpenRouter integration provides access to 10+ AI models with cost optimization:

**Cost-Effective Models:**
- Google Gemini 2.0 Flash (FREE during preview!)
- DeepSeek Chat ($0.14 per 1M tokens)
- Qwen 2.5 72B ($0.35 per 1M tokens)
- Llama 3.1 70B ($0.52 per 1M tokens)

**Premium Models:**
- GPT-4o ($2.50 per 1M tokens)
- Claude 3.5 Sonnet ($3.00 per 1M tokens)
- GPT-4 Turbo ($10.00 per 1M tokens)

### 3. **Circuit Breaker Pattern** 🔌
Automatic failover and resilience:
- Fails after 5 consecutive errors
- 5-minute timeout before retry
- Automatic fallback to alternative providers
- Provider health monitoring

### 4. **Position Reconciliation** 🔄
Ensures database matches Kalshi API state:
- Detects missing positions
- Fixes quantity discrepancies
- Auto-closes positions no longer in Kalshi
- Runs automatically every trading cycle

### 5. **Actual Cost Tracking** 💰
No more estimates - track exact costs:
- Per-request cost calculation
- Per-model cost breakdown
- Daily spending limits with hard stops
- Cost reporting and analytics

---

## 📦 Installation & Upgrade Steps

### Step 1: Update Dependencies

```bash
# Backup your current environment
pip freeze > old_requirements.txt

# Install updated dependencies
pip install -r requirements.txt

# Verify critical packages
pip list | grep -E "aiohttp|cryptography|httpx|openrouter|pybreaker|tenacity"
```

**Expected versions:**
- aiohttp: 3.11.10+ (fixes CVEs)
- cryptography: 44.0.0+ (security updates)
- httpx: 0.28.1+
- pybreaker: 1.2.0 (NEW - circuit breaker)
- tenacity: 9.0.0 (NEW - retry logic)
- openrouter: latest (NEW - multi-model access)

### Step 2: Update Environment Configuration

```bash
# Copy your existing .env
cp .env .env.backup

# Review the new env.template
cat env.template

# Add OpenRouter API key (REQUIRED if not using xAI)
echo "OPENROUTER_API_KEY=your_key_here" >> .env

# Set default provider
echo "DEFAULT_AI_PROVIDER=openrouter" >> .env

# Set preferred model (or leave empty for auto-selection)
echo "PREFERRED_MODEL=deepseek/deepseek-chat" >> .env

# Set conservative risk level
echo "RISK_LEVEL=conservative" >> .env

# Set daily cost limit
echo "DAILY_AI_COST_LIMIT=5.00" >> .env
```

### Step 3: Get OpenRouter API Key

1. Go to https://openrouter.ai/
2. Sign up / Log in
3. Go to https://openrouter.ai/keys
4. Create a new API key
5. Add credits ($5-10 recommended to start)
6. Add key to `.env` file

### Step 4: Test the New System

```bash
# Test OpenRouter connection
python -c "
import asyncio
from src.clients.openrouter_client import OpenRouterClient

async def test():
    client = OpenRouterClient()
    messages = [{'role': 'user', 'content': 'Say hello'}]
    result, cost, usage = await client.chat_completion(messages, model='deepseek/deepseek-chat')
    print(f'Response: {result}')
    print(f'Cost: ${cost:.4f}')
    await client.close()

asyncio.run(test())
"

# Test unified AI client
python -c "
import asyncio
from src.clients.unified_ai_client import UnifiedAIClient

async def test():
    client = UnifiedAIClient()
    result = await client.get_completion('Test prompt')
    print(f'Response: {result[:100]}...')
    summary = client.get_cost_summary()
    print(f'Cost Summary: {summary}')
    await client.close()

asyncio.run(test())
"
```

### Step 5: Run Position Reconciliation

```bash
# One-time reconciliation to ensure DB matches Kalshi
python -c "
import asyncio
from src.utils.database import DatabaseManager
from src.clients.kalshi_client import KalshiClient
from src.utils.position_reconciliation import run_reconciliation_check

async def reconcile():
    db = DatabaseManager()
    await db.initialize()
    kalshi = KalshiClient()

    result = await run_reconciliation_check(db, kalshi, auto_fix=True)

    print(f'Reconciliation Result:')
    print(f'  Kalshi Positions: {result.kalshi_positions}')
    print(f'  DB Positions: {result.db_positions}')
    print(f'  Matched: {result.matched}')
    print(f'  Discrepancies: {len(result.discrepancies)}')
    print(f'  Success: {result.success}')

    await kalshi.close()

asyncio.run(reconcile())
"
```

### Step 6: Review Configuration Changes

Check `src/config/settings.py` to see all conservative parameter changes:

```python
# Key conservative settings
max_position_size_pct: 3.0          # Down from 5.0
min_confidence_to_trade: 0.65       # Up from 0.50
kelly_fraction: 0.25                # Down from 0.75
max_positions: 10                   # Down from 15
daily_ai_budget: 5.0               # Down from 15.0
min_volume: 500.0                  # Up from 200.0
```

---

## 🚀 Using the New Multi-Model System

### Option 1: Automatic Model Selection (Recommended)

The system will automatically select the best model based on cost and task:

```python
from src.clients.unified_ai_client import UnifiedAIClient

client = UnifiedAIClient()

# Will use cheapest available model (likely DeepSeek or Gemini)
decision = await client.get_trading_decision(
    market_data=market_data,
    portfolio_data=portfolio_data,
    news_summary=news_summary
)
```

### Option 2: Force Premium Models

For high-confidence trades, use premium models:

```python
# Will use GPT-4o or Claude 3.5 Sonnet
decision = await client.get_trading_decision(
    market_data=market_data,
    portfolio_data=portfolio_data,
    news_summary=news_summary,
    prefer_premium=True  # 👈 Use premium model
)
```

### Option 3: Specific Model Selection

Use a specific model via OpenRouter:

```python
from src.clients.openrouter_client import OpenRouterClient

client = OpenRouterClient()

messages = [{"role": "user", "content": "Analyze this market..."}]

# Use specific model
response, cost, usage = await client.chat_completion(
    messages=messages,
    model="openai/gpt-4o-mini",  # Specific model
    temperature=0.1
)
```

---

## 💰 Cost Optimization Strategies

### 1. Use Free/Cheap Models for Initial Screening

```python
# Use free Gemini for initial market screening
model = "google/gemini-2.0-flash-exp"  # FREE

# Use DeepSeek for analysis ($0.14/1M tokens)
model = "deepseek/deepseek-chat"
```

### 2. Reserve Premium Models for High-Value Trades

```python
# Only use GPT-4o when confidence > 80%
if estimated_confidence > 0.80:
    decision = await client.get_trading_decision(
        ...,
        prefer_premium=True
    )
else:
    decision = await client.get_trading_decision(
        ...,
        prefer_premium=False  # Use cheap model
    )
```

### 3. Monitor Daily Costs

```python
# Check cost summary
summary = client.get_cost_summary()
print(f"Total Cost Today: ${summary['total_cost']:.2f}")
print(f"Cost by Model: {summary.get('openrouter_model_costs', {})}")

# System will auto-stop at daily limit
```

### 4. Adjust Cost Limits

Edit `.env`:
```bash
# Conservative (recommended for testing)
DAILY_AI_COST_LIMIT=5.00

# Moderate
DAILY_AI_COST_LIMIT=10.00

# Aggressive (only if proven profitable)
DAILY_AI_COST_LIMIT=25.00
```

---

## 🔧 Troubleshooting

### Issue: OpenRouter API Errors

**Solution:**
```bash
# Check API key
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('OPENROUTER_API_KEY'))"

# Verify credits at https://openrouter.ai/credits

# Test with simple request
python -c "
import httpx
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENROUTER_API_KEY')

response = httpx.post(
    'https://openrouter.ai/api/v1/chat/completions',
    headers={'Authorization': f'Bearer {api_key}'},
    json={
        'model': 'deepseek/deepseek-chat',
        'messages': [{'role': 'user', 'content': 'Test'}]
    }
)
print(response.status_code)
print(response.json())
"
```

### Issue: Circuit Breaker Opens

**Solution:**
```bash
# Check circuit breaker status in logs
grep "Circuit breaker" logs/trading_system.log

# Manually reset if needed (will auto-reset after timeout)
# Or adjust threshold in .env:
echo "CIRCUIT_BREAKER_FAIL_MAX=10" >> .env
```

### Issue: Position Reconciliation Failures

**Solution:**
```bash
# Run manual reconciliation with verbose logging
python -c "
import asyncio
import logging
from src.utils.database import DatabaseManager
from src.clients.kalshi_client import KalshiClient
from src.utils.position_reconciliation import run_reconciliation_check

logging.basicConfig(level=logging.DEBUG)

async def reconcile():
    db = DatabaseManager()
    await db.initialize()
    kalshi = KalshiClient()

    # Don't auto-fix, just report
    result = await run_reconciliation_check(db, kalshi, auto_fix=False)

    for disc in result.discrepancies:
        print(f'Discrepancy: {disc}')

    await kalshi.close()

asyncio.run(reconcile())
"
```

---

## 📊 Model Performance Comparison

Based on OpenRouter benchmarks:

| Model | Cost (1M tokens) | Speed | Quality | Best For |
|-------|-----------------|-------|---------|----------|
| Gemini 2.0 Flash | $0.00 | ⚡⚡⚡ | ⭐⭐⭐ | Testing, high-volume |
| DeepSeek Chat | $0.14 | ⚡⚡ | ⭐⭐⭐⭐ | Production (cheap) |
| GPT-4o Mini | $0.15 | ⚡⚡⚡ | ⭐⭐⭐⭐ | Fast decisions |
| Qwen 2.5 72B | $0.35 | ⚡⚡ | ⭐⭐⭐⭐ | Complex analysis |
| Llama 3.1 70B | $0.52 | ⚡⚡ | ⭐⭐⭐⭐ | Balanced |
| GPT-4o | $2.50 | ⚡⚡ | ⭐⭐⭐⭐⭐ | High-stakes trades |
| Claude 3.5 Sonnet | $3.00 | ⚡ | ⭐⭐⭐⭐⭐ | Best quality |

**Recommendation for starting:**
1. Use DeepSeek Chat as default ($0.14/1M tokens)
2. Fall back to Gemini 2.0 Flash if budget limited (FREE)
3. Reserve GPT-4o Mini for quick high-confidence decisions

---

## 🎯 Next Steps

1. **Test in Paper Trading Mode**
   ```bash
   # Ensure paper trading is enabled
   echo "LIVE_TRADING_ENABLED=false" >> .env

   # Run bot for 7-14 days
   python beast_mode_bot.py
   ```

2. **Monitor Performance**
   ```bash
   # Check daily AI costs
   tail -f logs/daily_openrouter_usage.pkl

   # Review trading decisions
   python view_strategy_performance.py
   ```

3. **Optimize Model Selection**
   - Track which models perform best
   - Adjust `PREFERRED_MODEL` in `.env`
   - Consider using multiple models for consensus

4. **Gradually Increase Risk** (Only After Proven Success)
   - Start with conservative settings
   - Monitor for 30+ days
   - Gradually increase position sizes if profitable
   - Never exceed comfortable risk tolerance

---

## ⚠️ Important Reminders

1. **Start Small**: Use conservative settings and small capital
2. **Monitor Daily**: Check logs and performance regularly
3. **Respect Limits**: Never override daily cost limits
4. **Paper Trade First**: Test for at least 14 days before live trading
5. **Understand the Code**: Know what the bot is doing
6. **You Accept Risk**: This is experimental software

---

## 📞 Support

- **Issues**: https://github.com/anthropics/claude-code/issues
- **OpenRouter Docs**: https://openrouter.ai/docs
- **Kalshi API Docs**: https://trading-api.readme.io/

---

**Last Updated**: December 2024
**Version**: 2.0.0-conservative
