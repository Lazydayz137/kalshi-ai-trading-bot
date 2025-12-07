# Quick Start: OpenRouter Multi-Model Setup

## 1. Get OpenRouter API Key (5 minutes)

1. Go to https://openrouter.ai/
2. Sign up with Google/GitHub
3. Go to https://openrouter.ai/keys
4. Click "Create Key"
5. Copy the key
6. Add $5-10 in credits at https://openrouter.ai/credits

## 2. Configure Environment

```bash
# Add to your .env file
echo "OPENROUTER_API_KEY=sk-or-v1-YOUR-KEY-HERE" >> .env
echo "DEFAULT_AI_PROVIDER=openrouter" >> .env
echo "PREFERRED_MODEL=deepseek/deepseek-chat" >> .env
```

## 3. Install Dependencies

```bash
pip install --upgrade -r requirements.txt
```

## 4. Test It Works

```bash
python -c "
import asyncio
from src.clients.openrouter_client import OpenRouterClient

async def test():
    client = OpenRouterClient()
    messages = [{'role': 'user', 'content': 'Say hello in 5 words'}]
    result, cost, usage = await client.chat_completion(messages)
    print(f'✅ Success! Response: {result}')
    print(f'💰 Cost: \${cost:.4f}')
    await client.close()

asyncio.run(test())
"
```

## 5. Run the Bot

```bash
# Paper trading (recommended)
python beast_mode_bot.py

# Live trading (after testing!)
python beast_mode_bot.py --live
```

## Cost-Effective Model Recommendations

**Free (Testing):**
- `google/gemini-2.0-flash-exp` - Free during preview

**Cheap (Production):**
- `deepseek/deepseek-chat` - $0.14/1M tokens
- `openai/gpt-4o-mini` - $0.15/1M tokens

**Premium (High-Stakes):**
- `openai/gpt-4o` - $2.50/1M tokens
- `anthropic/claude-3.5-sonnet` - $3.00/1M tokens

## Daily Cost Estimates

With `deepseek/deepseek-chat` at ~1000 tokens per decision:

- 10 decisions/day = ~$0.001 = **$0.03/month**
- 50 decisions/day = ~$0.007 = **$0.21/month**
- 100 decisions/day = ~$0.014 = **$0.42/month**

The system is configured for max $5/day limit as safety net.

## Monitoring Costs

```bash
# Check today's costs
python -c "
import pickle
with open('logs/daily_openrouter_usage.pkl', 'rb') as f:
    tracker = pickle.load(f)
    print(f'Total Cost Today: \${tracker.total_cost:.2f}')
    print(f'Requests: {tracker.request_count}')
    print(f'By Model: {tracker.model_costs}')
"
```

Done! You're now using multi-model AI with cost optimization.
