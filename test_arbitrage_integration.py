import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from src.strategies.arbitrage_engine import ArbitrageEngine, PolymarketEvent
from src.strategies.portfolio_optimization import _get_fast_ai_prediction
from src.utils.database import Market

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_arbitrage")

async def test_arbitrage_injection():
    print("\n--- Testing Arbitrage Data Injection ---\n")
    
    # 1. Setup Arbitrage Engine with Mock Data
    engine = ArbitrageEngine()
    
    # Create a fake Polymarket event
    fake_poly_event = PolymarketEvent(
        title="Kamala Harris wins 2024 Presidential Election",
        yes_price=0.45,
        no_price=0.55,
        volume=1000000.0,
        url="https://polymarket.com/event/test"
    )
    
    # Inject directly into cache (bypass API fetching)
    engine.cached_events = [fake_poly_event]
    print(f"✅ Injected Polymarket Event: '{fake_poly_event.title}' (Price: {fake_poly_event.yes_price})")
    
    # 2. Setup Mock XAI Client
    mock_xai = AsyncMock()
    mock_xai.get_completion.return_value = '{"probability": 0.6, "confidence": 0.8, "reasoning": "Test"}'
    
    # 3. Create a Kalshi Market that SHOULD match
    kalshi_market = Market(
        market_id="kalshi_kamala_win",
        title="Kamala Harris to be president", 
        category="Politics",
        yes_price=30,
        no_price=70,
        volume=1000,
        expiration_ts=1704067200, # Dummy ts
        status="active",
        last_updated="2024-01-01"
    )
    market_price = 0.30  # Kalshi price 30¢ vs Polymarket 45¢ -> Big Arbitrage
    
    print(f"✅ Created Kalshi Market: '{kalshi_market.title}' (Price: {market_price})")
    
    # 4. Call the prediction function
    print("\n⏳ Calling _get_fast_ai_prediction...")
    
    await _get_fast_ai_prediction(
        market=kalshi_market,
        xai_client=mock_xai,
        market_prob=market_price,
        odds_context="",
        db_manager=None,
        arbitrage_engine=engine  # PASS THE ENGINE!
    )
    
    # 5. Inspect the PROMPT passed to XAI
    # get_completion was called with (prompt, ...)
    call_args = mock_xai.get_completion.call_args
    if not call_args:
        print("❌ XAI Client was NOT called!")
        return

    args, kwargs = call_args
    prompt_sent = kwargs.get('prompt')
    if not prompt_sent and args:
         prompt_sent = args[0]
    
    print("\n--- Prompt Inspection ---")
    if "POLYMARKET ARBITRAGE DATA" in prompt_sent:
        print("✅ SUCCESS! 'POLYMARKET ARBITRAGE DATA' found in prompt.")
        print("-" * 50)
        # Extract the arbitrage section for verification
        lines = prompt_sent.split('\n')
        for line in lines:
            if "Polymarket" in line or "Matched" in line:
                print(line.strip())
        print("-" * 50)
    else:
        print("❌ FAILURE! Arbitrage data NOT found in prompt.")
        print("Full Prompt:")
        print(prompt_sent)

if __name__ == "__main__":
    asyncio.run(test_arbitrage_injection())
