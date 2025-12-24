import asyncio
import os
import sys

# Add src to path
sys.path.append(os.getcwd())

from src.jobs.decide import make_decision_for_market
from src.utils.database import DatabaseManager
from src.clients.xai_client import XAIClient
from src.clients.kalshi_client import KalshiClient
from src.config.settings import settings
from tests.test_helpers import find_suitable_test_market

# Use a temp db for testing
TEST_DB = "test_manual_decide.db"

async def main():
    print("🚀 Starting Manual Decision Test")
    
    # Initialize DB
    db_manager = DatabaseManager(db_path=TEST_DB)
    await db_manager.initialize()
    print("✅ Database initialized")
    
    kalshi_client = KalshiClient()
    
    # Choose AI client based on available keys (logic from bot)
    if settings.api.xai_api_key:
        xai_client = XAIClient(db_manager=db_manager)
        print("✅ Using XAIClient")
    elif settings.api.openai_api_key:
        print("✅ Using OpenAIClient (OpenRouter)")
        from src.clients.openai_client import OpenAIClient
        xai_client = OpenAIClient(db_manager=db_manager)
    else:
        print("❌ No valid AI API key found")
        return

    try:
        # Get a suitable test market
        print("🔍 Searching for suitable test market...")
        # Relax constraints if needed (override in memory if we could, but here we call the helper)
        # We can also manually fetch if helper fails
        test_market = await find_suitable_test_market()
        
        if not test_market:
            print("⚠️ No suitable markets found with helper criteria (Volume > 500).")
            print("Trying to find ANY active market...")
            markets = await kalshi_client.get_markets(limit=20)
            active_markets = [m for m in markets.get('markets', []) if m.get('status') == 'active']
            
            if active_markets:
                m_data = active_markets[0]
                from datetime import datetime
                from src.utils.database import Market
                test_market = Market(
                    market_id=m_data['ticker'],
                    title=m_data['title'],
                    yes_price=(m_data.get('yes_bid', 0) + m_data.get('yes_ask', 0)) / 200,
                    no_price=(m_data.get('no_bid', 0) + m_data.get('no_ask', 0)) / 200,
                    volume=m_data.get('volume', 0),
                    expiration_ts=m_data.get('close_ts', 0),
                    category=m_data.get('category', 'test'),
                    status='active',
                    last_updated=datetime.now(),
                    has_position=False
                )
                print(f"⚠️ Found fallback market: {test_market.title} (Vol: {test_market.volume})")
            else:
                print("❌ No active markets found at all!")
                return
        else:
            print(f"✅ Found suitable test market: {test_market.title} ({test_market.market_id})")
        
        # Store market in database
        await db_manager.upsert_markets([test_market])
        
        # Test the decision making process
        print(f"🤖 Requesting decision for market: {test_market.title}")
        position = await make_decision_for_market(
            test_market, db_manager, xai_client, kalshi_client
        )
        
        if position:
            print(f"🎉 DECISION MADE: {position.side} {position.quantity} @ ${position.entry_price}")
        else:
            print("🤔 AI decided NOT to trade (this is a valid result)")
            
    except Exception as e:
        print(f"❌ Error during manual test: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await kalshi_client.close()
        # await xai_client.close() # OpenAIClient might not have close method or it's different
        if hasattr(xai_client, 'close'):
             await xai_client.close()
             
        # Clean up test database
        if os.path.exists(TEST_DB):
            try:
                os.remove(TEST_DB)
                print("🧹 Test DB cleaned up")
            except:
                pass

if __name__ == "__main__":
    asyncio.run(main())
