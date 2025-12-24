
import asyncio
from dotenv import load_dotenv
from src.clients.kalshi_client import KalshiClient

async def test_series():
    load_dotenv()
    kalshi = KalshiClient()
    
    tickers = ["NBA", "NFL", "NHL", "GOLF", "EPL", "SOCCER"]
    
    print("🚀 Testing Series Ticker Fetching...")
    
    for t in tickers:
        print(f"\n🔍 Searching for Series: {t}")
        # Try series_ticker
        resp = await kalshi.get_markets(series_ticker=t, status="open", limit=50)
        markets = resp.get('markets', [])
        
        if markets:
            print(f"   ✅ Found {len(markets)} markets for {t}!")
            print(f"      Example: {markets[0]['title']} (Exp: {markets[0]['expiration_time']})")
        else:
            print(f"   ❌ No markets found for {t}")
            
    await kalshi.close()

if __name__ == "__main__":
    asyncio.run(test_series())
