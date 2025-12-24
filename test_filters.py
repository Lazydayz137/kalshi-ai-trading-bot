import asyncio
import aiosqlite
from src.utils.database import Market
from src.strategies.filters import MarketFilter

async def test_filters():
    print("Connecting to DB...")
    async with aiosqlite.connect("trading_system.db") as db:
        db.row_factory = aiosqlite.Row
        
        # Simulate logic from UnifiedTradingSystem -> "get_eligible_markets"
        # volume > 200, active, not expired
        cursor = await db.execute("""
            SELECT * FROM markets 
            WHERE volume > 200 AND status = 'active'
            ORDER BY volume DESC 
            LIMIT 50
        """)
        rows = await cursor.fetchall()
        
        markets = []
        for row in rows:
            d = dict(row)
            # Fix types if needed
            markets.append(Market(**d))
            
    print(f"Fetched {len(markets)} top markets.")
    
    passed_junk = 0
    passed_spread = 0
    
    for m in markets:
        print(f"Checking: {m.title} ({m.volume})")
        
        # 1. Junk Filter
        if MarketFilter.is_junk_market(m):
            print(f"  -> REJECTED BY JUNK FILTER")
            continue
        passed_junk += 1
        
        # 2. Spread Filter (Simulating with DB prices)
        # In actual bot, this uses fresh API data. Here we use DB yes_price/no_price as proxy.
        # DB yes_price is 0.0-1.0. Bot filter uses cents (1-99)? 
        # Wait, if yes_price is 0.74 (74 cents).
        # We need to assume the API returns what?
        # Let's assume DB price is accurate.
        
        # If DB stores 0.74, and API returns 74? Or 0.74?
        # Kalshi API v2 usually returns Cents (1-99) for 'bid'/'ask'.
        # BUT 'last_price' might be cents too.
        # DB 'yes_price' seems to be float 0.74.
        
        # Let's simulate calculation:
        # If API returns cents: 74.
        # Spread > 15.
        
        # We can't perfectly simulate API spread without API, but we can check if filter logic is sound.
        pass
        
    print(f"\nSummary:")
    print(f"Total: {len(markets)}")
    print(f"Passed Junk Filter: {passed_junk}")

if __name__ == "__main__":
    asyncio.run(test_filters())
