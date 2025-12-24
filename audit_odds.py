
import asyncio
import os
import time
from datetime import datetime
from collections import Counter
from dotenv import load_dotenv
from src.clients.kalshi_client import KalshiClient
from src.clients.odds_client import OddsClient

# Mock class to simulate the Market structure used in portfolio_optimization
class MockMarket:
    def __init__(self, title, category, market_id):
        self.title = title
        self.category = category
        self.market_id = market_id

async def audit_matching():
    load_dotenv()
    
    print("🚀 Starting Odds API Audit...")
    
    # 1. Initialize Clients
    kalshi = KalshiClient()
    odds_client = OddsClient()
    
    if not odds_client.api_key:
        print("❌ PRO TIP: No Odds API Key found in .env!")
        return

    # 2. Fetch ALL Active Markets (Kalshi)
    print("\n📊 Fetching ALL Kalshi Markets (Paginating)...")
    all_markets = []
    cursor = None
    page_count = 0
    
    while True:
        # NOTE: Using 'open' status as confirmed correct via previous tests
        markets_resp = await kalshi.get_markets(status="open", limit=100, cursor=cursor)
        batch = markets_resp.get('markets', [])
        all_markets.extend(batch)
        page_count += 1
        print(f"   Page {page_count}: Fetched {len(batch)} markets (Total: {len(all_markets)})")
        
        cursor = markets_resp.get("cursor")
        if not cursor:
            break
            
    # Sort by volume
    all_markets_sorted = sorted(all_markets, key=lambda x: x.get('volume', 0), reverse=True)
    
    print(f"\n   Found {len(all_markets)} total markets.")
    
    # 3. Analyze Expiry
    now_ts = time.time()
    thirty_days = 30 * 24 * 60 * 60
    
    expiry_counts = {"<30d": 0, ">30d": 0}
    short_term_markets = []
    
    print(f"\n⏳ Expiry Analysis (Bot default limit: 30 days):")
    
    for m in all_markets:
        exp_time = m.get('expiration_time')
        if exp_time:
             try:
                 # expected ISO format e.g. 2024-01-01T00:00:00Z
                 ts = datetime.fromisoformat(exp_time.replace("Z", "+00:00")).timestamp()
                 if ts - now_ts < thirty_days:
                     expiry_counts["<30d"] += 1
                     short_term_markets.append(m)
                 else:
                     expiry_counts[">30d"] += 1
             except:
                 pass

    print(f"   - Short Term (<30d): {expiry_counts['<30d']} (Eligible for Bot)")
    print(f"   - Long Term (>30d):  {expiry_counts['>30d']} (Filtered out)")
    
    # Analyze Categories of Short Term Markets
    short_term_cats = [m.get('category', 'UNKNOWN') for m in short_term_markets]
    cat_counts = Counter(short_term_cats)
    
    print(f"\n📊 Short Term Market Categories (n={len(short_term_markets)}):")
    for cat, count in cat_counts.most_common():
        print(f"   - '{cat}': {count}")
    
    # 4. Fetch Active Sports (Odds API)
    print("\n🌍 Fetching Active Sports from The-Odds-API...")
    active_sports = await odds_client.get_active_sports()
    print(f"   Found {len(active_sports)} active sports keys.")

    # 5. Simulate Matching Logic on SHORT TERM markets
    print("\n🔍 MATCHING SIMULATION (Top 50 Short Term Markets by Volume):")
    print("-" * 60)
    
    matched_count = 0
    markets_to_audit = sorted(short_term_markets, key=lambda x: x.get('volume', 0), reverse=True)[:50]
    
    for m in markets_to_audit:
        market = MockMarket(m['title'], m['category'], m['ticker'])
        expiry = m.get('expiration_time', 'UNKNOWN')
        
        # --- LOGIC FROM portfolio_optimization.py ---
        matching_sports = []
        market_title_lower = market.title.lower()
        # Handle None/Empty category
        market_cat_lower = str(market.category).lower() if market.category else ""
        
        for sport in active_sports:
            sport_key = sport['key']
            sport_title = sport['title'].lower()
            sport_group = sport.get('group', '').lower()
            
            # Direct Match
            if sport_title in market_title_lower or (market_cat_lower and sport_title in market_cat_lower):
                matching_sports.append(sport_key)
            # Group Match
            if sport_group and (sport_group in market_title_lower or (market_cat_lower and sport_group in market_cat_lower)):
                matching_sports.append(sport_key)
            # Specific Aliases
            if "nfl" in market_title_lower and "americanfootball_nfl" == sport_key:
                matching_sports.append(sport_key)
            elif "nba" in market_title_lower and "basketball_nba" == sport_key:
                matching_sports.append(sport_key)
            elif "nhl" in market_title_lower and "icehockey_nhl" == sport_key:
                matching_sports.append(sport_key)
            elif "golf" in market_cat_lower and "golf" in sport_key: 
                matching_sports.append(sport_key)

        unique_matches = list(set(matching_sports))
        
        if unique_matches:
            matched_count += 1
            print(f"✅ MATCHED: [{market.category}] {market.title[:60]}... -> {unique_matches}")
        
    print("-" * 60)
    print(f"Summary: Matched {matched_count}/{len(markets_to_audit)} short-term markets to a sport key.")

    # Properly close clients
    await kalshi.close() 
    await odds_client.close()

if __name__ == "__main__":
    asyncio.run(audit_matching())
