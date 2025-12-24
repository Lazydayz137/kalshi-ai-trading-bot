
import asyncio
from dotenv import load_dotenv
load_dotenv()

from src.clients.odds_client import OddsClient

async def check_odds():
    client = OddsClient()
    print(f"Key loaded: {client.api_key[:5]}...")
    sports = await client.get_active_sports()
    print(f"Found {len(sports)} active sports")
    for s in sports[:3]:
        print(f"- {s['title']} ({s['key']})")
    
    if sports:
        # Try finding NBA
        nba = next((s for s in sports if s['key'] == 'basketball_nba'), None)
        key = nba['key'] if nba else sports[0]['key']
        
        print(f"Fetching odds for {key}...")
        odds = await client.get_odds(key)
        print(f"Found {len(odds)} events for {key}")
        if odds:
            print(f"Sample Event: {odds[0].home_team} vs {odds[0].away_team}")
            print(f"Bookmakers: {len(odds[0].bookmakers)}")
            for b in odds[0].bookmakers[:2]:
                print(f"  - {b['title']}")

    await client.close()

if __name__ == "__main__":
    asyncio.run(check_odds())
