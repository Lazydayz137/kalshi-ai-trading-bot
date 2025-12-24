import aiohttp
import asyncio

async def test():
    base = "https://gamma-api.polymarket.com/markets"
    
    # List of test URLs
    tests = [
        f"{base}?limit=5&closed=false&sort=volume",
    ]
    
    async with aiohttp.ClientSession() as session:
        for url in tests:
            print(f"Testing: {url} ... ", end="")
            async with session.get(url) as resp:
                print(f"Status: {resp.status}")
                if resp.status == 200:
                    try:
                        data = await resp.json()
                        if len(data) > 0:
                            print(f"  > First ID: {data[0].get('id')}")
                            print(f"  > Prices Type: {type(data[0].get('outcomePrices'))}")
                            print(f"  > Prices Value: {data[0].get('outcomePrices')}")
                    except Exception as e:
                        print(e)

if __name__ == "__main__":
    asyncio.run(test())
