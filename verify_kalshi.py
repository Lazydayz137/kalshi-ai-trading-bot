import asyncio
import sys
import logging
from src.config.settings import settings
from src.clients.kalshi_client import KalshiClient

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_kalshi")

async def main():
    logger.info("Starting Kalshi Verification")
    
    # 1. Verify Settings
    logger.info(f"Kalshi API Key provided: {'Yes' if settings.api.kalshi_api_key else 'No'}")
    logger.info(f"Base URL: {settings.api.kalshi_base_url}")
    
    if not settings.api.kalshi_api_key:
        logger.error("❌ Kalshi API Key is missing from settings!")
        return

    # 2. Initialize Client
    try:
        # Note: KalshiClient expects the private key file to verify signatures
        client = KalshiClient()
        logger.info("✅ KalshiClient initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize KalshiClient: {e}")
        return

    # 3. Test Connection (Get Balance)
    try:
        logger.info("Testing connection by fetching balance...")
        balance = await client.get_balance()
        logger.info(f"✅ Connection successful! Balance response: {balance}")
        
    except Exception as e:
        logger.error(f"❌ Connection test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
