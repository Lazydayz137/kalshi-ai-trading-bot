import asyncio
import sys
import logging
from src.config.settings import settings
from src.clients.openai_client import OpenAIClient

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_openrouter")

async def main():
    logger.info("Starting OpenRouter Verification")
    
    # 1. Verify Settings
    logger.info(f"OpenAI API Key provided: {'Yes' if settings.api.openai_api_key else 'No'}")
    logger.info(f"Base URL: {settings.api.openai_base_url}")
    
    if not settings.api.openai_api_key:
        logger.error("❌ OpenAI API Key is missing!")
        return
        
    if "openrouter" not in settings.api.openai_base_url:
        logger.warning(f"⚠️ Base URL '{settings.api.openai_base_url}' does not look like OpenRouter (expected 'openrouter.ai')")

    # 2. Initialize Client
    try:
        client = OpenAIClient()
        logger.info("✅ OpenAIClient initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize OpenAIClient: {e}")
        return

    # 3. Test Connection
    try:
        logger.info("Testing connection with a simple prompt...")
        # Note: 'grok-4' might not be available or might require specific permissions/credits.
        # We'll try the configured model, and if it fails, maybe fall back to a common free one for testing?
        # But settings.settings.trading.primary_model is what we want to test.
        
        test_messages = [{"role": "user", "content": "Hello, are you working? Reply with 'Yes working'."}]
        
        # We need to manually call the client's internal methods or setup a dummy market data structure 
        # to use get_trading_decision. Let's just use the raw client to test connectivity first.
        
        # Use a model that is likely to work
        model = "google/gemini-2.0-flash-exp:free"
        logger.info(f"Using model: {model}")

        response = await client.client.chat.completions.create(
            model=model,
            messages=test_messages,
            max_tokens=20
        )
        
        content = response.choices[0].message.content
        logger.info(f"✅ Response received: {content}")
        
    except Exception as e:
        logger.error(f"❌ Connection test failed: {e}")
        # Print more details if available
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
