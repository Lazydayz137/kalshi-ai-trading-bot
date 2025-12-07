"""
OpenRouter API client for multi-model AI access with cost optimization.
Provides access to various AI models through OpenRouter's unified API.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import pickle
import os

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.settings import settings
from src.utils.logging_setup import TradingLoggerMixin


@dataclass
class ModelPricing:
    """Pricing information for an AI model."""
    model_id: str
    model_name: str
    prompt_price_per_million: float  # Cost per million prompt tokens
    completion_price_per_million: float  # Cost per million completion tokens
    context_window: int  # Maximum context length
    supports_streaming: bool = True
    supports_function_calling: bool = False


# OpenRouter model pricing (as of December 2024)
OPENROUTER_MODEL_PRICING = {
    # GPT Models
    "openai/gpt-4o": ModelPricing(
        model_id="openai/gpt-4o",
        model_name="GPT-4o",
        prompt_price_per_million=2.50,
        completion_price_per_million=10.00,
        context_window=128000,
        supports_function_calling=True
    ),
    "openai/gpt-4o-mini": ModelPricing(
        model_id="openai/gpt-4o-mini",
        model_name="GPT-4o Mini",
        prompt_price_per_million=0.15,
        completion_price_per_million=0.60,
        context_window=128000,
        supports_function_calling=True
    ),
    "openai/gpt-4-turbo": ModelPricing(
        model_id="openai/gpt-4-turbo",
        model_name="GPT-4 Turbo",
        prompt_price_per_million=10.00,
        completion_price_per_million=30.00,
        context_window=128000,
        supports_function_calling=True
    ),
    # Claude Models
    "anthropic/claude-3.5-sonnet": ModelPricing(
        model_id="anthropic/claude-3.5-sonnet",
        model_name="Claude 3.5 Sonnet",
        prompt_price_per_million=3.00,
        completion_price_per_million=15.00,
        context_window=200000,
        supports_function_calling=True
    ),
    "anthropic/claude-3-haiku": ModelPricing(
        model_id="anthropic/claude-3-haiku",
        model_name="Claude 3 Haiku",
        prompt_price_per_million=0.25,
        completion_price_per_million=1.25,
        context_window=200000
    ),
    # Gemini Models
    "google/gemini-2.0-flash-exp": ModelPricing(
        model_id="google/gemini-2.0-flash-exp",
        model_name="Gemini 2.0 Flash",
        prompt_price_per_million=0.00,  # Free during preview
        completion_price_per_million=0.00,
        context_window=1000000
    ),
    "google/gemini-pro-1.5": ModelPricing(
        model_id="google/gemini-pro-1.5",
        model_name="Gemini Pro 1.5",
        prompt_price_per_million=1.25,
        completion_price_per_million=5.00,
        context_window=2000000
    ),
    # DeepSeek Models (very cost effective)
    "deepseek/deepseek-chat": ModelPricing(
        model_id="deepseek/deepseek-chat",
        model_name="DeepSeek Chat",
        prompt_price_per_million=0.14,
        completion_price_per_million=0.28,
        context_window=64000
    ),
    # Llama Models (open source, cheap)
    "meta-llama/llama-3.1-70b-instruct": ModelPricing(
        model_id="meta-llama/llama-3.1-70b-instruct",
        model_name="Llama 3.1 70B",
        prompt_price_per_million=0.52,
        completion_price_per_million=0.75,
        context_window=131072
    ),
    "meta-llama/llama-3.1-8b-instruct": ModelPricing(
        model_id="meta-llama/llama-3.1-8b-instruct",
        model_name="Llama 3.1 8B",
        prompt_price_per_million=0.06,
        completion_price_per_million=0.06,
        context_window=131072
    ),
    # Mistral Models
    "mistralai/mistral-large": ModelPricing(
        model_id="mistralai/mistral-large",
        model_name="Mistral Large",
        prompt_price_per_million=2.00,
        completion_price_per_million=6.00,
        context_window=128000
    ),
    # Qwen Models (very cheap, good performance)
    "qwen/qwen-2.5-72b-instruct": ModelPricing(
        model_id="qwen/qwen-2.5-72b-instruct",
        model_name="Qwen 2.5 72B",
        prompt_price_per_million=0.35,
        completion_price_per_million=0.40,
        context_window=131072
    ),
}


@dataclass
class DailyUsageTracker:
    """Track daily AI usage and costs across all providers."""
    date: str
    total_cost: float = 0.0
    request_count: int = 0
    model_costs: Dict[str, float] = None  # Track cost per model
    daily_limit: float = 50.0
    is_exhausted: bool = False
    last_exhausted_time: Optional[datetime] = None

    def __post_init__(self):
        if self.model_costs is None:
            self.model_costs = {}


class OpenRouterClient(TradingLoggerMixin):
    """
    OpenRouter API client for multi-model AI access.
    Provides intelligent model selection based on cost, performance, and task requirements.
    """

    def __init__(self, api_key: Optional[str] = None, db_manager=None):
        """
        Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key (defaults to settings)
            db_manager: Optional DatabaseManager for logging queries
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.base_url = "https://openrouter.ai/api/v1"
        self.db_manager = db_manager

        # HTTP client with timeouts
        self.client = httpx.AsyncClient(
            timeout=120.0,  # 2 minute timeout
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
        )

        # Model pricing database
        self.model_pricing = OPENROUTER_MODEL_PRICING

        # Default model preferences (ordered by cost-effectiveness for trading)
        self.default_model_priority = [
            "google/gemini-2.0-flash-exp",  # Free during preview
            "deepseek/deepseek-chat",  # Very cheap, good quality
            "openai/gpt-4o-mini",  # Cheap, reliable
            "qwen/qwen-2.5-72b-instruct",  # Cheap, strong
            "anthropic/claude-3-haiku",  # Fast, efficient
            "meta-llama/llama-3.1-70b-instruct",  # Good balance
            "openai/gpt-4o",  # Expensive but best
            "anthropic/claude-3.5-sonnet",  # Premium
        ]

        # Cost tracking
        self.total_cost = 0.0
        self.request_count = 0

        # Daily usage tracking
        self.daily_tracker = self._load_daily_tracker()
        self.usage_file = "logs/daily_openrouter_usage.pkl"

        self.logger.info(
            "OpenRouter client initialized",
            available_models=len(self.model_pricing),
            daily_limit=self.daily_tracker.daily_limit,
            today_cost=self.daily_tracker.total_cost,
            today_requests=self.daily_tracker.request_count
        )

    def _load_daily_tracker(self) -> DailyUsageTracker:
        """Load or create daily usage tracker."""
        today = datetime.now().strftime("%Y-%m-%d")
        usage_file = "logs/daily_openrouter_usage.pkl"

        os.makedirs("logs", exist_ok=True)

        try:
            if os.path.exists(usage_file):
                with open(usage_file, 'rb') as f:
                    tracker = pickle.load(f)

                # Reset if new day
                if tracker.date != today:
                    tracker = DailyUsageTracker(
                        date=today,
                        daily_limit=tracker.daily_limit
                    )
                return tracker
        except Exception as e:
            self.logger.warning(f"Failed to load daily tracker: {e}")

        # Create new tracker
        daily_limit = getattr(settings.trading, 'daily_ai_cost_limit', 50.0)
        return DailyUsageTracker(date=today, daily_limit=daily_limit)

    def _save_daily_tracker(self):
        """Save daily usage tracker to disk."""
        try:
            os.makedirs("logs", exist_ok=True)
            with open(self.usage_file, 'wb') as f:
                pickle.dump(self.daily_tracker, f)
        except Exception as e:
            self.logger.error(f"Failed to save daily tracker: {e}")

    def _update_daily_cost(self, cost: float, model_id: str):
        """Update daily cost tracking."""
        self.daily_tracker.total_cost += cost
        self.daily_tracker.request_count += 1

        # Track per-model costs
        if model_id not in self.daily_tracker.model_costs:
            self.daily_tracker.model_costs[model_id] = 0.0
        self.daily_tracker.model_costs[model_id] += cost

        self._save_daily_tracker()

        # Check if we've hit daily limit
        if self.daily_tracker.total_cost >= self.daily_tracker.daily_limit:
            self.daily_tracker.is_exhausted = True
            self.daily_tracker.last_exhausted_time = datetime.now()
            self._save_daily_tracker()

            self.logger.warning(
                "Daily AI cost limit reached!",
                daily_cost=self.daily_tracker.total_cost,
                daily_limit=self.daily_tracker.daily_limit,
                requests_today=self.daily_tracker.request_count
            )

    def select_best_model(
        self,
        task_type: str = "general",
        max_cost_per_million_tokens: float = 5.0,
        min_context_window: int = 8000,
        require_function_calling: bool = False
    ) -> str:
        """
        Select the best model for a task based on cost and requirements.

        Args:
            task_type: Type of task ("general", "analysis", "quick", "premium")
            max_cost_per_million_tokens: Maximum acceptable cost
            min_context_window: Minimum required context window
            require_function_calling: Whether function calling is required

        Returns:
            Model ID to use
        """
        # Filter models by requirements
        eligible_models = []

        for model_id, pricing in self.model_pricing.items():
            # Check context window
            if pricing.context_window < min_context_window:
                continue

            # Check function calling
            if require_function_calling and not pricing.supports_function_calling:
                continue

            # Check cost
            avg_cost = (pricing.prompt_price_per_million + pricing.completion_price_per_million) / 2
            if avg_cost > max_cost_per_million_tokens:
                continue

            eligible_models.append((model_id, pricing, avg_cost))

        if not eligible_models:
            # Fallback to cheapest available
            self.logger.warning("No models meet criteria, using cheapest available")
            return "google/gemini-2.0-flash-exp"

        # Task-specific selection
        if task_type == "quick":
            # Prioritize speed and low cost
            eligible_models.sort(key=lambda x: x[2])  # Sort by cost
            return eligible_models[0][0]

        elif task_type == "premium":
            # Prioritize quality
            premium_order = [
                "openai/gpt-4o",
                "anthropic/claude-3.5-sonnet",
                "openai/gpt-4-turbo",
                "google/gemini-pro-1.5"
            ]
            for model_id in premium_order:
                if any(m[0] == model_id for m in eligible_models):
                    return model_id
            return eligible_models[0][0]

        else:
            # Default: best cost-effectiveness
            # Use our priority list
            for model_id in self.default_model_priority:
                if any(m[0] == model_id for m in eligible_models):
                    return model_id

            # Fallback to cheapest
            eligible_models.sort(key=lambda x: x[2])
            return eligible_models[0][0]

    def calculate_cost(
        self,
        model_id: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """
        Calculate exact cost for a request.

        Args:
            model_id: Model ID used
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens

        Returns:
            Cost in USD
        """
        if model_id not in self.model_pricing:
            self.logger.warning(f"Unknown model {model_id}, estimating cost")
            return (prompt_tokens + completion_tokens) * 0.00001  # Conservative estimate

        pricing = self.model_pricing[model_id]

        prompt_cost = (prompt_tokens / 1_000_000) * pricing.prompt_price_per_million
        completion_cost = (completion_tokens / 1_000_000) * pricing.completion_price_per_million

        return prompt_cost + completion_cost

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Tuple[str, float, Dict[str, int]]:
        """
        Make a chat completion request with automatic cost tracking.

        Args:
            messages: List of message dicts with "role" and "content"
            model: Model ID (auto-selected if None)
            temperature: Sampling temperature
            max_tokens: Max completion tokens
            **kwargs: Additional model parameters

        Returns:
            Tuple of (response_content, cost_usd, usage_dict)
        """
        # Check daily limits
        if self.daily_tracker.is_exhausted:
            now = datetime.now()
            if self.daily_tracker.date != now.strftime("%Y-%m-%d"):
                # New day - reset
                self.daily_tracker = DailyUsageTracker(
                    date=now.strftime("%Y-%m-%d"),
                    daily_limit=self.daily_tracker.daily_limit
                )
                self._save_daily_tracker()
            else:
                self.logger.warning("Daily limit exhausted, skipping request")
                return "", 0.0, {}

        # Auto-select model if not specified
        if model is None:
            model = self.select_best_model(task_type="general")

        # Prepare request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/yourusername/kalshi-ai-trading-bot",  # Optional
            "X-Title": "Kalshi AI Trading Bot"  # Optional
        }

        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens:
            data["max_tokens"] = max_tokens

        # Add any additional parameters
        data.update(kwargs)

        try:
            start_time = time.time()

            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data
            )

            response.raise_for_status()
            result = response.json()

            processing_time = time.time() - start_time

            # Extract response
            content = result["choices"][0]["message"]["content"]

            # Extract usage and calculate cost
            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

            cost = self.calculate_cost(model, prompt_tokens, completion_tokens)

            # Update tracking
            self.total_cost += cost
            self.request_count += 1
            self._update_daily_cost(cost, model)

            self.logger.info(
                "OpenRouter completion successful",
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost=f"${cost:.4f}",
                processing_time=f"{processing_time:.2f}s"
            )

            return content, cost, usage

        except httpx.HTTPStatusError as e:
            self.logger.error(
                f"OpenRouter API error: {e.response.status_code}",
                response=e.response.text
            )
            raise
        except Exception as e:
            self.logger.error(f"OpenRouter request failed: {str(e)}")
            raise

    async def get_trading_decision(
        self,
        market_data: Dict,
        portfolio_data: Dict,
        news_summary: str = "",
        use_premium: bool = False
    ) -> Optional[Dict]:
        """
        Get a trading decision using the most cost-effective model.

        Args:
            market_data: Market information
            portfolio_data: Portfolio state
            news_summary: News context
            use_premium: Whether to use premium models

        Returns:
            Trading decision dict or None
        """
        # Select model based on importance
        model = self.select_best_model(
            task_type="premium" if use_premium else "general",
            max_cost_per_million_tokens=20.0 if use_premium else 5.0,
            min_context_window=16000
        )

        # Build prompt
        prompt = self._build_trading_prompt(market_data, portfolio_data, news_summary)

        messages = [
            {"role": "user", "content": prompt}
        ]

        try:
            content, cost, usage = await self.chat_completion(
                messages=messages,
                model=model,
                temperature=0.1,
                max_tokens=2000
            )

            # Parse JSON response
            decision = self._parse_trading_decision(content)

            # Log if database available
            if self.db_manager and decision:
                await self._log_query(
                    strategy="openrouter_trading",
                    query_type="trading_decision",
                    market_id=market_data.get("ticker"),
                    prompt=prompt,
                    response=content,
                    tokens_used=usage.get("total_tokens"),
                    cost_usd=cost,
                    confidence_extracted=decision.get("confidence"),
                    decision_extracted=decision.get("action")
                )

            return decision

        except Exception as e:
            self.logger.error(f"Failed to get trading decision: {str(e)}")
            return None

    def _build_trading_prompt(
        self,
        market_data: Dict,
        portfolio_data: Dict,
        news_summary: str
    ) -> str:
        """Build trading decision prompt."""
        from src.utils.prompts import SIMPLIFIED_PROMPT_TPL

        return SIMPLIFIED_PROMPT_TPL.format(
            ticker=market_data.get("ticker", "UNKNOWN"),
            title=market_data.get("title", "Unknown Market"),
            yes_price=market_data.get("yes_price", 50),
            no_price=market_data.get("no_price", 50),
            volume=market_data.get("volume", 0),
            close_time=market_data.get("close_time", "Unknown"),
            days_to_expiry=market_data.get("days_to_expiry", 30),
            news_summary=news_summary[:1000],
            cash=portfolio_data.get("cash", 1000),
            balance=portfolio_data.get("balance", 1000),
            existing_positions=len(portfolio_data.get("positions", [])),
            ev_threshold=10,
            max_trade_value=portfolio_data.get("max_trade_value", 100),
            max_position_pct=portfolio_data.get("max_position_pct", 5)
        )

    def _parse_trading_decision(self, response_text: str) -> Optional[Dict]:
        """Parse trading decision from response."""
        import re

        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group(0))
                return decision
            return None
        except Exception as e:
            self.logger.error(f"Failed to parse decision: {str(e)}")
            return None

    async def _log_query(self, **kwargs):
        """Log query to database if available."""
        if self.db_manager:
            try:
                from src.utils.database import LLMQuery

                llm_query = LLMQuery(
                    timestamp=datetime.now(),
                    **kwargs
                )

                asyncio.create_task(self.db_manager.log_llm_query(llm_query))
            except Exception as e:
                self.logger.error(f"Failed to log query: {e}")

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
        self.logger.info(
            "OpenRouter client closed",
            total_cost=self.total_cost,
            total_requests=self.request_count
        )

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
