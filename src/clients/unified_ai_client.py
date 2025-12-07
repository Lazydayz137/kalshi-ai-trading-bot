"""
Unified AI Client with intelligent model routing and cost optimization.
Routes requests between xAI (Grok), OpenRouter (multi-model), and other providers.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from pybreaker import CircuitBreaker, CircuitBreakerError
from tenacity import retry, stop_after_attempt, wait_exponential

from src.clients.xai_client import XAIClient, TradingDecision
from src.clients.openrouter_client import OpenRouterClient
from src.config.settings import settings
from src.utils.logging_setup import TradingLoggerMixin


class AIProvider(Enum):
    """Available AI providers."""
    XAI = "xai"  # Grok models
    OPENROUTER = "openrouter"  # Multi-model access
    DIRECT_OPENAI = "openai"  # Direct OpenAI
    DIRECT_ANTHROPIC = "anthropic"  # Direct Anthropic


@dataclass
class ProviderHealth:
    """Health status for an AI provider."""
    provider: AIProvider
    is_available: bool = True
    error_count: int = 0
    last_error: Optional[str] = None
    last_success: Optional[datetime] = None
    total_cost: float = 0.0
    total_requests: int = 0
    avg_latency_ms: float = 0.0


@dataclass
class ModelSelectionStrategy:
    """Strategy for selecting which model/provider to use."""
    prefer_cost_effective: bool = True  # Prioritize cheap models
    allow_premium_on_confidence: bool = True  # Use premium for high-confidence trades
    premium_confidence_threshold: float = 0.80  # Confidence level to trigger premium
    max_cost_per_request: float = 0.10  # Maximum cost per request
    enable_fallback: bool = True  # Fall back to cheaper models on failure
    prefer_local_models: bool = False  # Future: prefer self-hosted models


class UnifiedAIClient(TradingLoggerMixin):
    """
    Unified AI client that intelligently routes between multiple providers.

    Features:
    - Automatic provider selection based on cost, latency, and availability
    - Circuit breaker pattern for resilience
    - Fallback chains for reliability
    - Cost optimization and tracking
    - Model performance monitoring
    """

    def __init__(self, db_manager=None):
        """
        Initialize unified AI client with all providers.

        Args:
            db_manager: Optional DatabaseManager for logging
        """
        self.db_manager = db_manager

        # Initialize providers
        self.xai_client = None
        self.openrouter_client = None

        # Provider health tracking
        self.provider_health: Dict[AIProvider, ProviderHealth] = {
            AIProvider.XAI: ProviderHealth(AIProvider.XAI),
            AIProvider.OPENROUTER: ProviderHealth(AIProvider.OPENROUTER),
        }

        # Circuit breakers for each provider
        self.circuit_breakers: Dict[AIProvider, CircuitBreaker] = {
            AIProvider.XAI: CircuitBreaker(
                fail_max=5,  # Open after 5 failures
                timeout_duration=300,  # 5 minute timeout
                name="xai_breaker"
            ),
            AIProvider.OPENROUTER: CircuitBreaker(
                fail_max=5,
                timeout_duration=300,
                name="openrouter_breaker"
            ),
        }

        # Selection strategy
        self.strategy = ModelSelectionStrategy()

        # Initialize available providers
        self._initialize_providers()

        self.logger.info(
            "Unified AI client initialized",
            available_providers=len([h for h in self.provider_health.values() if h.is_available]),
            strategy=self.strategy
        )

    def _initialize_providers(self):
        """Initialize available AI providers."""
        try:
            # Try to initialize xAI (Grok)
            if hasattr(settings.api, 'xai_api_key') and settings.api.xai_api_key:
                self.xai_client = XAIClient(db_manager=self.db_manager)
                self.provider_health[AIProvider.XAI].is_available = True
                self.logger.info("xAI (Grok) provider initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize xAI: {e}")
            self.provider_health[AIProvider.XAI].is_available = False

        try:
            # Try to initialize OpenRouter
            import os
            if os.getenv("OPENROUTER_API_KEY"):
                self.openrouter_client = OpenRouterClient(db_manager=self.db_manager)
                self.provider_health[AIProvider.OPENROUTER].is_available = True
                self.logger.info("OpenRouter provider initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize OpenRouter: {e}")
            self.provider_health[AIProvider.OPENROUTER].is_available = False

    def select_provider(
        self,
        task_type: str = "trading_decision",
        confidence_estimate: Optional[float] = None,
        prefer_premium: bool = False
    ) -> Tuple[AIProvider, Any]:
        """
        Select the best provider for a task.

        Args:
            task_type: Type of task to perform
            confidence_estimate: Estimated confidence (for premium selection)
            prefer_premium: Force premium model selection

        Returns:
            Tuple of (provider, client)
        """
        # Check for premium requirement
        use_premium = prefer_premium or (
            confidence_estimate is not None and
            confidence_estimate >= self.strategy.premium_confidence_threshold and
            self.strategy.allow_premium_on_confidence
        )

        # Build priority list
        if use_premium:
            # Premium: prefer xAI Grok-4
            priority = [AIProvider.XAI, AIProvider.OPENROUTER]
        elif self.strategy.prefer_cost_effective:
            # Cost-effective: prefer OpenRouter (can select cheap models)
            priority = [AIProvider.OPENROUTER, AIProvider.XAI]
        else:
            # Default: xAI then OpenRouter
            priority = [AIProvider.XAI, AIProvider.OPENROUTER]

        # Select first available provider
        for provider in priority:
            health = self.provider_health[provider]
            breaker = self.circuit_breakers[provider]

            # Check if provider is available and circuit is closed
            if health.is_available and breaker.current_state == "closed":
                client = self._get_client(provider)
                if client:
                    self.logger.debug(
                        f"Selected provider: {provider.value}",
                        task_type=task_type,
                        use_premium=use_premium
                    )
                    return provider, client

        # No provider available
        self.logger.error("No AI providers available!")
        raise RuntimeError("No AI providers available")

    def _get_client(self, provider: AIProvider) -> Optional[Any]:
        """Get client instance for a provider."""
        if provider == AIProvider.XAI:
            return self.xai_client
        elif provider == AIProvider.OPENROUTER:
            return self.openrouter_client
        return None

    async def get_trading_decision(
        self,
        market_data: Dict,
        portfolio_data: Dict,
        news_summary: str = "",
        prefer_premium: bool = False
    ) -> Optional[TradingDecision]:
        """
        Get a trading decision with intelligent provider selection.

        Args:
            market_data: Market information
            portfolio_data: Portfolio state
            news_summary: News context
            prefer_premium: Whether to prefer premium models

        Returns:
            TradingDecision or None
        """
        # Select provider
        try:
            provider, client = self.select_provider(
                task_type="trading_decision",
                prefer_premium=prefer_premium
            )
        except RuntimeError as e:
            self.logger.error(f"Provider selection failed: {e}")
            return None

        # Make request with circuit breaker
        breaker = self.circuit_breakers[provider]
        health = self.provider_health[provider]

        try:
            # Execute with circuit breaker protection
            decision = await self._execute_with_breaker(
                breaker,
                self._get_decision_from_provider,
                provider,
                client,
                market_data,
                portfolio_data,
                news_summary
            )

            # Update health on success
            health.last_success = datetime.now()
            health.total_requests += 1

            return decision

        except CircuitBreakerError:
            self.logger.warning(
                f"Circuit breaker open for {provider.value}, trying fallback"
            )

            # Try fallback provider if enabled
            if self.strategy.enable_fallback:
                return await self._try_fallback_decision(
                    market_data,
                    portfolio_data,
                    news_summary,
                    excluded_provider=provider
                )
            return None

        except Exception as e:
            self.logger.error(
                f"Error getting decision from {provider.value}: {e}"
            )
            health.error_count += 1
            health.last_error = str(e)

            # Try fallback
            if self.strategy.enable_fallback:
                return await self._try_fallback_decision(
                    market_data,
                    portfolio_data,
                    news_summary,
                    excluded_provider=provider
                )
            return None

    async def _execute_with_breaker(self, breaker, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        return await breaker.call_async(func, *args, **kwargs)

    async def _get_decision_from_provider(
        self,
        provider: AIProvider,
        client: Any,
        market_data: Dict,
        portfolio_data: Dict,
        news_summary: str
    ) -> Optional[TradingDecision]:
        """Get trading decision from specific provider."""
        if provider == AIProvider.XAI:
            # Use xAI client
            return await client.get_trading_decision(
                market_data=market_data,
                portfolio_data=portfolio_data,
                news_summary=news_summary
            )
        elif provider == AIProvider.OPENROUTER:
            # Use OpenRouter client
            result = await client.get_trading_decision(
                market_data=market_data,
                portfolio_data=portfolio_data,
                news_summary=news_summary
            )

            # Convert dict to TradingDecision if needed
            if result and isinstance(result, dict):
                return TradingDecision(
                    action=result.get("action", "SKIP"),
                    side=result.get("side", "YES"),
                    confidence=result.get("confidence", 0.5),
                    limit_price=result.get("limit_price", 50)
                )
            return result
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    async def _try_fallback_decision(
        self,
        market_data: Dict,
        portfolio_data: Dict,
        news_summary: str,
        excluded_provider: AIProvider
    ) -> Optional[TradingDecision]:
        """Try to get decision from fallback providers."""
        for provider in AIProvider:
            if provider == excluded_provider:
                continue

            health = self.provider_health.get(provider)
            if not health or not health.is_available:
                continue

            breaker = self.circuit_breakers.get(provider)
            if not breaker or breaker.current_state != "closed":
                continue

            try:
                client = self._get_client(provider)
                if not client:
                    continue

                self.logger.info(f"Trying fallback provider: {provider.value}")

                decision = await self._get_decision_from_provider(
                    provider,
                    client,
                    market_data,
                    portfolio_data,
                    news_summary
                )

                if decision:
                    self.logger.info(f"Fallback successful: {provider.value}")
                    return decision

            except Exception as e:
                self.logger.warning(f"Fallback to {provider.value} failed: {e}")
                continue

        return None

    async def get_completion(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        prefer_premium: bool = False
    ) -> Optional[str]:
        """
        Get a simple completion with intelligent routing.

        Args:
            prompt: Prompt text
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            prefer_premium: Whether to prefer premium models

        Returns:
            Completion text or None
        """
        try:
            provider, client = self.select_provider(
                task_type="completion",
                prefer_premium=prefer_premium
            )
        except RuntimeError:
            return None

        try:
            if provider == AIProvider.XAI:
                return await client.get_completion(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            elif provider == AIProvider.OPENROUTER:
                messages = [{"role": "user", "content": prompt}]
                content, cost, usage = await client.chat_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return content
            else:
                return None

        except Exception as e:
            self.logger.error(f"Completion failed: {e}")
            return None

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get summary of costs across all providers."""
        summary = {
            "total_cost": 0.0,
            "total_requests": 0,
            "by_provider": {}
        }

        for provider, health in self.provider_health.items():
            summary["total_cost"] += health.total_cost
            summary["total_requests"] += health.total_requests
            summary["by_provider"][provider.value] = {
                "cost": health.total_cost,
                "requests": health.total_requests,
                "available": health.is_available,
                "error_count": health.error_count
            }

        # Add specific provider details
        if self.xai_client:
            summary["xai_daily_cost"] = self.xai_client.daily_tracker.total_cost

        if self.openrouter_client:
            summary["openrouter_daily_cost"] = self.openrouter_client.daily_tracker.total_cost
            summary["openrouter_model_costs"] = self.openrouter_client.daily_tracker.model_costs

        return summary

    async def close(self):
        """Close all provider clients."""
        if self.xai_client:
            await self.xai_client.close()
        if self.openrouter_client:
            await self.openrouter_client.close()

        # Log final summary
        summary = self.get_cost_summary()
        self.logger.info(
            "Unified AI client closed",
            total_cost=summary["total_cost"],
            total_requests=summary["total_requests"],
            providers=summary["by_provider"]
        )

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
