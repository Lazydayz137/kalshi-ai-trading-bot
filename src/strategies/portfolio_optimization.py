"""
Advanced Portfolio Optimization - Kelly Criterion Extensions

This module implements cutting-edge portfolio optimization techniques:
1. Kelly Criterion Extension (KCE) for prediction markets
2. Risk Parity allocation 
3. Dynamic position sizing based on market conditions
4. Cross-correlation analysis between markets
5. Multi-objective optimization (return vs risk vs drawdown)

Based on latest research:
- Kelly Criterion Extension for dynamic markets (Kim, 2024)
- Fractional Kelly strategies for risk management
- Portfolio optimization for prediction markets

Key innovations:
- Uses Kelly Criterion for fund managers (not direct asset investment)
- Adapts to both favorable and unfavorable market conditions  
- Implements dynamic rebalancing based on market state
- Risk-adjusted allocation rather than equal capital weights
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from scipy.optimize import minimize, minimize_scalar
import warnings
warnings.filterwarnings('ignore')

from src.utils.database import DatabaseManager, Market, Position
from src.clients.kalshi_client import KalshiClient
from src.clients.xai_client import XAIClient
from src.clients.odds_client import OddsClient
from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger


@dataclass
class MarketOpportunity:
    """Represents a trading opportunity with all required metrics for optimization."""
    market_id: str
    market_title: str
    predicted_probability: float
    market_probability: float
    confidence: float
    edge: float  # predicted_prob - market_prob
    volatility: float
    expected_return: float
    max_loss: float
    time_to_expiry: float  # in days
    correlation_score: float  # correlation with portfolio
    
    # Kelly metrics
    kelly_fraction: float
    fractional_kelly: float  # Conservative Kelly
    risk_adjusted_fraction: float
    
    # Portfolio metrics  
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_contribution: float
    
    # Data collection
    decision_id: Optional[int] = None


@dataclass
class PortfolioAllocation:
    """Optimal portfolio allocation across opportunities."""
    allocations: Dict[str, float]  # market_id -> allocation fraction
    total_capital_used: float
    expected_portfolio_return: float
    portfolio_volatility: float
    portfolio_sharpe: float
    max_portfolio_drawdown: float
    diversification_ratio: float
    
    # Risk metrics
    portfolio_var_95: float  # Value at Risk
    portfolio_cvar_95: float  # Conditional Value at Risk
    
    # Kelly metrics
    aggregate_kelly_fraction: float
    portfolio_growth_rate: float


class AdvancedPortfolioOptimizer:
    """
    Advanced portfolio optimization using Kelly Criterion Extensions and modern portfolio theory.
    
    This implements the latest research in prediction market portfolio optimization:
    - Kelly Criterion Extension (KCE) for dynamic market conditions
    - Risk parity allocation for balanced risk exposure  
    - Multi-factor optimization considering correlation, volatility, and drawdown
    - Dynamic rebalancing based on market regime detection
    """
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        kalshi_client: KalshiClient,
        xai_client: XAIClient,
        odds_client: Optional[OddsClient] = None
    ):
        self.db_manager = db_manager
        self.kalshi_client = kalshi_client
        self.xai_client = xai_client
        self.odds_client = odds_client
        self.logger = get_trading_logger("portfolio_optimizer")
        
        # Portfolio parameters
        self.total_capital = getattr(settings.trading, 'total_capital', 10000)
        self.max_position_fraction = getattr(settings.trading, 'max_single_position', 0.25)
        self.min_position_size = getattr(settings.trading, 'min_position_size', 5)
        self.kelly_fraction_multiplier = getattr(settings.trading, 'kelly_fraction', 0.25)  # Fractional Kelly
        
        # Risk management
        self.max_portfolio_volatility = getattr(settings.trading, 'max_volatility', 0.20)
        self.max_correlation = getattr(settings.trading, 'max_correlation', 0.70)
        self.target_sharpe_ratio = getattr(settings.trading, 'target_sharpe', 2.0)
        
        # Market regime detection
        self.market_state = "normal"  # normal, volatile, trending
        self.regime_lookback = 30  # days
        
        # Performance tracking
        self.historical_allocations = []
        self.realized_returns = []
        self.portfolio_metrics = {}

    async def optimize_portfolio(
        self, 
        opportunities: List[MarketOpportunity]
    ) -> PortfolioAllocation:
        """
        Main portfolio optimization using advanced Kelly Criterion and risk parity.
        
        Process:
        1. Calculate Kelly fractions for each opportunity
        2. Apply risk adjustments and correlations
        3. Optimize using multi-objective function
        4. Apply risk constraints and position limits
        5. Return optimal allocation
        """
        self.logger.info(f"Optimizing portfolio across {len(opportunities)} opportunities")
        
        if not opportunities:
            return self._empty_allocation()
        
        # Limit opportunities to prevent optimization complexity
        max_opportunities = getattr(settings.trading, 'max_opportunities_per_batch', 50)
        if len(opportunities) > max_opportunities:
            # Sort by confidence * expected_return and take top N
            opportunities = sorted(
                opportunities, 
                key=lambda x: x.confidence * x.expected_return, 
                reverse=True
            )[:max_opportunities]
            self.logger.info(f"Limited to top {max_opportunities} opportunities for optimization")
        
        try:
            # Step 1: Enhance opportunities with portfolio metrics
            enhanced_opportunities = await self._enhance_opportunities_with_metrics(opportunities)
            
            # Step 2: Detect market regime and adjust parameters
            await self._detect_market_regime()
            
            # Step 3: Calculate Kelly fractions
            kelly_fractions = self._calculate_kelly_fractions(enhanced_opportunities)
            
            # Step 3.5: Update opportunities with Kelly fractions
            for opp in enhanced_opportunities:
                kelly_val = kelly_fractions.get(opp.market_id, 0.0)
                # Update the opportunity object in place
                opp.kelly_fraction = kelly_val
                opp.fractional_kelly = kelly_val * 0.5  # Conservative Kelly
                opp.risk_adjusted_fraction = kelly_val
            
            # Step 4: Apply correlation adjustments
            correlation_matrix = await self._estimate_correlation_matrix(enhanced_opportunities)
            adjusted_fractions = self._apply_correlation_adjustments(kelly_fractions, correlation_matrix)
            
            # Step 5: Multi-objective optimization
            optimal_allocation = self._multi_objective_optimization(
                enhanced_opportunities, adjusted_fractions, correlation_matrix
            )
            
            # Step 6: Apply risk constraints
            final_allocation = self._apply_risk_constraints(optimal_allocation, enhanced_opportunities)
            
            # Step 7: Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(
                final_allocation, enhanced_opportunities, correlation_matrix
            )
            
            result = PortfolioAllocation(
                allocations=final_allocation,
                **portfolio_metrics
            )
            
            self.logger.info(
                f"Portfolio optimization complete: "
                f"Capital used: ${result.total_capital_used:.0f}, "
                f"Expected return: {result.expected_portfolio_return:.1%}, "
                f"Sharpe ratio: {result.portfolio_sharpe:.2f}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in portfolio optimization: {e}")
            return self._empty_allocation()

    async def _enhance_opportunities_with_metrics(
        self, 
        opportunities: List[MarketOpportunity]
    ) -> List[MarketOpportunity]:
        """
        Enhance opportunities with additional portfolio metrics.
        """
        enhanced = []
        
        for opp in opportunities:
            try:
                # Calculate advanced metrics
                sharpe_ratio = self._calculate_sharpe_ratio(opp)
                sortino_ratio = self._calculate_sortino_ratio(opp)
                max_dd_contribution = self._estimate_max_drawdown_contribution(opp)
                
                # Create enhanced opportunity
                enhanced_opp = MarketOpportunity(
                    market_id=opp.market_id,
                    market_title=opp.market_title,
                    predicted_probability=opp.predicted_probability,
                    market_probability=opp.market_probability,
                    confidence=opp.confidence,
                    edge=opp.edge,
                    volatility=opp.volatility,
                    expected_return=opp.expected_return,
                    max_loss=opp.max_loss,
                    time_to_expiry=opp.time_to_expiry,
                    correlation_score=0.0,  # Will be calculated later
                    kelly_fraction=0.0,
                    fractional_kelly=0.0,
                    risk_adjusted_fraction=0.0,
                    sharpe_ratio=sharpe_ratio,
                    sortino_ratio=sortino_ratio,
                    max_drawdown_contribution=max_dd_contribution
                )
                
                enhanced.append(enhanced_opp)
                
            except Exception as e:
                self.logger.error(f"Error enhancing opportunity {opp.market_id}: {e}")
                continue
        
        return enhanced

    def _calculate_kelly_fractions(self, opportunities: List[MarketOpportunity]) -> Dict[str, float]:
        """
        Calculate Kelly fractions using the Kelly Criterion Extension (KCE).
        
        Implements the advanced Kelly Criterion that adapts to market conditions:
        - Standard Kelly for high-confidence, low-correlation opportunities
        - Fractional Kelly for moderate confidence
        - Kelly Criterion Extension for dynamic market environments
        """
        kelly_fractions = {}
        
        for opp in opportunities:
            try:
                # Calculate basic Kelly fraction: f* = (bp - q) / b
                # Where p = win probability, q = lose probability, b = odds
                
                win_prob = opp.predicted_probability
                lose_prob = 1 - win_prob
                
                # Calculate odds from market price
                if opp.market_probability > 0 and opp.market_probability < 1:
                    odds = (1 - opp.market_probability) / opp.market_probability
                else:
                    odds = 1.0
                
                # Standard Kelly calculation
                if opp.edge > 0 and win_prob > 0.5:
                    kelly_standard = (odds * win_prob - lose_prob) / odds
                else:
                    kelly_standard = 0.0
                
                # Apply Kelly Criterion Extension for dynamic markets
                # Adjust for market regime and time decay
                regime_multiplier = self._get_regime_multiplier()
                time_decay_factor = max(0.1, min(1.0, opp.time_to_expiry / 30))  # Decay over 30 days
                
                kelly_kce = kelly_standard * regime_multiplier * time_decay_factor
                
                # Apply confidence adjustment
                confidence_adjusted = kelly_kce * opp.confidence
                
                # Apply fractional Kelly (typically 25-50% of full Kelly)
                fractional_kelly = confidence_adjusted * self.kelly_fraction_multiplier
                
                # Ensure reasonable bounds
                final_kelly = max(0.0, min(self.max_position_fraction, fractional_kelly))
                
                # Store calculations
                opp.kelly_fraction = kelly_standard
                opp.fractional_kelly = fractional_kelly
                opp.risk_adjusted_fraction = final_kelly
                
                kelly_fractions[opp.market_id] = final_kelly
                
                self.logger.debug(
                    f"Kelly calculation for {opp.market_id}: "
                    f"Standard: {kelly_standard:.3f}, "
                    f"KCE: {kelly_kce:.3f}, "
                    f"Final: {final_kelly:.3f}"
                )
                
            except Exception as e:
                self.logger.error(f"Error calculating Kelly for {opp.market_id}: {e}")
                kelly_fractions[opp.market_id] = 0.0
        
        return kelly_fractions

    async def _estimate_correlation_matrix(
        self, 
        opportunities: List[MarketOpportunity]
    ) -> np.ndarray:
        """
        Estimate correlation matrix between markets for portfolio optimization.
        
        Uses multiple approaches:
        1. Category-based correlations
        2. Time-based correlations (similar expiry dates)
        3. Content similarity analysis
        4. Historical price correlations where available
        """
        n = len(opportunities)
        if n <= 1:
            return np.eye(1)
            
        correlation_matrix = np.eye(n)
        
        try:
            for i, opp1 in enumerate(opportunities):
                for j, opp2 in enumerate(opportunities):
                    if i != j:
                        correlation = await self._estimate_pairwise_correlation(opp1, opp2)
                        correlation_matrix[i, j] = correlation
                        
                        # Update correlation score in opportunity
                        opp1.correlation_score = max(opp1.correlation_score, abs(correlation))
            
            # Ensure matrix is positive semidefinite
            correlation_matrix = self._ensure_positive_semidefinite(correlation_matrix)
            
        except Exception as e:
            self.logger.error(f"Error estimating correlation matrix: {e}")
            correlation_matrix = np.eye(n)  # Fall back to identity matrix
        
        return correlation_matrix

    async def _estimate_pairwise_correlation(
        self, 
        opp1: MarketOpportunity, 
        opp2: MarketOpportunity
    ) -> float:
        """
        Estimate correlation between two market opportunities.
        """
        try:
            correlation = 0.0
            
            # 1. Category-based correlation (if markets are in same category)
            category_corr = await self._get_category_correlation(opp1.market_id, opp2.market_id)
            
            # 2. Time-based correlation (similar expiry times)
            time_diff = abs(opp1.time_to_expiry - opp2.time_to_expiry)
            time_corr = max(0, 1 - (time_diff / 30))  # Decay over 30 days
            
            # 3. Content similarity (use AI to assess)
            content_corr = await self._get_content_similarity(opp1, opp2)
            
            # 4. Volatility similarity
            vol_diff = abs(opp1.volatility - opp2.volatility)
            vol_corr = max(0, 1 - vol_diff)
            
            # Combine correlations with weights
            correlation = (
                0.4 * category_corr +
                0.2 * time_corr +
                0.3 * content_corr +
                0.1 * vol_corr
            )
            
            # Cap maximum correlation
            correlation = min(self.max_correlation, correlation)
            
            return correlation
            
        except Exception as e:
            self.logger.error(f"Error estimating pairwise correlation: {e}")
            return 0.1  # Small default correlation

    def _apply_correlation_adjustments(
        self, 
        kelly_fractions: Dict[str, float], 
        correlation_matrix: np.ndarray
    ) -> Dict[str, float]:
        """
        Adjust Kelly fractions based on correlations to reduce portfolio risk.
        
        High correlations reduce effective diversification, so we scale down allocations
        to highly correlated markets.
        """
        adjusted_fractions = kelly_fractions.copy()
        market_ids = list(kelly_fractions.keys())
        
        try:
            for i, market_id in enumerate(market_ids):
                # Calculate average correlation with other markets
                avg_correlation = np.mean([
                    abs(correlation_matrix[i, j]) 
                    for j in range(len(market_ids)) 
                    if i != j
                ])
                
                # Adjust fraction based on correlation
                # Higher correlation -> lower allocation
                correlation_penalty = 1 - (avg_correlation * 0.5)  # Max 50% penalty
                
                adjusted_fractions[market_id] *= correlation_penalty
                
                self.logger.debug(
                    f"Correlation adjustment for {market_id}: "
                    f"Avg corr: {avg_correlation:.3f}, "
                    f"Penalty: {correlation_penalty:.3f}"
                )
        
        except Exception as e:
            self.logger.error(f"Error applying correlation adjustments: {e}")
        
        return adjusted_fractions

    def _multi_objective_optimization(
        self,
        opportunities: List[MarketOpportunity],
        kelly_fractions: Dict[str, float],
        correlation_matrix: np.ndarray
    ) -> Dict[str, float]:
        """
        Multi-objective optimization balancing return, risk, and diversification.
        
        Objective function combines:
        1. Expected return (maximize)
        2. Portfolio volatility (minimize)  
        3. Maximum drawdown (minimize)
        4. Diversification ratio (maximize)
        5. Sharpe ratio (maximize)
        """
        try:
            n = len(opportunities)
            if n == 0:
                return {}
            
            # Initial allocation from Kelly fractions
            initial_weights = np.array([kelly_fractions.get(opp.market_id, 0) for opp in opportunities])
            
            # Normalize to sum to 1 or less
            if initial_weights.sum() > 1.0:
                initial_weights = initial_weights / initial_weights.sum()
            
            # If initial weights are all zero or very small, use simple fallback
            if initial_weights.sum() < 0.001:
                self.logger.warning("Initial weights too small, using simple allocation fallback")
                return self._simple_allocation_fallback(opportunities)
            
            # Expected returns vector
            expected_returns = np.array([opp.expected_return for opp in opportunities])
            
            # Volatilities vector
            volatilities = np.array([opp.volatility for opp in opportunities])
            
            # Create covariance matrix from correlations and volatilities
            covariance_matrix = np.outer(volatilities, volatilities) * correlation_matrix
            
            def objective_function(weights):
                """Multi-objective function to minimize."""
                try:
                    # Ensure weights are valid
                    if np.any(weights < 0) or np.sum(weights) > 1.0001:  # Small tolerance
                        return 1e6
                    
                    # Portfolio return
                    portfolio_return = np.dot(weights, expected_returns)
                    
                    # Portfolio volatility
                    portfolio_vol = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
                    
                    # Diversification ratio (higher is better)
                    weighted_vol = np.dot(weights, volatilities)
                    diversification_ratio = weighted_vol / (portfolio_vol + 1e-8)
                    
                    # Sharpe ratio approximation
                    sharpe = (portfolio_return / (portfolio_vol + 1e-8))
                    
                    # Maximum drawdown approximation
                    max_dd = self._estimate_portfolio_max_drawdown(weights, opportunities)
                    
                    # Multi-objective score (higher is better, so we minimize negative)
                    # FIXED: Much less conservative - focus on returns, light risk penalty
                    score = -(
                        10.0 * portfolio_return +     # Heavy weight on returns
                        2.0 * sharpe +                # Moderate Sharpe weight
                        0.5 * diversification_ratio - # Light diversification bonus
                        0.2 * portfolio_vol -         # Light volatility penalty (was 1.0)
                        0.1 * max_dd                  # Very light drawdown penalty (was 1.0)
                    )
                    
                    return score
                    
                except Exception as e:
                    return 1e6  # Large penalty for errors
            
            # Try optimization with more relaxed constraints
            try:
                # Constraints - force meaningful allocation
                constraints = [
                    {'type': 'ineq', 'fun': lambda w: 0.80 - np.sum(w)},  # Sum <= 80% (reasonable max)
                    {'type': 'ineq', 'fun': lambda w: np.sum(w) - 0.05},  # Sum >= 5% (force minimum allocation)
                ]
                
                # Bounds for each weight
                bounds = [(0, min(self.max_position_fraction, 0.3)) for _ in range(n)]  # Cap at 30%
                
                # Optimize
                result = minimize(
                    objective_function,
                    initial_weights,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 500, 'ftol': 1e-6}  # More relaxed tolerance
                )
                
                if result.success and np.sum(result.x) > 0.001:
                    optimal_weights = result.x
                    self.logger.info(f"Optimization successful with sum: {np.sum(optimal_weights):.3f}")
                else:
                    self.logger.warning(f"Optimization failed: {result.message}")
                    raise Exception("Optimization failed")
                    
            except Exception as e:
                self.logger.warning(f"Scipy optimization failed: {e}, using simple fallback")
                return self._simple_allocation_fallback(opportunities)
            
            # Convert back to dictionary
            optimal_allocation = {
                opp.market_id: float(optimal_weights[i]) 
                for i, opp in enumerate(opportunities)
                if optimal_weights[i] > 0.001  # Filter tiny allocations
            }
            
            return optimal_allocation
            
        except Exception as e:
            self.logger.error(f"Error in multi-objective optimization: {e}")
            # Fall back to simple allocation
            return self._simple_allocation_fallback(opportunities)

    def _simple_allocation_fallback(self, opportunities: List[MarketOpportunity]) -> Dict[str, float]:
        """
        Simple fallback allocation when optimization fails.
        Allocates based on expected return * confidence, subject to position limits.
        """
        try:
            if not opportunities:
                return {}
            
            # Calculate scores for each opportunity
            scores = []
            for opp in opportunities:
                # Score based on expected return, confidence, and edge
                score = opp.expected_return * opp.confidence * max(0, abs(opp.edge))
                scores.append((opp.market_id, score))
            
            # Sort by score
            scores.sort(key=lambda x: x[1], reverse=True)
            
            # Simple allocation: give more to higher scoring opportunities
            allocation = {}
            total_allocation = 0.0
            max_positions = min(5, len(opportunities))  # Limit to top 5 positions
            
            for i, (market_id, score) in enumerate(scores[:max_positions]):
                if score <= 0:
                    continue
                    
                # Allocate more to higher ranked opportunities
                weight = max(0.05, 0.25 - (i * 0.03))  # Start at 25%, decrease by 3% each rank
                
                # Don't exceed total capital
                if total_allocation + weight <= 0.8:  # Max 80% allocation
                    allocation[market_id] = weight
                    total_allocation += weight
                else:
                    remaining = 0.8 - total_allocation
                    if remaining > 0.01:  # Only if meaningful allocation left
                        allocation[market_id] = remaining
                    break
            
            self.logger.info(f"Simple fallback allocation: {len(allocation)} positions, {total_allocation:.1%} capital")
            return allocation
            
        except Exception as e:
            self.logger.error(f"Error in simple allocation fallback: {e}")
            return {}

    def _apply_risk_constraints(
        self, 
        allocation: Dict[str, float], 
        opportunities: List[MarketOpportunity]
    ) -> Dict[str, float]:
        """
        Apply final risk constraints and position sizing limits.
        """
        try:
            constrained_allocation = {}
            total_allocation = 0.0
            
            for market_id, fraction in allocation.items():
                # Find the opportunity
                opp = next((o for o in opportunities if o.market_id == market_id), None)
                if not opp:
                    continue
                
                # Apply minimum position size
                dollar_allocation = fraction * self.total_capital
                if dollar_allocation < self.min_position_size:
                    continue
                
                # Apply maximum position fraction
                final_fraction = min(fraction, self.max_position_fraction)
                
                # Check if this would exceed total capital
                if total_allocation + final_fraction <= 1.0:
                    constrained_allocation[market_id] = final_fraction
                    total_allocation += final_fraction
                else:
                    # Scale down to fit remaining capital
                    remaining = 1.0 - total_allocation
                    if remaining > 0.001:
                        constrained_allocation[market_id] = remaining
                        total_allocation = 1.0
                    break
            
            self.logger.info(f"Risk constraints applied. Total allocation: {total_allocation:.3f}")
            self.logger.info(f"Final constrained allocations: {constrained_allocation}")
            
            return constrained_allocation
            
        except Exception as e:
            self.logger.error(f"Error applying risk constraints: {e}")
            return allocation

    def _calculate_portfolio_metrics(
        self,
        allocation: Dict[str, float],
        opportunities: List[MarketOpportunity],
        correlation_matrix: np.ndarray
    ) -> Dict:
        """
        Calculate comprehensive portfolio metrics.
        """
        try:
            if not allocation:
                return self._empty_portfolio_metrics()
            
            # Get vectors for allocated opportunities
            allocated_opps = [opp for opp in opportunities if opp.market_id in allocation]
            weights = np.array([allocation[opp.market_id] for opp in allocated_opps])
            returns = np.array([opp.expected_return for opp in allocated_opps])
            volatilities = np.array([opp.volatility for opp in allocated_opps])
            
            # Portfolio return
            portfolio_return = np.dot(weights, returns)
            
            # Portfolio volatility
            n = len(allocated_opps)
            if n > 1:
                allocated_corr_matrix = correlation_matrix[:n, :n]
                covariance_matrix = np.outer(volatilities, volatilities) * allocated_corr_matrix
                portfolio_vol = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
            else:
                portfolio_vol = volatilities[0] * weights[0] if len(volatilities) > 0 else 0.0
            
            # Sharpe ratio
            portfolio_sharpe = portfolio_return / (portfolio_vol + 1e-8)
            
            # Diversification ratio
            weighted_vol = np.dot(weights, volatilities)
            diversification_ratio = weighted_vol / (portfolio_vol + 1e-8)
            
            # Capital usage
            total_capital_used = sum(allocation.values()) * self.total_capital
            
            # Risk metrics (simplified)
            portfolio_var_95 = portfolio_vol * 1.645  # 95% VaR
            portfolio_cvar_95 = portfolio_var_95 * 1.2  # Approximate CVaR
            
            # Kelly metrics
            aggregate_kelly = sum(
                opp.kelly_fraction * allocation[opp.market_id] 
                for opp in allocated_opps
            )
            
            portfolio_growth_rate = portfolio_return - (portfolio_vol ** 2) / 2  # Geometric return approximation
            
            # Maximum drawdown
            max_portfolio_drawdown = self._estimate_portfolio_max_drawdown(weights, allocated_opps)
            
            return {
                'total_capital_used': total_capital_used,
                'expected_portfolio_return': portfolio_return,
                'portfolio_volatility': portfolio_vol,
                'portfolio_sharpe': portfolio_sharpe,
                'max_portfolio_drawdown': max_portfolio_drawdown,
                'diversification_ratio': diversification_ratio,
                'portfolio_var_95': portfolio_var_95,
                'portfolio_cvar_95': portfolio_cvar_95,
                'aggregate_kelly_fraction': aggregate_kelly,
                'portfolio_growth_rate': portfolio_growth_rate
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {e}")
            return self._empty_portfolio_metrics()

    # Helper methods
    
    async def _detect_market_regime(self):
        """Detect current market regime for Kelly adjustments."""
        # Simplified regime detection - in production would use more sophisticated methods
        self.market_state = "normal"  # Default
    
    def _get_regime_multiplier(self) -> float:
        """Get Kelly multiplier based on market regime."""
        regime_multipliers = {
            "normal": 1.0,
            "volatile": 0.7,  # Reduce Kelly in volatile markets
            "trending": 1.2   # Increase Kelly in trending markets
        }
        return regime_multipliers.get(self.market_state, 1.0)
    
    def _calculate_sharpe_ratio(self, opp: MarketOpportunity) -> float:
        """Calculate Sharpe ratio for opportunity."""
        return opp.expected_return / (opp.volatility + 1e-8)
    
    def _calculate_sortino_ratio(self, opp: MarketOpportunity) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        # Simplified - assumes normal distribution
        downside_vol = opp.volatility * 0.7  # Approximation
        return opp.expected_return / (downside_vol + 1e-8)
    
    def _estimate_max_drawdown_contribution(self, opp: MarketOpportunity) -> float:
        """Estimate maximum drawdown contribution."""
        # Simplified approximation
        return opp.volatility * 2.0
    
    async def _get_category_correlation(self, market_id1: str, market_id2: str) -> float:
        """Get correlation based on market categories."""
        # Simplified - would query market metadata
        return 0.1  # Default low correlation
    
    async def _get_content_similarity(self, opp1: MarketOpportunity, opp2: MarketOpportunity) -> float:
        """Get content similarity using AI."""
        # Simplified - would use embeddings or AI analysis
        return 0.1  # Default low similarity
    
    def _ensure_positive_semidefinite(self, matrix: np.ndarray) -> np.ndarray:
        """Ensure correlation matrix is positive semidefinite."""
        try:
            eigenvals, eigenvecs = np.linalg.eigh(matrix)
            eigenvals = np.maximum(eigenvals, 0.001)  # Ensure positive
            return eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        except:
            return np.eye(matrix.shape[0])
    
    def _estimate_portfolio_max_drawdown(self, weights: np.ndarray, opportunities: List[MarketOpportunity]) -> float:
        """Estimate portfolio maximum drawdown."""
        # Simplified approximation
        individual_mdd = np.array([opp.max_drawdown_contribution for opp in opportunities])
        return np.dot(weights, individual_mdd) * 0.8  # Diversification benefit
    
    def _empty_allocation(self) -> PortfolioAllocation:
        """Return empty portfolio allocation."""
        return PortfolioAllocation(
            allocations={},
            **self._empty_portfolio_metrics()
        )
    
    def _empty_portfolio_metrics(self) -> Dict:
        """Return empty portfolio metrics."""
        return {
            'total_capital_used': 0.0,
            'expected_portfolio_return': 0.0,
            'portfolio_volatility': 0.0,
            'portfolio_sharpe': 0.0,
            'max_portfolio_drawdown': 0.0,
            'diversification_ratio': 1.0,
            'portfolio_var_95': 0.0,
            'portfolio_cvar_95': 0.0,
            'aggregate_kelly_fraction': 0.0,
            'portfolio_growth_rate': 0.0
        }


async def create_market_opportunities_from_markets(
    markets: List[Market],
    xai_client: XAIClient,
    kalshi_client: KalshiClient,
    db_manager: DatabaseManager = None,
    total_capital: float = 10000,
    odds_client: Optional[OddsClient] = None,
    arbitrage_engine: Optional[Any] = None
) -> List[MarketOpportunity]:
    """
    Convert Market objects to MarketOpportunity objects with all required metrics.
    """
    logger = get_trading_logger("portfolio_opportunities")
    opportunities = []
    
    # Limit markets to prevent excessive AI costs and focus on best opportunities
    # Limit markets to prevent excessive AI costs and focus on best opportunities
    max_markets_to_analyze = 50  # INCREASED: Analyze top 50 markets (was 10) to find more opportunities
    if len(markets) > max_markets_to_analyze:
        # Sort by volume and take top markets
        markets = sorted(markets, key=lambda m: m.volume, reverse=True)[:max_markets_to_analyze]
        logger.info(f"Limited to top {max_markets_to_analyze} markets by volume for AI analysis")
    
    # Pre-fetch sports dict if odds client available
    active_sports = []
    if odds_client:
        try:
            active_sports = await odds_client.get_active_sports()
            logger.info(f"Fetched {len(active_sports)} active sports from Odds API")
        except Exception as e:
            logger.warning(f"Failed to fetch sports: {e}")

    for market in markets:
        try:
            # 🛑 1. FAST FAIL: JUNK FILTER (User Request: "stupid questions")
            # Must run inside loop
            from src.strategies.filters import MarketFilter
            if MarketFilter.is_junk_market(market):
                continue

            # Get current market data
            market_data = await kalshi_client.get_market(market.market_id)
            if not market_data:
                continue
            
            # FIXED: Extract from nested 'market' object (same fix as immediate trading)
            market_info = market_data.get('market', {})
            
            # CHECK SPREAD (Critical Fix for "Instant Loss")
            yes_bid = market_info.get('yes_bid', 0)
            yes_ask = market_info.get('yes_ask', 0)
            spread = yes_ask - yes_bid
            
            if spread > 15:  # Relaxed to 15 cent spread to allow more trades
                logger.warning(f"Skipping {market.market_id}: Spread too wide ({spread}¢). Bid: {yes_bid}, Ask: {yes_ask}")
                continue

            # CHECK SPREAD PERCENTAGE (Prevent 3¢ loss on 15¢ contract = 20% loss)
            # Use Ask price as denimator (Cost basis)
            if yes_ask > 0:
                spread_pct = spread / yes_ask
                if spread_pct > 0.10: # Max 10% instant loss allowed
                     logger.warning(f"Skipping {market.market_id}: Spread % too high ({spread_pct:.1%}). Bid: {yes_bid}, Ask: {yes_ask}")
                     continue
                
            # Use Mid-Price for fairer probability estimation, or Last Price as fallback
            if yes_ask > 0 and yes_bid > 0:
                market_prob = ((yes_bid + yes_ask) / 2) / 100
            else:
                market_prob = market_info.get('yes_price', 50) / 100
            
            # Skip markets with extreme prices (too risky for portfolio)
            if market_prob < 0.05 or market_prob > 0.95:
                continue
            
            # Context from Odds API
            odds_context = ""
            if odds_client and active_sports:
                # Dynamic mapping logic
                matching_sports = []
                
                # normalize strings for matching
                market_title_lower = market.title.lower()
                market_cat_lower = market.category.lower()
                
                for sport in active_sports:
                    sport_key = sport['key']
                    sport_title = sport['title'].lower()
                    sport_group = sport.get('group', '').lower()
                    
                    # Direct Match
                    if sport_title in market_title_lower or sport_title in market_cat_lower:
                        matching_sports.append(sport_key)
                        continue
                        
                    # Group Match (e.g. "Basketball" matches "NBA")
                    if sport_group and (sport_group in market_title_lower or sport_group in market_cat_lower):
                        matching_sports.append(sport_key)
                        continue
                
                # Fetch odds for all matching sports (usually just 1)
                # Limit to 3 to avoid spamming API if generic match (e.g. "soccer")
                for sport_key in list(set(matching_sports))[:3]:
                    # Request H2H AND Outrights (Winner) to cover Golf/Futures
                    odds = await odds_client.get_odds(sport_key, markets="h2h,outrights")
                    
                    # Fuzzy match event
                    for event in odds:
                        # Check if team names present in Kalshi title
                        # For Golf/Outrights, home_team might be the Player Name?
                        # In Outrights, 'home_team' is often the event name, and bookmakers have 'outcomes' with player names.
                        
                        is_match = False
                        
                        # 1. Event Name Match
                        if event.home_team in market.title or event.home_team in market_title_lower:
                            is_match = True
                        elif event.away_team and (event.away_team in market.title):
                            is_match = True
                            
                        # 2. Player Name Scan (Crucial for Golf/Tennis Outrights)
                        # We need to peek into bookmakers to find player names if it's an outright market
                        found_outcome_match = False
                        for booky in event.bookmakers:
                                    for outcome in mkt['outcomes']:
                                        if outcome['name'] in market.title:
                                            found_outcome_match = True
                                            odds_context += f"DraftKings Odds ({event.sport_title}): {outcome['name']} {outcome['price']} (Implied: {1/outcome['price']:.2%}); "

            # Get REAL AI prediction using fast analysis
            predicted_prob, confidence, decision_id = await _get_fast_ai_prediction(
                market, xai_client, market_prob, odds_context, db_manager, arbitrage_engine
            )

            opportunity = MarketOpportunity(
                market_id=market.market_id,
                question=market.title,
                answer="Yes",
                current_price=market_prob,
                consensus_probability=predicted_prob,
                confidence=confidence,
                volume=market.volume,
                liquidity=market.volume * market_prob * 100,
                category=market.category,
                decision_id=decision_id
            )
            opportunities.append(opportunity)
            
        except Exception as e:
            logger.error(f"Error processing market {market.market_id}: {e}")
            continue

    return opportunities


async def run_portfolio_optimization(
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient,
    xai_client: XAIClient,
    odds_client: Optional[OddsClient] = None
) -> PortfolioAllocation:
    try:
        # Initialize optimizer
        optimizer = AdvancedPortfolioOptimizer(db_manager, kalshi_client, xai_client, odds_client)
        
        # 🟢 POLL REAL BALANCE (User Request: "poll actual balance")
        try:
             # Default capital baseline (Budget)
             current_capital = settings.trading.daily_budget
             
             # Fetch real balance from Kalshi
             balance_data = await kalshi_client.get_balance()
             if balance_data and 'balance' in balance_data:
                 real_balance_dollars = balance_data['balance'] / 100.0
                 logger.info(f"💰 Account Balance: ${real_balance_dollars:.2f} (Budget: ${current_capital})")
                 
                 # Apply Reality Check: Can't trade more than account balance.
                 # Even in simulation, sizing positions based on $10k when you have $50 leads to failure.
                 if real_balance_dollars < current_capital and not settings.trading.paper_mode:
                     logger.warning(f"⚠️ Adjusting capital base: ${current_capital} -> ${real_balance_dollars:.2f} (capped by actual funds)")
                     current_capital = real_balance_dollars
                 elif real_balance_dollars < current_capital and settings.trading.paper_mode:
                     # In paper mode, we might WANT to simulate a larger bankroll. 
                     # But user asked to "know what to apply kelly criterion to".
                     # Let's log it.
                     pass

             # Update optimizer's capital base for Kelly calculations
             optimizer.total_capital = current_capital
             
        except Exception as e:
            logger.warning(f"Failed to check real balance: {e}")

        # Get markets
        markets = await db_manager.get_eligible_markets(
            volume_min=settings.trading.min_volume,
            max_days_to_expiry=settings.trading.max_time_to_expiry_days
        )
        if not markets:
            logger.warning("No eligible markets for portfolio optimization")
            return optimizer._empty_allocation()
        
        # Convert to opportunities (no immediate trading in batch mode)
        opportunities = await create_market_opportunities_from_markets(
            markets, xai_client, kalshi_client, None, 0, odds_client
        )
        
        if not opportunities:
            logger.warning("No valid opportunities for portfolio optimization")
            return optimizer._empty_allocation()
        
        logger.info(f"Running portfolio optimization on {len(opportunities)} opportunities")
        
        # Optimize portfolio
        allocation = await optimizer.optimize_portfolio(opportunities)
        
        logger.info(
            f"Portfolio optimization complete: "
            f"{len(allocation.allocations)} positions, "
            f"${allocation.total_capital_used:.0f} allocated"
        )
        
        return allocation
        
    except Exception as e:
        logger.error(f"Error in portfolio optimization: {e}")
        return AdvancedPortfolioOptimizer(db_manager, kalshi_client, xai_client)._empty_allocation() 


async def _get_fast_ai_prediction(
    market: Market,
    xai_client: XAIClient,
    market_prob: float,
    odds_context: str,
    db_manager: Optional[DatabaseManager] = None,
    arbitrage_engine: Optional[Any] = None
) -> Tuple[float, float, Optional[int]]:
    """
    Get a quick AI prediction using a cheaper/faster prompt.
    Returns: (predicted_probability, confidence, decision_id)
    """
    
    # 🔍 POLYMARKET ARBITRAGE CHECK
    arbitrage_context = ""
    if arbitrage_engine:
        # Check for fuzzy match
        arb_match = arbitrage_engine.get_polymarket_match(market.title)
        if arb_match:
            arbitrage_context = (
                f"\n\n🚨 ***POLYMARKET ARBITRAGE DATA*** 🚨\n"
                f"MATCHED EVENT: {arb_match.title}\n"
                f"POLYMARKET PRICES: YES {arb_match.yes_price}, NO {arb_match.no_price}\n"
                f"VOLUME: ${arb_match.volume:,.0f}\n"
                f"IMPLIED ODDS: {arb_match.yes_price:.2%}\n"
                f"KALSHI ODDS: {market_prob:.2%}\n"
                f"SPREAD: {abs(arb_match.yes_price - market_prob):.2%}\n"
                f"Strategy: If the spread is > 5%, TRUST POLYMARKET as the 'True Price'. arbitrage away the difference.\n"
            )

    system_prompt = "You are a professional super-forecaster. Analyze the market and provide a probability estimate."
    
    user_prompt = f"""
    Analyze this prediction market:
    Question: {market.title}
    Category: {market.category}
    Current Market Price (Probability): {market_prob:.2%}
    
    External Odds Context: {odds_context}
    {arbitrage_context}
    
    Instructions:
    1. CRITICAL: If 'External Odds Context' or 'Arbitrage Data' is present, YOU MUST ALIGN YOUR PREDICTION WITH IT.
    2. Do not invent reasons to disagree with the market unless you have overwhelming proof.
    3. If the spread (Arbitrage) is > 5%, TRUST THE EXTERNAL ODDS.
    4. If no odds, use general knowledge.
    5. If Sport/Politics and NO odds, be extremely conservative (Confidence < 0.2).
    
    Output JSON:
    {{
        "probability": 0.0 to 1.0,
        "confidence": 0.0 to 1.0,
        "reasoning": "brief explanation"
    }}
    """
    
    try:
        response = await xai_client.get_completion(
            prompt=user_prompt,
            system_prompt=system_prompt,
            response_format={"type": "json_object"}
        )
        
        # Parse response
        import json
        data = json.loads(response)
        
        pred_prob = float(data.get('probability', 0.5))
        confidence = float(data.get('confidence', 0.5))
        reasoning = data.get('reasoning', 'No reasoning provided')
        
        # 🚨 SAFETY LATCH: FORCE ALIGNMENT WITH REAL DATA 🚨
        if arbitrage_engine and arbitrage_context:
            # Extract implied odds from context (hacky but effective)
            # Re-fetch is safer but expensive. Let's rely on the engine we passed.
            arb_match = arbitrage_engine.get_polymarket_match(market.title)
            if arb_match:
                poly_price = arb_match.yes_price
                diff = abs(pred_prob - poly_price)
                if diff > 0.10: # If AI deviates > 10% from Polymarket
                    # Force correction
                    reasoning += f" [AUTO-CORRECTED: AI deviated {diff:.2%} from Polymarket ({poly_price:.2%}). Snapping to reality.]"
                    pred_prob = poly_price # Trust the market
                    confidence = 0.90 # High confidence in the arbitrage

        
        # 💾 Save decision to DB if db_manager provided
        decision_id = None
        if db_manager:
            from src.utils.database import TrainingExample
            from datetime import datetime
            
            # Simple decision log
            decision_id = await db_manager.log_ai_analysis(
                market_id=market.market_id,
                market_question=market.title,
                ai_response=response,
                prompt_used=user_prompt,
                initial_price=market_prob,
                predicted_prob=pred_prob,
                confidence=confidence,
                decision_pnl=0.0 # Unknown yet
            )
            
            # Detailed Training Example
            example = TrainingExample(
                market_ticker=market.market_id,
                market_question=market.title,
                timestamp=datetime.now().isoformat(),
                current_price=market_prob,
                news_context=odds_context + arbitrage_context,
                ai_reasoning=reasoning,
                ai_prediction=pred_prob,
                ai_confidence=confidence,
                decision_id=decision_id
            )
            await db_manager.save_training_example(example)
            
        return pred_prob, confidence, decision_id
        
    except Exception as e:
        import traceback
        # logger is not available in this scope? It's a method on self? No, this is a standalone function?
        # It's a static method or standalone?
        # Let's check imports.
        # It's an instance method? No, indentation suggests it's a standalone function.
        # I need to get a logger.
        from src.utils.logging_setup import get_trading_logger
        l = get_trading_logger("fast_ai_prediction")
        l.error(f"Prediction failed for {market.market_id}: {e}")
        l.error(traceback.format_exc())
        return 0.5, 0.0, None