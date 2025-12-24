from dataclasses import dataclass
from typing import Optional

@dataclass
class TradingDecision:
    """Represents an AI trading decision."""
    action: str  # "buy", "sell", "hold" / "SKIP"
    side: str    # "yes", "no"
    confidence: float  # 0.0 to 1.0
    reasoning: str
    limit_price: Optional[int] = None # The limit price for the order in cents (XAI specific)
    position_size_pct: float = 0.0  # Percentage of available capital to use (OpenAI specific)
    max_price: Optional[float] = None  # Maximum price willing to pay
    stop_loss: Optional[float] = None  # Stop loss price
    take_profit: Optional[float] = None  # Take profit price
    expected_return: Optional[float] = None  # Expected return percentage
    risk_assessment: Optional[str] = None  # Risk level assessment


@dataclass
class AIDecisionResult:
    """Contains the decision and full context for training data."""
    decision: Optional[TradingDecision]
    prompt: str
    response_text: str
    model_name: str

    def __bool__(self):
        return self.decision is not None
