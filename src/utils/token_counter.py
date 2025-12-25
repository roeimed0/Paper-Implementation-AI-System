"""
Token Counter - Accurate Token Counting & Cost Estimation

Counts tokens using tiktoken (OpenAI's tokenizer) for cost estimation.

Key Learning Concepts:
- Token counting for cost management
- Different encodings for different models
- Cost calculation from token usage
- Budget tracking

Why This Matters:
- LLMs charge per token, not per character
- Accurate counting prevents surprise bills
- Essential for production cost management
"""

import tiktoken
from typing import Dict, Optional


class TokenCounter:
    """
    Count tokens and estimate costs for LLM usage.

    Uses tiktoken for accurate token counting.
    Supports Claude (approximation) and GPT models.

    Example:
        counter = TokenCounter(model="claude-sonnet-4-5")

        tokens = counter.count_tokens("Hello, world!")
        cost = counter.estimate_cost(
            prompt_tokens=100,
            completion_tokens=500
        )
        print(f"Cost: ${cost:.4f}")
    """

    # Pricing per million tokens (USD)
    PRICING = {
        "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
        "claude-opus-4-5": {"input": 15.00, "output": 75.00},
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "mock": {"input": 0.00, "output": 0.00},
    }

    def __init__(self, model: str = "mock"):
        """
        Initialize token counter.

        Args:
            model: Model name for pricing lookup
        """
        self.model = model

        # Get encoding for tokenization
        # Claude uses same tokenizer as GPT-4 (approximation)
        if "claude" in model.lower():
            encoding_name = "cl100k_base"  # GPT-4 encoding
        elif "gpt-4" in model.lower():
            encoding_name = "cl100k_base"
        elif "gpt-3.5" in model.lower():
            encoding_name = "cl100k_base"
        else:
            encoding_name = "cl100k_base"  # Default

        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception:
            # Fallback to simple approximation
            self.encoding = None

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count

        Returns:
            Number of tokens

        Example:
            tokens = counter.count_tokens("Hello, world!")
            # Returns: ~4 tokens
        """
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # Fallback: approximate 1 token per 4 characters
            return max(1, len(text) // 4)

    def estimate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: Optional[str] = None
    ) -> float:
        """
        Estimate cost in USD for token usage.

        Args:
            prompt_tokens: Input tokens
            completion_tokens: Output tokens
            model: Model name (optional, uses instance model if not provided)

        Returns:
            Estimated cost in USD

        Example:
            cost = counter.estimate_cost(
                prompt_tokens=1000,
                completion_tokens=500
            )
            # For Claude Sonnet: (1000 * $3 + 500 * $15) / 1M = $0.0105
        """
        model_name = model or self.model

        # Get pricing for this model
        pricing = self.PRICING.get(model_name, {"input": 0.0, "output": 0.0})

        # Calculate cost (pricing is per million tokens)
        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    def get_pricing(self, model: Optional[str] = None) -> Dict[str, float]:
        """
        Get pricing for a model.

        Args:
            model: Model name (optional)

        Returns:
            Dictionary with input/output pricing per million tokens
        """
        model_name = model or self.model
        return self.PRICING.get(model_name, {"input": 0.0, "output": 0.0})


class BudgetTracker:
    """
    Track spending against a budget.

    Prevents exceeding cost limits.

    Example:
        tracker = BudgetTracker(budget_usd=100.00)

        tracker.add_usage(prompt_tokens=1000, completion_tokens=500, model="claude-sonnet-4-5")

        if tracker.is_over_budget():
            print("Budget exceeded!")

        print(f"Spent: ${tracker.get_total_cost():.2f}")
    """

    def __init__(self, budget_usd: float = 500.00, alert_threshold: float = 0.8):
        """
        Initialize budget tracker.

        Args:
            budget_usd: Total budget in USD
            alert_threshold: Alert when this fraction of budget is used (0.8 = 80%)
        """
        self.budget_usd = budget_usd
        self.alert_threshold = alert_threshold
        self.total_cost = 0.0
        self.usage_log = []
        self.counter = TokenCounter()

    def add_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
        metadata: Optional[Dict] = None
    ) -> float:
        """
        Add token usage and calculate cost.

        Args:
            prompt_tokens: Input tokens used
            completion_tokens: Output tokens used
            model: Model name
            metadata: Optional metadata to log

        Returns:
            Cost of this usage in USD
        """
        cost = self.counter.estimate_cost(prompt_tokens, completion_tokens, model)
        self.total_cost += cost

        # Log usage
        self.usage_log.append({
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "model": model,
            "cost": cost,
            "metadata": metadata or {}
        })

        return cost

    def get_total_cost(self) -> float:
        """Get total cost so far."""
        return self.total_cost

    def get_remaining_budget(self) -> float:
        """Get remaining budget."""
        return max(0.0, self.budget_usd - self.total_cost)

    def get_budget_used_percent(self) -> float:
        """Get percentage of budget used."""
        return (self.total_cost / self.budget_usd) * 100 if self.budget_usd > 0 else 0.0

    def is_over_budget(self) -> bool:
        """Check if over budget."""
        return self.total_cost >= self.budget_usd

    def should_alert(self) -> bool:
        """Check if should alert (threshold exceeded)."""
        return self.total_cost >= (self.budget_usd * self.alert_threshold)

    def get_stats(self) -> Dict:
        """Get budget statistics."""
        return {
            "budget_usd": self.budget_usd,
            "total_cost": round(self.total_cost, 4),
            "remaining": round(self.get_remaining_budget(), 4),
            "percent_used": round(self.get_budget_used_percent(), 2),
            "total_requests": len(self.usage_log),
            "over_budget": self.is_over_budget(),
            "should_alert": self.should_alert(),
        }
