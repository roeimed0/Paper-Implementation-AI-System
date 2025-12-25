"""
Rate Limiter - Token Bucket Algorithm

Prevents exceeding API rate limits by controlling request frequency.

Key Learning Concepts:
- Token bucket algorithm (industry standard)
- Async/await for waiting
- Time-based resource management
- Prevents API throttling

Used by: AWS, Stripe, GitHub, all major APIs
"""

import asyncio
import time
from typing import Optional


class RateLimiter:
    """
    Token bucket rate limiter.

    Algorithm:
    1. Bucket holds tokens (up to max capacity)
    2. Tokens refill at constant rate
    3. Each request consumes tokens
    4. If no tokens available, wait until refill

    Example:
        # Allow 50 requests per minute
        limiter = RateLimiter(requests_per_minute=50)

        async def call_api():
            await limiter.acquire()  # Waits if rate limit hit
            response = await api_call()
            return response
    """

    def __init__(
        self,
        requests_per_minute: int = 50,
        tokens_per_minute: int = 100000,
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Max requests per minute
            tokens_per_minute: Max tokens per minute (for LLMs)
        """
        # Request-based limiting
        self.requests_per_minute = requests_per_minute
        self.request_interval = 60.0 / requests_per_minute  # Seconds between requests
        self.request_tokens = requests_per_minute  # Current tokens
        self.request_capacity = requests_per_minute  # Max tokens
        self.last_request_update = time.time()

        # Token-based limiting (for LLM token usage)
        self.tokens_per_minute = tokens_per_minute
        self.token_interval = 60.0 / tokens_per_minute  # Seconds per token
        self.token_tokens = tokens_per_minute  # Current tokens
        self.token_capacity = tokens_per_minute  # Max tokens
        self.last_token_update = time.time()

        # Lock for thread safety
        self._lock = asyncio.Lock()

    def _refill_request_tokens(self) -> None:
        """
        Refill request tokens based on elapsed time.

        This is the core of the token bucket algorithm!
        """
        now = time.time()
        elapsed = now - self.last_request_update

        # Calculate tokens to add (based on elapsed time)
        tokens_to_add = elapsed / self.request_interval

        # Add tokens, but don't exceed capacity
        self.request_tokens = min(
            self.request_capacity,
            self.request_tokens + tokens_to_add
        )

        self.last_request_update = now

    def _refill_token_tokens(self, tokens_needed: int) -> None:
        """Refill token-usage tokens."""
        now = time.time()
        elapsed = now - self.last_token_update

        tokens_to_add = elapsed / self.token_interval

        self.token_tokens = min(
            self.token_capacity,
            self.token_tokens + tokens_to_add
        )

        self.last_token_update = now

    async def acquire(self, tokens: int = 0) -> None:
        """
        Acquire permission to make a request.

        Waits if rate limit would be exceeded.

        Args:
            tokens: Number of LLM tokens for this request (optional)

        Example:
            await limiter.acquire()  # Request-based limiting
            await limiter.acquire(tokens=1500)  # Also check token limit
        """
        async with self._lock:
            # Check request-based limit
            self._refill_request_tokens()

            if self.request_tokens < 1.0:
                # Need to wait for tokens to refill
                wait_time = (1.0 - self.request_tokens) * self.request_interval
                await asyncio.sleep(wait_time)
                self._refill_request_tokens()

            # Consume one request token
            self.request_tokens -= 1.0

            # Check token-based limit (if provided)
            if tokens > 0:
                self._refill_token_tokens(tokens)

                if self.token_tokens < tokens:
                    # Need to wait for enough tokens
                    wait_time = (tokens - self.token_tokens) * self.token_interval
                    await asyncio.sleep(wait_time)
                    self._refill_token_tokens(tokens)

                # Consume tokens
                self.token_tokens -= tokens

    def get_stats(self) -> dict:
        """
        Get current rate limiter statistics.

        Returns:
            Dictionary with current state
        """
        self._refill_request_tokens()
        self._refill_token_tokens(0)

        return {
            "request_tokens_available": round(self.request_tokens, 2),
            "request_capacity": self.request_capacity,
            "token_tokens_available": round(self.token_tokens, 2),
            "token_capacity": self.token_capacity,
            "requests_per_minute": self.requests_per_minute,
            "tokens_per_minute": self.tokens_per_minute,
        }

    def reset(self) -> None:
        """Reset rate limiter to full capacity."""
        self.request_tokens = self.request_capacity
        self.token_tokens = self.token_capacity
        self.last_request_update = time.time()
        self.last_token_update = time.time()


class NoOpRateLimiter(RateLimiter):
    """
    No-op rate limiter for testing/mocks.

    Never limits - all requests pass through immediately.
    """

    async def acquire(self, tokens: int = 0) -> None:
        """Allow all requests immediately."""
        pass  # Do nothing - no limiting!
