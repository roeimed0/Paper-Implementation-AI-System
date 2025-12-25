"""
Base LLM Client - Abstract Interface

Defines the contract that ALL LLM clients must implement.
This is the core abstraction that makes the system provider-agnostic.

Key Design Patterns:
- Abstract Base Class (ABC) for interface definition
- Async/await for non-blocking I/O
- Pydantic models for type safety
- Context manager support for resource cleanup
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """
    Configuration for LLM clients.

    This is provider-agnostic - different clients use different fields.
    """
    provider: str = Field(..., description="LLM provider: 'mock', 'claude', 'openai', 'ollama'")
    model: str = Field(..., description="Model identifier (e.g., 'claude-sonnet-4-5-20250929')")
    api_key: Optional[str] = Field(None, description="API key (not needed for mock/ollama)")
    base_url: Optional[str] = Field(None, description="Custom API endpoint (for Ollama)")

    # Generation parameters
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=4000, gt=0, description="Maximum tokens to generate")
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Nucleus sampling")

    # Rate limiting
    requests_per_minute: int = Field(default=50, gt=0)
    tokens_per_minute: int = Field(default=100000, gt=0)

    # Caching
    enable_cache: bool = Field(default=True, description="Enable response caching")
    cache_ttl_seconds: int = Field(default=86400, description="Cache time-to-live")

    # Retry logic
    max_retries: int = Field(default=3, ge=0, description="Max retry attempts on failure")
    retry_delay_seconds: float = Field(default=1.0, gt=0, description="Base delay between retries")

    class Config:
        frozen = True  # Immutable after creation


@dataclass
class LLMResponse:
    """
    Standardized response from any LLM client.

    Contains the generated text plus metadata for tracking and debugging.
    """
    content: str
    model: str
    provider: str

    # Token usage
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    # Metadata
    latency_ms: float
    timestamp: datetime
    cached: bool = False

    # Raw response (for debugging)
    raw_response: Optional[Dict[str, Any]] = None

    @property
    def cost_usd(self) -> float:
        """
        Estimate cost in USD.

        Note: Mock client returns 0.0
        Real clients calculate based on provider pricing.
        """
        return 0.0  # Override in subclasses


class BaseLLMClient(ABC):
    """
    Abstract base class for all LLM clients.

    Design Philosophy:
    - All LLM interactions go through this interface
    - Concrete clients (Mock, Claude, OpenAI) implement this contract
    - Makes swapping providers a config change, not a code change

    Usage:
        config = LLMConfig(provider="mock", model="mock-v1")
        async with MockLLMClient(config) as client:
            response = await client.generate("Hello, world!")
            print(response.content)
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize LLM client with configuration.

        Args:
            config: LLMConfig instance with provider-specific settings
        """
        self.config = config
        self._initialized = False

    async def __aenter__(self):
        """Async context manager entry - initialize resources."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup resources."""
        await self.cleanup()
        return False  # Don't suppress exceptions

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize client resources (connections, rate limiters, cache).

        Called automatically when using async context manager.
        Must be idempotent (safe to call multiple times).
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """
        Cleanup client resources.

        Called automatically when exiting async context manager.
        Should gracefully handle partial initialization.
        """
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text completion from prompt.

        This is the primary interface for LLM interactions.

        Args:
            prompt: User prompt (main input)
            system_prompt: System/instruction prompt (optional)
            temperature: Override config temperature
            max_tokens: Override config max_tokens
            stop_sequences: Sequences that stop generation
            **kwargs: Provider-specific parameters

        Returns:
            LLMResponse with generated text and metadata

        Raises:
            LLMError: On generation failure (after retries)
            RateLimitError: If rate limit exceeded
        """
        pass

    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using provider's tokenizer.

        Args:
            text: Text to tokenize

        Returns:
            Number of tokens
        """
        pass

    @abstractmethod
    async def stream_generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        """
        Generate text with streaming (yields chunks as they arrive).

        This is optional - not all providers/use-cases need it.
        Mock client will simulate streaming.

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            **kwargs: Provider-specific parameters

        Yields:
            str: Text chunks as they're generated
        """
        pass

    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about the provider and model.

        Returns:
            Dictionary with provider metadata
        """
        return {
            "provider": self.config.provider,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }


class LLMError(Exception):
    """Base exception for LLM client errors."""
    pass


class RateLimitError(LLMError):
    """Raised when rate limit is exceeded."""
    pass


class ValidationError(LLMError):
    """Raised when response validation fails."""
    pass


class TimeoutError(LLMError):
    """Raised when request times out."""
    pass
