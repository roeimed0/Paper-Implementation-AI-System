"""
LLM Client Abstraction Layer

Provides provider-agnostic interface for LLM interactions.
Supports: Mock, Claude, OpenAI, Ollama (future).
"""

from .base import (
    BaseLLMClient,
    LLMConfig,
    LLMResponse,
    # Exceptions
    LLMError,
    RetryableError,
    PermanentError,
    RateLimitError,
    TimeoutError,
    NetworkError,
    AuthenticationError,
    ValidationError,
    ContentPolicyError,
    ModelNotFoundError,
    ContextLengthError,
)
from .mock_client import MockLLMClient
from .factory import ClientFactory

__all__ = [
    # Core classes
    "BaseLLMClient",
    "LLMConfig",
    "LLMResponse",
    "MockLLMClient",
    "ClientFactory",
    # Exceptions
    "LLMError",
    "RetryableError",
    "PermanentError",
    "RateLimitError",
    "TimeoutError",
    "NetworkError",
    "AuthenticationError",
    "ValidationError",
    "ContentPolicyError",
    "ModelNotFoundError",
    "ContextLengthError",
]
