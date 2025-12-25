"""
LLM Client Abstraction Layer

Provides provider-agnostic interface for LLM interactions.
Supports: Mock, Claude, OpenAI, Ollama (future).
"""

from .base import BaseLLMClient, LLMConfig, LLMResponse
from .mock_client import MockLLMClient

__all__ = [
    "BaseLLMClient",
    "LLMConfig",
    "LLMResponse",
    "MockLLMClient",
]
