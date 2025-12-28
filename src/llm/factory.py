"""
LLM Client Factory

Provides factory pattern for creating LLM clients from configuration.
Enables config-driven provider switching without code changes.

Key Learning Concepts:
- Factory pattern (creates objects based on config)
- Dependency injection (decouple creation from usage)
- Plugin architecture (easy to add new providers)

Why This Matters:
- Switch providers with config change, not code change
- Centralized client creation logic
- Easy to extend with new providers
"""

from typing import Optional
from .base import BaseLLMClient, LLMConfig
from .mock_client import MockLLMClient


class ClientFactory:
    """
    Factory for creating LLM clients based on configuration.

    Usage:
        # From config object
        config = LLMConfig(provider="mock", model="mock-v1")
        client = ClientFactory.create(config)

        # From UnifiedConfig
        from src.config import get_config
        config = get_config()
        client = ClientFactory.create_from_unified_config(config)

        # Use client
        async with client:
            response = await client.generate("Hello!")
    """

    @staticmethod
    def create(config: LLMConfig) -> BaseLLMClient:
        """
        Create LLM client from configuration.

        Args:
            config: LLMConfig with provider and settings

        Returns:
            Concrete LLM client instance

        Raises:
            ValueError: If provider is not supported
        """
        provider = config.provider.lower()

        if provider == "mock":
            return MockLLMClient(config)

        elif provider == "claude":
            # NOT IMPLEMENTED - Claude client placeholder exists but not implemented
            # Import here to avoid dependency if not using Claude
            try:
                from .claude_client import ClaudeClient  # type: ignore[attr-defined]
                return ClaudeClient(config)  # type: ignore[possibly-undefined]
            except (ImportError, AttributeError):
                raise ValueError(
                    "Claude provider not implemented yet. "
                    "See src/llm/claude_client.py for implementation guide. "
                    "Will require: pip install anthropic"
                )

        elif provider == "openai":
            # NOT IMPLEMENTED - OpenAI client placeholder exists but not implemented
            # Import here to avoid dependency if not using OpenAI
            try:
                from .openai_client import OpenAIClient  # type: ignore[attr-defined]
                return OpenAIClient(config)  # type: ignore[possibly-undefined]
            except (ImportError, AttributeError):
                raise ValueError(
                    "OpenAI provider not implemented yet. "
                    "See src/llm/openai_client.py for implementation guide. "
                    "Will require: pip install openai"
                )

        elif provider == "ollama":
            # NOT IMPLEMENTED - Ollama client not created yet
            # Import here to avoid dependency if not using Ollama
            try:
                from .ollama_client import OllamaClient  # type: ignore[import-not-found,attr-defined]
                return OllamaClient(config)  # type: ignore[possibly-undefined]
            except (ImportError, AttributeError):
                raise ValueError(
                    "Ollama provider not implemented yet. "
                    "Create src/llm/ollama_client.py following BaseLLMClient interface. "
                    "Will require: pip install ollama"
                )

        else:
            supported = ["mock", "claude", "openai", "ollama"]
            raise ValueError(
                f"Unsupported provider: {provider}. "
                f"Supported providers: {', '.join(supported)}"
            )

    @staticmethod
    def create_from_unified_config(
        unified_config,
        provider: Optional[str] = None
    ) -> BaseLLMClient:
        """
        Create LLM client from UnifiedConfig.

        Args:
            unified_config: UnifiedConfig instance
            provider: Override provider (uses config default if None)

        Returns:
            Concrete LLM client instance
        """
        llm_config = unified_config.get_llm_config(provider=provider)
        return ClientFactory.create(llm_config)

    @staticmethod
    def list_available_providers() -> list[str]:
        """
        List all available LLM providers.

        Returns:
            List of provider names that can be used
        """
        available = ["mock"]  # Mock is always available

        # Check for optional providers
        try:
            import anthropic  # type: ignore[import-not-found]
            available.append("claude")
        except ImportError:
            pass

        try:
            import openai  # type: ignore[import-not-found]
            available.append("openai")
        except ImportError:
            pass

        try:
            import ollama  # type: ignore[import-not-found]
            available.append("ollama")
        except ImportError:
            pass

        return available

    @staticmethod
    def get_provider_info(provider: str) -> dict:
        """
        Get information about a provider.

        Args:
            provider: Provider name

        Returns:
            Dictionary with provider information
        """
        info = {
            "mock": {
                "name": "Mock LLM",
                "description": "Simulated LLM for testing and development (zero cost)",
                "requires_api_key": False,
                "cost": "Free",
                "package": None,
            },
            "claude": {
                "name": "Anthropic Claude",
                "description": "Claude models (Sonnet, Opus, Haiku)",
                "requires_api_key": True,
                "cost": "Paid API",
                "package": "anthropic",
            },
            "openai": {
                "name": "OpenAI",
                "description": "GPT models (GPT-4, GPT-3.5)",
                "requires_api_key": True,
                "cost": "Paid API",
                "package": "openai",
            },
            "ollama": {
                "name": "Ollama",
                "description": "Local LLM server (Llama, Mistral, etc.)",
                "requires_api_key": False,
                "cost": "Free (local)",
                "package": "ollama",
            },
        }

        return info.get(provider.lower(), {"error": f"Unknown provider: {provider}"})
