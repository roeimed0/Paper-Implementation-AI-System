"""
Unified Configuration System

Loads configuration from multiple sources (priority order):
1. Environment variables (highest priority)
2. YAML configuration files
3. Default values (lowest priority)

Key Learning Concepts:
- Configuration management patterns
- Environment variable handling
- YAML file parsing
- Singleton pattern (optional)
- Type safety with Pydantic
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from src.llm.base import LLMConfig


class CacheConfig(BaseModel):
    """Cache configuration."""
    enable: bool = True
    ttl_seconds: int = 86400
    directory: str = "data/cache"


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "json"
    directory: str = "data/logs"
    log_prompts: bool = True
    log_responses: bool = True
    log_token_usage: bool = True


class CostTrackingConfig(BaseModel):
    """Cost tracking configuration."""
    enable: bool = True
    budget_usd: float = 500.00
    alert_threshold: float = 0.8
    pricing: Dict[str, Dict[str, float]] = Field(default_factory=dict)


class UnifiedConfig:
    """
    Unified configuration for the entire system.

    Loads from:
    - config/model_config.yaml
    - Environment variables (.env)

    Usage:
        config = UnifiedConfig()
        llm_config = config.get_llm_config()

        # Or use the singleton
        config = get_config()
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to YAML config file (default: config/model_config.yaml)
        """
        # Load environment variables from .env file
        load_dotenv()

        # Determine config file path
        if config_path is None:
            # Find project root (where .env.example is)
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent
            config_path = project_root / "config" / "model_config.yaml"

        self.config_path = config_path
        self._config_data: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._config_data = yaml.safe_load(f) or {}

    def get_provider(self) -> str:
        """
        Get the LLM provider to use.

        Can be overridden by LLM_PROVIDER environment variable.

        Returns:
            Provider name: "mock", "claude", "openai", or "ollama"
        """
        return os.getenv("LLM_PROVIDER", self._config_data.get("provider", "mock"))

    def get_llm_config(self, provider: Optional[str] = None) -> LLMConfig:
        """
        Get LLM configuration for the specified provider.

        Args:
            provider: Provider name (default: uses configured provider)

        Returns:
            LLMConfig instance ready to use

        Example:
            config = UnifiedConfig()
            llm_config = config.get_llm_config()  # Uses provider from YAML
            # OR
            llm_config = config.get_llm_config("claude")  # Force Claude
        """
        if provider is None:
            provider = self.get_provider()

        # Get provider-specific config
        provider_config = self._config_data.get(provider, {})

        # Get API key from environment (if needed)
        api_key = None
        if provider == "claude":
            api_key = os.getenv("CLAUDE_API_KEY")
        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")

        # Build LLMConfig
        return LLMConfig(
            provider=provider,
            model=provider_config.get("model", f"{provider}-default"),
            api_key=api_key,
            base_url=provider_config.get("base_url"),
            temperature=provider_config.get("temperature", 0.0),
            max_tokens=provider_config.get("max_tokens", 4000),
            top_p=provider_config.get("top_p"),
            requests_per_minute=provider_config.get("requests_per_minute", 50),
            tokens_per_minute=provider_config.get("tokens_per_minute", 100000),
            enable_cache=self.get_cache_config().enable,
            cache_ttl_seconds=self.get_cache_config().ttl_seconds,
            max_retries=provider_config.get("max_retries", 3),
            retry_delay_seconds=provider_config.get("retry_delay_seconds", 1.0),
        )

    def get_cache_config(self) -> CacheConfig:
        """Get cache configuration."""
        cache_data = self._config_data.get("cache", {})
        return CacheConfig(**cache_data)

    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration."""
        logging_data = self._config_data.get("logging", {})
        return LoggingConfig(**logging_data)

    def get_cost_tracking_config(self) -> CostTrackingConfig:
        """Get cost tracking configuration."""
        cost_data = self._config_data.get("cost_tracking", {})
        return CostTrackingConfig(**cost_data)

    def get_mock_settings(self) -> Dict[str, Any]:
        """
        Get mock-specific settings.

        Returns settings like simulate_delay, error_rate, etc.
        """
        return self._config_data.get("mock", {})

    def reload(self) -> None:
        """Reload configuration from file."""
        self._load_config()

    def __repr__(self) -> str:
        return f"UnifiedConfig(provider={self.get_provider()}, config_path={self.config_path})"


# Singleton instance (optional - use if you want global config)
_global_config: Optional[UnifiedConfig] = None


def get_config(force_reload: bool = False) -> UnifiedConfig:
    """
    Get global configuration instance (singleton pattern).

    Args:
        force_reload: Force reload config from file

    Returns:
        UnifiedConfig singleton instance

    Example:
        from src.config import get_config

        config = get_config()
        llm_config = config.get_llm_config()
    """
    global _global_config

    if _global_config is None or force_reload:
        _global_config = UnifiedConfig()

    return _global_config
