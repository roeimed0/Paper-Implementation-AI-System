"""
Test Configuration System

Demonstrates how to use the unified configuration system
to switch between different LLM providers.

Run with: python examples/test_config.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import UnifiedConfig, get_config


def demo_basic_config():
    """Demo 1: Load and inspect configuration."""
    print("=" * 60)
    print("DEMO 1: Basic Configuration Loading")
    print("=" * 60)

    config = UnifiedConfig()

    print(f"\nConfig file: {config.config_path}")
    print(f"Current provider: {config.get_provider()}")

    llm_config = config.get_llm_config()
    print(f"\nLLM Configuration:")
    print(f"  Provider: {llm_config.provider}")
    print(f"  Model: {llm_config.model}")
    print(f"  Temperature: {llm_config.temperature}")
    print(f"  Max Tokens: {llm_config.max_tokens}")
    print(f"  Cache Enabled: {llm_config.enable_cache}")


def demo_provider_switching():
    """Demo 2: Switch between providers."""
    print("\n" + "=" * 60)
    print("DEMO 2: Provider Switching")
    print("=" * 60)

    config = UnifiedConfig()

    providers = ["mock", "claude", "openai", "ollama"]

    for provider in providers:
        print(f"\n{provider.upper()} Configuration:")
        try:
            llm_config = config.get_llm_config(provider)
            print(f"  Model: {llm_config.model}")
            print(f"  Temperature: {llm_config.temperature}")
            print(f"  Max Tokens: {llm_config.max_tokens}")
            print(f"  Has API Key: {llm_config.api_key is not None}")
        except Exception as e:
            print(f"  Error: {e}")


def demo_cache_config():
    """Demo 3: Cache configuration."""
    print("\n" + "=" * 60)
    print("DEMO 3: Cache Configuration")
    print("=" * 60)

    config = UnifiedConfig()
    cache_config = config.get_cache_config()

    print(f"\nCache Settings:")
    print(f"  Enabled: {cache_config.enable}")
    print(f"  TTL: {cache_config.ttl_seconds}s ({cache_config.ttl_seconds / 3600:.1f} hours)")
    print(f"  Directory: {cache_config.directory}")


def demo_logging_config():
    """Demo 4: Logging configuration."""
    print("\n" + "=" * 60)
    print("DEMO 4: Logging Configuration")
    print("=" * 60)

    config = UnifiedConfig()
    log_config = config.get_logging_config()

    print(f"\nLogging Settings:")
    print(f"  Level: {log_config.level}")
    print(f"  Format: {log_config.format}")
    print(f"  Directory: {log_config.directory}")
    print(f"  Log Prompts: {log_config.log_prompts}")
    print(f"  Log Responses: {log_config.log_responses}")
    print(f"  Log Token Usage: {log_config.log_token_usage}")


def demo_cost_tracking():
    """Demo 5: Cost tracking configuration."""
    print("\n" + "=" * 60)
    print("DEMO 5: Cost Tracking Configuration")
    print("=" * 60)

    config = UnifiedConfig()
    cost_config = config.get_cost_tracking_config()

    print(f"\nCost Tracking Settings:")
    print(f"  Enabled: {cost_config.enable}")
    print(f"  Budget: ${cost_config.budget_usd:.2f}")
    print(f"  Alert Threshold: {cost_config.alert_threshold * 100:.0f}%")

    print(f"\n  Pricing (per million tokens):")
    for model, prices in cost_config.pricing.items():
        print(f"    {model}:")
        print(f"      Input: ${prices['input']:.2f}")
        print(f"      Output: ${prices['output']:.2f}")


def demo_singleton():
    """Demo 6: Singleton pattern."""
    print("\n" + "=" * 60)
    print("DEMO 6: Singleton Pattern")
    print("=" * 60)

    # Get config twice - should be same instance
    config1 = get_config()
    config2 = get_config()

    print(f"\nSame instance? {config1 is config2}")
    print(f"Provider: {config1.get_provider()}")

    # Force reload
    config3 = get_config(force_reload=True)
    print(f"After reload - same instance? {config1 is config3}")


def demo_mock_settings():
    """Demo 7: Mock-specific settings."""
    print("\n" + "=" * 60)
    print("DEMO 7: Mock-Specific Settings")
    print("=" * 60)

    config = UnifiedConfig()
    mock_settings = config.get_mock_settings()

    print(f"\nMock LLM Settings:")
    print(f"  Simulate Delay: {mock_settings.get('simulate_delay')}")
    print(f"  Min Delay: {mock_settings.get('min_delay_ms')}ms")
    print(f"  Max Delay: {mock_settings.get('max_delay_ms')}ms")
    print(f"  Error Rate: {mock_settings.get('error_rate') * 100:.1f}%")


def main():
    """Run all configuration demos."""
    print("\n🔧 CONFIGURATION SYSTEM DEMOS")
    print("Learn how to manage LLM settings!")

    demo_basic_config()
    demo_provider_switching()
    demo_cache_config()
    demo_logging_config()
    demo_cost_tracking()
    demo_singleton()
    demo_mock_settings()

    print("\n" + "=" * 60)
    print("✅ ALL CONFIGURATION DEMOS COMPLETE!")
    print("=" * 60)

    print("\n💡 Key Takeaways:")
    print("1. Configuration loaded from YAML + environment variables")
    print("2. Switch providers by changing ONE line in YAML")
    print("3. Environment variables override YAML settings")
    print("4. Type-safe configuration with Pydantic")
    print("5. Singleton pattern for global config access")

    print("\n🎯 To switch from mock to real LLM:")
    print("   1. Edit config/model_config.yaml")
    print("   2. Change provider: 'mock' → 'claude'")
    print("   3. Set CLAUDE_API_KEY in .env")
    print("   4. Done! No code changes needed!")


if __name__ == "__main__":
    main()
