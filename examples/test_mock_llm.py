"""
Test Mock LLM Client - Learning Example

This script demonstrates:
1. How to use the LLM abstraction layer
2. Async/await patterns
3. Mock LLM client capabilities
4. Zero API costs!

Run with: python examples/test_mock_llm.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm import MockLLMClient, LLMConfig


async def demo_basic_usage():
    """Demo 1: Basic mock LLM usage."""
    print("=" * 60)
    print("DEMO 1: Basic Mock LLM Usage")
    print("=" * 60)

    # Create configuration
    config = LLMConfig(
        provider="mock",
        model="mock-v1",
        temperature=0.0,
        max_tokens=1000
    )

    # Use async context manager (best practice!)
    async with MockLLMClient(config) as client:
        # Generate a response
        response = await client.generate("Implement quicksort algorithm")

        print(f"\nPrompt: 'Implement quicksort algorithm'")
        print(f"\nResponse:")
        print(response.content)
        print(f"\nMetadata:")
        print(f"  - Tokens: {response.total_tokens} ({response.prompt_tokens} + {response.completion_tokens})")
        print(f"  - Latency: {response.latency_ms:.2f}ms")
        print(f"  - Cached: {response.cached}")
        print(f"  - Cost: ${response.cost_usd:.4f}")


async def demo_caching():
    """Demo 2: Response caching."""
    print("\n" + "=" * 60)
    print("DEMO 2: Response Caching")
    print("=" * 60)

    config = LLMConfig(provider="mock", model="mock-v1", enable_cache=True)

    async with MockLLMClient(config) as client:
        prompt = "Explain binary search"

        # First call - not cached
        print("\nFirst call (not cached):")
        response1 = await client.generate(prompt)
        print(f"  Latency: {response1.latency_ms:.2f}ms")
        print(f"  Cached: {response1.cached}")

        # Second call - should be cached!
        print("\nSecond call (should be cached):")
        response2 = await client.generate(prompt)
        print(f"  Latency: {response2.latency_ms:.2f}ms")
        print(f"  Cached: {response2.cached}")

        # Safe division - handle zero latency case
        if response2.latency_ms > 0:
            speedup = response1.latency_ms / response2.latency_ms
            print(f"\nSpeedup: {speedup:.1f}x faster!")
        else:
            print(f"\nSpeedup: Instant! (cached response took ~0ms)")


async def demo_streaming():
    """Demo 3: Streaming generation."""
    print("\n" + "=" * 60)
    print("DEMO 3: Streaming Generation")
    print("=" * 60)

    config = LLMConfig(provider="mock", model="mock-v1")

    async with MockLLMClient(config) as client:
        print("\nStreaming response: ", end="", flush=True)

        async for chunk in client.stream_generate("Write a test for quicksort"):
            print(chunk, end="", flush=True)

        print("\n\nDone streaming!")


async def demo_multiple_prompts():
    """Demo 4: Different prompt types."""
    print("\n" + "=" * 60)
    print("DEMO 4: Smart Pattern Matching")
    print("=" * 60)

    config = LLMConfig(provider="mock", model="mock-v1")

    prompts = [
        "Extract problem definition from this algorithm",
        "Generate tests for the sorting function",
        "Critique this implementation for potential bugs"
    ]

    async with MockLLMClient(config) as client:
        for i, prompt in enumerate(prompts, 1):
            print(f"\n{i}. Prompt: {prompt}")
            response = await client.generate(prompt)
            print(f"   Response preview: {response.content[:100]}...")


async def demo_stats():
    """Demo 5: Client statistics."""
    print("\n" + "=" * 60)
    print("DEMO 5: Usage Statistics")
    print("=" * 60)

    config = LLMConfig(provider="mock", model="mock-v1")

    client = MockLLMClient(config)
    await client.initialize()

    # Make several calls
    for i in range(5):
        await client.generate(f"Test prompt {i}")

    # Get stats
    stats = client.get_stats()
    print(f"\nClient Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    await client.cleanup()


async def main():
    """Run all demos."""
    print("\n🎓 MOCK LLM CLIENT - LEARNING DEMOS")
    print("Zero API cost! Learn professional LLM patterns!\n")

    await demo_basic_usage()
    await demo_caching()
    await demo_streaming()
    await demo_multiple_prompts()
    await demo_stats()

    print("\n" + "=" * 60)
    print("✅ ALL DEMOS COMPLETE!")
    print("=" * 60)
    print("\n💡 Key Takeaways:")
    print("1. LLM abstraction layer works with ANY provider")
    print("2. Async/await enables non-blocking operations")
    print("3. Caching dramatically improves performance")
    print("4. Streaming provides better UX for long responses")
    print("5. Mock client teaches patterns without API costs!")
    print("\nNext: Swap to real LLM with ONE config change 🚀")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
