"""
Test Utility Modules

Demonstrates rate limiting, token counting, caching, and logging.

Run with: python examples/test_utils.py
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    RateLimiter,
    TokenCounter,
    BudgetTracker,
    CacheManager,
    get_logger,
)


async def demo_rate_limiter():
    """Demo 1: Rate limiter with token bucket algorithm."""
    print("=" * 60)
    print("DEMO 1: Rate Limiter (Token Bucket Algorithm)")
    print("=" * 60)

    # Allow 5 requests per minute (fast for demo)
    limiter = RateLimiter(requests_per_minute=5)

    print("\nMaking 7 requests (limit is 5/minute)...")
    print("First 5 should be fast, then we'll see waiting...")

    for i in range(7):
        start = time.time()
        await limiter.acquire()
        elapsed = (time.time() - start) * 1000

        print(f"  Request {i+1}: {elapsed:.0f}ms wait")

        # Show stats
        if i == 4:  # After 5th request
            stats = limiter.get_stats()
            print(f"\n  Stats after 5 requests: {stats['request_tokens_available']:.2f} tokens left")

    print("\n✅ Rate limiter prevents API throttling!")


def demo_token_counter():
    """Demo 2: Token counting and cost estimation."""
    print("\n" + "=" * 60)
    print("DEMO 2: Token Counter & Cost Estimation")
    print("=" * 60)

    counter = TokenCounter(model="claude-sonnet-4-5")

    # Count tokens in text
    text = "Hello, world! This is a test of the token counter."
    tokens = counter.count_tokens(text)

    print(f"\nText: '{text}'")
    print(f"Tokens: {tokens}")
    print(f"Characters: {len(text)}")
    print(f"Ratio: ~{len(text) / tokens:.1f} chars per token")

    # Estimate costs
    prompt_tokens = 1000
    completion_tokens = 500

    cost = counter.estimate_cost(prompt_tokens, completion_tokens)

    print(f"\nCost Estimation:")
    print(f"  Prompt tokens: {prompt_tokens}")
    print(f"  Completion tokens: {completion_tokens}")
    print(f"  Total tokens: {prompt_tokens + completion_tokens}")
    print(f"  Cost: ${cost:.4f}")

    # Compare models
    print(f"\nSame request on different models:")
    for model in ["claude-sonnet-4-5", "claude-opus-4-5", "gpt-4", "gpt-3.5-turbo"]:
        counter_model = TokenCounter(model=model)
        cost_model = counter_model.estimate_cost(prompt_tokens, completion_tokens)
        print(f"  {model}: ${cost_model:.4f}")

    print("\n✅ Token counting prevents surprise bills!")


def demo_budget_tracker():
    """Demo 3: Budget tracking."""
    print("\n" + "=" * 60)
    print("DEMO 3: Budget Tracker")
    print("=" * 60)

    tracker = BudgetTracker(budget_usd=10.00, alert_threshold=0.8)

    print(f"\nBudget: ${tracker.budget_usd:.2f}")
    print(f"Alert threshold: {tracker.alert_threshold * 100:.0f}%\n")

    # Simulate API calls
    api_calls = [
        (1000, 500, "claude-sonnet-4-5"),
        (1500, 750, "claude-sonnet-4-5"),
        (2000, 1000, "gpt-4"),
        (500, 250, "gpt-3.5-turbo"),
    ]

    for i, (prompt, completion, model) in enumerate(api_calls, 1):
        cost = tracker.add_usage(prompt, completion, model)
        stats = tracker.get_stats()

        print(f"Call {i} ({model}):")
        print(f"  Tokens: {prompt + completion}")
        print(f"  Cost: ${cost:.4f}")
        print(f"  Total spent: ${stats['total_cost']:.4f} ({stats['percent_used']:.1f}% of budget)")

        if tracker.should_alert() and not tracker.is_over_budget():
            print(f"  ⚠️  ALERT: {stats['percent_used']:.0f}% of budget used!")

        if tracker.is_over_budget():
            print(f"  ❌ BUDGET EXCEEDED! Stop making calls!")
            break

        print()

    print("✅ Budget tracking prevents overspending!")


def demo_cache():
    """Demo 4: File-based caching with TTL."""
    print("\n" + "=" * 60)
    print("DEMO 4: File-Based Cache with TTL")
    print("=" * 60)

    cache = CacheManager(cache_dir="data/cache/test", ttl_seconds=5)

    # Clear any old data
    cache.clear()

    print(f"\nCache directory: {cache.cache_dir}")
    print(f"TTL: {cache.ttl_seconds} seconds\n")

    # Store data
    print("Storing data...")
    cache.set("user_123", {"name": "Alice", "score": 100})
    cache.set("user_456", {"name": "Bob", "score": 85})

    # Retrieve data
    print("\nRetrieving data (should hit):")
    data1 = cache.get("user_123")
    print(f"  user_123: {data1} (hit: {data1 is not None})")

    data2 = cache.get("user_456")
    print(f"  user_456: {data2} (hit: {data2 is not None})")

    data3 = cache.get("user_999")
    print(f"  user_999: {data3} (hit: {data3 is not None})")

    # Stats
    stats = cache.get_stats()
    print(f"\nCache stats:")
    print(f"  Entries: {stats['total_entries']}")
    print(f"  Size: {stats['total_size_mb']} MB")
    print(f"  Hits: {stats['hits']}")
    print(f"  Misses: {stats['misses']}")
    print(f"  Hit rate: {stats['hit_rate_percent']:.1f}%")

    print("\n✅ Caching saves API costs and improves speed!")


async def demo_combined():
    """Demo 5: All utilities working together."""
    print("\n" + "=" * 60)
    print("DEMO 5: All Utilities Together")
    print("=" * 60)

    logger = get_logger(__name__)
    limiter = RateLimiter(requests_per_minute=10)
    counter = TokenCounter(model="claude-sonnet-4-5")
    cache = CacheManager()
    tracker = BudgetTracker(budget_usd=100.00)

    print("\nSimulating LLM API calls with all protections:\n")

    prompts = [
        "Explain quicksort",
        "Write a binary search function",
        "Explain quicksort",  # Duplicate - should hit cache!
    ]

    for i, prompt in enumerate(prompts, 1):
        logger.info(f"Processing request {i}", prompt=prompt[:30])

        # Check cache first
        cache_key = f"response:{prompt}"
        cached_response = cache.get(cache_key)

        if cached_response:
            logger.info("Cache hit!", prompt=prompt[:30])
            print(f"  Request {i}: CACHED (free!)")
            continue

        # Rate limit
        await limiter.acquire()

        # Simulate API call
        await asyncio.sleep(0.1)

        # Count tokens and track cost
        prompt_tokens = counter.count_tokens(prompt)
        completion_tokens = 500  # Simulated

        cost = tracker.add_usage(prompt_tokens, completion_tokens, "claude-sonnet-4-5")

        # Cache response
        cache.set(cache_key, {"response": "Mock response"})

        stats = tracker.get_stats()
        print(f"  Request {i}: ${cost:.4f} (total: ${stats['total_cost']:.4f})")

    print(f"\n✅ Production-ready LLM application patterns!")


async def main():
    """Run all demos."""
    print("\n🔧 UTILITY MODULES DEMOS")
    print("Production patterns for LLM applications!\n")

    await demo_rate_limiter()
    demo_token_counter()
    demo_budget_tracker()
    demo_cache()
    await demo_combined()

    print("\n" + "=" * 60)
    print("✅ ALL UTILITY DEMOS COMPLETE!")
    print("=" * 60)

    print("\n💡 Key Takeaways:")
    print("1. Rate limiting prevents API throttling")
    print("2. Token counting enables cost management")
    print("3. Budget tracking prevents overspending")
    print("4. Caching saves money and improves speed")
    print("5. Together: Production-ready LLM apps!")

    print("\n🎯 These are the same patterns used by:")
    print("   - ChatGPT")
    print("   - Stripe API")
    print("   - AWS services")
    print("   - All major production systems")


if __name__ == "__main__":
    asyncio.run(main())
