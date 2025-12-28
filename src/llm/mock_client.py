"""
Mock LLM Client - Learning Tool

A sophisticated mock LLM client for development and testing.
Simulates real LLM behavior without API costs.

Key Learning Concepts:
- Implements BaseLLMClient interface (teaches abstraction)
- Simulates async delays (teaches async/await)
- Pattern-based responses (teaches prompt engineering)
- Error simulation (teaches error handling)
- Token counting (teaches cost management)

This is YOUR PRIMARY LEARNING TOOL - no API key needed!
"""

import asyncio
import time
import re
from datetime import datetime
from typing import Optional, List, Dict, Any
from .base import BaseLLMClient, LLMConfig, LLMResponse, LLMError
from ..utils import CacheManager


class MockLLMClient(BaseLLMClient):
    """
    Mock LLM client for development, testing, and learning.

    Simulates realistic LLM behavior:
    - Pattern-based responses (detects keywords in prompts)
    - Simulated API delays
    - Token counting
    - Occasional errors (configurable)
    - Cache simulation

    Usage:
        config = LLMConfig(provider="mock", model="mock-v1")
        async with MockLLMClient(config) as client:
            response = await client.generate("Explain quicksort")
            print(response.content)  # Gets a smart mock response!
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._call_count = 0

        # Use persistent file-based cache (replaces in-memory dict)
        self._cache = CacheManager(
            cache_dir="data/cache/mock_llm",
            ttl_seconds=config.cache_ttl_seconds if hasattr(config, 'cache_ttl_seconds') else 86400
        )

        # Simulation settings (configurable)
        self.simulate_delay = True
        self.min_delay_ms = 100
        self.max_delay_ms = 500
        self.error_rate = 0.0  # 0.0 = never fail, 0.1 = 10% chance of error

    async def initialize(self) -> None:
        """Initialize mock client (instant, no real resources needed)."""
        if self._initialized:
            return

        await asyncio.sleep(0.01)  # Simulate tiny initialization delay
        self._initialized = True

    async def cleanup(self) -> None:
        """Cleanup mock client (optionally clear cache)."""
        # Note: We don't auto-clear cache on cleanup - it persists across sessions
        # This is intentional: cached responses save costs in future runs
        self._initialized = False

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
        Generate mock response based on prompt patterns.

        This teaches you:
        - How to structure LLM generate() methods
        - Async/await patterns
        - Error handling
        - Response construction
        """
        start_time = time.time()
        self._call_count += 1

        # Acquire rate limit permission (prevents API throttling)
        await self._acquire_rate_limit()

        # Check cache first (teaching: caching pattern)
        cache_key = f"mock:{prompt}:{temperature}:{max_tokens}"
        if self.config.enable_cache:
            cached_response = self._cache.get(cache_key)
            if cached_response:
                # Cache hit! Return cached content
                return self._build_response(
                    content=cached_response,
                    prompt_tokens=await self.count_tokens(prompt),
                    completion_tokens=await self.count_tokens(cached_response),
                    latency_ms=(time.time() - start_time) * 1000,
                    cached=True
                )

        # Simulate occasional errors (teaching: error handling)
        if self.error_rate > 0 and (self._call_count % int(1 / self.error_rate)) == 0:
            raise LLMError("Mock API error: Simulated failure for testing")

        # Simulate API delay (teaching: async operations take time)
        if self.simulate_delay:
            delay = (self.min_delay_ms +
                    (self.max_delay_ms - self.min_delay_ms) * (temperature or self.config.temperature)) / 1000
            await asyncio.sleep(delay)

        # Generate mock response based on prompt patterns
        content = self._generate_smart_response(prompt, system_prompt)

        # Cache response (persists to disk for future runs)
        if self.config.enable_cache:
            self._cache.set(cache_key, content)

        # Build standardized response
        prompt_tokens = await self.count_tokens(prompt)
        completion_tokens = await self.count_tokens(content)
        latency_ms = (time.time() - start_time) * 1000

        return self._build_response(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            cached=False
        )

    def _generate_smart_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate contextually relevant mock responses.

        Teaching: This shows how to analyze prompts and generate appropriate responses.
        In a real LLM, this is where the model does its magic.
        """
        prompt_lower = prompt.lower()
        system_lower = (system_prompt or "").lower()

        # JSON extraction patterns (check system prompt for JSON instructions)
        if system_lower and any(kw in system_lower for kw in ["json", "output format", "structured"]):
            # This is a structured extraction request
            return self._mock_json_extraction_response(prompt, system_prompt)

        # Algorithm/code generation patterns
        if any(kw in prompt_lower for kw in ["algorithm", "implement", "code", "function"]):
            # But NOT if asking for JSON output
            if "json" not in prompt_lower:
                if "quicksort" in prompt_lower or "sort" in prompt_lower:
                    return self._mock_quicksort_response()
                elif "binary search" in prompt_lower:
                    return self._mock_binary_search_response()
                else:
                    return self._mock_generic_algorithm_response()

        # Problem extraction patterns (for Reader Agent)
        if any(kw in prompt_lower for kw in ["extract problem", "identify inputs", "define problem"]):
            return self._mock_problem_extraction_response()

        # Test generation patterns
        if "test" in prompt_lower and ("generate" in prompt_lower or "write" in prompt_lower):
            return self._mock_test_generation_response()

        # Critique/review patterns (for Critic Agent)
        if any(kw in prompt_lower for kw in ["critique", "review", "analyze", "verify"]):
            return self._mock_critique_response()

        # Default: Generic helpful response
        return f"Mock response to: {prompt[:100]}...\n\nThis is a simulated LLM response. In a real implementation, this would be generated by the actual model."

    def _mock_quicksort_response(self) -> str:
        """Mock response for quicksort algorithm."""
        return """def quicksort(arr):
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quicksort(left) + middle + quicksort(right)"""

    def _mock_binary_search_response(self) -> str:
        """Mock response for binary search."""
        return """def binary_search(arr, target):
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1"""

    def _mock_generic_algorithm_response(self) -> str:
        """Generic algorithm response."""
        return """def algorithm_implementation():
    # Mock implementation
    # This demonstrates the structure of an algorithm
    result = process_input()
    return result"""

    def _mock_problem_extraction_response(self) -> str:
        """Mock response for problem extraction (Reader Agent)."""
        return """{
    "problem_definition": "Sort an array of integers in ascending order",
    "inputs": ["array of integers"],
    "outputs": ["sorted array"],
    "constraints": ["in-place sorting allowed", "stable sort not required"],
    "edge_cases": ["empty array", "single element", "all duplicates"]
}"""

    def _mock_test_generation_response(self) -> str:
        """Mock response for test generation."""
        return """def test_algorithm():
    # Test case 1: Normal input
    assert algorithm([3, 1, 4, 1, 5]) == [1, 1, 3, 4, 5]

    # Test case 2: Empty input
    assert algorithm([]) == []

    # Test case 3: Single element
    assert algorithm([42]) == [42]"""

    def _mock_critique_response(self) -> str:
        """Mock response for code critique (Critic Agent)."""
        return """CRITIQUE:
1. ✅ Algorithm correctness: Implementation follows expected logic
2. ✅ Edge cases: Handles empty and single-element cases
3. ⚠️  Time complexity: O(n²) in worst case - could be optimized
4. ✅ Code clarity: Well-structured and readable

SUGGESTED IMPROVEMENTS:
- Add input validation
- Consider optimizing worst-case complexity
- Add docstring documentation"""

    def _mock_json_extraction_response(self, prompt: str, system_prompt: Optional[str]) -> str:
        """
        Mock response for JSON extraction requests.

        Analyzes the prompt to determine what kind of structured data to return.
        """
        prompt_lower = prompt.lower()

        # Algorithm information extraction
        if any(kw in prompt_lower for kw in ["binary search", "binarysearch"]):
            return """{
  "algorithm_name": "Binary Search",
  "purpose": "Find the index of a target value in a sorted array",
  "inputs": ["sorted array of comparable elements", "target value to find"],
  "outputs": ["integer index (0-based) of target, or -1 if not found"],
  "time_complexity": "O(log n)"
}"""
        elif any(kw in prompt_lower for kw in ["merge sort", "mergesort"]):
            return """{
  "algorithm_name": "Merge Sort",
  "purpose": "Sort an array of elements using divide-and-conquer",
  "inputs": ["unsorted array of comparable elements"],
  "outputs": ["sorted array in ascending order"],
  "time_complexity": "O(n log n) in all cases"
}"""
        elif any(kw in prompt_lower for kw in ["dijkstra"]):
            return """{
  "algorithm_name": "Dijkstra's Shortest Path",
  "purpose": "Find shortest paths from source to all nodes in weighted graph",
  "inputs": ["weighted graph with non-negative edges", "source vertex"],
  "outputs": ["dictionary mapping vertices to shortest distances"],
  "time_complexity": "O((V + E) log V) with binary heap"
}"""

        # Problem definition extraction (Reader Agent pattern)
        elif "problem definition" in prompt_lower or "extract the problem" in prompt_lower:
            # Analyze the prompt text to extract problem info
            if "merge sort" in prompt_lower or "merging" in prompt_lower:
                return """{
  "problem_statement": "Sort an array by dividing it into halves and merging sorted subarrays",
  "inputs": ["unsorted array of comparable elements"],
  "outputs": ["sorted array maintaining original elements"],
  "constraints": ["requires O(n) extra space", "works on any comparable type"],
  "edge_cases": ["empty array", "single element", "already sorted", "reverse sorted"],
  "assumptions": ["elements are comparable", "stable sort preserves equal element order"]
}"""
            else:
                # Generic problem extraction
                return """{
  "problem_statement": "Solve the algorithmic problem described in the text",
  "inputs": ["input data as described"],
  "outputs": ["computed result"],
  "constraints": ["constraints as stated"],
  "edge_cases": ["boundary conditions mentioned"],
  "assumptions": ["implicit assumptions from context"]
}"""

        # Default: Generic structured response
        return """{
  "extracted_info": "Mock structured data based on the prompt",
  "note": "This is a mock response - real LLM would analyze the actual content"
}"""

    async def count_tokens(self, text: str) -> int:
        """
        Count tokens (simulated - approximately 1 token per 4 characters).

        Teaching: Real tokenizers are more complex, but this shows the concept.
        """
        # Simple approximation: ~1 token per 4 characters
        return max(1, len(text) // 4)

    async def stream_generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        """
        Simulate streaming generation (yields chunks).

        Teaching: This shows how streaming APIs work - they yield text chunks
        as they're generated rather than waiting for the complete response.
        """
        # Generate full response
        response = await self.generate(prompt, system_prompt=system_prompt, **kwargs)
        content = response.content

        # Simulate streaming by yielding chunks
        chunk_size = 20
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i+chunk_size]
            await asyncio.sleep(0.05)  # Simulate streaming delay
            yield chunk

    def _build_response(
        self,
        content: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        cached: bool = False
    ) -> LLMResponse:
        """
        Build standardized LLMResponse.

        Teaching: This shows how to construct consistent response objects
        that work across all LLM providers.
        """
        return LLMResponse(
            content=content,
            model=self.config.model,
            provider=self.config.provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency_ms,
            timestamp=datetime.now(),
            cached=cached,
            raw_response={"mock": True, "call_count": self._call_count}
        )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get mock client statistics.

        Useful for debugging and understanding usage patterns.
        """
        cache_stats = self._cache.get_stats()
        return {
            "call_count": self._call_count,
            "cache_entries": cache_stats["total_entries"],
            "cache_hit_rate": cache_stats["hit_rate_percent"],
            "cache_size_mb": cache_stats["total_size_mb"],
            "provider": self.config.provider,
            "model": self.config.model,
        }
