"""
Prompt Engineering Lesson 1: Structured Outputs

Learn how to get LLMs to output valid JSON with specific schemas.

Key Concepts:
1. System prompts define the LLM's role and output format
2. Clear schema specifications prevent malformed responses
3. Few-shot examples show the LLM what you want
4. Pydantic validates the output

Run with: python examples/prompt_engineering_lesson_1.py
"""

import asyncio
import json
import sys
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm import MockLLMClient, LLMConfig


# ==============================================================================
# STEP 1: Define Your Output Schema with Pydantic
# ==============================================================================

class AlgorithmInfo(BaseModel):
    """
    This defines EXACTLY what structure we want from the LLM.

    Pydantic will:
    - Validate types (str, List[str], etc.)
    - Check required fields
    - Provide clear error messages
    """
    algorithm_name: str = Field(..., description="Name of the algorithm")
    purpose: str = Field(..., description="What the algorithm does")
    inputs: List[str] = Field(..., description="Required inputs")
    outputs: List[str] = Field(..., description="What it returns")
    time_complexity: str = Field(..., description="Big-O time complexity")

    class Config:
        # Generate JSON schema for the LLM to follow
        schema_extra = {
            "example": {
                "algorithm_name": "Binary Search",
                "purpose": "Find element in sorted array",
                "inputs": ["sorted array", "target value"],
                "outputs": ["index of element or -1"],
                "time_complexity": "O(log n)"
            }
        }


# ==============================================================================
# STEP 2: Craft System Prompt with Clear Instructions
# ==============================================================================

def create_extraction_prompt(algorithm_description: str) -> tuple[str, str]:
    """
    Create a system + user prompt for structured extraction.

    Best Practices:
    1. Be EXPLICIT about output format
    2. Show exact JSON structure
    3. List all required fields
    4. Give one clear example
    """

    system_prompt = """You are an expert algorithm analyzer. Your task is to extract structured information about algorithms.

CRITICAL: You MUST respond with ONLY valid JSON. No other text before or after.

Required JSON structure:
{
  "algorithm_name": "string - name of the algorithm",
  "purpose": "string - what the algorithm does in one sentence",
  "inputs": ["array of strings - required inputs"],
  "outputs": ["array of strings - what it returns"],
  "time_complexity": "string - Big-O notation"
}

Example output:
{
  "algorithm_name": "Quicksort",
  "purpose": "Sort an array of elements in place",
  "inputs": ["unsorted array"],
  "outputs": ["sorted array"],
  "time_complexity": "O(n log n) average, O(n²) worst"
}

Remember: ONLY output valid JSON, nothing else."""

    user_prompt = f"""Analyze this algorithm and extract the information:

{algorithm_description}

Output the JSON now:"""

    return system_prompt, user_prompt


# ==============================================================================
# STEP 3: Parse and Validate LLM Response
# ==============================================================================

async def extract_algorithm_info(description: str) -> AlgorithmInfo:
    """
    Complete workflow: Prompt → LLM → Parse → Validate
    """
    # Create prompts
    system_prompt, user_prompt = create_extraction_prompt(description)

    # Call LLM
    config = LLMConfig(provider="mock", model="mock-v1")
    async with MockLLMClient(config) as client:
        response = await client.generate(
            user_prompt,
            system_prompt=system_prompt,
            temperature=0.0  # Lower = more consistent
        )

    print(f"\n📤 LLM Response (raw):\n{response.content}\n")

    # Parse JSON
    try:
        data = json.loads(response.content)
        print(f"✅ Valid JSON parsed")
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing failed: {e}")
        raise

    # Validate with Pydantic
    try:
        algorithm_info = AlgorithmInfo(**data)
        print(f"✅ Pydantic validation passed")
        return algorithm_info
    except ValidationError as e:
        print(f"❌ Pydantic validation failed:\n{e}")
        raise


# ==============================================================================
# EXAMPLES TO TRY
# ==============================================================================

EXAMPLE_ALGORITHMS = {
    "binary_search": """
Binary search finds an element in a sorted array by repeatedly dividing
the search interval in half. It compares the target value to the middle
element and eliminates half of the array. This continues until the element
is found or the interval is empty. Time complexity is O(log n).
    """,

    "merge_sort": """
Merge sort is a divide-and-conquer algorithm that divides the array into
halves, recursively sorts them, and merges the sorted halves. It requires
O(n) extra space and runs in O(n log n) time for all cases.
    """,

    "dijkstra": """
Dijkstra's algorithm finds the shortest path from a source node to all
other nodes in a weighted graph with non-negative edge weights. It uses
a priority queue to greedily select the closest unvisited node. Time
complexity is O((V + E) log V) with a binary heap.
    """
}


async def main():
    """Run the lesson examples."""
    print("=" * 70)
    print("PROMPT ENGINEERING LESSON 1: Structured Outputs")
    print("=" * 70)

    print("\n📚 WHAT YOU'LL LEARN:")
    print("1. How to write system prompts for structured extraction")
    print("2. How to specify exact JSON schemas")
    print("3. How to validate LLM outputs with Pydantic")
    print("4. Why temperature=0 gives consistent results")

    # Example 1: Binary Search
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Extracting Binary Search Info")
    print("=" * 70)

    result1 = await extract_algorithm_info(EXAMPLE_ALGORITHMS["binary_search"])

    print(f"\n📊 Extracted Information:")
    print(f"  Algorithm: {result1.algorithm_name}")
    print(f"  Purpose: {result1.purpose}")
    print(f"  Inputs: {', '.join(result1.inputs)}")
    print(f"  Outputs: {', '.join(result1.outputs)}")
    print(f"  Complexity: {result1.time_complexity}")

    # Example 2: Different algorithm
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Extracting Dijkstra's Algorithm Info")
    print("=" * 70)

    result2 = await extract_algorithm_info(EXAMPLE_ALGORITHMS["dijkstra"])

    print(f"\n📊 Extracted Information:")
    print(f"  Algorithm: {result2.algorithm_name}")
    print(f"  Purpose: {result2.purpose}")
    print(f"  Inputs: {', '.join(result2.inputs)}")
    print(f"  Outputs: {', '.join(result2.outputs)}")
    print(f"  Complexity: {result2.time_complexity}")

    # Key Takeaways
    print("\n" + "=" * 70)
    print("🎯 KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. SYSTEM PROMPT = Define role + output format
   - Be EXPLICIT: "Output ONLY valid JSON"
   - Show exact structure with example

2. PYDANTIC SCHEMA = Your contract with the LLM
   - Defines required fields and types
   - Validates automatically
   - Gives clear errors

3. TEMPERATURE = 0 for consistency
   - Lower temperature = more deterministic
   - Use 0.0 for structured extraction
   - Use 0.7+ for creative text

4. ERROR HANDLING is critical
   - LLMs can output malformed JSON
   - Always try/except JSON parsing
   - Always validate with Pydantic

5. MOCK LLM lets you practice for FREE
   - No API costs
   - Instant feedback
   - Same patterns work with real LLMs
    """)

    print("\n" + "=" * 70)
    print("✅ LESSON 1 COMPLETE!")
    print("=" * 70)
    print("\nNEXT STEPS:")
    print("1. Modify AlgorithmInfo schema - add new fields")
    print("2. Try extracting from your own algorithm descriptions")
    print("3. See what happens with incomplete descriptions")
    print("4. Move to Lesson 2: Few-Shot Learning")


if __name__ == "__main__":
    asyncio.run(main())
