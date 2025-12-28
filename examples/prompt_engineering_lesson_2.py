"""
Prompt Engineering Lesson 2: Few-Shot Learning

Learn how to use examples to guide LLM behavior.

Key Concepts:
1. Few-shot = Show examples of desired behavior
2. Examples teach format, style, and edge cases
3. Quality over quantity (2-3 good examples > 10 mediocre)
4. Examples reduce ambiguity

Run with: python examples/prompt_engineering_lesson_2.py
"""

import asyncio
import json
import sys
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm import MockLLMClient, LLMConfig


# ==============================================================================
# Schema for Problem Extraction (Reader Agent Preview!)
# ==============================================================================

class ProblemDefinition(BaseModel):
    """
    What the Reader Agent will extract from papers.

    This is a PREVIEW of Phase 1 - you'll use this pattern!
    """
    problem_statement: str = Field(..., description="Clear problem description")
    inputs: List[str] = Field(..., description="Required inputs with types")
    outputs: List[str] = Field(..., description="Expected outputs with types")
    constraints: List[str] = Field(..., description="Constraints and limitations")
    edge_cases: List[str] = Field(..., description="Special cases to handle")
    assumptions: List[str] = Field(default_factory=list, description="Implicit assumptions made")


# ==============================================================================
# COMPARISON: Zero-Shot vs Few-Shot
# ==============================================================================

def create_zero_shot_prompt(text: str) -> tuple[str, str]:
    """
    Zero-shot = No examples, just instructions.

    Works for simple tasks, struggles with nuance.
    """
    system_prompt = """You are a problem extraction expert. Extract the problem definition from the text and output as JSON.

Output format:
{
  "problem_statement": "...",
  "inputs": ["..."],
  "outputs": ["..."],
  "constraints": ["..."],
  "edge_cases": ["..."],
  "assumptions": ["..."]
}"""

    user_prompt = f"Extract the problem definition:\n\n{text}"

    return system_prompt, user_prompt


def create_few_shot_prompt(text: str) -> tuple[str, str]:
    """
    Few-shot = Provide 2-3 examples.

    MUCH better at understanding what you want!
    """
    system_prompt = """You are a problem extraction expert. Extract the problem definition from algorithm descriptions.

Here are examples of good extractions:

Example 1:
Input: "Binary search finds an element in a sorted array. Given a sorted array and target value, return the index or -1 if not found. Array must be sorted."

Output:
{
  "problem_statement": "Find the index of a target value in a sorted array",
  "inputs": ["sorted array of integers", "target integer value"],
  "outputs": ["integer index (0-based) or -1 if not found"],
  "constraints": ["array must be pre-sorted", "array can be empty"],
  "edge_cases": ["empty array returns -1", "target not in array returns -1", "duplicate values"],
  "assumptions": ["array contains comparable elements", "stable ordering"]
}

Example 2:
Input: "Dijkstra's shortest path algorithm finds minimum distance from source to all nodes in weighted graph with non-negative edges."

Output:
{
  "problem_statement": "Find shortest paths from a source node to all other nodes in a weighted graph",
  "inputs": ["weighted graph with vertices and edges", "source vertex", "non-negative edge weights"],
  "outputs": ["dictionary mapping each vertex to shortest distance from source"],
  "constraints": ["edge weights must be non-negative", "graph can be disconnected"],
  "edge_cases": ["source node has distance 0", "unreachable nodes have infinite distance"],
  "assumptions": ["graph is represented as adjacency list", "vertices are uniquely identifiable"]
}

Now extract from the input below. Output ONLY the JSON, nothing else."""

    user_prompt = f"Extract the problem definition:\n\n{text}"

    return system_prompt, user_prompt


# ==============================================================================
# Test Both Approaches
# ==============================================================================

async def extract_with_prompt(text: str, prompt_creator, approach_name: str) -> ProblemDefinition:
    """Extract using specified prompt approach."""
    system_prompt, user_prompt = prompt_creator(text)

    print(f"\n{'='*70}")
    print(f"APPROACH: {approach_name}")
    print(f"{'='*70}")
    print(f"\n📝 System Prompt Preview (first 200 chars):")
    print(f"{system_prompt[:200]}...")

    config = LLMConfig(provider="mock", model="mock-v1")
    async with MockLLMClient(config) as client:
        response = await client.generate(
            user_prompt,
            system_prompt=system_prompt,
            temperature=0.0
        )

    print(f"\n📤 LLM Response:\n{response.content}\n")

    data = json.loads(response.content)
    problem = ProblemDefinition(**data)

    return problem


def display_problem(problem: ProblemDefinition):
    """Pretty print the extracted problem."""
    print(f"\n📊 EXTRACTED PROBLEM:")
    print(f"\n  Statement: {problem.problem_statement}")
    print(f"\n  Inputs:")
    for inp in problem.inputs:
        print(f"    • {inp}")
    print(f"\n  Outputs:")
    for out in problem.outputs:
        print(f"    • {out}")
    print(f"\n  Constraints:")
    for con in problem.constraints:
        print(f"    • {con}")
    print(f"\n  Edge Cases:")
    for edge in problem.edge_cases:
        print(f"    • {edge}")
    if problem.assumptions:
        print(f"\n  Assumptions:")
        for assumption in problem.assumptions:
            print(f"    • {assumption}")


# ==============================================================================
# Test Algorithm Description
# ==============================================================================

TEST_ALGORITHM = """
The merge sort algorithm divides an array into two halves, recursively sorts each half,
then merges the sorted halves together. It requires additional space proportional to
the array size. The algorithm works on any comparable elements and maintains stability
(equal elements keep their relative order). It handles empty and single-element arrays
as base cases.
"""


async def main():
    """Compare zero-shot vs few-shot."""
    print("=" * 70)
    print("PROMPT ENGINEERING LESSON 2: Few-Shot Learning")
    print("=" * 70)

    print("\n🎯 LEARNING OBJECTIVES:")
    print("1. Understand zero-shot vs few-shot prompting")
    print("2. See how examples improve extraction quality")
    print("3. Learn how to write effective few-shot examples")
    print("4. Understand when to use each approach")

    print(f"\n📖 TEST ALGORITHM DESCRIPTION:")
    print(f"{TEST_ALGORITHM}")

    # Zero-shot extraction
    print("\n" + "="*70)
    print("TEST 1: ZERO-SHOT (No Examples)")
    print("="*70)
    problem_zero = await extract_with_prompt(
        TEST_ALGORITHM,
        create_zero_shot_prompt,
        "Zero-Shot"
    )
    display_problem(problem_zero)

    # Few-shot extraction
    print("\n" + "="*70)
    print("TEST 2: FEW-SHOT (With Examples)")
    print("="*70)
    problem_few = await extract_with_prompt(
        TEST_ALGORITHM,
        create_few_shot_prompt,
        "Few-Shot"
    )
    display_problem(problem_few)

    # Comparison
    print("\n" + "="*70)
    print("📊 COMPARISON: Zero-Shot vs Few-Shot")
    print("="*70)
    print(f"""
Zero-Shot Results:
  - Inputs: {len(problem_zero.inputs)} items
  - Constraints: {len(problem_zero.constraints)} items
  - Edge Cases: {len(problem_zero.edge_cases)} items
  - Assumptions: {len(problem_zero.assumptions)} items

Few-Shot Results:
  - Inputs: {len(problem_few.inputs)} items
  - Constraints: {len(problem_few.constraints)} items
  - Edge Cases: {len(problem_few.edge_cases)} items
  - Assumptions: {len(problem_few.assumptions)} items

Notice:
  - Few-shot typically extracts MORE detail
  - Few-shot follows the example format more closely
  - Few-shot catches implicit information (assumptions)
  - Few-shot is more consistent across different inputs
    """)

    # Key Lessons
    print("\n" + "="*70)
    print("🎓 KEY LESSONS")
    print("="*70)
    print("""
1. FEW-SHOT WINS for complex extraction
   - Shows LLM EXACTLY what you want
   - Reduces ambiguity
   - Handles edge cases better

2. QUALITY OF EXAMPLES MATTERS
   - Pick diverse examples (simple + complex)
   - Show edge cases you care about
   - Match your actual use case

3. HOW MANY EXAMPLES?
   - 2-3 is usually optimal
   - 1 example = might overfit to that pattern
   - 5+ examples = diminishing returns + token cost

4. WHEN TO USE EACH:
   - Zero-shot: Simple, well-defined tasks
   - Few-shot: Complex extraction, specific format
   - Fine-tuning: Same task repeated 1000s of times

5. EXAMPLES = IMPLICIT INSTRUCTIONS
   - "Extract constraints" → unclear
   - Example showing constraints → clear!
   - Show what you want, don't just describe it
    """)

    print("\n" + "="*70)
    print("🔧 HANDS-ON PRACTICE")
    print("="*70)
    print("""
Try these exercises:

1. Add a 3rd example to the few-shot prompt
   - Pick a different algorithm (e.g., quicksort)
   - Make it distinct from the first 2

2. Test with a tricky algorithm description
   - One with implicit assumptions
   - See if few-shot catches them

3. Create bad examples and see what breaks
   - Example with missing fields
   - Example with wrong format
   - Learn what makes a good example

4. Compare performance on 5 different algorithms
   - Count which approach extracts more info
   - Note where each approach fails
    """)

    print("\n" + "="*70)
    print("✅ LESSON 2 COMPLETE!")
    print("="*70)
    print("\nNEXT STEPS:")
    print("1. Practice writing few-shot examples")
    print("2. Move to Lesson 3: Chain-of-Thought Reasoning")
    print("3. Start designing your Reader Agent schema")


if __name__ == "__main__":
    asyncio.run(main())
