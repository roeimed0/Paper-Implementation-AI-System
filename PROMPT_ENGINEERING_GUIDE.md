# Prompt Engineering Guide
**For the Paper→Code AI System**

A hands-on guide to mastering prompt engineering for GenAI systems. Each concept is taught through runnable examples with your mock LLM.

---

## 🎯 Learning Path

### Phase 1: Foundations (Week 1)
- [ ] **Lesson 1**: Structured Outputs - Get valid JSON from LLMs
- [ ] **Lesson 2**: Few-Shot Learning - Use examples to guide behavior
- [ ] **Lesson 3**: Chain-of-Thought - Make LLMs show their reasoning
- [ ] **Lesson 4**: Error Recovery - Handle malformed outputs

### Phase 2: Advanced (Week 2)
- [ ] **Lesson 5**: Multi-Step Extraction - Break complex tasks into stages
- [ ] **Lesson 6**: Validation Prompts - Self-check LLM outputs
- [ ] **Lesson 7**: Prompt Optimization - Iterate for better results
- [ ] **Lesson 8**: Cost Optimization - Reduce tokens, maintain quality

---

## Lesson 1: Structured Outputs ✅ Ready

**File:** [`examples/prompt_engineering_lesson_1.py`](examples/prompt_engineering_lesson_1.py)

### What You'll Learn
- How to write system prompts that demand JSON output
- Pydantic schemas as contracts with LLMs
- Parsing and validating LLM responses
- Why temperature matters for consistency

### Key Pattern
```python
system_prompt = """You are an expert. Output ONLY valid JSON.

Required structure:
{
  "field1": "type - description",
  "field2": ["array - description"]
}

Example:
{
  "field1": "example value",
  "field2": ["item1", "item2"]
}"""
```

### Run It
```bash
python examples/prompt_engineering_lesson_1.py
```

### Practice Exercises
1. Add new fields to `AlgorithmInfo` schema
2. Extract from your own algorithm descriptions
3. See what happens with incomplete inputs
4. Try different temperature values (0.0, 0.5, 1.0)

---

## Lesson 2: Few-Shot Learning ✅ Ready

**File:** [`examples/prompt_engineering_lesson_2.py`](examples/prompt_engineering_lesson_2.py)

### What You'll Learn
- Difference between zero-shot and few-shot prompting
- How to write effective examples
- Optimal number of examples (2-3 usually best)
- When to use each approach

### Key Pattern
```python
system_prompt = """Task: Extract problem definition.

Example 1:
Input: "Binary search..."
Output: {"problem": "...", "inputs": [...]}

Example 2:
Input: "Merge sort..."
Output: {"problem": "...", "inputs": [...]}

Now extract from the input below."""
```

### Run It
```bash
python examples/prompt_engineering_lesson_2.py
```

### Practice Exercises
1. Add a 3rd example to few-shot prompt
2. Test with tricky algorithm descriptions
3. Create intentionally bad examples - see what breaks
4. Compare quality on 5 different algorithms

---

## Lesson 3: Chain-of-Thought Reasoning (Coming Soon)

### Concept
Ask the LLM to "think step-by-step" before answering. This improves accuracy on complex reasoning tasks.

### Pattern
```python
system_prompt = """Think step-by-step before extracting.

Steps:
1. Identify the main problem
2. List explicit inputs/outputs
3. Find implicit constraints
4. Identify edge cases
5. Output final JSON

Example thought process:
"First, I see this is a sorting problem...
The inputs are clearly stated as...
The constraint about sorted output implies..."
"""
```

### When to Use
- Complex reasoning required
- Multiple interpretation possibilities
- Need to catch implicit assumptions
- Debugging why extraction failed

---

## Lesson 4: Error Recovery (Coming Soon)

### Concept
LLMs sometimes output malformed JSON. Build retry logic with correction prompts.

### Pattern
```python
async def extract_with_retry(text, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = await llm.generate(prompt)
            return parse_and_validate(response)
        except ValidationError as e:
            if attempt == max_retries - 1:
                raise
            # Create correction prompt
            prompt = f"Previous output had errors: {e}\nPlease fix and output valid JSON."
```

### Error Types
1. **JSON Parse Errors** - Invalid JSON syntax
2. **Schema Violations** - Missing required fields
3. **Type Mismatches** - String instead of array
4. **Incomplete Outputs** - Truncated response

---

## Best Practices Summary

### 1. System Prompts
✅ **DO:**
- Be explicit: "Output ONLY valid JSON"
- Show exact structure with examples
- Use consistent formatting
- Define clear role for LLM

❌ **DON'T:**
- Assume LLM knows format
- Use ambiguous instructions
- Mix multiple tasks in one prompt
- Skip examples for complex tasks

### 2. Temperature Settings
- **0.0** - Structured extraction (most consistent)
- **0.3** - Slight variation, still reliable
- **0.7** - Creative text generation
- **1.0+** - Maximum creativity, least consistent

### 3. Token Optimization
- Use few-shot, not 10-shot (2-3 examples optimal)
- Cache static system prompts
- Compress verbose instructions
- Remove unnecessary explanations in examples

### 4. Validation
Always validate in this order:
1. JSON parsing (syntax)
2. Pydantic validation (schema)
3. Business logic validation (domain rules)
4. Completeness checks (all required info present)

---

## Real-World Prompt Templates

### Template 1: Problem Extraction (Reader Agent)

```python
SYSTEM_PROMPT = """You are a scientific algorithm analyzer. Extract structured problem definitions from algorithm descriptions.

OUTPUT FORMAT (JSON only):
{
  "problem_statement": "one sentence problem description",
  "inputs": ["input1 with type", "input2 with type"],
  "outputs": ["output1 with type"],
  "constraints": ["constraint1", "constraint2"],
  "edge_cases": ["case1", "case2"],
  "assumptions": ["assumption1", "assumption2"]
}

EXAMPLE:
Input: "Binary search finds element in sorted array..."
Output: {
  "problem_statement": "Find index of target in sorted array",
  "inputs": ["sorted integer array", "target integer"],
  "outputs": ["integer index or -1"],
  "constraints": ["array must be sorted", "can be empty"],
  "edge_cases": ["empty array", "target not present"],
  "assumptions": ["elements are comparable"]
}

CRITICAL: Output ONLY the JSON, no other text."""
```

### Template 2: Algorithm Reconstruction (Planner Agent)

```python
SYSTEM_PROMPT = """You are an algorithm design expert. Convert problem definitions into step-by-step algorithms.

OUTPUT FORMAT (JSON only):
{
  "algorithm_name": "descriptive name",
  "steps": [
    {
      "step_number": 1,
      "description": "what to do",
      "inputs": ["data needed"],
      "outputs": ["data produced"],
      "control_flow": "sequential|loop|conditional"
    }
  ],
  "termination_condition": "when algorithm stops"
}

Think step-by-step:
1. What's the high-level approach?
2. What are the sequential steps?
3. What loops are needed?
4. What conditionals handle edge cases?

Output ONLY the JSON."""
```

### Template 3: Code Critique (Critic Agent)

```python
SYSTEM_PROMPT = """You are a code quality expert. Compare implementations against specifications and find discrepancies.

OUTPUT FORMAT (JSON only):
{
  "correctness_score": 0-100,
  "issues": [
    {
      "severity": "critical|major|minor",
      "category": "logic|performance|style",
      "description": "what's wrong",
      "location": "line number or function",
      "suggestion": "how to fix"
    }
  ],
  "missing_requirements": ["requirement1", "requirement2"],
  "additional_logic": ["extra code not in spec"]
}

Be thorough:
- Check every requirement is implemented
- Verify edge cases are handled
- Flag performance issues
- Note style violations

Output ONLY the JSON."""
```

---

## Debugging Prompts

### When LLM Outputs Wrong Format

**Problem:** LLM adds explanatory text before/after JSON

**Solution:** Emphasize more strongly
```python
system_prompt = """CRITICAL: Your ENTIRE response must be ONLY the JSON object.
Do NOT include:
- Explanations before the JSON
- Comments after the JSON
- Markdown code blocks
- Any other text

START your response with { and END with }"""
```

### When LLM Misses Required Fields

**Problem:** JSON valid but missing required fields

**Solution:** List requirements explicitly
```python
system_prompt = """REQUIRED FIELDS (all must be present):
1. problem_statement (string)
2. inputs (array of strings)
3. outputs (array of strings)
4. constraints (array of strings)
5. edge_cases (array of strings)

Missing ANY field = invalid response."""
```

### When LLM Hallucinates Information

**Problem:** LLM adds information not in source

**Solution:** Emphasize source-only extraction
```python
system_prompt = """Extract ONLY information explicitly stated in the text.

DO NOT:
- Add information from general knowledge
- Infer unstated constraints
- Assume standard implementations

If information is missing, use empty array []."""
```

---

## Advanced Techniques

### 1. Multi-Turn Extraction

For very complex extractions, break into multiple prompts:

```python
# Turn 1: Extract basic info
basic_info = await extract_basic(text)

# Turn 2: Extract detailed constraints (with context)
detailed = await extract_detailed(text, context=basic_info)

# Combine results
final = merge(basic_info, detailed)
```

### 2. Self-Validation

Ask LLM to validate its own output:

```python
# First extraction
extraction = await extract(text)

# Self-validation prompt
validation_prompt = f"""
You extracted: {extraction}

Check for:
1. All required fields present?
2. Information matches source text?
3. No hallucinated information?
4. Edge cases identified?

Output: {{"valid": true/false, "issues": [...]}}
"""
```

### 3. Iterative Refinement

```python
for iteration in range(3):
    extraction = await extract(text)
    issues = await validate(extraction)
    if not issues:
        break
    # Refine prompt based on issues
    prompt = create_refined_prompt(issues)
```

---

## Measuring Prompt Quality

### Metrics to Track

1. **Parse Success Rate**
   - % of responses that parse as valid JSON
   - Target: >95%

2. **Schema Compliance Rate**
   - % of parsed responses that pass Pydantic validation
   - Target: >90%

3. **Completeness Score**
   - % of required fields actually filled
   - Target: >85%

4. **Accuracy Score**
   - % of extracted info that matches ground truth
   - Requires manual labeling

### A/B Testing Prompts

```python
# Test prompt variants
results_v1 = await test_prompt(prompt_v1, test_cases)
results_v2 = await test_prompt(prompt_v2, test_cases)

# Compare metrics
compare_metrics(results_v1, results_v2)
```

---

## Next Steps

1. **Run Lesson 1 & 2**
   ```bash
   python examples/prompt_engineering_lesson_1.py
   python examples/prompt_engineering_lesson_2.py
   ```

2. **Practice with Your Own Examples**
   - Find 3 algorithm descriptions
   - Write extraction prompts
   - Iterate until high quality

3. **Study Anthropic's Guide**
   - [Anthropic Prompt Engineering](https://docs.anthropic.com/claude/docs/prompt-engineering)
   - Focus on: structured outputs, few-shot, chain-of-thought

4. **Design Reader Agent Schema**
   - What fields do you need?
   - What Pydantic models?
   - What validation rules?

5. **Move to Phase 1 Implementation**
   - Build Reader Agent
   - Apply these patterns
   - Iterate on prompts

---

## Resources

### Official Docs
- [Anthropic Prompt Engineering](https://docs.anthropic.com/claude/docs/prompt-engineering)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [JSON Schema Guide](https://json-schema.org/)

### Example Prompts
- [Anthropic Cookbook](https://github.com/anthropics/anthropic-cookbook)
- [LangChain Prompts](https://python.langchain.com/docs/modules/model_io/prompts/)

### Your Codebase
- [`mock_client.py`](src/llm/mock_client.py) - Pattern-based responses
- [`test_mock_llm.py`](examples/test_mock_llm.py) - Mock usage examples
- All prompt lessons in [`examples/`](examples/)

---

**Remember:** Prompt engineering is an iterative skill. Your first prompt won't be perfect - test, measure, refine!
