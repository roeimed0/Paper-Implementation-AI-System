---
name: genai-project-architecture
description: Project structure and organization patterns for the Paper→Code GenAI System. Enforces the 9-stage pipeline architecture, module organization, data flow patterns, and configuration standards. Use this skill when organizing code, creating new modules, or deciding where files belong.
---

# GenAI Project Architecture

**Purpose:** Enforce consistent project structure and organization for the Paper→Code AI System

**Last Updated:** 2025-12-25

---

## When to Use This Skill

This skill **auto-activates** when:
- Creating or organizing modules in `src/`
- Working with `data/` directory structure
- Setting up configuration files
- Asking about project organization ("where should I put...?")
- Designing pipeline stages
- Organizing prompts, outputs, or datasets

This skill **should be used** for:
- ✅ Deciding where new code belongs
- ✅ Creating new pipeline stages
- ✅ Organizing agent modules
- ✅ Structuring data directories
- ✅ Configuration file organization
- ✅ Understanding data flow between stages

**Do NOT use** for:
- ❌ Python implementation details (use python-llm-dev-guidelines)
- ❌ LLM API integration (use python-llm-dev-guidelines)
- ❌ Testing strategies (use python-llm-dev-guidelines)

---

## Quick Reference: Project Structure

### High-Level Architecture

```
Paper-Implementation-AI-System/
│
├── src/                    # All source code
│   ├── agents/            # 4 LLM agents (Reader, Planner, Implementer, Critic)
│   ├── generators/        # Code generators (Pseudocode, Tests)
│   ├── verification/      # Verification and cross-validation
│   ├── evaluation/        # Metrics and scoring
│   ├── dataset/           # Dataset logging and export
│   ├── pipeline/          # Pipeline orchestration
│   ├── input_acquisition/ # PDF parsing and text extraction
│   ├── llm/               # LLM client implementations
│   ├── utils/             # Utilities (rate limiting, caching, logging)
│   ├── config/            # Configuration loading
│   ├── handlers/          # Error handling
│   └── schemas/           # Pydantic data models
│
├── data/                  # All data files
│   ├── prompts/           # Agent prompts (versioned)
│   ├── cache/             # API response cache
│   ├── outputs/           # Generated implementations
│   ├── logs/              # Application logs
│   └── datasets/          # Logged pipeline runs
│
├── config/                # Configuration files (YAML)
│   ├── model_config.yaml
│   ├── prompt_templates.yaml
│   └── logging_config.yaml
│
├── tests/                 # Test suite
│   ├── unit/
│   ├── integration/
│   └── property/
│
├── notebooks/             # Jupyter notebooks for analysis
├── examples/              # Example usage scripts
└── dev/                   # Development documentation
    └── active/            # Active dev-docs
```

---

## 9-Stage Pipeline Architecture

**Core Principle:** Each stage has a clear input, output, and responsibility. No stage is skipped.

```
Stage 1: Input Acquisition
    Module: src/input_acquisition/
    Input:  PDF/Text file
    Output: Structured text (JSON)
    ↓
Stage 2: Problem Extraction (Reader Agent)
    Module: src/agents/reader_agent.py
    Input:  Structured text
    Output: Problem definition (JSON)
    ↓
Stage 3: Algorithm Reconstruction (Planner Agent)
    Module: src/agents/planner_agent.py
    Input:  Problem definition
    Output: Algorithm steps (JSON)
    ↓
Stage 4: Pseudocode Generation
    Module: src/generators/pseudocode_generator.py
    Input:  Algorithm steps
    Output: Language-agnostic pseudocode
    ↓
Stage 5: Code Implementation (Implementer Agent)
    Module: src/agents/implementer_agent.py
    Input:  Pseudocode
    Output: Python implementation
    ↓
Stage 6: Test Generation
    Module: src/generators/test_generator.py
    Input:  Python code
    Output: Test suite (unit, property, reference)
    ↓
Stage 7: Verification & Critique (Critic Agent)
    Module: src/agents/critic_agent.py
    Input:  All previous stages
    Output: Critique report + validation results
    ↓
Stage 8: Evaluation & Scoring
    Module: src/evaluation/evaluator.py
    Input:  Test results + critique
    Output: Metrics and quality scores
    ↓
Stage 9: Dataset Logging
    Module: src/dataset/logger.py
    Input:  All stage outputs
    Output: Logged dataset entry
```

---

## Module Organization Patterns

### Rule 1: One Module Per Pipeline Stage

**Good:**
```
src/agents/reader_agent.py       # Stage 2
src/agents/planner_agent.py      # Stage 3
src/agents/implementer_agent.py  # Stage 5
src/agents/critic_agent.py       # Stage 7
```

**Bad:**
```
src/agents.py                    # ❌ All agents in one file
src/reader_planner.py            # ❌ Mixing stages
```

### Rule 2: Shared Functionality Goes to Base Classes or Utils

**Good:**
```
src/llm/base.py                  # Base LLM client
src/agents/base_agent.py         # Base agent (if needed)
src/utils/prompt_loader.py       # Shared prompt loading
```

**Bad:**
```
src/agents/reader_agent.py       # ❌ Contains LLM client code
src/generators/utils.py          # ❌ Vague naming
```

### Rule 3: Validators Live With Their Domain

**Good:**
```
src/agents/validators/problem_validator.py      # Validates Stage 2 output
src/agents/validators/algorithm_validator.py    # Validates Stage 3 output
src/agents/validators/code_validator.py         # Validates Stage 5 output
```

**Pattern:** `{module}/validators/{domain}_validator.py`

---

## Data Organization Patterns

### Prompts: Organized by Agent

```
data/prompts/
├── reader/
│   ├── system_prompt.txt           # Main system prompt
│   ├── few_shot_examples.json      # Example inputs/outputs
│   ├── v1_2024-12-25/              # Versioned prompts
│   └── performance_log.json        # Track effectiveness
├── planner/
│   └── ...
├── implementer/
│   └── ...
└── critic/
    └── ...
```

**Naming Convention:**
- System prompts: `system_prompt.txt`
- Few-shot: `few_shot_examples.json`
- Versions: `v{version}_{date}/`

### Cache: Organized by LLM Provider

```
data/cache/
├── claude/
│   ├── reader_responses/
│   ├── planner_responses/
│   └── ...
└── gpt/
    └── ...
```

**Cache Key Format:** `{agent}_{hash(prompt)}.json`

### Outputs: Organized by Pipeline Run

```
data/outputs/
├── run_001_upgma_2024-12-25/
│   ├── 1_input.json
│   ├── 2_problem.json
│   ├── 3_algorithm.json
│   ├── 4_pseudocode.txt
│   ├── 5_implementation.py
│   ├── 6_tests.py
│   ├── 7_critique.json
│   ├── 8_evaluation.json
│   └── 9_dataset_entry.json
├── run_002_kmeans_2024-12-26/
│   └── ...
```

**Pattern:** `run_{id}_{algorithm}_{date}/`

### Logs: Organized by Module and Date

```
data/logs/
├── app_2024-12-25.log              # Main application log
├── llm_api_2024-12-25.log          # API calls
├── pipeline_2024-12-25.log         # Pipeline execution
└── errors_2024-12-25.log           # Error-only log
```

---

## Configuration Patterns

### Configuration Files Live in config/

```
config/
├── model_config.yaml           # LLM settings, API keys
├── prompt_templates.yaml       # Reusable prompt fragments
├── logging_config.yaml         # Logging configuration
└── pipeline_config.yaml        # Pipeline orchestration settings
```

### model_config.yaml Structure

```yaml
llm_providers:
  claude:
    api_key: ${CLAUDE_API_KEY}  # From environment
    model: claude-sonnet-4-5-20250929
    max_tokens: 4000
    temperature: 0.0
    timeout: 60

  gpt:
    api_key: ${OPENAI_API_KEY}
    model: gpt-4
    max_tokens: 4000
    temperature: 0.0

rate_limiting:
  requests_per_minute: 50
  tokens_per_minute: 100000

caching:
  enabled: true
  ttl_seconds: 86400  # 24 hours

cost_tracking:
  budget_usd: 500
  alert_threshold: 0.8
```

### Loading Configuration

```python
from src.config.unified_config import UnifiedConfig

config = UnifiedConfig.load()
api_key = config.llm_providers.claude.api_key
max_tokens = config.llm_providers.claude.max_tokens
```

---

## Schema Organization

### All Pydantic Schemas in src/schemas/

```
src/schemas/
├── input_schema.py          # Stage 1 output
├── problem_schema.py        # Stage 2 output
├── algorithm_schema.py      # Stage 3 output
├── code_schema.py           # Stage 5 output (metadata)
├── test_schema.py           # Stage 6 output
├── critique_schema.py       # Stage 7 output
├── evaluation_schema.py     # Stage 8 output
└── dataset_schema.py        # Stage 9 output
```

**Pattern:** `{stage_name}_schema.py`

### Schema Naming Convention

```python
# Good
class ProblemDefinition(BaseModel): ...
class AlgorithmStep(BaseModel): ...
class CritiqueReport(BaseModel): ...

# Bad
class Problem(BaseModel): ...      # ❌ Too generic
class Data(BaseModel): ...         # ❌ Meaningless
class Output(BaseModel): ...       # ❌ Vague
```

---

## Where Does New Code Go?

### Decision Tree

```
Are you creating a NEW pipeline stage?
├─ Yes → src/{stage_type}/
│         Examples: src/agents/, src/generators/, src/verification/
│
└─ No → Is it SHARED across stages?
        ├─ Yes → Is it LLM-related?
        │        ├─ Yes → src/llm/
        │        └─ No  → src/utils/
        │
        └─ No → Put it WITH the stage that uses it
                 Example: src/agents/validators/problem_validator.py
```

### Examples

**Q:** Where does the Reader agent go?
**A:** `src/agents/reader_agent.py` (it's a pipeline stage agent)

**Q:** Where does prompt loading code go?
**A:** `src/utils/prompt_loader.py` (shared utility)

**Q:** Where does Claude API client go?
**A:** `src/llm/claude_client.py` (LLM integration)

**Q:** Where does problem validation go?
**A:** `src/agents/validators/problem_validator.py` (specific to Reader agent)

**Q:** Where do Pydantic models go?
**A:** `src/schemas/{domain}_schema.py` (centralized schemas)

**Q:** Where do pytest tests go?
**A:** `tests/{test_type}/test_{module}.py` (mirrors src/ structure)

**Q:** Where do agent prompts go?
**A:** `data/prompts/{agent_name}/` (data, not code)

**Q:** Where do generated implementations go?
**A:** `data/outputs/run_{id}_{algorithm}_{date}/` (pipeline outputs)

---

## Data Flow Patterns

### Stage-to-Stage Communication

**Pattern:** Each stage reads from previous stage's output

```python
# Stage 2: Reader Agent
input_text = load_json("data/outputs/run_001/1_input.json")
problem = reader_agent.extract(input_text)
save_json(problem, "data/outputs/run_001/2_problem.json")

# Stage 3: Planner Agent
problem = load_json("data/outputs/run_001/2_problem.json")
algorithm = planner_agent.reconstruct(problem)
save_json(algorithm, "data/outputs/run_001/3_algorithm.json")
```

**Key Points:**
- Each stage is independent
- Communication via JSON files
- Enables replay, debugging, and analysis

### Pipeline Orchestration

```python
# src/pipeline/orchestrator.py

class PipelineOrchestrator:
    def run(self, input_file: str, output_dir: str):
        # Stage 1
        structured_text = self.input_acquisition.process(input_file)

        # Stage 2
        problem = self.reader_agent.extract(structured_text)

        # Stage 3
        algorithm = self.planner_agent.reconstruct(problem)

        # ... continue through all stages

        # Save all outputs
        self.save_pipeline_run(output_dir, all_outputs)
```

---

## Testing Organization

### Test Directory Mirrors src/

```
tests/
├── unit/
│   ├── test_reader_agent.py
│   ├── test_planner_agent.py
│   └── test_claude_client.py
├── integration/
│   ├── test_pipeline_e2e.py
│   └── test_agent_integration.py
└── property/
    └── test_algorithm_properties.py
```

**Pattern:** `tests/{test_type}/test_{module}.py`

### Test Data Organization

```
tests/fixtures/
├── sample_papers/
│   ├── upgma_paper.pdf
│   └── kmeans_paper.pdf
├── expected_outputs/
│   ├── upgma_problem.json
│   └── kmeans_algorithm.json
└── mock_responses/
    └── claude_reader_response.json
```

---

## Documentation Organization

### Active Development Docs

```
dev/active/
├── paper-to-code-ai-system/
│   ├── plan.md           # Strategic plan
│   ├── context.md        # Key decisions
│   └── tasks.md          # Checklist
└── {other-tasks}/
```

### Reference Documentation

```
docs/
├── architecture.md       # Overall architecture
├── api.md               # API documentation
├── deployment.md        # Deployment guide
└── contributing.md      # Contribution guidelines
```

---

## Common Patterns

### Pattern 1: Agent Module Structure

```
src/agents/
├── base_agent.py                    # Abstract base (if needed)
├── reader_agent.py                  # Concrete agent
├── planner_agent.py
├── implementer_agent.py
├── critic_agent.py
└── validators/                      # Agent-specific validators
    ├── problem_validator.py
    ├── algorithm_validator.py
    └── code_validator.py
```

### Pattern 2: Generator Module Structure

```
src/generators/
├── base_generator.py                # Abstract base (if needed)
├── pseudocode_generator.py
├── test_generator.py
└── templates/                       # Generation templates
    ├── pseudocode_templates.py
    └── test_templates.py
```

### Pattern 3: Utilities Module Structure

```
src/utils/
├── rate_limiter.py
├── token_counter.py
├── cache.py
├── logger.py
└── prompt_loader.py
```

---

## Anti-Patterns (What NOT to Do)

### ❌ Don't Mix Pipeline Stages

**Bad:**
```python
# src/agents/reader_planner.py
class ReaderPlanner:  # ❌ Two stages in one
    def extract_and_plan(self): ...
```

**Good:**
```python
# src/agents/reader_agent.py
class ReaderAgent: ...

# src/agents/planner_agent.py
class PlannerAgent: ...
```

### ❌ Don't Put Data in src/

**Bad:**
```
src/
├── agents/
│   ├── reader_agent.py
│   └── prompts/              # ❌ Data in source code
│       └── system_prompt.txt
```

**Good:**
```
src/agents/reader_agent.py
data/prompts/reader/system_prompt.txt
```

### ❌ Don't Create Vague Module Names

**Bad:**
```
src/helpers.py       # ❌ What helpers?
src/misc.py          # ❌ Meaningless
src/stuff.py         # ❌ No
```

**Good:**
```
src/utils/prompt_loader.py
src/utils/json_validator.py
```

### ❌ Don't Duplicate Pipeline Logic

**Bad:**
```python
# Multiple files with pipeline orchestration
src/agents/reader_agent.py    # Has pipeline code ❌
src/run.py                     # Has pipeline code ❌
cli.py                         # Has pipeline code ❌
```

**Good:**
```python
# Single source of truth
src/pipeline/orchestrator.py  # All orchestration here ✅
```

---

## Resource Files

For detailed patterns, see:
- [Project Structure](resources/project-structure.md) - Complete directory tree
- [Module Organization](resources/module-organization.md) - Detailed module patterns
- [Data Management](resources/data-management.md) - Data organization strategies

---

## Quick Checklist

Before creating a new file/module, ask:

- [ ] Which pipeline stage does this belong to?
- [ ] Is this shared across stages? → `src/utils/` or `src/llm/`
- [ ] Is this stage-specific? → `src/{stage_type}/{name}.py`
- [ ] Is this a schema? → `src/schemas/{domain}_schema.py`
- [ ] Is this data? → `data/{category}/`
- [ ] Is this a test? → `tests/{test_type}/test_{module}.py`
- [ ] Does this follow naming conventions?
- [ ] Is this documented in dev-docs?

---

**Remember:** When in doubt, refer to the 9-stage pipeline architecture. Every module should map to a stage or be shared infrastructure.
