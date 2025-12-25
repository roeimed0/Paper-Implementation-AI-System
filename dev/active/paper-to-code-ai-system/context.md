# Paper→Code AI System - Context

**Last Updated:** 2025-12-25
**Status:** Phase 0 - Mock-First Learning Approach

---

## 🎯 STRATEGIC PIVOT (2025-12-25)

### Primary Goal Shift
**FROM:** "Build a production paper-to-code system with Claude API"
**TO:** "Learn professional GenAI development skills while building portfolio project"

### Key Insight
User wants to **learn GenAI development for career/market entry**, not spend money on API calls during learning. This is actually a BETTER approach for skill development!

### Mock-First Development Philosophy
Build professional LLM abstractions with mocks first:
- Zero API costs during development
- Learn architecture patterns (transferable to any LLM)
- Faster iteration (no network delays)
- Better testing practices
- Swap in real LLMs later with ONE config change

This is how professional teams build LLM apps!

---

## SESSION PROGRESS

### ✅ COMPLETED
- Reviewed GenAI Project Architecture (README.md)
- Reviewed Claude Code Infrastructure Architecture
- Created comprehensive 12-week implementation plan
- Created 2 custom skills (genai-project-architecture, python-llm-dev-guidelines)
- **Phase 0.1 Environment Setup:**
  - Created requirements.txt + requirements-minimal.txt (no Jupyter due to Windows paths)
  - Set up Python 3.11 conda environment
  - Created .env.example, SETUP.md
  - Updated .gitignore for Python
  - Created data directory structure
- **Strategic Pivot:**
  - Updated plan.md for mock-first development
  - Refocused on learning outcomes vs production deployment

### 🟡 IN PROGRESS
- Updating dev-docs to reflect mock-first approach
- File: `plan.md` ✅ UPDATED
- File: `context.md` (this file) - IN PROGRESS
- File: `tasks.md` - NEEDS UPDATE

### 📋 NEXT STEPS
1. Update tasks.md with mock-first priorities
2. Commit Phase 0.2 - Learning-focused architecture updates
3. Begin Phase 0 implementation:
   - Build BaseLLMClient abstract class
   - Build MockLLMClient (PRIMARY LEARNING TOOL)
   - Build configuration system
   - Add optional Claude client later

### ⚠️ BLOCKERS
None - mock-first approach removes API key blocker!

---

## Key Decisions Made

### 1. LLM Architecture (UPDATED 2025-12-25)
**Decision:** Build provider-agnostic abstraction layer with mock-first development
**Primary LLM:** MockLLMClient (sophisticated mock, zero cost)
**Secondary LLM:** Optional - Claude/GPT/Ollama (swap in later)

**Reasoning:**
- Learn professional architecture patterns (most valuable skill!)
- Zero API costs during development
- Faster iteration without network calls
- Build once, works with ANY LLM provider
- Optional: Add real LLM when ready with ONE config change

**Alternatives Considered:**
- ~~Claude API first~~ → Deferred (costs money, not needed for learning)
- ~~GPT-4~~ → Deferred (costs money)
- ✅ Mock LLM → PRIMARY (free, teaches architecture!)
- Open-source (Ollama) → Optional secondary path

### 2. Verification Strategy
**Decision:** Layered verification stack
**Components:**
1. Unit tests with mocks (mandatory baseline)
2. Property-based testing (invariants)
3. Integration tests (mock → real LLM swapping)
4. Reference cross-validation (when using real LLMs)

**Reasoning:** Each layer catches different failure types, mocks enable comprehensive testing

### 3. Development Sequence
**Decision:** Dev-docs first, skills second, mock implementation third
**Reasoning:**
- Requirements-driven infrastructure
- Build foundational abstractions before agents
- Professional software engineering practice
- Skills guide actual needs from plan

### 4. Learning Focus Over Production (NEW)
**Decision:** Optimize for skill development and portfolio value
**Priorities:**
1. Learn abstraction architecture patterns
2. Master async/await with LLMs
3. Build testable AI systems
4. Create portfolio-ready codebase
5. Optional: Deploy with real LLMs later

**Reasoning:** Market wants developers who understand architecture, not just API usage

### 4. Project Scope (MVP)
**Decision:** 12-week timeline for full 9-stage pipeline
**Scope:**
- All 9 pipeline stages implemented
- 3 test algorithms successfully transformed
- Dataset logging active
- Evaluation metrics defined

**Explicitly OUT of MVP:**
- Formal verification (Z3/Coq)
- Fine-tuning custom models
- Multi-language code generation
- Web interface

---

## Architecture Overview

### 9-Stage Pipeline

```
1. Input Acquisition
   └─ PDF/Text → Clean, Tagged Text

2. Problem Extraction (Reader Agent)
   └─ Text → Problem Definition JSON

3. Algorithm Reconstruction (Planner Agent)
   └─ Problem → Algorithm Steps JSON

4. Pseudocode Generation
   └─ Steps → Language-Agnostic Pseudocode

5. Code Implementation (Implementer Agent)
   └─ Pseudocode → Python Code

6. Test Generation
   └─ Code → Test Suite (unit/property/reference)

7. Verification & Critique
   └─ Tests → Validation + Critique Report

8. Evaluation & Scoring
   └─ Results → Objective Metrics

9. Dataset Logging
   └─ All Stages → Training Dataset
```

### Agent Architecture

**3 Primary Agents:**
1. **Reader Agent** - Problem extraction from papers
2. **Planner Agent** - Algorithm reconstruction
3. **Implementer Agent** - Code generation

**1 Validation Agent:**
4. **Critic Agent** - Cross-stage verification

### Technology Stack

**Core:**
- Python 3.11+
- Claude API (anthropic SDK)
- Pydantic (validation)
- pytest (testing)

**Utilities:**
- hypothesis (property testing)
- pypdf (PDF parsing)
- tiktoken (token counting)
- aiohttp (async HTTP)

**Storage:**
- SQLite (dataset logging)
- JSON (stage outputs)
- Git (version control)

---

## Key Files and Their Purpose

### Project Structure

#### Configuration Files
**config/model_config.yaml**
- API keys (Claude, GPT)
- Model parameters (temperature, max_tokens)
- Cost budgets and limits
- Status: Empty, needs implementation (Phase 0)

**config/prompt_templates.yaml**
- Reusable prompt templates
- Few-shot examples
- Agent system prompts
- Status: Empty, needs population (Phases 2-5)

**config/logging_config.yaml**
- Log levels and formats
- Output destinations
- Status: Empty, needs configuration (Phase 0)

#### Source Code

**src/llm/base.py**
- Abstract base class for LLM clients
- Common interface: generate(), count_tokens(), etc.
- Status: Empty, needs implementation (Phase 0, Task 4)

**src/llm/claude_client.py**
- Claude API client implementation
- Async calls, error handling, rate limiting
- Status: Empty, needs implementation (Phase 0, Task 5)

**src/llm/openai_client.py**
- OpenAI GPT client (for comparison)
- Status: Empty, deferred to later phase

**src/utils/rate_limiter.py**
- Token bucket rate limiting
- Status: Empty, needs implementation (Phase 0, Task 6)

**src/utils/token_counter.py**
- Accurate token counting for cost tracking
- Status: Empty, needs implementation (Phase 0, Task 6)

**src/utils/cache.py**
- Response caching with TTL
- Status: Empty, needs implementation (Phase 0, Task 6)

**src/utils/logger.py**
- Structured logging setup
- Status: Empty, needs implementation (Phase 0, Task 3)

#### Agents (to be created)

**src/agents/reader_agent.py**
- Problem extraction from papers
- Outputs: Problem definition JSON
- Status: Not created yet (Phase 2)

**src/agents/planner_agent.py**
- Algorithm reconstruction from problem
- Outputs: Algorithm steps JSON
- Status: Not created yet (Phase 3)

**src/agents/implementer_agent.py**
- Code generation from pseudocode
- Outputs: Python implementation
- Status: Not created yet (Phase 5)

**src/agents/critic_agent.py**
- Cross-stage verification
- Outputs: Critique report
- Status: Not created yet (Phase 7)

#### Generators (to be created)

**src/generators/pseudocode_generator.py**
- Algorithm steps → pseudocode
- Status: Not created yet (Phase 4)

**src/generators/test_generator.py**
- Code → test suite generation
- Status: Not created yet (Phase 6)

#### Verification (to be created)

**src/verification/cross_validator.py**
- Reference implementation comparison
- Status: Not created yet (Phase 6)

#### Evaluation (to be created)

**src/evaluation/evaluator.py**
- Metrics calculation and reporting
- Status: Not created yet (Phase 8)

#### Dataset (to be created)

**src/dataset/logger.py**
- Pipeline output logging
- Status: Not created yet (Phase 9)

#### Pipeline (to be created)

**src/pipeline/orchestrator.py**
- End-to-end pipeline execution
- Status: Not created yet (Phase 10)

---

## Data Organization

### data/prompts/
**Purpose:** Store all agent prompts and templates
**Structure:**
```
data/prompts/
├── reader/
│   ├── system_prompt.txt
│   └── few_shot_examples.json
├── planner/
│   ├── system_prompt.txt
│   └── few_shot_examples.json
├── implementer/
│   └── ...
└── critic/
    └── ...
```
**Status:** Directory exists, empty

### data/cache/
**Purpose:** Cache LLM API responses
**Structure:**
```
data/cache/
├── claude_responses/
└── gpt_responses/
```
**Status:** Directory exists, empty

### data/outputs/
**Purpose:** Store pipeline outputs
**Structure:**
```
data/outputs/
├── implementations/
│   ├── algorithm_name_v1.py
│   └── algorithm_name_v2.py
├── tests/
│   └── test_algorithm_name.py
└── evaluations/
    └── algorithm_name_eval.json
```
**Status:** Directory exists, empty

### data/logs/
**Purpose:** Application logs
**Status:** Not created yet (Phase 0)

---

## Claude Code Infrastructure Status

### Existing (Ready to Use)
1. **Agents (10 available):**
   - code-architecture-reviewer
   - documentation-architect
   - plan-reviewer
   - web-research-specialist
   - Others...

2. **Commands:**
   - /dev-docs (used to create this plan)
   - /dev-docs-update

3. **Hooks:**
   - skill-activation-prompt.sh (active)
   - post-tool-use-tracker.sh (active)
   - tsc-check.sh (TypeScript-specific, needs replacement)

### To Be Created
1. **Skills (2 needed):**
   - genai-project-architecture
   - python-llm-dev-guidelines

2. **Hooks (Python-specific):**
   - python-lint-check.sh (ruff/black/mypy)
   - requirements-check.sh

---

## Patterns and Conventions

### Code Style
- Type hints on all functions
- Google-style docstrings
- Max line length: 100 characters
- Use Pydantic for data validation
- Async/await for all I/O operations

### Naming Conventions
- Files: snake_case.py
- Classes: PascalCase
- Functions/variables: snake_case
- Constants: UPPER_SNAKE_CASE
- Private methods: _leading_underscore

### Error Handling
- Custom exceptions in src/handlers/error_handler.py
- Never silent failures
- Log all errors with context
- Retry logic for transient API failures

### Testing
- Test files: test_*.py
- Use pytest fixtures
- Property tests in separate files: test_*_properties.py
- Reference tests: test_*_reference.py

### Prompts
- Store in data/prompts/{agent_name}/
- Version control all prompts
- Include metadata (date, version, performance)

### Logging
- Structured JSON logs
- Include: timestamp, level, module, message, context
- Separate log files per module

---

## Quick Resume Instructions

**If returning to this project after a break:**

1. **Read this file** (context.md) for current state
2. **Check plan.md** for overall strategy
3. **Review tasks.md** for what's done and what's next
4. **Look at SESSION PROGRESS** section above
5. **Continue from NEXT STEPS**

**Current Status:** Plan created, about to identify skill requirements and create targeted skills

**Immediate Next Action:**
- Review plan with user
- Identify skill needs from plan
- Create genai-project-architecture skill
- Create python-llm-dev-guidelines skill
- Begin Phase 0 implementation

---

## Important Constraints

### Budget Constraints
- API costs: Estimate $200-500 for MVP
- Use caching aggressively
- Token counting for all requests
- Set hard limits in config

### Time Constraints
- 12-week MVP target
- Weekly milestones
- Phase-based delivery

### Quality Constraints
- 80%+ test pass rate minimum
- 100% code → paper traceability
- All assumptions explicitly documented
- No silent failures

### Technical Constraints
- Python 3.11+ required
- Async/await mandatory for API calls
- Pydantic for all data validation
- Type hints required

---

## Resources and References

### Documentation
- Claude API: https://docs.anthropic.com/
- Pydantic: https://docs.pydantic.dev/
- Hypothesis: https://hypothesis.readthedocs.io/
- pytest: https://docs.pytest.org/

### Example Papers for Testing
- UPGMA algorithm (phylogenetics)
- k-means clustering
- PageRank algorithm
- Needleman-Wunsch alignment

### Reference Implementations
- BioPython (bio algorithms)
- SciPy (scientific computing)
- scikit-learn (ML algorithms)
- NetworkX (graph algorithms)

---

## Notes for Future Sessions

### What Went Well
- Clear architectural decisions made early
- Comprehensive planning before implementation
- Strategic choice of Claude API (simplifies MVP)
- Layered verification approach is sound

### What to Watch
- API costs - monitor closely
- Prompt effectiveness - log and measure
- Test generation quality - validate manually
- Critique agent leniency - strict prompts needed

### Ideas for Later
- Fine-tune on logged dataset (post-MVP)
- Multi-language code generation (Python → C++/Java)
- Web interface for paper upload
- Formal verification with Z3 for core logic
- Automated paper corpus processing

---

**End of Context File**
