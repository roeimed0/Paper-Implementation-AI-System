# Paper→Code AI System - Implementation Plan

**Last Updated:** 2025-12-25
**Status:** Phase 0 - Foundation (Mock-First Learning Approach)

---

## 🎯 STRATEGIC PIVOT - LEARNING-FOCUSED APPROACH

### New Primary Goal: Learn Professional GenAI Development
This project has been refocused from "build a production system" to **"learn market-ready GenAI development skills"** while building a portfolio-worthy project.

### Why Mock-First Development?
1. **Zero API Costs** - Learn without spending money on Claude/GPT API calls
2. **Professional Patterns** - Build the same architecture real companies use
3. **Faster Iteration** - No waiting for API calls during development
4. **Transferable Skills** - Everything you learn works with ANY LLM provider
5. **Portfolio Ready** - Demonstrates architecture skills, not just API usage

### Development Philosophy
> "Build abstractions that work with mocks first, then swap in real LLMs with a config change"

This is how professional teams build LLM applications - mock-driven development, clean abstractions, dependency injection.

---

## Executive Summary

### Project Vision
Build an AI-driven system that converts scientific algorithm descriptions (papers, lecture notes, method sections) into correct, testable, and verifiable software implementations through structured reasoning, multi-stage validation, and automated evaluation.

**Learning Focus:** Master the architecture, patterns, and practices of production GenAI systems.

### Core Principle
This is **NOT** a single-prompt solution. This is a **systems engineering project** that decomposes the translation task into explicit reasoning stages, each with clear responsibilities, validation points, and failure surfaces.

**Development Principle:** Build with mocks first, validate architecture, then integrate real LLMs.

### Success Criteria
- ✅ Professional LLM abstraction layer (works with any provider)
- ✅ Sophisticated mock LLM for testing (no API costs)
- ✅ Multi-agent orchestration architecture
- ✅ Comprehensive test suites (unit, property-based, integration)
- ✅ Portfolio-ready codebase demonstrating GenAI expertise
- ✅ Optional: Real LLM integration (Claude/GPT) when ready
- ✅ Measurable correctness metrics (not just "it runs")

---

## Current State Analysis

### What Exists
1. **Project Structure** (scaffolded):
   - `src/llm/` - Empty LLM client modules
   - `src/prompt_engineering/` - Empty prompt management modules
   - `src/utils/` - Empty utility modules
   - `config/` - Empty YAML configuration files
   - `data/` - Directory structure for cache/prompts/outputs

2. **GenAI Project Architecture** (documented):
   - Modular structure for LLM integration
   - Provider-agnostic design (Claude, GPT)
   - Configuration-driven approach

3. **Claude Code Infrastructure** (active):
   - 10 pre-configured agents
   - Skills system (TypeScript-focused, needs Python adaptation)
   - Dev-docs pattern
   - Hooks for automation

### What's Missing (Everything Critical)
1. ❌ LLM client implementations
2. ❌ Multi-agent orchestration system
3. ❌ Pipeline stages (9 stages defined below)
4. ❌ Verification framework
5. ❌ Evaluation metrics
6. ❌ Dataset logging system
7. ❌ Python-specific skills for guidance

---

## Proposed Future State

### System Architecture

```
Scientific Paper/Algorithm Description
            ↓
    [1. Input Acquisition]
    PDF/Text → Clean, Tagged Text
            ↓
    [2. Problem Extraction (Reader Agent)]
    Text → Structured Problem Definition (JSON)
            ↓
    [3. Algorithm Reconstruction (Planner Agent)]
    Problem → Deterministic Algorithm Steps (JSON)
            ↓
    [4. Pseudocode Generation]
    Steps → Language-Agnostic Pseudocode
            ↓
    [5. Code Implementation (Implementer Agent)]
    Pseudocode → Executable Python Code
            ↓
    [6. Test Generation]
    Code → Test Suite (unit/property/reference)
            ↓
    [7. Verification & Critique Loop]
    Tests → Validation Results + Critique
            ↓
    [8. Evaluation & Scoring]
    Results → Metrics (pass rate, completeness, consistency)
            ↓
    [9. Dataset Logging]
    All Stages → Training Dataset + Benchmark
```

### Technology Stack

**LLM Architecture:** Provider-Agnostic Abstraction Layer
- **Primary (Development):** Mock LLM Client (zero cost, full control)
- **Secondary (Learning):** Optional Ollama integration (free, local)
- **Production (Optional):** Claude API or GPT-4 (swap in later)
- **Design:** Abstract base class + dependency injection

**Language:** Python 3.11+
- Async/await for API calls (real and mocked)
- Type hints throughout (enforced with mypy)
- Pydantic for data validation
- ABC (Abstract Base Classes) for LLM abstraction

**Key Libraries:**
- `anthropic` - Claude API client
- `pydantic` - Schema validation
- `pytest` - Testing framework
- `hypothesis` - Property-based testing
- `pypdf` - PDF parsing
- `tiktoken` - Token counting
- `aiohttp` - Async HTTP

**Verification Stack:**
1. Unit tests (pytest)
2. Property-based tests (hypothesis)
3. Reference cross-validation (compare to known implementations)
4. (Optional) Formal methods for core logic (Z3, future phase)

---

## Implementation Phases

### Phase 0: Foundation (Week 1) - MOCK-FIRST APPROACH
**Goal:** Build production-ready LLM abstraction layer with sophisticated mocking (ZERO API COST)

**Philosophy:** Learn professional patterns by building with mocks first, swap in real LLMs later.

#### Tasks (Reprioritized for Learning):

1. **Environment Setup** ✅ COMPLETED
   - Created `requirements-minimal.txt` (without Jupyter due to Windows path issues)
   - Set up Python 3.11 conda environment
   - Configured `.env.example` template
   - Acceptance: Dependencies install successfully

2. **Base LLM Client Architecture** (CRITICAL - Do This First!)
   - Implement `src/llm/base.py` abstract base class
   - Define common interface: `async def generate(prompt, **kwargs) -> str`
   - Add error handling patterns (retry, timeout, fallback)
   - Support both mock and real implementations
   - Acceptance: Abstract base class with clear contract

3. **Mock LLM Client Implementation** (PRIMARY LEARNING TOOL)
   - Implement `src/llm/mock_client.py`
   - Sophisticated mock responses based on prompt patterns
   - Simulates: API delays, token counting, rate limits, occasional errors
   - Configuration: Can set response templates, failure rates
   - Acceptance: Mock client passes all abstract base class tests

   **Why Mock First?**
   - Zero API costs during development
   - Full control over responses for testing
   - Learn abstraction patterns (most valuable skill!)
   - Faster iteration (no network calls)

4. **Configuration System**
   - Implement `config/model_config.yaml` schema
   - Create `UnifiedConfig` class for loading configs
   - LLM provider selection: "mock", "claude", "openai", "ollama"
   - Acceptance: Config switches between mock/real with ONE parameter

5. **Logging System**
   - Implement structured logging (`src/utils/logger.py`)
   - Log mock vs real LLM calls differently
   - Track simulated costs for mocks
   - Acceptance: Clear visibility into which LLM is being used

6. **Claude Client Implementation** (OPTIONAL - Add When Ready)
   - Implement `src/llm/claude_client.py` (same interface as mock!)
   - Only needed if you want to test with real API
   - Can skip entirely for learning
   - Acceptance: Swappable with mock via config change

7. **Utility Modules**
   - `src/utils/rate_limiter.py` - Works with mock AND real clients
   - `src/utils/token_counter.py` - Simulated for mocks, real for API
   - `src/utils/cache.py` - Caches both mock and real responses
   - Acceptance: All utilities work with mock client

**Learning Outcomes:**
- ✅ Understand LLM abstraction architecture
- ✅ Master async/await patterns
- ✅ Learn dependency injection
- ✅ Build testable AI systems
- ✅ Zero API costs

**Risks:**
- Over-engineering mocks → Solution: Start simple, add sophistication as needed
- Mock responses unrealistic → Solution: Add prompt pattern matching

**Estimated Time:** 3-5 days (same time, better learning!)

---

### Phase 1: Input Acquisition (Week 2)
**Goal:** Convert papers/PDFs into clean, machine-readable, tagged text

#### Tasks:
1. **PDF Text Extraction**
   - Implement PDF → raw text extraction
   - Handle multi-column layouts
   - Extract equations (as text markers initially)
   - Acceptance: Can extract text from sample papers

2. **Text Cleaning & Normalization**
   - Remove headers/footers/page numbers
   - Normalize whitespace
   - Handle Unicode issues
   - Acceptance: Clean text ready for LLM processing

3. **Section Detection (Rule-Based)**
   - Identify: Abstract, Introduction, Methods, Results, Discussion
   - Tag algorithm descriptions
   - Mark mathematical sections
   - Acceptance: Sections correctly tagged in 80%+ of test papers

4. **Output Schema**
   - Define JSON schema for structured input
   - Sections with metadata
   - Algorithm text isolated
   - Acceptance: Valid JSON schema, passes validation

**Deliverable:** `src/input_acquisition/` module

**Risks:**
- PDF parsing quality → Solution: Multiple parsing libraries fallback
- Section detection accuracy → Solution: Start with simple heuristics, improve iteratively

**Estimated Time:** 4-6 days

---

### Phase 2: Problem Extraction - Reader Agent (Week 3)
**Goal:** Make implicit knowledge explicit - extract formal problem definition

#### Tasks:
1. **Reader Agent Prompt Design**
   - System prompt for problem extraction
   - Few-shot examples (2-3 algorithm papers)
   - JSON output schema definition
   - Acceptance: Prompt clearly defines extraction task

2. **Extraction Schema Design**
   ```json
   {
     "problem_definition": "...",
     "inputs": ["..."],
     "outputs": ["..."],
     "constraints": ["..."],
     "assumptions": ["..."],
     "edge_cases": ["..."],
     "mathematical_properties": ["..."]
   }
   ```
   - Pydantic models for validation
   - Acceptance: Schema validates correctly

3. **Reader Agent Implementation**
   - `src/agents/reader_agent.py`
   - Call Claude API with extraction prompt
   - Parse and validate JSON response
   - Handle malformed responses
   - Acceptance: Successfully extracts from 3 test papers

4. **Validation Layer**
   - Check completeness (all required fields)
   - Verify logical consistency
   - Flag ambiguities
   - Acceptance: Validation catches malformed extractions

**Deliverable:** `src/agents/reader_agent.py` + prompts in `data/prompts/reader/`

**Risks:**
- LLM misses implicit assumptions → Solution: Explicit prompt instructions, few-shot examples
- JSON parsing failures → Solution: Retry with schema correction prompts

**Estimated Time:** 5-7 days

---

### Phase 3: Algorithm Reconstruction - Planner Agent (Week 4)
**Goal:** Convert narrative algorithm descriptions into deterministic, unambiguous steps

#### Tasks:
1. **Planner Agent Prompt Design**
   - System prompt for algorithm reconstruction
   - Instructions: ordered steps, control flow, termination
   - Few-shot examples
   - Acceptance: Prompt produces clear step-by-step algorithms

2. **Algorithm Schema Design**
   ```json
   {
     "algorithm_name": "...",
     "steps": [
       {
         "step_number": 1,
         "description": "...",
         "inputs": ["..."],
         "outputs": ["..."],
         "control_flow": "sequential|loop|conditional",
         "termination_condition": "..."
       }
     ],
     "ambiguities_resolved": ["..."],
     "missing_information": ["..."]
   }
   ```
   - Acceptance: Schema covers all algorithm patterns

3. **Planner Agent Implementation**
   - `src/agents/planner_agent.py`
   - Takes problem extraction JSON as input
   - Produces deterministic algorithm description
   - Flags ambiguities explicitly
   - Acceptance: Reconstructs 3 test algorithms correctly

4. **Ambiguity Resolution Strategy**
   - Explicit marking of assumptions made
   - Flag missing information
   - Suggest alternatives for unclear steps
   - Acceptance: Ambiguities never silently ignored

**Deliverable:** `src/agents/planner_agent.py` + prompts

**Risks:**
- Algorithm ambiguities missed → Solution: Dedicated ambiguity-checking pass
- Over-specification → Solution: Validate against original paper

**Estimated Time:** 6-8 days

---

### Phase 4: Pseudocode Generation (Week 5)
**Goal:** Bridge reasoning and implementation with language-agnostic pseudocode

#### Tasks:
1. **Pseudocode Generator Prompt**
   - Convert algorithm steps → pseudocode
   - Explicit loops, conditions, data structures
   - No natural language ambiguity
   - Acceptance: Pseudocode is executable-like

2. **Pseudocode Templates**
   - Standard control flow patterns
   - Data structure initialization
   - Update rules
   - Acceptance: Templates cover common patterns

3. **Generator Implementation**
   - `src/generators/pseudocode_generator.py`
   - Takes algorithm JSON → produces pseudocode
   - Validates against algorithm steps
   - Acceptance: Pseudocode matches algorithm steps 1:1

4. **Validation Against Algorithm**
   - Every step from algorithm appears in pseudocode
   - No extra logic introduced
   - Control flow preserved
   - Acceptance: Automated validation passes

**Deliverable:** `src/generators/pseudocode_generator.py`

**Risks:**
- Pseudocode too abstract → Solution: Require explicit data structures
- Pseudocode too language-specific → Solution: Review against language-agnostic criteria

**Estimated Time:** 4-5 days

---

### Phase 5: Code Implementation - Implementer Agent (Week 6-7)
**Goal:** Produce clean, executable, testable Python code

#### Tasks:
1. **Implementer Agent Prompt Design**
   - Pseudocode → Python translation
   - Clean functions with docstrings
   - Type hints throughout
   - No paper-specific wording in code
   - Acceptance: Generated code is maintainable

2. **Code Generation Templates**
   - Function signatures
   - Docstring format (Google style)
   - Error handling patterns
   - Acceptance: Templates enforce consistency

3. **Implementer Agent Implementation**
   - `src/agents/implementer_agent.py`
   - Takes pseudocode → produces Python code
   - Validates syntax
   - Basic linting
   - Acceptance: Generated code runs without syntax errors

4. **Code Quality Validation**
   - Syntax check (compile)
   - Type checking (mypy)
   - Linting (ruff)
   - Acceptance: Generated code passes quality checks

5. **Traceability System**
   - Link code functions → pseudocode steps → algorithm steps → paper
   - Comments with step references
   - Acceptance: Full traceability implemented

**Deliverable:** `src/agents/implementer_agent.py` + generated code in `data/outputs/implementations/`

**Risks:**
- Generated code has bugs → Solution: Testing phase catches this
- Code not maintainable → Solution: Strict quality requirements in prompt

**Estimated Time:** 7-10 days

---

### Phase 6: Automated Test Generation (Week 8)
**Goal:** Verify correctness through comprehensive testing

#### Tasks:

#### 6.1: Unit Test Generation
1. **Test Generator Prompt**
   - Generate pytest tests
   - Small deterministic examples
   - Known expected outputs
   - Edge cases from problem extraction
   - Acceptance: Generates runnable tests

2. **Test Generator Implementation**
   - `src/generators/test_generator.py`
   - Synthetic test case creation
   - Expected output generation
   - Acceptance: Tests are executable

#### 6.2: Property-Based Testing
1. **Property Identification**
   - Extract invariants from algorithm
   - Mathematical properties
   - Structural constraints
   - Acceptance: Properties match algorithm guarantees

2. **Hypothesis Test Generation**
   - Generate property-based tests
   - Input generators with constraints
   - Property assertions
   - Acceptance: Property tests catch violations

#### 6.3: Reference Cross-Validation
1. **Reference Implementation Matching**
   - Identify comparable implementations (SciPy, BioPython, etc.)
   - Create comparison tests
   - Define tolerance thresholds
   - Acceptance: Comparison tests validate behavior

2. **Cross-Validation Framework**
   - `src/verification/cross_validator.py`
   - Run same inputs through both implementations
   - Compare outputs with tolerance
   - Acceptance: Framework validates correctly

**Deliverable:** `src/generators/test_generator.py` + `src/verification/` module

**Risks:**
- Generated tests don't catch real bugs → Solution: Require multiple test types
- Property tests too weak → Solution: Manual review of properties

**Estimated Time:** 6-8 days

---

### Phase 7: Verification & Critique Loop (Week 9)
**Goal:** Catch hallucinations, mismatches, and silent errors

#### Tasks:
1. **Critic Agent Prompt Design**
   - Compare paper ↔ algorithm ↔ pseudocode ↔ code
   - Flag inconsistencies
   - Check for missing assumptions
   - Verify mathematical correctness
   - Acceptance: Prompt catches common failure modes

2. **Critic Agent Implementation**
   - `src/agents/critic_agent.py`
   - Multi-stage comparison
   - Generates critique report
   - Acceptance: Identifies known inconsistencies

3. **Critique Validation**
   - Check completeness of critique
   - Verify findings are actionable
   - Acceptance: Critique provides clear next steps

4. **Revision Loop**
   - Failure triggers agent re-runs
   - Maximum iteration limit
   - Log all revisions
   - Acceptance: System doesn't accept failures

**Deliverable:** `src/agents/critic_agent.py` + revision framework

**Risks:**
- Critic too lenient → Solution: Strict critique prompts with examples
- Infinite revision loops → Solution: Max iterations + human escalation

**Estimated Time:** 5-7 days

---

### Phase 8: Evaluation & Scoring (Week 10)
**Goal:** Measure quality objectively with quantifiable metrics

#### Tasks:
1. **Metrics Definition**
   ```python
   metrics = {
       "test_pass_rate": float,          # % of tests passing
       "assumption_completeness": float,  # % of assumptions captured
       "structural_similarity": float,    # Code ↔ reference similarity
       "stage_consistency": float,        # Paper → code consistency
       "ambiguity_resolution": int,       # # of ambiguities flagged
       "critique_severity": int           # # of critical issues
   }
   ```
   - Acceptance: All metrics have clear calculation methods

2. **Evaluation Framework**
   - `src/evaluation/evaluator.py`
   - Calculate all metrics
   - Generate evaluation report
   - Store results in database
   - Acceptance: Framework runs on all pipeline outputs

3. **Scoring Thresholds**
   - Define "pass" thresholds for each metric
   - Overall quality score
   - Acceptance criteria for implementations
   - Acceptance: Thresholds validated against sample algorithms

4. **Comparison Framework**
   - Compare across different:
     - LLM models (Claude vs GPT)
     - Prompt variations
     - Algorithm types
   - Acceptance: Can objectively compare approaches

**Deliverable:** `src/evaluation/` module + metrics dashboard

**Risks:**
- Metrics don't capture real quality → Solution: Validate against human evaluation
- Gaming metrics → Solution: Multiple independent metrics

**Estimated Time:** 4-6 days

---

### Phase 9: Dataset Logging & Learning (Week 11)
**Goal:** Turn all outputs into reusable datasets for improvement

#### Tasks:
1. **Logging Schema Design**
   ```json
   {
     "paper_id": "...",
     "timestamp": "...",
     "stages": {
       "input": {...},
       "problem_extraction": {...},
       "algorithm_reconstruction": {...},
       "pseudocode": {...},
       "code": {...},
       "tests": {...},
       "critique": {...},
       "evaluation": {...}
     },
     "failures": [...],
     "revisions": [...],
     "final_metrics": {...}
   }
   ```
   - Acceptance: Schema captures full pipeline

2. **Dataset Storage**
   - SQLite database for structured data
   - JSON files for stage outputs
   - Git for code versioning
   - Acceptance: All data persisted correctly

3. **Dataset Export**
   - Export to training-friendly formats
   - Filtering and querying
   - Acceptance: Can export subsets for analysis

4. **Analysis Tools**
   - Failure pattern analysis
   - Success factor identification
   - Prompt effectiveness tracking
   - Acceptance: Can identify improvement opportunities

**Deliverable:** `src/dataset/` module + analysis notebooks

**Risks:**
- Too much data → Solution: Efficient storage, sampling strategies
- Data not useful for training → Solution: Design with future fine-tuning in mind

**Estimated Time:** 5-6 days

---

### Phase 10: Integration & End-to-End Testing (Week 12)
**Goal:** Orchestrate all stages into working pipeline

#### Tasks:
1. **Pipeline Orchestrator**
   - `src/pipeline/orchestrator.py`
   - Runs all stages sequentially
   - Handles failures gracefully
   - Progress tracking
   - Acceptance: Full pipeline runs on test paper

2. **End-to-End Tests**
   - Test complete pipeline on 3-5 papers
   - Verify all stages execute
   - Check output quality
   - Acceptance: Pipeline produces valid implementations

3. **Error Recovery**
   - Retry logic for transient failures
   - Fallback strategies
   - Human intervention points
   - Acceptance: Pipeline doesn't crash on errors

4. **CLI Interface**
   - Command-line tool for running pipeline
   - Configuration options
   - Progress display
   - Acceptance: Usable CLI for end-users

**Deliverable:** Working end-to-end system

**Estimated Time:** 5-7 days

---

## Risk Assessment

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| LLM API instability | Medium | High | Retry logic, fallback to GPT |
| JSON parsing failures | High | Medium | Schema validation, retry with corrections |
| Test generation insufficient | Medium | High | Multiple test types, manual review |
| Paper parsing quality | High | Medium | Multiple parsing libraries, manual fallback |
| Cost overruns (API) | Medium | Medium | Token budgets, caching, monitoring |
| Verification too lenient | Medium | Critical | Strict prompts, human validation samples |

### Project Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Scope creep | High | Medium | Stick to 9-stage pipeline, defer enhancements |
| Over-engineering | Medium | Medium | MVP mindset, iterate based on results |
| Time underestimation | Medium | Low | Phase-based approach allows re-estimation |

---

## Success Metrics

### MVP Success Criteria (End of Week 12)
- ✅ Pipeline runs end-to-end on 3 test algorithms
- ✅ Generated code passes 80%+ of tests
- ✅ All 9 stages implemented and integrated
- ✅ Evaluation metrics track quality
- ✅ Dataset logging captures all transformations

### Quality Metrics
- **Correctness:** 80%+ test pass rate
- **Completeness:** 90%+ assumptions captured
- **Traceability:** 100% code → paper linkage
- **Reproducibility:** Same input produces same output

### Learning Metrics
- Dataset contains 10+ paper → code transformations
- Failure patterns documented and categorized
- Prompt effectiveness measured and tracked

---

## Timeline Estimates

### Optimistic (12 weeks)
- All phases on schedule
- Minimal blockers
- Clean execution

### Realistic (14-16 weeks)
- Expected debugging time
- Prompt iteration
- Integration challenges

### Pessimistic (18-20 weeks)
- Significant technical challenges
- Multiple prompt redesigns
- Verification issues

---

## Next Steps

### Immediate (This Week)
1. Review this plan with stakeholders
2. Set up development environment (Phase 0, Task 1)
3. Create Python-specific Claude Code skills
4. Begin implementation of Phase 0

### Week 2
- Complete Phase 0 (foundation)
- Begin Phase 1 (input acquisition)
- Create first test papers dataset

### Month 1 Goal
- Phases 0-2 complete
- Can extract problems from papers
- Can reconstruct algorithms

### Month 3 Goal
- Full pipeline MVP
- 3 successful paper → code transformations
- Documented lessons learned

---

## Open Questions

1. **Algorithm Selection:** Which papers/algorithms for initial testing?
   - Suggestion: Start with well-known algorithms (UPGMA, k-means, PageRank)

2. **Reference Implementations:** Which libraries for cross-validation?
   - BioPython, SciPy, scikit-learn, NetworkX

3. **Formal Verification:** Include in MVP or defer to Phase 2?
   - Recommendation: Defer to post-MVP, focus on test-based verification

4. **Cost Management:** API budget for development?
   - Estimate: $200-500 for MVP with caching

5. **Prompt Library:** Build reusable prompts from start?
   - Recommendation: Yes, store in `data/prompts/` with version control

---

## Conclusion

This is a **12-week minimum viable product** that demonstrates:
- AI systems engineering capability
- Multi-agent orchestration
- Verification-driven development
- Dataset creation for future improvement

The project is **ambitious but achievable** with disciplined execution and the right infrastructure (Claude Code skills, dev-docs tracking, automated testing).

**Next Action:** Review plan, adjust as needed, then begin Phase 0 implementation.
