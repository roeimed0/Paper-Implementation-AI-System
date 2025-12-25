# Project Structure - Complete Reference

**Purpose:** Complete directory tree with file purposes and relationships

---

## Full Directory Tree

```
Paper-Implementation-AI-System/
│
├── .claude/                           # Claude Code infrastructure
│   ├── agents/                        # Specialized agents (10)
│   ├── skills/                        # Development skills
│   │   ├── genai-project-architecture/
│   │   └── python-llm-dev-guidelines/
│   ├── hooks/                         # Automation hooks
│   ├── commands/                      # Custom commands
│   └── settings.json                  # Claude Code config
│
├── src/                               # All source code
│   │
│   ├── agents/                        # Pipeline stage agents
│   │   ├── __init__.py
│   │   ├── base_agent.py             # Abstract base (optional)
│   │   ├── reader_agent.py           # Stage 2: Problem extraction
│   │   ├── planner_agent.py          # Stage 3: Algorithm reconstruction
│   │   ├── implementer_agent.py      # Stage 5: Code generation
│   │   ├── critic_agent.py           # Stage 7: Verification
│   │   └── validators/               # Stage-specific validation
│   │       ├── __init__.py
│   │       ├── problem_validator.py
│   │       ├── algorithm_validator.py
│   │       └── code_validator.py
│   │
│   ├── generators/                    # Code generators
│   │   ├── __init__.py
│   │   ├── pseudocode_generator.py   # Stage 4: Pseudocode
│   │   ├── test_generator.py         # Stage 6: Test generation
│   │   └── templates/
│   │       ├── pseudocode_templates.py
│   │       └── test_templates.py
│   │
│   ├── verification/                  # Verification systems
│   │   ├── __init__.py
│   │   └── cross_validator.py        # Reference validation
│   │
│   ├── evaluation/                    # Metrics and scoring
│   │   ├── __init__.py
│   │   ├── evaluator.py              # Stage 8: Evaluation
│   │   └── metrics.py                # Metric calculations
│   │
│   ├── dataset/                       # Dataset logging
│   │   ├── __init__.py
│   │   ├── logger.py                 # Stage 9: Logging
│   │   └── exporter.py               # Dataset export
│   │
│   ├── pipeline/                      # Pipeline orchestration
│   │   ├── __init__.py
│   │   └── orchestrator.py           # Stage 10: E2E pipeline
│   │
│   ├── input_acquisition/             # PDF/text processing
│   │   ├── __init__.py
│   │   ├── pdf_extractor.py          # Stage 1: PDF parsing
│   │   ├── text_cleaner.py           # Text normalization
│   │   ├── section_detector.py       # Section tagging
│   │   └── pipeline.py               # Input pipeline
│   │
│   ├── llm/                           # LLM client implementations
│   │   ├── __init__.py
│   │   ├── base.py                   # Abstract LLM client
│   │   ├── claude_client.py          # Claude API
│   │   ├── openai_client.py          # GPT API (future)
│   │   └── utils.py                  # LLM utilities
│   │
│   ├── utils/                         # Shared utilities
│   │   ├── __init__.py
│   │   ├── rate_limiter.py           # API rate limiting
│   │   ├── token_counter.py          # Token counting
│   │   ├── cache.py                  # Response caching
│   │   ├── logger.py                 # Logging setup
│   │   └── prompt_loader.py          # Prompt loading
│   │
│   ├── config/                        # Configuration management
│   │   ├── __init__.py
│   │   └── unified_config.py         # Config loader
│   │
│   ├── handlers/                      # Error handling
│   │   ├── __init__.py
│   │   └── error_handler.py          # Custom exceptions
│   │
│   └── schemas/                       # Pydantic data models
│       ├── __init__.py
│       ├── input_schema.py           # Stage 1 output
│       ├── problem_schema.py         # Stage 2 output
│       ├── algorithm_schema.py       # Stage 3 output
│       ├── code_schema.py            # Stage 5 metadata
│       ├── test_schema.py            # Stage 6 output
│       ├── critique_schema.py        # Stage 7 output
│       ├── evaluation_schema.py      # Stage 8 output
│       └── dataset_schema.py         # Stage 9 output
│
├── data/                              # All data files (not code)
│   │
│   ├── prompts/                       # Agent prompts (versioned)
│   │   ├── reader/
│   │   │   ├── system_prompt.txt
│   │   │   ├── few_shot_examples.json
│   │   │   ├── v1_2024-12-25/
│   │   │   └── performance_log.json
│   │   ├── planner/
│   │   │   └── ...
│   │   ├── implementer/
│   │   │   └── ...
│   │   └── critic/
│   │       └── ...
│   │
│   ├── cache/                         # API response cache
│   │   ├── claude/
│   │   │   ├── reader_responses/
│   │   │   ├── planner_responses/
│   │   │   └── ...
│   │   └── gpt/
│   │       └── ...
│   │
│   ├── outputs/                       # Generated implementations
│   │   ├── run_001_upgma_2024-12-25/
│   │   │   ├── 1_input.json
│   │   │   ├── 2_problem.json
│   │   │   ├── 3_algorithm.json
│   │   │   ├── 4_pseudocode.txt
│   │   │   ├── 5_implementation.py
│   │   │   ├── 6_tests.py
│   │   │   ├── 7_critique.json
│   │   │   ├── 8_evaluation.json
│   │   │   └── 9_dataset_entry.json
│   │   └── run_002_kmeans_2024-12-26/
│   │       └── ...
│   │
│   ├── logs/                          # Application logs
│   │   ├── app_2024-12-25.log
│   │   ├── llm_api_2024-12-25.log
│   │   ├── pipeline_2024-12-25.log
│   │   └── errors_2024-12-25.log
│   │
│   └── datasets/                      # Logged pipeline runs
│       ├── paper_to_code.db          # SQLite database
│       └── exports/
│           ├── training_data.jsonl
│           └── evaluation_set.jsonl
│
├── config/                            # Configuration files (YAML)
│   ├── model_config.yaml             # LLM settings, API keys
│   ├── prompt_templates.yaml         # Reusable prompt fragments
│   ├── logging_config.yaml           # Logging configuration
│   └── pipeline_config.yaml          # Pipeline settings
│
├── tests/                             # Test suite
│   ├── __init__.py
│   ├── conftest.py                   # Pytest configuration
│   │
│   ├── unit/                         # Unit tests
│   │   ├── test_reader_agent.py
│   │   ├── test_planner_agent.py
│   │   ├── test_implementer_agent.py
│   │   ├── test_critic_agent.py
│   │   ├── test_claude_client.py
│   │   ├── test_rate_limiter.py
│   │   └── ...
│   │
│   ├── integration/                  # Integration tests
│   │   ├── test_pipeline_e2e.py
│   │   ├── test_agent_integration.py
│   │   └── ...
│   │
│   ├── property/                     # Property-based tests
│   │   ├── test_algorithm_properties.py
│   │   └── ...
│   │
│   └── fixtures/                     # Test data
│       ├── sample_papers/
│       │   ├── upgma_paper.pdf
│       │   └── kmeans_paper.pdf
│       ├── expected_outputs/
│       │   ├── upgma_problem.json
│       │   └── kmeans_algorithm.json
│       └── mock_responses/
│           └── claude_reader_response.json
│
├── notebooks/                         # Jupyter notebooks
│   ├── dataset_analysis.ipynb
│   ├── prompt_testing.ipynb
│   ├── model_comparison.ipynb
│   └── failure_analysis.ipynb
│
├── examples/                          # Example usage scripts
│   ├── basic_pipeline_run.py
│   ├── single_agent_test.py
│   └── batch_processing.py
│
├── dev/                               # Development documentation
│   ├── README.md                     # Dev-docs pattern guide
│   ├── active/                       # Active work
│   │   └── paper-to-code-ai-system/
│   │       ├── plan.md
│   │       ├── context.md
│   │       └── tasks.md
│   └── archive/                      # Completed work (optional)
│
├── docs/                              # Reference documentation
│   ├── architecture.md
│   ├── api.md
│   ├── deployment.md
│   └── contributing.md
│
├── .env.example                       # Environment template
├── .gitignore
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Python project config
├── README.md                          # Project overview
└── LICENSE
```

---

## File Purposes by Category

### Core Pipeline Files

| File | Stage | Purpose |
|------|-------|---------|
| `src/input_acquisition/pdf_extractor.py` | 1 | PDF → text extraction |
| `src/agents/reader_agent.py` | 2 | Problem extraction from text |
| `src/agents/planner_agent.py` | 3 | Algorithm reconstruction |
| `src/generators/pseudocode_generator.py` | 4 | Pseudocode generation |
| `src/agents/implementer_agent.py` | 5 | Python code generation |
| `src/generators/test_generator.py` | 6 | Test suite generation |
| `src/agents/critic_agent.py` | 7 | Cross-stage verification |
| `src/evaluation/evaluator.py` | 8 | Metrics calculation |
| `src/dataset/logger.py` | 9 | Dataset logging |
| `src/pipeline/orchestrator.py` | 10 | E2E orchestration |

### Infrastructure Files

| File | Purpose |
|------|---------|
| `src/llm/base.py` | Abstract LLM client interface |
| `src/llm/claude_client.py` | Claude API implementation |
| `src/utils/rate_limiter.py` | API rate limiting |
| `src/utils/cache.py` | Response caching |
| `src/utils/token_counter.py` | Token counting & cost tracking |
| `src/utils/logger.py` | Structured logging |
| `src/config/unified_config.py` | Configuration loading |
| `src/handlers/error_handler.py` | Custom exceptions |

### Schema Files

| File | Purpose |
|------|---------|
| `src/schemas/input_schema.py` | Validates Stage 1 output |
| `src/schemas/problem_schema.py` | Validates Stage 2 output |
| `src/schemas/algorithm_schema.py` | Validates Stage 3 output |
| `src/schemas/code_schema.py` | Code metadata schema |
| `src/schemas/test_schema.py` | Test suite schema |
| `src/schemas/critique_schema.py` | Critique report schema |
| `src/schemas/evaluation_schema.py` | Evaluation metrics schema |
| `src/schemas/dataset_schema.py` | Dataset entry schema |

---

## Relationships and Dependencies

### Agent → LLM Client

```
src/agents/reader_agent.py
    → imports src/llm/claude_client.py
    → imports src/utils/prompt_loader.py
    → imports src/schemas/problem_schema.py
```

### Pipeline → All Stages

```
src/pipeline/orchestrator.py
    → imports src/input_acquisition/pipeline.py
    → imports src/agents/reader_agent.py
    → imports src/agents/planner_agent.py
    → imports src/generators/pseudocode_generator.py
    → imports src/agents/implementer_agent.py
    → imports src/generators/test_generator.py
    → imports src/agents/critic_agent.py
    → imports src/evaluation/evaluator.py
    → imports src/dataset/logger.py
```

### Config → Everything

```
src/config/unified_config.py
    ← used by ALL modules for settings
```

---

## Growth Patterns

### Adding a New Agent

1. Create `src/agents/new_agent.py`
2. Create `src/schemas/new_output_schema.py`
3. Create `data/prompts/new_agent/`
4. Add validator: `src/agents/validators/new_validator.py`
5. Add tests: `tests/unit/test_new_agent.py`
6. Update pipeline: `src/pipeline/orchestrator.py`

### Adding a New Utility

1. Create `src/utils/new_utility.py`
2. Add tests: `tests/unit/test_new_utility.py`
3. Import where needed

### Adding a New Data Category

1. Create `data/new_category/`
2. Document structure in this file
3. Update .gitignore if needed

---

**Key Principle:** Every file has a clear purpose and location. No orphaned files.
