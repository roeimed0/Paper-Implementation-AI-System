# Data Management Patterns

**Purpose:** Guidelines for organizing and managing data files

---

## Core Principle

**Data is separate from code:**
- Code lives in `src/`
- Data lives in `data/`
- Never mix

---

## Prompt Management

### Directory Structure

```
data/prompts/
├── reader/
│   ├── system_prompt.txt
│   ├── few_shot_examples.json
│   ├── v1_2024-12-25/          # Versioned
│   │   ├── system_prompt.txt
│   │   └── metadata.json
│   └── performance_log.json    # Track effectiveness
├── planner/
├── implementer/
└── critic/
```

### Versioning Pattern

**When to version:**
- Major prompt changes
- Performance improvements/regressions
- Experimentation

**Version format:** `v{number}_{date}/`

**Metadata:**
```json
{
  "version": "v1",
  "date": "2024-12-25",
  "description": "Initial reader prompt",
  "performance": {
    "test_accuracy": 0.85,
    "avg_cost_usd": 0.02
  }
}
```

### Prompt Loading

```python
from src.utils.prompt_loader import PromptLoader

prompts = PromptLoader("data/prompts/reader")
system_prompt = prompts.load("system_prompt.txt")
examples = prompts.load_json("few_shot_examples.json")
```

---

## Cache Management

### Structure

```
data/cache/
├── claude/
│   ├── reader_responses/
│   │   └── {hash}.json
│   ├── planner_responses/
│   └── implementer_responses/
└── gpt/
    └── ...
```

### Cache Key Generation

**Pattern:** `{agent}_{hash(prompt + params)}.json`

**Example:**
```python
import hashlib

def cache_key(agent: str, prompt: str, **params) -> str:
    content = f"{prompt}{sorted(params.items())}"
    hash_val = hashlib.sha256(content.encode()).hexdigest()[:16]
    return f"{agent}_{hash_val}.json"
```

### Cache Expiry

**TTL:** 24 hours (configurable)

**Cleanup:** Daily cron job or on-demand

---

## Output Management

### Run Organization

**Pattern:** `run_{id}_{algorithm}_{date}/`

**Structure:**
```
data/outputs/
└── run_001_upgma_2024-12-25/
    ├── 1_input.json          # Stage 1 output
    ├── 2_problem.json        # Stage 2 output
    ├── 3_algorithm.json      # Stage 3 output
    ├── 4_pseudocode.txt      # Stage 4 output
    ├── 5_implementation.py   # Stage 5 output
    ├── 6_tests.py            # Stage 6 output
    ├── 7_critique.json       # Stage 7 output
    ├── 8_evaluation.json     # Stage 8 output
    ├── 9_dataset_entry.json  # Stage 9 output
    └── metadata.json         # Run metadata
```

### Metadata Format

```json
{
  "run_id": "001",
  "algorithm": "upgma",
  "date": "2024-12-25",
  "paper_source": "path/to/paper.pdf",
  "total_cost_usd": 0.15,
  "total_tokens": 15000,
  "duration_seconds": 120,
  "success": true,
  "final_metrics": {
    "test_pass_rate": 0.95,
    "critique_severity": "low"
  }
}
```

---

## Log Management

### Log Files

```
data/logs/
├── app_2024-12-25.log           # Main application
├── llm_api_2024-12-25.log       # API calls only
├── pipeline_2024-12-25.log      # Pipeline execution
└── errors_2024-12-25.log        # Errors only
```

### Log Rotation

**Daily rotation:** New file per day
**Retention:** 30 days
**Compression:** After 7 days

### Log Format

**JSON structured logging:**
```json
{
  "timestamp": "2024-12-25T10:30:45Z",
  "level": "INFO",
  "module": "reader_agent",
  "message": "Problem extraction started",
  "extra": {
    "stage": "reader",
    "run_id": "001",
    "text_length": 5000
  }
}
```

---

## Dataset Management

### Database Schema

**SQLite:** `data/datasets/paper_to_code.db`

**Tables:**
```sql
CREATE TABLE pipeline_runs (
    run_id TEXT PRIMARY KEY,
    algorithm TEXT,
    date TIMESTAMP,
    success BOOLEAN,
    metrics JSON
);

CREATE TABLE stage_outputs (
    id INTEGER PRIMARY KEY,
    run_id TEXT,
    stage_number INTEGER,
    output JSON,
    FOREIGN KEY(run_id) REFERENCES pipeline_runs(run_id)
);

CREATE TABLE failures (
    id INTEGER PRIMARY KEY,
    run_id TEXT,
    stage_number INTEGER,
    error_message TEXT,
    timestamp TIMESTAMP
);
```

### Export Formats

**Training data (JSONL):**
```jsonl
{"input": "...", "output": "...", "stage": "reader"}
{"input": "...", "output": "...", "stage": "planner"}
```

**Evaluation set:**
```json
[
  {
    "algorithm": "upgma",
    "paper": "...",
    "expected_output": {...},
    "metrics": {...}
  }
]
```

---

## Backup Strategy

### What to Backup

**Critical:**
- `data/prompts/` - Version controlled prompts
- `data/datasets/` - Logged pipeline runs
- `data/outputs/` - Generated implementations

**Non-critical (regenerable):**
- `data/cache/` - API response cache
- `data/logs/` - Application logs

### Backup Schedule

- **Prompts:** Git version control (continuous)
- **Datasets:** Daily backup
- **Outputs:** Weekly backup

---

## Cleanup Policies

### Cache

**When:** Cache exceeds 1GB or daily
**Keep:** Most recent 7 days
**Delete:** Expired entries (TTL > 24h)

### Logs

**When:** Daily rotation
**Keep:** 30 days
**Compress:** After 7 days
**Delete:** After 30 days

### Outputs

**When:** Manual or when storage full
**Keep:** Successful runs
**Archive:** Failed runs (for analysis)
**Delete:** After analysis complete

---

## .gitignore Patterns

**Ignore:**
```gitignore
# Cache
data/cache/**

# Logs
data/logs/**

# Outputs (generated)
data/outputs/**

# Environment
.env

# Datasets (large)
data/datasets/*.db
```

**Track:**
```gitignore
# Prompts (versioned)
!data/prompts/**

# Example outputs (small, for testing)
!data/outputs/examples/**

# Dataset schema
!data/datasets/schema.sql
```

---

## Data Access Patterns

### Reading Prompts

```python
prompts = PromptLoader("data/prompts/reader")
system_prompt = prompts.load("system_prompt.txt")
```

### Writing Outputs

```python
output_dir = f"data/outputs/run_{run_id}_{algorithm}_{date}"
Path(output_dir).mkdir(parents=True, exist_ok=True)

with open(f"{output_dir}/2_problem.json", "w") as f:
    json.dump(problem.model_dump(), f, indent=2)
```

### Logging Dataset Entry

```python
from src.dataset.logger import DatasetLogger

logger = DatasetLogger("data/datasets/paper_to_code.db")
logger.log_run(run_id, all_stages, metadata)
```

---

**Principle:** Data is organized, versioned, and managed separately from code. Clear retention and cleanup policies.
