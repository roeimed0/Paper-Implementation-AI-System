# Module Organization Patterns

**Purpose:** Detailed guidelines for organizing code within modules

---

## Module Structure Patterns

### Agent Module Pattern

**Structure:**
```
src/agents/
├── base_agent.py          # Optional: Shared agent functionality
├── {agent_name}_agent.py  # Concrete agent implementation
└── validators/            # Agent-specific validators
    └── {domain}_validator.py
```

**Example:**
```python
# src/agents/reader_agent.py
from src.llm.claude_client import ClaudeClient
from src.utils.prompt_loader import PromptLoader
from src.schemas.problem_schema import ProblemDefinition

class ReaderAgent:
    def __init__(self, llm_client: ClaudeClient):
        self.llm = llm_client
        self.prompts = PromptLoader("data/prompts/reader")

    async def extract(self, text: str) -> ProblemDefinition:
        prompt = self.prompts.load("system_prompt.txt")
        response = await self.llm.generate_json(prompt, ProblemDefinition)
        return response
```

---

## Dependency Injection Pattern

**Always use constructor injection:**

```python
# Good ✅
class PlannerAgent:
    def __init__(self, llm_client: ClaudeClient, validator: AlgorithmValidator):
        self.llm = llm_client
        self.validator = validator

# Bad ❌
class PlannerAgent:
    def __init__(self):
        self.llm = ClaudeClient()  # Hard-coded dependency
```

---

## Schema Import Pattern

**Always import from src/schemas/:**

```python
# Good ✅
from src.schemas.problem_schema import ProblemDefinition
from src.schemas.algorithm_schema import Algorithm

# Bad ❌
class ProblemDefinition(BaseModel):  # Defined in agent file
    ...
```

---

## Naming Conventions

### Files
- Agents: `{name}_agent.py`
- Generators: `{name}_generator.py`
- Validators: `{domain}_validator.py`
- Utilities: `{function}.py`
- Schemas: `{domain}_schema.py`

### Classes
- Agents: `{Name}Agent`
- Generators: `{Name}Generator`
- Validators: `{Domain}Validator`
- Schemas: `{Domain}` or `{Domain}Definition`

### Functions
- snake_case
- Descriptive verbs: `extract_`, `generate_`, `validate_`

---

## Import Organization

**Order:**
1. Standard library
2. Third-party
3. Local project

**Example:**
```python
# Standard library
import asyncio
from pathlib import Path
from typing import Dict, List

# Third-party
from pydantic import BaseModel
import anthropic

# Local project
from src.llm.claude_client import ClaudeClient
from src.schemas.problem_schema import ProblemDefinition
from src.utils.logger import get_logger
```

---

## Error Handling Pattern

**Use custom exceptions:**

```python
from src.handlers.error_handler import (
    LLMAPIError,
    ValidationError,
    PipelineError
)

try:
    result = await self.llm.generate(prompt)
except anthropic.APIError as e:
    raise LLMAPIError(f"Claude API failed: {e}")
except ValidationError as e:
    raise PipelineError(f"Stage 2 validation failed: {e}")
```

---

## Logging Pattern

**Use structured logging:**

```python
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ReaderAgent:
    async def extract(self, text: str) -> ProblemDefinition:
        logger.info("Starting problem extraction", extra={
            "stage": "reader",
            "text_length": len(text)
        })

        try:
            result = await self.llm.generate_json(...)
            logger.info("Extraction successful", extra={
                "problem_size": len(result.model_dump())
            })
            return result
        except Exception as e:
            logger.error("Extraction failed", extra={
                "error": str(e),
                "stage": "reader"
            })
            raise
```

---

## Configuration Access Pattern

**Load config once, pass as dependency:**

```python
# Good ✅
from src.config.unified_config import UnifiedConfig

config = UnifiedConfig.load()
llm_client = ClaudeClient(
    api_key=config.llm_providers.claude.api_key,
    model=config.llm_providers.claude.model
)

# Bad ❌
class ClaudeClient:
    def __init__(self):
        config = UnifiedConfig.load()  # Loads config every time
```

---

## Testing Pattern

**Mirror src/ structure in tests/:**

```
src/agents/reader_agent.py
→ tests/unit/test_reader_agent.py

src/utils/rate_limiter.py
→ tests/unit/test_rate_limiter.py
```

**Use fixtures for common setup:**

```python
# tests/conftest.py
import pytest
from src.llm.claude_client import ClaudeClient

@pytest.fixture
def mock_llm_client():
    return MockClaudeClient()

@pytest.fixture
def reader_agent(mock_llm_client):
    return ReaderAgent(mock_llm_client)
```

---

## When to Create a New Module

**Create new module when:**
- ✅ Clear, single responsibility
- ✅ Reusable across multiple files
- ✅ More than 100 lines of code
- ✅ Logically independent

**Don't create when:**
- ❌ Only used once
- ❌ Fewer than 30 lines
- ❌ Tightly coupled to parent
- ❌ No clear purpose

---

## Module Size Guidelines

- **Agents:** 100-300 lines per agent
- **Generators:** 150-400 lines
- **Utilities:** 50-200 lines
- **Schemas:** 30-100 lines per schema file

**If exceeding:** Consider splitting into sub-modules

---

**Principle:** Every module should have a clear, single purpose and be independently testable.
