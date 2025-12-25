---
name: python-llm-dev-guidelines
description: Python implementation patterns for LLM-based applications. Covers async/await, LLM client patterns, prompt engineering, rate limiting, caching, error handling, and testing strategies for AI agents.
---

# Python LLM Development Guidelines

**Purpose:** Best practices for implementing LLM-based systems in Python

**Last Updated:** 2025-12-25

---

## When to Use This Skill

This skill **auto-activates** when:
- Writing Python code in `src/`
- Working with LLM APIs (Claude, GPT)
- Implementing agents or generators
- Writing async/await code
- Handling API errors or rate limits
- Creating tests for LLM code

This skill **should be used** for:
- ✅ LLM client implementation
- ✅ Agent development patterns
- ✅ Prompt loading and management
- ✅ Async/await best practices
- ✅ Error handling for APIs
- ✅ Rate limiting and caching
- ✅ Testing LLM code

**Do NOT use** for:
- ❌ Project structure decisions (use genai-project-architecture)
- ❌ Where files belong (use genai-project-architecture)
- ❌ Pipeline organization (use genai-project-architecture)

---

## Quick Reference: Core Patterns

### 1. LLM Client Pattern (Base Class)

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Type
from pydantic import BaseModel

class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        **kwargs
    ) -> str:
        """Generate text from prompt."""
        pass

    @abstractmethod
    async def generate_json(
        self,
        prompt: str,
        schema: Type[BaseModel],
        **kwargs
    ) -> BaseModel:
        """Generate JSON matching Pydantic schema."""
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        pass

    @abstractmethod
    def estimate_cost(self, tokens: int) -> float:
        """Estimate cost for token count."""
        pass
```

### 2. Claude Client Implementation

```python
import anthropic
from typing import Type
from pydantic import BaseModel
from src.llm.base import BaseLLMClient
from src.utils.rate_limiter import RateLimiter
from src.utils.cache import ResponseCache
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ClaudeClient(BaseLLMClient):
    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-5-20250929",
        rate_limiter: RateLimiter | None = None,
        cache: ResponseCache | None = None
    ):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model
        self.rate_limiter = rate_limiter
        self.cache = cache

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 4000,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        # Check cache first
        if self.cache:
            cached = await self.cache.get(prompt, max_tokens, temperature)
            if cached:
                logger.info("Cache hit", extra={"prompt_length": len(prompt)})
                return cached

        # Rate limit
        if self.rate_limiter:
            await self.rate_limiter.acquire()

        # Call API with retry logic
        try:
            response = await self._call_with_retry(
                prompt, max_tokens, temperature, **kwargs
            )
            result = response.content[0].text

            # Cache response
            if self.cache:
                await self.cache.set(
                    prompt, max_tokens, temperature, result
                )

            logger.info("API call successful", extra={
                "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
                "cost_usd": self.estimate_cost(response.usage.input_tokens + response.usage.output_tokens)
            })

            return result

        except anthropic.APIError as e:
            logger.error("API call failed", extra={"error": str(e)})
            raise LLMAPIError(f"Claude API failed: {e}")

    async def generate_json(
        self,
        prompt: str,
        schema: Type[BaseModel],
        max_tokens: int = 4000,
        **kwargs
    ) -> BaseModel:
        # Add JSON instructions to prompt
        json_prompt = f"""{prompt}

IMPORTANT: Respond with valid JSON matching this schema:
{schema.model_json_schema()}

Respond ONLY with JSON, no other text."""

        response_text = await self.generate(
            json_prompt, max_tokens, temperature=0.0, **kwargs
        )

        # Parse and validate JSON
        try:
            return schema.model_validate_json(response_text)
        except Exception as e:
            logger.error("JSON parsing failed", extra={
                "error": str(e),
                "response": response_text[:500]
            })
            # Retry with schema correction
            return await self._retry_with_correction(
                prompt, schema, response_text
            )

    async def _call_with_retry(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        max_retries: int = 3,
        **kwargs
    ):
        for attempt in range(max_retries):
            try:
                return await self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs
                )
            except anthropic.RateLimitError:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Rate limited, waiting {wait_time}s")
                    await asyncio.sleep(wait_time)
                else:
                    raise
            except anthropic.APIConnectionError:
                if attempt < max_retries - 1:
                    logger.warning(f"Connection error, retry {attempt + 1}")
                    await asyncio.sleep(1)
                else:
                    raise

    def count_tokens(self, text: str) -> int:
        # Use tiktoken or anthropic's token counter
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4")  # Approximation
        return len(enc.encode(text))

    def estimate_cost(self, tokens: int) -> float:
        # Claude Sonnet 4.5 pricing (example)
        input_cost_per_1m = 3.00
        output_cost_per_1m = 15.00
        # Simplified: assume 50/50 split
        return (tokens / 1_000_000) * ((input_cost_per_1m + output_cost_per_1m) / 2)
```

### 3. Agent Implementation Pattern

```python
from src.llm.claude_client import ClaudeClient
from src.utils.prompt_loader import PromptLoader
from src.schemas.problem_schema import ProblemDefinition
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ReaderAgent:
    """Extracts problem definition from paper text (Stage 2)."""

    def __init__(
        self,
        llm_client: ClaudeClient,
        prompt_dir: str = "data/prompts/reader"
    ):
        self.llm = llm_client
        self.prompts = PromptLoader(prompt_dir)

    async def extract(self, text: str) -> ProblemDefinition:
        """Extract problem definition from paper text."""
        logger.info("Starting problem extraction", extra={
            "text_length": len(text)
        })

        # Load prompts
        system_prompt = self.prompts.load("system_prompt.txt")
        examples = self.prompts.load_json("few_shot_examples.json")

        # Build full prompt
        full_prompt = self._build_prompt(system_prompt, examples, text)

        # Call LLM
        try:
            problem = await self.llm.generate_json(
                full_prompt,
                schema=ProblemDefinition,
                max_tokens=4000
            )

            # Validate output
            validation_errors = self._validate(problem)
            if validation_errors:
                logger.warning("Validation issues", extra={
                    "errors": validation_errors
                })
                # Could retry or fix here

            logger.info("Extraction successful", extra={
                "assumptions_count": len(problem.assumptions),
                "constraints_count": len(problem.constraints)
            })

            return problem

        except Exception as e:
            logger.error("Extraction failed", extra={"error": str(e)})
            raise

    def _build_prompt(
        self,
        system: str,
        examples: list,
        text: str
    ) -> str:
        examples_str = "\n\n".join([
            f"Example {i+1}:\nInput: {ex['input']}\nOutput: {ex['output']}"
            for i, ex in enumerate(examples)
        ])

        return f"""{system}

{examples_str}

Now extract from this paper:
{text}"""

    def _validate(self, problem: ProblemDefinition) -> list[str]:
        errors = []
        if not problem.problem_definition:
            errors.append("Missing problem definition")
        if not problem.inputs:
            errors.append("No inputs specified")
        if not problem.outputs:
            errors.append("No outputs specified")
        return errors
```

### 4. Async/Await Patterns

**Always use async for I/O operations:**

```python
# Good ✅
async def process_paper(paper_path: str):
    text = await extract_text(paper_path)        # Async I/O
    problem = await reader_agent.extract(text)   # Async API call
    algorithm = await planner_agent.plan(problem)# Async API call
    return algorithm

# Bad ❌
def process_paper(paper_path: str):
    text = extract_text(paper_path)              # Blocking
    problem = reader_agent.extract(text)         # Would be blocking
    return problem
```

**Concurrent API calls when independent:**

```python
# Good ✅ - Run in parallel
import asyncio

async def process_multiple():
    results = await asyncio.gather(
        reader_agent.extract(text1),
        reader_agent.extract(text2),
        reader_agent.extract(text3)
    )
    return results

# Bad ❌ - Sequential when could be parallel
async def process_multiple():
    result1 = await reader_agent.extract(text1)
    result2 = await reader_agent.extract(text2)  # Waits for result1
    result3 = await reader_agent.extract(text3)  # Waits for result2
    return [result1, result2, result3]
```

### 5. Error Handling for LLM APIs

```python
from src.handlers.error_handler import LLMAPIError, ValidationError

async def safe_llm_call(agent, input_data):
    try:
        result = await agent.process(input_data)
        return result

    except anthropic.RateLimitError:
        logger.warning("Rate limit hit, backing off")
        await asyncio.sleep(60)
        return await agent.process(input_data)  # Retry once

    except anthropic.APIConnectionError as e:
        raise LLMAPIError(f"Network error: {e}")

    except ValidationError as e:
        logger.error("Validation failed", extra={"error": str(e)})
        # Could retry with corrected prompt
        raise

    except Exception as e:
        logger.exception("Unexpected error")
        raise LLMAPIError(f"Unexpected error: {e}")
```

### 6. Rate Limiting Pattern

```python
import asyncio
import time

class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(
        self,
        requests_per_minute: int = 50,
        tokens_per_minute: int = 100_000
    ):
        self.rpm = requests_per_minute
        self.tpm = tokens_per_minute
        self.request_tokens = requests_per_minute
        self.token_tokens = tokens_per_minute
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 0):
        """Acquire permission to make API call."""
        async with self._lock:
            await self._refill()

            # Wait if no tokens available
            while self.request_tokens < 1 or self.token_tokens < tokens:
                await asyncio.sleep(0.1)
                await self._refill()

            self.request_tokens -= 1
            self.token_tokens -= tokens

    async def _refill(self):
        """Refill tokens based on time elapsed."""
        now = time.time()
        elapsed = now - self.last_update

        # Refill tokens (per second rate)
        self.request_tokens = min(
            self.rpm,
            self.request_tokens + (self.rpm / 60) * elapsed
        )
        self.token_tokens = min(
            self.tpm,
            self.token_tokens + (self.tpm / 60) * elapsed
        )

        self.last_update = now
```

### 7. Response Caching Pattern

```python
import hashlib
import json
from pathlib import Path
import time

class ResponseCache:
    """Cache LLM responses to avoid redundant API calls."""

    def __init__(self, cache_dir: str, ttl_seconds: int = 86400):
        self.cache_dir = Path(cache_dir)
        self.ttl = ttl_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def get(self, prompt: str, *args, **kwargs) -> str | None:
        """Get cached response if exists and not expired."""
        key = self._make_key(prompt, args, kwargs)
        cache_file = self.cache_dir / f"{key}.json"

        if not cache_file.exists():
            return None

        # Check expiry
        with open(cache_file) as f:
            data = json.load(f)

        if time.time() - data["timestamp"] > self.ttl:
            cache_file.unlink()  # Expired, delete
            return None

        return data["response"]

    async def set(self, prompt: str, *args, response: str, **kwargs):
        """Cache response."""
        key = self._make_key(prompt, args, kwargs)
        cache_file = self.cache_dir / f"{key}.json"

        data = {
            "timestamp": time.time(),
            "prompt": prompt[:500],  # Store snippet for debugging
            "response": response
        }

        with open(cache_file, "w") as f:
            json.dump(data, f)

    def _make_key(self, prompt: str, args, kwargs) -> str:
        """Generate cache key from prompt and parameters."""
        content = f"{prompt}{args}{sorted(kwargs.items())}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
```

### 8. Prompt Loading Pattern

```python
from pathlib import Path
import json

class PromptLoader:
    """Load prompts from data/prompts/ directory."""

    def __init__(self, prompt_dir: str):
        self.dir = Path(prompt_dir)

    def load(self, filename: str) -> str:
        """Load text prompt file."""
        path = self.dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Prompt not found: {path}")

        with open(path) as f:
            return f.read()

    def load_json(self, filename: str) -> dict | list:
        """Load JSON prompt file."""
        path = self.dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Prompt not found: {path}")

        with open(path) as f:
            return json.load(f)

    def load_versioned(self, version: str, filename: str) -> str:
        """Load specific version of prompt."""
        path = self.dir / version / filename
        if not path.exists():
            raise FileNotFoundError(f"Versioned prompt not found: {path}")

        with open(path) as f:
            return f.read()
```

### 9. Testing LLM Code

**Unit tests with mocks:**

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.agents.reader_agent import ReaderAgent
from src.schemas.problem_schema import ProblemDefinition

@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.generate_json.return_value = ProblemDefinition(
        problem_definition="Test problem",
        inputs=["input1"],
        outputs=["output1"],
        constraints=[],
        assumptions=[],
        edge_cases=[],
        mathematical_properties=[]
    )
    return llm

@pytest.fixture
def reader_agent(mock_llm):
    return ReaderAgent(mock_llm)

@pytest.mark.asyncio
async def test_reader_extracts_problem(reader_agent, mock_llm):
    result = await reader_agent.extract("sample text")

    assert isinstance(result, ProblemDefinition)
    assert result.problem_definition == "Test problem"
    mock_llm.generate_json.assert_called_once()
```

**Integration tests (slow, real API):**

```python
import pytest
from src.llm.claude_client import ClaudeClient
from src.agents.reader_agent import ReaderAgent

@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
async def test_reader_real_api():
    llm = ClaudeClient(api_key=os.getenv("CLAUDE_API_KEY"))
    agent = ReaderAgent(llm)

    with open("tests/fixtures/sample_paper.txt") as f:
        text = f.read()

    result = await agent.extract(text)

    assert result.problem_definition
    assert len(result.inputs) > 0
    assert len(result.outputs) > 0
```

---

## Key Principles

1. **Always async for I/O** - Use `async def` and `await` for all API calls and file I/O
2. **Use Pydantic for validation** - All LLM outputs must validate against schemas
3. **Dependency injection** - Pass LLM clients and utilities via constructor
4. **Structured logging** - Log with context (agent, stage, tokens, cost)
5. **Error handling** - Catch specific exceptions, retry transient errors
6. **Rate limiting** - Respect API limits with token bucket
7. **Caching** - Cache responses to reduce costs
8. **Testing** - Mock for unit tests, real API for integration tests

---

## Common Anti-Patterns to Avoid

### ❌ Synchronous API calls
```python
# Bad
def call_llm(prompt):
    return claude.generate(prompt)  # Blocking
```

### ❌ No error handling
```python
# Bad
async def process():
    return await llm.generate(prompt)  # Will crash on error
```

### ❌ No validation
```python
# Bad
response = await llm.generate(prompt)
# Use response directly without validating structure
```

### ❌ Hard-coded API keys
```python
# Bad
client = ClaudeClient(api_key="sk-ant-...")  # Never!
```

### ❌ No rate limiting
```python
# Bad
for item in large_list:
    await llm.generate(...)  # Will hit rate limits
```

### ❌ Ignoring costs
```python
# Bad
# No token counting, no cost tracking, no budgets
```

---

## Resource Files

For detailed patterns, see:
- [LLM Client Patterns](resources/llm-client-patterns.md)
- [Prompt Engineering](resources/prompt-engineering.md)
- [Async Patterns](resources/async-patterns.md)
- [Testing LLM Code](resources/testing-llm-code.md)

---

**Remember:** LLM code requires special handling - async I/O, validation, error handling, rate limiting, and caching are not optional.
