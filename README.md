# Paper→Code AI System

An AI-driven system that converts scientific algorithm descriptions into verified, executable implementations through multi-agent reasoning and validation.

**Status:** Phase 0 Complete (Foundation Infrastructure) ✅

## 🎯 Project Goals

- **Learning Focus:** Master professional GenAI development patterns
- **Mock-First Development:** Build with zero API costs, swap to real LLMs later
- **Production Patterns:** Industry-standard architecture and best practices
- **Portfolio Ready:** Demonstrate senior-level GenAI engineering skills

## 🏗️ Architecture

### Mock-First Approach
This project is built using **mock-first development** - a professional pattern where you:
1. Build complete LLM abstraction layer
2. Develop with sophisticated mocks (zero cost)
3. Swap to real LLMs (Claude/GPT) with ONE config change

**Total Development Cost:** $0.00 💰

### Key Components

#### ✅ Phase 0: Foundation (COMPLETE)
- **LLM Abstraction Layer** - Provider-agnostic interface supporting mock, Claude, OpenAI, Ollama
- **Configuration System** - YAML + environment variables, hot-swappable providers
- **Rate Limiter** - Token bucket algorithm (same as AWS, Stripe, GitHub)
- **Token Counter** - Accurate cost estimation with tiktoken
- **Cache Manager** - TTL-based file caching with statistics
- **Logging System** - Rich console output + file logging

#### 🚧 Phase 1-10: Coming Soon
- Multi-agent pipeline (Reader, Planner, Implementer, Critic)
- PDF processing and text extraction
- Verification framework (unit, property-based, reference tests)
- Dataset logging and evaluation

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Conda or venv

### Installation

```bash
# Clone the repository
git clone https://github.com/roeimed0/Paper-Implementation-AI-System.git
cd Paper-Implementation-AI-System

# Create conda environment
conda create -n paper-to-code python=3.11
conda activate paper-to-code

# Install dependencies
pip install -r requirements-minimal.txt

# Test the infrastructure
python examples/test_mock_llm.py
python examples/test_config.py
python examples/test_utils.py
```

### Demo the Mock LLM

```python
import asyncio
from src.llm import MockLLMClient, LLMConfig

async def main():
    config = LLMConfig(provider="mock", model="mock-v1")

    async with MockLLMClient(config) as client:
        response = await client.generate("Implement quicksort algorithm")
        print(response.content)
        print(f"Tokens: {response.total_tokens}")
        print(f"Cost: ${response.cost_usd:.4f}")  # $0.00!

asyncio.run(main())
```

## 📚 Learning Outcomes

### Design Patterns
- ✅ Abstract Base Classes (provider-agnostic interfaces)
- ✅ Async/Await (non-blocking I/O)
- ✅ Context Managers (resource management)
- ✅ Singleton Pattern (global configuration)
- ✅ Token Bucket Algorithm (rate limiting)
- ✅ Dependency Injection (flexible architecture)

### Production Skills
- ✅ Pydantic for type safety
- ✅ Structured logging
- ✅ Cost tracking and budgets
- ✅ File-based caching with TTL
- ✅ Mock-driven development
- ✅ 12-Factor App configuration

## 🛠️ Technology Stack

- **Language:** Python 3.11+
- **LLM Clients:** Mock (dev), Claude API, OpenAI API, Ollama (optional)
- **Validation:** Pydantic
- **Async:** asyncio, aiohttp
- **Logging:** Rich, standard logging
- **Testing:** pytest, pytest-asyncio, hypothesis
- **Code Quality:** ruff, black, mypy

## 📊 Project Structure

```
Paper-Implementation-AI-System/
├── .claude/                    # Claude Code integration
│   ├── agents/                 # Specialized AI agents
│   ├── skills/                 # Custom development skills
│   └── settings.json           # Configuration
├── config/                     # System configuration
│   └── model_config.yaml       # LLM provider settings
├── src/
│   ├── llm/                    # LLM abstraction layer
│   │   ├── base.py            # Abstract interface
│   │   └── mock_client.py     # Mock implementation
│   ├── config/                 # Configuration management
│   └── utils/                  # Utilities
│       ├── rate_limiter.py    # Token bucket algorithm
│       ├── token_counter.py   # Cost estimation
│       ├── cache.py           # TTL caching
│       └── logger.py          # Logging setup
├── examples/                   # Working demos
│   ├── test_mock_llm.py       # LLM abstraction demo
│   ├── test_config.py         # Configuration demo
│   └── test_utils.py          # Utilities demo
└── dev/                        # Development docs
    └── active/
        └── paper-to-code-ai-system/
            ├── plan.md         # 12-week implementation plan
            ├── context.md      # Session progress tracking
            └── tasks.md        # Phase-by-phase checklist
```

## 🎓 For Employers/Portfolio

This project demonstrates:

- **Architecture Skills:** Built provider-agnostic LLM abstraction supporting multiple providers
- **Production Patterns:** Token bucket rate limiting, TTL caching, budget tracking
- **Cost Optimization:** Zero-cost development with sophisticated mocking
- **Type Safety:** Pydantic models throughout for validation
- **Async Programming:** Non-blocking I/O for concurrent LLM calls
- **Testing:** Mock-driven development enabling comprehensive testing

**Interview-Ready Example:**
> "I built a GenAI system with provider-agnostic LLM abstraction layer. Used abstract base classes to define interfaces, implemented both mock and real clients, and employed async/await for performance. The architecture supports swapping between Claude, GPT, or local models with just a config change. I also implemented token bucket rate limiting and TTL-based caching for production use."

## 📈 Development Progress

- [x] Phase 0: Foundation Infrastructure (100%)
  - [x] Environment setup
  - [x] LLM abstraction layer
  - [x] Configuration system
  - [x] Logging system
  - [x] Rate limiter
  - [x] Token counter
  - [x] Cache manager
- [ ] Phase 1: Input Acquisition
- [ ] Phase 2: Reader Agent
- [ ] Phase 3: Planner Agent
- [ ] Phase 4-10: Remaining pipeline stages

## 🤝 Contributing

This is a learning project. Feel free to:
- Fork and experiment
- Suggest improvements
- Share your own implementations

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Built using mock-first development methodology
- Inspired by production GenAI systems
- Developed with Claude Code for rapid iteration

---

**Status:** Active Development | Phase 0 Complete ✅ | Total Cost: $0.00 💰
