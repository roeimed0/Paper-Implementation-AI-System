# Paper→Code AI System - Setup Guide

**Last Updated:** 2025-12-25

---

## Prerequisites

- **Python 3.11+** (required)
- **Git** (for version control)
- **Claude API key** from [Anthropic](https://console.anthropic.com/)
- (Optional) **OpenAI API key** for model comparison

---

## Quick Start (5 minutes)

### 1. Clone Repository (if needed)

```bash
git clone <repository-url>
cd Paper-Implementation-AI-System
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate it
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API keys
# Required: CLAUDE_API_KEY
# Optional: OPENAI_API_KEY
```

### 5. Verify Setup

```bash
# Test Python version
python --version  # Should be 3.11+

# Test imports
python -c "import anthropic, pydantic, pytest; print('✅ All dependencies installed')"

# Run tests (when available)
pytest tests/ -v
```

---

## Detailed Setup Instructions

### Step 1: Python Environment

#### Check Python Version

```bash
python --version
```

**Required:** Python 3.11 or higher

#### Install Python 3.11+ (if needed)

**Windows:**
- Download from [python.org](https://www.python.org/downloads/)
- Check "Add Python to PATH" during installation

**macOS:**
```bash
brew install python@3.11
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv
```

---

### Step 2: Virtual Environment Setup

#### Why Virtual Environment?

- Isolates project dependencies
- Prevents conflicts with system Python
- Easy to reproduce environment

#### Create and Activate

```bash
# Create
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS/Linux)
source .venv/bin/activate

# Verify activation (should show .venv path)
which python  # macOS/Linux
where python  # Windows
```

#### Deactivate (when done)

```bash
deactivate
```

---

### Step 3: Install Dependencies

#### Install All Requirements

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**This installs:**
- anthropic (Claude API)
- pydantic (data validation)
- pytest (testing)
- asyncio libraries (async/await)
- And 20+ other dependencies

#### Verify Installation

```bash
pip list | grep anthropic
pip list | grep pydantic
pip list | grep pytest
```

---

### Step 4: API Keys Configuration

#### Get API Keys

1. **Claude API Key (Required)**
   - Go to [console.anthropic.com](https://console.anthropic.com/)
   - Sign up / Log in
   - Navigate to API Keys
   - Create new key
   - Copy the key (starts with `sk-ant-...`)

2. **OpenAI API Key (Optional)**
   - Go to [platform.openai.com](https://platform.openai.com/api-keys)
   - Sign up / Log in
   - Create new secret key
   - Copy the key (starts with `sk-proj-...`)

#### Configure .env File

```bash
# Copy template
cp .env.example .env

# Edit with your favorite editor
nano .env  # or vim, code, notepad, etc.
```

**Minimum configuration:**

```env
CLAUDE_API_KEY=sk-ant-api03-YOUR_ACTUAL_KEY_HERE
CLAUDE_MODEL=claude-sonnet-4-5-20250929
DEFAULT_TEMPERATURE=0.0
ENABLE_CACHE=true
LOG_LEVEL=INFO
```

**⚠️ IMPORTANT:** Never commit `.env` to Git! (It's in `.gitignore`)

---

### Step 5: Directory Structure

#### Verify Structure

```bash
tree -L 2  # or ls -R
```

**Should see:**

```
Paper-Implementation-AI-System/
├── .claude/                    # Claude Code infrastructure
├── config/                     # YAML config files
├── data/                       # Data directories
│   ├── cache/                 # API response cache
│   ├── logs/                  # Application logs
│   ├── outputs/               # Generated code
│   ├── datasets/              # Pipeline run data
│   └── prompts/               # Agent prompts
├── dev/active/                # Dev-docs
├── src/                       # Source code
├── tests/                     # Test suite
├── .env                       # Your config (git-ignored)
├── .env.example               # Template
├── requirements.txt           # Dependencies
└── README.md                  # Project overview
```

---

## Configuration Reference

### Environment Variables

#### Required

| Variable | Description | Example |
|----------|-------------|---------|
| `CLAUDE_API_KEY` | Anthropic API key | `sk-ant-api03-...` |

#### Recommended

| Variable | Description | Default |
|----------|-------------|---------|
| `CLAUDE_MODEL` | Model to use | `claude-sonnet-4-5-20250929` |
| `DEFAULT_TEMPERATURE` | Sampling temperature | `0.0` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |
| `ENABLE_CACHE` | Enable response caching | `true` |

#### Optional

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | None |
| `API_BUDGET_USD` | Cost limit | `500.00` |
| `CACHE_TTL_SECONDS` | Cache expiry | `86400` |
| `DRY_RUN` | No API calls (testing) | `false` |

See [.env.example](.env.example) for complete list.

---

## Verification Steps

### Test 1: Python Imports

```python
python -c "
import anthropic
import pydantic
import pytest
import asyncio
print('✅ All core imports successful')
"
```

### Test 2: API Connection (uses credits!)

```python
python -c "
import os
from anthropic import Anthropic

client = Anthropic(api_key=os.getenv('CLAUDE_API_KEY'))
response = client.messages.create(
    model='claude-sonnet-4-5-20250929',
    max_tokens=20,
    messages=[{'role': 'user', 'content': 'Say hello'}]
)
print(f'✅ API works: {response.content[0].text}')
"
```

### Test 3: Run Test Suite

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/unit/test_example.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## Troubleshooting

### Issue: "Python not found"

**Solution:**
- Ensure Python 3.11+ is installed
- Add Python to PATH
- Restart terminal

### Issue: "pip install fails"

**Solutions:**
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Use specific index
pip install -r requirements.txt --index-url https://pypi.org/simple

# Install one by one
pip install anthropic pydantic pytest
```

### Issue: "anthropic.AuthenticationError"

**Solution:**
- Check `CLAUDE_API_KEY` in `.env`
- Verify key is valid at [console.anthropic.com](https://console.anthropic.com/)
- Ensure `.env` is in project root
- Restart Python session after editing `.env`

### Issue: "Module not found"

**Solution:**
```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: "Permission denied"

**Solution (Windows):**
```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Solution (macOS/Linux):**
```bash
chmod +x setup.sh  # if using setup script
```

---

## Next Steps

Once setup is complete:

1. **Review the plan:**
   ```bash
   cat dev/active/paper-to-code-ai-system/plan.md
   ```

2. **Start Phase 0 implementation:**
   - Create configuration system
   - Implement base LLM client
   - Build utility modules

3. **Run first test:**
   ```bash
   pytest tests/ -v
   ```

4. **Begin development:**
   - Follow [dev/active/paper-to-code-ai-system/tasks.md](dev/active/paper-to-code-ai-system/tasks.md)
   - Track progress in dev-docs

---

## Development Workflow

### Daily Workflow

```bash
# 1. Activate environment
source .venv/bin/activate  # or .venv\Scripts\activate

# 2. Pull latest changes
git pull

# 3. Work on code
# ... development ...

# 4. Run tests
pytest tests/ -v

# 5. Check code quality
ruff check src/
black --check src/
mypy src/

# 6. Commit changes
git add .
git commit -m "Description of changes"

# 7. Push
git push
```

### Testing Workflow

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_claude_client.py

# Run with coverage
pytest --cov=src

# Run only fast tests (skip slow integration tests)
pytest -m "not slow"
```

---

## Additional Resources

### Documentation

- [Dev-Docs Plan](dev/active/paper-to-code-ai-system/plan.md) - 12-week implementation roadmap
- [Architecture Skill](.claude/skills/genai-project-architecture/SKILL.md) - Project structure guide
- [Python LLM Skill](.claude/skills/python-llm-dev-guidelines/SKILL.md) - LLM development patterns

### External Links

- [Anthropic Claude API Docs](https://docs.anthropic.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [pytest Documentation](https://docs.pytest.org/)
- [asyncio Guide](https://docs.python.org/3/library/asyncio.html)

---

## Support

### Getting Help

1. Check [dev/active/paper-to-code-ai-system/context.md](dev/active/paper-to-code-ai-system/context.md) for current state
2. Review [tasks.md](dev/active/paper-to-code-ai-system/tasks.md) for next steps
3. Use Claude Code agents for assistance:
   - `code-architecture-reviewer` - Code review
   - `documentation-architect` - Documentation
   - `web-research-specialist` - Research issues

### Common Commands

```bash
# Activate environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest

# Code quality
ruff check src/
black src/
mypy src/

# Check environment
python --version
pip list
```

---

**Setup complete?** ✅ Begin Phase 0 implementation!
