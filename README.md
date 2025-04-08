Here’s the refined English version of your README.md with improved hierarchy and structure:

```markdown
# DeeperLang

[![Built with uv](https://img.shields.io/badge/uv-%E2%9A%A1%EF%B8%8F-FFD242?logo=python&logoColor=white)](https://docs.astral.sh/uv/)

A next-generation Python project managed with [uv](https://docs.astral.sh/uv) for dependency and virtual environment management. Reproduce the development environment instantly by syncing locked dependencies from the `uv.lock` file—no manual installation from requirements files needed.

---

## 🛠️ Environment Setup

### Clone Repository
```bash
git clone <your-repo-url>
cd Deeperlang_Backend
```

### Configure Environment Variables
Create a `.env` file in the project root with your API keys:
```env
OPENAI_API_KEY=your_openai_api_key
METAPHOR_API_KEY=your_metaphor_api_key
QDRANT_API_KEY=your_qdrant_api
QDRANT_API_URL=your_qdrant_url
```

### Install uv Tool
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version
```

### Create Virtual Environment
```bash
# Create Python 3.11-based virtual environment
uv venv deeperlang --python 3.11

# Activate environment
source deeperlang/bin/activate
```

### Sync Dependencies
```bash
uv sync --active
```

---

## 🚦 Run the Project
Execute with Chainlit:
```bash
chainlit run main.py
```

---
```