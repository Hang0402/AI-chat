# Roleplay AI

**AI-powered character roleplay system — distill personas from web/text, chat with persistent memory, deploy with one command.**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> Scrape any character from the web → LLM distills into a structured persona card → Chat with streaming + memory → Beautiful WeChat-style UI.

---

## Quick Start

```bash
pip install -r requirements.txt

# Set API key
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.deepseek.com/v1"

# Start web server
python main.py serve
# Open http://localhost:8080

# Or CLI chat with pre-built character
python main.py chat characters/xiaolu.json
```

## Three Core Modules

### 1. Character Distillation

| Input | Method |
|-------|--------|
| Raw text | `python main.py distill --text` |
| Web URL | `python main.py distill --url <URL>` |
| Search query | `python main.py distill --search <query>` |

The LLM extracts: name, age, personality traits, speaking style, background, likes/dislikes, taboos, sample dialogues, emotion range — all as structured JSON.

### 2. Chat Engine

- **System prompt injection**: Character card becomes the system message
- **Short-term memory**: Last 20 turns in context window
- **Long-term memory**: Auto-summarizes key events every 6 turns
- **Streaming output**: WebSocket with real-time typing animation

### 3. WeChat-Style UI

- Responsive mobile-first design
- Sidebar character selection + in-app distillation
- Typing indicator + streaming text
- One-click deploy: `python main.py serve`

## Project Structure

```
roleplay-ai/
├── main.py                  # CLI: serve / distill / chat
├── src/
│   ├── distiller.py         # Web scrape + LLM character extraction
│   ├── engine.py            # Chat engine with dual memory
│   ├── server.py            # FastAPI + WebSocket server
│   └── static/
│       └── index.html       # WeChat-style chat UI
├── characters/
│   └── xiaolu.json          # Pre-built character (art student barista)
└── requirements.txt
```

## Deployment

```bash
# Local
python main.py serve

# Docker (coming soon)
# Or deploy to any Python host: Railway, Render, VPS
```

### WeChat Integration Path

This project exposes a WebSocket API — it can be connected to:
- **WeChat Official Account** via message callback
- **WeChat Work Bot** via webhook
- **Telegram/Discord Bot** via adapter

## Portfolio Context

| Project | Demonstrates |
|---------|-------------|
| [ai-api-tester](https://github.com/Hang0402/hang) | AI + traditional API testing |
| [llm-eval](https://github.com/Hang0402/hang/tree/llm-eval) | LLM output quality evaluation |
| **roleplay-ai** | AI application engineering + RAG + Prompt design |

Three projects = a complete AI Engineer portfolio.

## License

MIT
