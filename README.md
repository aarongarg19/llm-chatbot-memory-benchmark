# LLM Chatbot Memory Benchmark

A custom LLM chatbot project that aims to compare **Retrieval-Augmented Generation (RAG)** and **Open Format Knowledge (OFK)** approaches for conversational memory.

## Overview

The project uses a locally hosted language model to generate responses while maintaining conversation history and supporting persistent memory retrieval. It aims to evaluate different memory approaches based on retrieval quality, response latency, and personalization.

## Tech Stack

- **Python** — Backend development
- **FastAPI** — REST API
- **Ollama (Qwen 2.5 3B)** — Local LLM
- **SQLite** — Conversation and memory storage
- **Git and GitHub** — Version control

## Features

- Chat API with conversation IDs
- Persistent conversation history
- Common interface for memory providers
- User-specific memory retrieval
- Response timing measurements
- Planned comparison of RAG and OFK memory approaches

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/aarongarg19/llm-chatbot-memory-benchmark.git
cd llm-chatbot-memory-benchmark
```

**2. Set up the backend**

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**3. Download the LLM**

Install [Ollama](https://ollama.com/download), then run:

```bash
ollama pull qwen2.5:3b
```

**4. Start the backend**

```bash
uvicorn app.main:app --reload
```

API documentation: http://127.0.0.1:8000/docs

Health check: http://127.0.0.1:8000/health

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check backend status |
| POST | `/chat` | Send a chat message |
| GET | `/conversations/{conversation_id}/messages` | Retrieve conversation history |

## Development Status

- [x] FastAPI backend and Ollama integration
- [x] SQLite conversation persistence
- [x] Common memory-provider interface
- [x] Basic memory retrieval testing
- [ ] RAG and OFK provider integration
- [ ] Comparative benchmarking

## Repository

[GitHub: LLM Chatbot Memory Benchmark](https://github.com/aarongarg19/llm-chatbot-memory-benchmark)
