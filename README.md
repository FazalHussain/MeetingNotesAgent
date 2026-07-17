# Meeting Notes Agent

A conversational AI agent built with **LangGraph** and **RAG** (Retrieval-Augmented Generation) that answers questions about your meeting notes. It loads `.txt` meeting transcripts into a ChromaDB vector store, then lets users query them through a Streamlit chat interface powered by a tool-calling LLM agent.

## Demo

```
User: What decisions were made about the database optimization?
Assistant: In the "Database Optimization Discussion" meeting on July 4, 2026,
the team decided to:
1. Add indexes to frequently searched columns
2. Archive old transactional data
3. Enable database query monitoring

Participants included Sarah Chen (DBA), James Wilson (Backend Lead), ...
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│              Streamlit UI (main.py)              │
│   Chat interface · Session state · History       │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│         LangGraph Workflow (runnable.py)         │
│                                                  │
│   ┌───────────┐    should_continue    ┌───────┐ │
│   │   agent   │──────────────────────▶│ tools │ │
│   │ (LLM call)│◀──────────────────────│ (exec)│ │
│   └───────────┘     tools → agent     └───────┘ │
│                                                  │
│   Checkpointing: InMemorySaver                   │
└────────────────────┬────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
┌──────────────────┐  ┌──────────────────┐
│  LLM Provider    │  │  Tool Functions   │
│  (OpenAI-compat) │  │  (tools.py)       │
└──────────────────┘  └───────┬──────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
          ┌──────────────┐    ┌────────────────┐
          │ query_docs   │    │ get_current_   │
          │ (ChromaDB    │    │ time           │
          │  RAG search) │    └────────────────┘
          └──────┬───────┘
                 │
                 ▼
          ┌──────────────┐
          │  ChromaDB    │
          │  Vector Store│
          │  (./chroma_db)│
          └──────────────┘
```

### Data Flow

1. **Ingestion** — `rag_document_loader.py` reads `.txt` files from `meeting_notes/`, splits them into 400-character chunks (80-char overlap), computes embeddings with `all-MiniLM-L6-v2`, and persists them to `./chroma_db/`.
2. **Runtime** — `main.py` starts the Streamlit app; `tools.py` loads the existing ChromaDB instance and exposes `query_documents` and `get_current_time` as LangGraph tools.
3. **Query** — User asks a question → LangGraph agent decides whether to call a tool → tool retrieves top-5 similar chunks from Chroma → LLM synthesizes a grounded answer citing specific meetings.

## Project Structure

```
MeetingNotesAgent/
├── main.py                  # Streamlit entry point — chat UI, session state
├── runnable.py              # LangGraph StateGraph: agent ↔ tools loop
├── tools.py                 # @tool functions: query_documents, get_current_time
├── rag_document_loader.py   # One-shot script: ingest meeting notes → ChromaDB
├── rag_evaluation.ipynb     # Jupyter notebook for RAG evaluation
├── meeting_notes/           # Source .txt meeting transcripts
│   ├── 1.txt
│   └── 2.txt
├── chroma_db/               # Persisted Chroma vector store (generated)
├── requirements.txt         # Python dependencies
└── .env                     # Environment variables (not committed)
```

## Prerequisites

- **Python 3.10+**
- **An OpenAI-compatible LLM endpoint** — any provider that exposes an OpenAI-compatible API (OpenAI, Azure OpenAI, OmniRouter, Ollama, etc.)
- **~500 MB disk** for the embedding model download (first run only)

## Setup

### 1. Clone and install

```bash
git clone <your-repo-url>
cd MeetingNotesAgent

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file in the project root:

```env
# LLM provider (OpenAI-compatible API)
OMNIROUTER_API_KEY=your-api-key-here
OMNIROUTER_BASE_URL=https://your-provider.com/v1
LLM_MODEL=gpt-4o-mini

# Optional: LangSmith tracing
LANGSMITH_API_KEY=your-langsmith-key

# Optional: custom document directory (default: meeting_notes)
RAG_DIRECTORY=meeting_notes
```

### 3. Add meeting notes

Place `.txt` files in the `meeting_notes/` directory (or set `RAG_DIRECTORY` to a custom path). Each file should contain the transcript or summary of a single meeting.

### 4. Ingest documents

Run this once, and again whenever you add or update meeting notes:

```bash
python rag_document_loader.py
```

This loads your `.txt` files, chunks them, computes embeddings, and saves the vector store to `./chroma_db/`.

### 5. Launch the app

```bash
streamlit run main.py
```

The app opens at [http://localhost:8501](http://localhost:8501).

## Tools

The agent has access to two tools:

| Tool | Description |
|---|---|
| `query_documents(question)` | Searches ChromaDB for the top-5 most relevant document chunks and returns them with source filenames. This is the core RAG retrieval tool. |
| `get_current_time()` | Returns the current local timestamp. |

To add a new tool, decorate a function with `@tool` in `tools.py` and add it to the `available_functions` dict.

## How It Works

### LangGraph Workflow

The agent is built on a LangGraph `StateGraph` with two nodes:

- **`agent`** — Calls the LLM with the current conversation history and bound tools. If the LLM issues tool calls, the graph routes to the `tools` node.
- **`tools`** — Executes each tool call, wraps results as `ToolMessage`s, and routes back to `agent` for synthesis.

The `should_continue` conditional edge decides whether to loop back to tools or end the turn. `InMemorySaver` provides checkpointing for multi-turn conversations within a session.

### System Prompt

The agent uses a system prompt that enforces strict retrieval-grounded behavior:

1. **Always query documents** before answering — no hallucination from memory.
2. **Ground answers in retrieved documents** — no fabrication.
3. **Synthesize across all returned chunks** — a single query may span multiple meetings.
4. **Cite specific meeting names** from source metadata.
5. **List decisions, action items, and participants** when present.
6. **Admit uncertainty** when documents don't contain the answer.

### Embeddings & Chunking

- **Embedding model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional, ~80 MB, runs locally via PyTorch)
- **Chunk size**: 400 characters with 80-character overlap
- **Splitter**: `RecursiveCharacterTextSplitter` with `["\n\n", "\n", " "]` separators — respects paragraph and sentence boundaries
- **Retrieval**: Cosine similarity search, top-5 results

## Evaluation

The `rag_evaluation.ipynb` notebook provides an interactive environment for evaluating RAG retrieval quality. Use it to:

- Test queries against the vector store
- Inspect retrieved chunks for relevance
- Iterate on chunking parameters and retrieval settings

## Configuration Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `OMNIROUTER_API_KEY` | Yes | — | API key for your LLM provider |
| `OMNIROUTER_BASE_URL` | Yes | — | Base URL for the OpenAI-compatible API |
| `LLM_MODEL` | No | `gpt-4o-mini` | Model name to use |
| `LANGSMITH_API_KEY` | No | — | Enable LangSmith tracing/evaluation |
| `RAG_DIRECTORY` | No | `meeting_notes` | Directory containing `.txt` meeting notes |

## Troubleshooting

**"No documents found" / empty responses**
- Make sure you've run `python rag_document_loader.py` after adding notes
- Verify `./chroma_db/` exists and is not empty
- Check that `RAG_DIRECTORY` points to the correct folder

**LLM connection errors**
- Verify `OMNIROUTER_BASE_URL` and `OMNIROUTER_API_KEY` in `.env`
- Ensure the endpoint is OpenAI-compatible (supports `/chat/completions`)
- Check that `LLM_MODEL` matches a model available at your provider

**Embedding model download is slow**
- The first run downloads `all-MiniLM-L6-v2` (~80 MB) from HuggingFace
- Subsequent runs use the cached model in `~/.cache/huggingface/`

**Streamlit caching issues**
- If tools seem stale after code changes, restart the Streamlit server
- The ChromaDB instance is loaded at module import time in `tools.py`

## License

This project is for educational and internal use.
