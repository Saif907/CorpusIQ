# CorpusIQ

> AI-powered document intelligence platform for compliance, enterprise search, and e-discovery.

CorpusIQ is a production-oriented document processing platform that ingests large document collections, detects and redacts sensitive information, indexes content for semantic search, and enables grounded question answering using Retrieval-Augmented Generation (RAG).

The project is designed to demonstrate modern AI system architecture, including document ingestion, hybrid retrieval, vector search, agentic workflows, and Model Context Protocol (MCP) integration.

---

## Features (Planned)

- Document ingestion pipeline
- PII detection and redaction
- Thread-aware document chunking
- Semantic search using vector embeddings
- Hybrid retrieval (Vector + BM25)
- Retrieval-Augmented Generation (RAG)
- Citation-based responses
- Model Context Protocol (MCP) server
- Redis caching and rate limiting
- PostgreSQL metadata storage
- Qdrant vector database
- FastAPI REST API
- React-based frontend dashboard

---

## Tech Stack

### Backend

- FastAPI
- PostgreSQL
- SQLAlchemy
- Redis
- Qdrant
- LangChain
- Sentence Transformers
- OpenAI API
- Presidio
- spaCy
- MCP Python SDK

### Frontend (Planned)

- React
- TypeScript
- Tailwind CSS

---

## Project Structure

```text
CorpusIQ/
│
├── backend/
│   ├── app/
│   ├── tests/
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│
├── datasets/
│
├── docs/
│
├── docker-compose.yml
├── .gitignore
└── README.md
```

---

## Development Roadmap

### Version 1

- Enron Email Dataset
- Document ingestion
- PII detection
- Redaction
- Thread-aware chunking
- Embeddings
- Hybrid retrieval
- RAG
- MCP integration

### Version 2

- OCR support
- Scanned PDFs
- Image preprocessing
- Human review workflow

### Version 3

- Legal document processing
- Multi-agent workflows
- Advanced compliance analysis

---

## Architecture (High Level)

```text
Documents
    │
    ▼
Ingestion
    │
    ▼
PII Detection
    │
    ▼
Redaction
    │
    ▼
Chunking
    │
    ▼
Embeddings
    │
    ▼
Qdrant
    │
    ▼
Hybrid Retrieval
    │
    ▼
LLM / Agent
    │
    ▼
Grounded Response + Citations
```

---

## Getting Started

### Clone the repository

```bash
git clone https://github.com/<username>/CorpusIQ.git
cd CorpusIQ
```

### Backend

```bash
cd backend

python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

### Run the API

```bash
uvicorn app.main:app --reload
```

---

## Current Status

🚧 Project under active development.

The initial milestone focuses on building a production-ready backend around the Enron Email dataset, including ingestion, indexing, retrieval, and grounded question answering.

---

## License

This project is licensed under the MIT License.