# 🚗 AutoSupport AI — RAG Customer Support Chatbot

> Prototype submission for the AI Engineer Intern case study  
> Deadline: Tuesday 03.03.2026 · Language: English

---

## Table of Contents
1. [What this builds and why](#what-this-builds-and-why)
2. [Architecture](#architecture)
3. [Key technical decisions (with rationale)](#key-technical-decisions)
4. [Project structure](#project-structure)
5. [Setup and running locally](#setup-and-running-locally)
6. [How each requirement is satisfied](#how-each-requirement-is-satisfied)
7. [Limitations and known trade-offs](#limitations-and-known-trade-offs)
8. [Roadmap — from prototype to production](#roadmap)

---

## What this builds and why

Every day an automotive company receives hundreds of customer inquiries about the same recurring topics: vehicle features, service intervals, warranty terms. These answers already exist in internal documents — but customers and agents have to hunt through them manually.

**RAG (Retrieval-Augmented Generation)** solves this by:
1. Pre-indexing the knowledge base as searchable vectors
2. At query time, finding the most relevant document excerpts
3. Injecting those excerpts into an LLM prompt as grounded context
4. Returning a natural-language answer that cites its sources

**Why RAG instead of fine-tuning?**

| Approach | Pro | Con |
|---|---|---|
| Fine-tuning | No retrieval latency | Expensive; knowledge baked in; hard to update |
| RAG (this project) | Knowledge updated by editing .txt files; sources auditable | Slightly more infrastructure |
| Pure prompt stuffing | Simple | Knowledge base too large to fit in one prompt |

RAG is the right choice for a corporate knowledge base that changes regularly.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     INGESTION  (A1)                     │
│                                                         │
│  data/knowledge_base/*.txt                              │
│         │                                               │
│  DirectoryLoader / TextLoader                           │
│         │                                               │
│  RecursiveCharacterTextSplitter (400 chars, 50 overlap) │
│         │                                               │
│  OllamaEmbeddings  (nomic-embed-text, 768-dim)          │
│         │                                               │
│  ChromaDB              │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   RAG CHAIN  (A2)  — LCEL               │
│                                                         │
│  user_query                                             │
│       │                                                 │
│  Chroma retriever  (cosine similarity, Top-K)           │
│       │                                                 │
│  format_docs_with_sources()  →  context string          │
│       │                                                 │
│  ChatPromptTemplate  (system + human)                   │
│       │                                                 │
│  ChatOllama  (llama3.2, temp=0.1)                       │
│       │                                                 │
│  StrOutputParser  →  answer (str) + sources             │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                  STREAMLIT UI  (A3)                     │
│                                                         │
│  Sidebar: Top-K slider | Rebuild toggle | Clear history │
│  Chat area: message history + source expanders          │
└─────────────────────────────────────────────────────────┘
```

---

## Key technical decisions

### Model: `llama3.2` (3 B parameters)

- Fits the 1–4 B budget requested; runs on CPU without a GPU
- Reliable instruction-following — stays grounded in retrieved context
- ~2 GB RAM; laptop-safe for a local prototype
- Alternative considered: `mistral:7b` — better quality but 4 GB+ RAM, too slow on CPU for a live demo

### Embedding model: `nomic-embed-text`

- Purpose-built retrieval model; strong performance on Massive Text Embedding Benchmark (MTEB) semantic search benchmarks
- 274 MB and fast even on CPU
- 768-dimension vectors — good expressiveness without excess memory
- Alternative: `mxbai-embed-large` (1024-dim, marginally better but 669 MB and slower)

### Vector store: ChromaDB

- 100% local — no API keys, no cloud, no cost
- Python-native; zero extra services to run
- Fast enough for a knowledge base of this size
- Alternative FAISS: faster at scale but in-memory only by default

### Chunk size: 400 characters, 50 overlap

- Below 400 chars: chunks lose context; answers fragment
- Above 400 chars: one chunk spans multiple topics; retrieval precision drops
- 400 chars ≈ 1–2 dense paragraphs — matches the section structure of the documents
- 50-char overlap (~13%) prevents answers being cut at chunk boundaries

### Retrieval: cosine similarity, Top-K = 3 (configurable)

- `llama3.2` context window is ~4 K tokens; 3 × 400 chars ≈ 1200 tokens, leaving headroom for prompt + answer
- Top-K exposed as a runtime slider so the evaluator can experiment live
- Alternative search type MMR (Maximal Marginal Relevance): gives more diverse results but adds latency; useful only when documents overlap heavily — not the case here

### Framework: LangChain (LCEL)

- Each pipeline step is a composable `Runnable` — swap LLM, embedder, or DB with one line
- LCEL supports streaming out-of-the-box (call `.stream()` instead of `.invoke()`)
- Rich ecosystem: memory, agents, evaluation tools available when needed

### Temperature: 0.1

- Customer support requires factual, reproducible answers — not creativity
- Near-zero temperature keeps the model tightly grounded in the retrieved context
- Not 0.0 because fully deterministic decoding can produce repetitive output

### System prompt design

Three explicit rules baked into the system prompt:
1. **Answer only from context** — reduces hallucination
2. **Admit when it doesn't know** — builds user trust; avoids confident errors
3. **Always cite sources** — answers are auditable; satisfies A2 spec

---

## Project structure

```
rag-chatbot-customer-support/
├── data/
│   ├── knowledge_base/          # Source documents — edit these to update the KB
│   │   ├── warranty_policy.txt
│   │   ├── service_schedule.txt
│   │   ├── vehicle_features.txt
│   │   └── ordering_process.txt
├── src/
│   ├── config.py                # All tunable parameters
│   ├── rag_pipeline.py          # A1 ingestion + A2 RAG chain + RAGChatbot class
│   └── app.py                   # A3 Streamlit UI
├── requirements.txt
└── README.md
```

---

## Setup and running locally

### Prerequisites

- Python >= 3.10
- [Ollama](https://ollama.ai) installed and running

### 1. Pull the required models

```bash
ollama pull llama3.2           # chat model (~2 GB)
ollama pull nomic-embed-text   # embedding model (~274 MB)
```

### 2. Clone and install

```bash
git clone <git@github.com:nightingale18/rag-chatbot-customer-support.git>
cd rag-chatbot-customer-support

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Run

```bash
streamlit run src/app.py
```

The app opens at **http://localhost:8501**.

**First run:** The pipeline loads, splits, and embeds the knowledge base documents. 

### Adding new documents

1. Drop `.txt` files into `data/knowledge_base/`
2. In the sidebar, tick **Rebuild vector store** and refresh — or restart the app

---

- **Answer quality is not perfect** (as specified). With a 3B model on CPU, answers are not always perfectly phrased.
- **No persistent memory across sessions** — chat history is reset when the browser tab is closed. Adding LangChain `ConversationBufferMemory` would fix this.
- **No authentication** — fine for a prototype; production requires auth.
- **Single-user** — Streamlit's session state is per user, but ChromaDB is shared. For concurrent multi-user production use, switch to a dedicated vector DB service (Weaviate, FAISS).
- **Plain .txt only** — PDF, Word, and HTML documents would need additional loaders (all available in `langchain-community`).

---

## Roadmap

### Production
- Add PDF document loaders to expand the knowledge base
- Integrate LangChain `ConversationBufferMemory` for multi-turn dialogue
- Add a simple evaluation harness: ask known questions, measure retrieval hit rate
- Replace local Ollama with a hosted model API (e.g. GPT-4o) for scale
- Replace ChromaDB with a scalable vector database (Weaviate, FAISS) for multi-user concurrency
- Build an interface to add/update/delete knowledge base documents
