# 🚗 AutoSupport AI — RAG Customer Support Chatbot

A lightweight prototype designed to automate recurring automotive inquiries—such as warranty terms and service schedules—using local AI. This project implements **Retrieval-Augmented Generation (RAG)** to ensure responses are grounded in specific company documents, minimizing hallucinations and providing auditable sources.

---

## Architecture

The system is built as a three-stage pipeline to ensure performance on standard hardware:

1.  **Ingestion (A1):** Processes `.txt` documents from `data/knowledge_base/`, splitting them into 400-character chunks for precise retrieval.
2.  **RAG Chain (A2):** Uses a LangChain Expression Language (LCEL) pipeline to retrieve context via **cosine similarity** and generate answers using **Llama 3.2**.
3.  **UI (A3):** A Streamlit interface that provides chat history, a Top-K retrieval slider, and source transparency.


---

## Technical Decisions & Rationale

* **Model: Llama 3.2 (3B):** Selected for its efficient 1–4B parameter footprint, allowing it to follow instructions reliably on a standard CPU with only ~2 GB of RAM.
* **Embeddings: nomic-embed-text:** A purpose-built retrieval model that offers 768-dimension expressiveness while remaining fast and lightweight (274 MB).
* **Vector Store: ChromaDB:** A local, Python-native solution that requires no API keys or cloud infrastructure, perfect for a self-contained prototype.
* **Chunking Logic:** We use **400-character chunks** with a **60-character overlap** (15%). This size matches the typical paragraph structure of automotive FAQs, ensuring retrieval is neither too broad nor context-poor.
* **Temperature (0):** Set to zero to ensure deterministic, factual, and reproducible support answers.
* **System Prompt:** Explicitly programmed to admit ignorance if information is missing and to always cite document sources.

---

## Prompt Template

```

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", config.SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )

```

## Project Structure

```text
rag-chatbot-customer-support/
├── data/
│   ├── knowledge_base/     # Source documents (.txt)
│   └── chroma_db/          # Persistent vector store
├── src/
│   ├── config.py           # Central tunable parameters
│   ├── rag_pipeline.py     # Ingestion logic and RAG chain
│   └── app.py              # Streamlit interface
├── requirements.txt
└── README.md

```

---

## Setup and Running Locally

### 1. Prerequisites

* Python 3.10+
* [Ollama](https://ollama.ai) installed and running

### 2. Pull the Models

```bash
ollama pull llama3.2
ollama pull nomic-embed-text

```

### 3. Installation

```bash
git clone [https://github.com/nightingale18/rag-chatbot-customer-support.git](https://github.com/nightingale18/rag-chatbot-customer-support.git)
cd rag-chatbot-customer-support

# Setup environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt

```

### 4. Launch the App

```bash
streamlit run src/app.py

```

The app will be available at **http://localhost:8501**. On the first run, the system will automatically index the documents in your knowledge base.

---

## Roadmap & Trade-offs

* **Memory:** Currently, chat history resets per session. Future updates will include `ConversationBufferMemory`.
* **Scalability:** While ChromaDB is excellent for local use, moving to a hosted service like Weaviate or FAISS would support multi-user production loads.
* **Document Support:** Expanding beyond `.txt` to include PDF and Word loaders to handle a broader range of corporate documents.