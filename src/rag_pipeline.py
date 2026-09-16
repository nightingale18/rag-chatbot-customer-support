"""
Document ingestion and RAG chain logic.
"""

from __future__ import annotations

import sys
import config

from pathlib import Path
from typing import List, Tuple

# ── LangChain core ────────────────────────────────────────────────────────────
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ── LangChain integrations ────────────────────────────────────────────────────
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Retrievers ────────────────────────────────────────────────────────────────
from langchain_classic.retrievers import EnsembleRetriever, MultiQueryRetriever
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_community.retrievers import BM25Retriever

# ── Reranker ──────────────────────────────────────────────────────────────────
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Ensure local config is accessible
sys.path.insert(0, str(Path(__file__).parent))

# ──────────────────────────────────────────────
# A1: Ingestion Pipeline
# ──────────────────────────────────────────────


def load_documents(directory: Path = config.KNOWLEDGE_BASE_DIR) -> List[Document]:
    """
    Loads text files and extracts 'Title:' and 'Category:' headers from the
    first few lines to use as metadata for cleaner citations.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Knowledge base not found at: {directory}")

    raw_docs = []
    for path in sorted(directory.glob("**/*.txt")):
        lines = path.read_text(encoding="utf-8").strip().splitlines()

        title, category, body_start = path.stem, "General", 0
        for i, line in enumerate(lines[:4]):
            if line.startswith("Title:"):
                title = line.replace("Title:", "").strip()
                body_start = i + 1
            elif line.startswith("Category:"):
                category = line.replace("Category:", "").strip()
                body_start = i + 1
            elif not line.strip():
                body_start = i + 1
            else:
                break

        raw_docs.append(
            Document(
                page_content="\n".join(lines[body_start:]).strip(),
                metadata={"source": path.name, "title": title, "category": category},
            )
        )

    print(f"-> Loaded {len(raw_docs)} documents.")
    return raw_docs


def split_documents(docs: List[Document]) -> List[Document]:
    """
    Chunks text using a 400-char window. The overlap ensures
    sentences aren't awkwardly cut in half during retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=config.CHUNK_SEPARATORS,
    )
    return splitter.split_documents(docs)


def build_vector_store(chunks: List[Document]) -> Chroma:
    """
    Creates a local ChromaDB instance using Nomic embeddings.
    Chosen for high MTEB performance and lightweight for CPU.
    """
    embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(config.CHROMA_DB_DIR),
    )


def ingest_knowledge_base(force_rebuild: bool = False) -> Tuple[Chroma, List[Document]]:
    """
    Returns (vector_store, chunks).
    Handles initialization. If a DB exists and we load it from disk.
    """
    db_exists = config.CHROMA_DB_DIR.exists() and any(config.CHROMA_DB_DIR.iterdir())

    docs = load_documents()
    chunks = split_documents(docs)

    if db_exists and not force_rebuild:
        print("Loading ChromaDB from a disk.")
        vector_store = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=OllamaEmbeddings(model=config.EMBEDDING_MODEL),
            persist_directory=str(config.CHROMA_DB_DIR),
        )
    else:

        print("Building a new ChromaDB vector store.")
        vector_store = build_vector_store(chunks)

    return vector_store, chunks


# ══════════════════════════════════════════════════════════════════════════════
# A2: Enhanced RAG Pipeline
#
# Multi-Stage Retrieval & Refinement
# ══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE OVERVIEW:
#
# 1. Query Expansion (Multi-Query): LLM generates 3 variations of the input
#    to improve semantic coverage.
#
# 2. Ensemble Retrieval: Parallel search across Vector (Semantic) and
#    BM25 (Keyword) stores, merged via Reciprocal Rank Fusion (RRF).
#
# 3. Contextual Compression filters out everything except the specific
#    sentences that answer the question.
#
#  4. Re-Ranking (Cross-Encoder)
# ==============================================================================


def build_retriever(
    vector_store: Chroma,
    chunks: List[Document],
    llm: ChatOllama,
    top_k: int,
) -> ContextualCompressionRetriever:
    """
    Our enhanced retriever pipeline consists of several retrievers
    which are connected together. As a post-processing step we use
    re-ranking
    """
    # Step 1: Keyword search using BM25 (Best Matching 25)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = top_k
    print("BM25 retriever is loaded...")

    # Step 2: Semantic search is performed using
    # HNSW (Hierarchical Navigable Small World) algorithm
    chroma_retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    print("Chroma retriever is loaded...")

    # Step 3: Hybrid search using EnsembleRetriever
    # weights: 40% BM25, 60% Chroma.
    # Reciprocal Rank Fusion (RRF) merges the two ranked lists without
    # needing score normalization across different scales.
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=config.ENSEMBLE_WEIGHTS,  # [0.4, 0.6]
        id_key="source",
    )

    # Step 4: Multi-Query Retriever
    # LLM-as-a-Judge: Uses the LLM to generate 3 alternative phrasings
    # of the user's question.
    # Each rephrasing is run independently.

    # We want to rephrase the query to retrieve similar documents
    # from both keyword-based (BM25) and semantic (HNSW) searches.
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=ensemble_retriever,
        llm=llm,
    )
    print("MultiQueryRetriever is loaded.")

    # Step 5: Cross-Encoder Reranker
    # Provide more relevant relevance score of the documents
    #
    # ms-marco-MiniLM-L-6-v2
    # 22M params, fast on CPU
    # top_n keeps only the N best-ranked chunks, where N < K
    cross_encoder_model = HuggingFaceCrossEncoder(
        model_name=config.RERANKER_MODEL  # "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    reranker = CrossEncoderReranker(
        model=cross_encoder_model,
        top_n=top_k,
    )

    # Step 6: Contextual Compression Retriever
    # ContextualCompressionRetriever filters out everything
    # except the specific chunks that answers the question
    final_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=multi_query_retriever,
    )
    print("Contextual Compression Retriever & Cross-Encoder Reranker is loaded.")

    return final_retriever


# ──────────────────────────────────────────────
# A3: RAG Chain
# ──────────────────────────────────────────────


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(
        f"[Source: {d.metadata['title']}]\n{d.page_content}" for d in docs
    )


def build_rag_chain(vector_store: Chroma, chunks: List[Document], top_k: int):
    """LCEL chain: Question -> Retrieve -> Format -> Prompt -> LLM -> Parse."""

    llm = ChatOllama(model=config.CHAT_MODEL, temperature=config.LLM_TEMPERATURE)

    retriever = build_retriever(
        vector_store=vector_store,
        chunks=chunks,
        llm=llm,
        top_k=top_k,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", config.SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


# ──────────────────────────────────────────────
# Main interface
# ──────────────────────────────────────────────


class RAGChatbot:
    """
    The primary engine for the chatbot.
    """

    def __init__(self, top_k: int = config.DEFAULT_TOP_K, force_rebuild: bool = False):
        self._top_k = top_k
        self.vector_store, self.chunks = ingest_knowledge_base(force_rebuild)
        self.chain, self.retriever = build_rag_chain(
            self.vector_store, self.chunks, top_k
        )

    def ask(self, question: str) -> Tuple[str, List[Document]]:
        """Returns the LLM's answer and the raw documents used for context."""
        sources = self.retriever.invoke(question)
        answer = self.chain.invoke(question)
        return answer, sources

    @property
    def top_k(self) -> int:
        return self._top_k

    @top_k.setter
    def top_k(self, new_k: int):
        if new_k != self._top_k:
            self._top_k = new_k
            self.chain, self.retriever = build_rag_chain(self.vector_store, new_k)
