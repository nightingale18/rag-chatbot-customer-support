"""
rag_pipeline.py – Document ingestion and RAG chain logic.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import List, Tuple

# Ensure local config is accessible
sys.path.insert(0, str(Path(__file__).parent))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

import config

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
    Chosen for high MTEB performance while staying lightweight for CPU.
    """
    embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(config.CHROMA_DB_DIR),
    )


def ingest_knowledge_base(force_rebuild: bool = False) -> Chroma:
    """
    Handles initialization. If a DB exists and we aren't forcing
    a rebuild, it loads from disk to save time on embeddings.
    """
    db_exists = config.CHROMA_DB_DIR.exists() and any(config.CHROMA_DB_DIR.iterdir())

    if db_exists and not force_rebuild:
        return Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=OllamaEmbeddings(model=config.EMBEDDING_MODEL),
            persist_directory=str(config.CHROMA_DB_DIR),
        )

    print("Building fresh vector store...")
    return build_vector_store(split_documents(load_documents()))


# ──────────────────────────────────────────────
# A2: RAG Chain
# ──────────────────────────────────────────────


def format_docs(docs: List[Document]) -> str:
    """Combines retrieved chunks into a context block for the LLM."""
    return "\n\n".join(
        f"[Source: {d.metadata['title']}]\n{d.page_content}" for d in docs
    )


def build_rag_chain(vector_store: Chroma, top_k: int):
    """
    Assembles the LCEL chain.
    Flow: Question -> Retrieve -> Format -> Prompt -> LLM -> Parse.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", config.SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )
    llm = ChatOllama(model=config.CHAT_MODEL, temperature=config.LLM_TEMPERATURE)

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
    The primary engine for the chatbot. Manages state and allows
    Top-K adjustments without re-loading the entire DB.
    """

    def __init__(self, top_k: int = config.DEFAULT_TOP_K, force_rebuild: bool = False):
        self._top_k = top_k
        self.vector_store = ingest_knowledge_base(force_rebuild)
        self.chain, self.retriever = build_rag_chain(self.vector_store, top_k)

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
