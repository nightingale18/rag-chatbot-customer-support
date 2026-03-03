"""
evaluate.py – Offline evaluation for the RAG chatbot.

Metrics implemented:

  1. Faithfulness  – Does the answer contain only facts from the retrieved context?
                     Scored by an LLM judge (0-1 per claim).
  2. Answer Relevance – How well does the answer address the original question?
                        Scored by an LLM judge (0-1).
  3. Context Recall   – Were the right chunks retrieved at all?
                        Measured by string overlap with a reference answer.

Results are printed to the terminal and saved to data/json/eval_results.json.
"""

from __future__ import annotations

import json
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

import config
from rag_pipeline import RAGChatbot


# ──────────────────────────────────────────────
# Evaluation Dataset
#
# Structure:
#   question    - defined by the user
#   reference   - expected answer
# ──────────────────────────────────────────────

EVAL_DATASET: list[dict[str, str]] = [
    {
        "question": "How long is the standard warranty?",
        "reference": "The standard warranty lasts 2 years with unlimited mileage.",
    },
    {
        "question": "What does the EV battery warranty cover?",
        "reference": (
            "The EV battery warranty lasts 8 years or 160,000 km, whichever comes first. "
            "It covers battery capacity falling below 70% and battery cell defects."
        ),
    },
    {
        "question": "How often should I change the engine oil?",
        "reference": "Engine oil should be changed every 15,000 km or 12 months, whichever comes first.",
    },
    {
        "question": "What are my charging options for an electric vehicle?",
        "reference": (
            "You can charge at home with a Wallbox (11 or 22 kW), "
            "at public charging points across Europe, or use DC fast charging up to 200 kW."
        ),
    },
    {
        "question": "How do I place a vehicle order?",
        "reference": (
            "Configure your car online or at a dealer, sign the purchase agreement, "
            "pay a 10% deposit, and track production via the mobile app."
        ),
    },
    {
        "question": "Is roadside assistance available on weekends?",
        "reference": "Yes, roadside assistance is available 24/7, 365 days a year.",
    },
]


# ──────────────────────────────────────────────
# LLM Judge Prompts
# ──────────────────────────────────────────────

FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            textwrap.dedent(
                """\
        You are a strict evaluation judge.
        Your job is to check whether an answer is faithful to a given context.

        Rules:
        - Read the CONTEXT carefully.
        - Check every factual claim in the ANSWER.
        - If ALL claims are supported by the context → score 1.0
        - If SOME claims are not in the context → score 0.5
        - If the answer contradicts or ignores the context → score 0.0

        Respond with ONLY a JSON object, no explanation:
        {{"score": <0.0, 0.5, or 1.0>, "reason": "<one sentence>"}}
    """
            ),
        ),
        ("human", "CONTEXT:\n{context}\n\nANSWER:\n{answer}"),
    ]
)

RELEVANCE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            textwrap.dedent(
                """\
        You are a strict evaluation judge.
        Your job is to score how well an answer addresses a question.

        Rules:
        - Score 1.0 if the answer directly and completely answers the question.
        - Score 0.5 if the answer is related but incomplete or off-topic in parts.
        - Score 0.0 if the answer does not address the question at all.

        Respond with ONLY a JSON object, no explanation:
        {{"score": <0.0, 0.5, or 1.0>, "reason": "<one sentence>"}}
    """
            ),
        ),
        ("human", "QUESTION:\n{question}\n\nANSWER:\n{answer}"),
    ]
)


# ──────────────────────────────────────────────
# Metric Functions
# ──────────────────────────────────────────────


def _parse_judge_response(raw: str) -> dict[str, Any]:
    """Extract JSON from the judge LLM's response."""
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r'"score"\s*:\s*([0-9.]+)', cleaned)
        score = float(match.group(1)) if match else 0.0
        return {"score": score, "reason": "Could not parse judge response."}


def score_faithfulness(
    judge_llm: ChatOllama,
    answer: str,
    context_docs: list,
) -> dict[str, Any]:
    """
    Ask the LLM judge whether every claim in `answer` is supported by
    the retrieved context chunks.
    """
    context_text = "\n\n".join(
        f"[{d.metadata.get('title', 'Unknown')}]\n{d.page_content}"
        for d in context_docs
    )
    chain = FAITHFULNESS_PROMPT | judge_llm
    raw = chain.invoke({"context": context_text, "answer": answer}).content
    return _parse_judge_response(raw)


def score_relevance(
    judge_llm: ChatOllama,
    question: str,
    answer: str,
) -> dict[str, Any]:
    """
    Ask the LLM judge how well `answer` addresses `question`.
    """
    chain = RELEVANCE_PROMPT | judge_llm
    raw = chain.invoke({"question": question, "answer": answer}).content
    return _parse_judge_response(raw)


def score_context_recall(
    answer: str,
    reference: str,
) -> float:
    """
    What fraction of key reference words appear in the answer?

    Returns a float 0.0–1.0.
    """
    ref_words = set(reference.lower().split())
    ans_words = set(answer.lower().split())
    # Ignore common stop words for cleaner signal
    stop = {
        "the",
        "a",
        "an",
        "is",
        "it",
        "of",
        "to",
        "and",
        "or",
        "for",
        "in",
        "with",
        "that",
        "this",
        "are",
        "be",
        "as",
        "at",
        "by",
    }
    ref_keywords = ref_words - stop
    if not ref_keywords:
        return 1.0
    overlap = ref_keywords & ans_words
    return round(len(overlap) / len(ref_keywords), 3)


# ──────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────


def run_evaluation(top_k: int = config.DEFAULT_TOP_K) -> list[dict]:
    """
    Runs the full eval dataset through the chatbot and scores each response.
    Prints a summary table.
    """
    print("=" * 60)
    print("  RAG Evaluation — Faithfulness · Relevance · Recall")
    print("=" * 60)

    # Init chatbot and llama3.2 judge
    chatbot = RAGChatbot(top_k=top_k)
    judge = ChatOllama(model=config.CHAT_MODEL, temperature=0)
    print("Chatbot and judge is loaded")

    results = []

    for i, sample in enumerate(EVAL_DATASET, 1):
        question = sample["question"]
        reference = sample["reference"]

        print(f"\n[{i}/{len(EVAL_DATASET)}] Q: {question}")

        # 1. Get chatbot answer & retrieved docs
        answer, source_docs = chatbot.ask(question)

        # 2. Score
        faith = score_faithfulness(judge, answer, source_docs)
        relev = score_relevance(judge, question, answer)
        recall = score_context_recall(answer, reference)

        result = {
            "question": question,
            "answer": answer,
            "reference": reference,
            "sources": [d.metadata.get("source", "?") for d in source_docs],
            "faithfulness": faith,
            "relevance": relev,
            "context_recall": recall,
        }
        results.append(result)

        # Print evaluation metrics
        print(f"   Faithfulness : {faith['score']:.1f}  — {faith['reason']}")
        print(f"   Relevance    : {relev['score']:.1f}  — {relev['reason']}")
        print(f"   Context Recall (heuristic): {recall:.2f}")

    # Summary
    avg_faith = sum(r["faithfulness"]["score"] for r in results) / len(results)
    avg_relev = sum(r["relevance"]["score"] for r in results) / len(results)
    avg_recall = sum(r["context_recall"] for r in results) / len(results)

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Avg Faithfulness   : {avg_faith:.2f} / 1.0")
    print(f"  Avg Relevance      : {avg_relev:.2f} / 1.0")
    print(f"  Avg Context Recall : {avg_recall:.2f} / 1.0")
    print("=" * 60)

    # Save to disk
    output_path = Path(__file__).parent.parent / "data" / "eval_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": {
                    "avg_faithfulness": round(avg_faith, 3),
                    "avg_relevance": round(avg_relev, 3),
                    "avg_context_recall": round(avg_recall, 3),
                },
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n  Results saved → {output_path}")

    return results


if __name__ == "__main__":
    run_evaluation()
