"""
Streamlit Chat Interface for the RAG Chatbot.

Run with: streamlit run src/app.py
"""

import streamlit as st
from rag_pipeline import RAGChatbot

from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# UI Components
# ──────────────────────────────────────────────

"""Render sidebar settings. Returns a dict of parameters."""


def render_sidebar() -> dict:
    with st.sidebar:
        st.header("⚙️ Settings")

        top_k = st.slider(
            "Retrieved chunks (Top-K)",
            min_value=1,
            max_value=10,
            value=3,
            help="How many document chunks to retrieve per query.",
        )

        st.divider()
        st.markdown("**How it works**")
        st.markdown(
            "1. Your question is embedded\n"
            "2. Relevant document chunks are retrieved\n"
            "3. An LLM generates an answer based on the context"
        )

    return {"top_k": top_k}


"""Render a single chat message."""


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📄 Sources"):
                for source in message["sources"]:
                    st.markdown(f"- {source}")


"""Display all messages stored in session state."""


def render_chat_history() -> None:
    for message in st.session_state.messages:
        render_message(message)


""" Generate a chatbot response for the given query."""


def get_bot_response(query: str, top_k: int) -> tuple[str, list[str]]:
    rag_chatbot = RAGChatbot(top_k)
    answer, sources = rag_chatbot.ask(query)
    return answer, sources


def main():
    st.set_page_config(
        page_title="Customer Service Chatbot",
        page_icon="🚗",
        layout="centered",
    )

    st.title("🚗 Customer Service Chatbot")
    st.caption("Ask questions about vehicles, services, warranty, and more.")

    settings = render_sidebar()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    render_chat_history()

    if prompt := st.chat_input("Ask a question..."):
        # User message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        answer, sources = get_bot_response(prompt, top_k=settings["top_k"])

        response = {"role": "assistant", "content": answer, "sources": sources}
        render_message(response)
        st.session_state.messages.append(response)


if __name__ == "__main__":
    main()
