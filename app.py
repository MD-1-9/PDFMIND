"""
Streamlit Frontend Application

Provides a user interface for:
- Uploading and processing PDF documents
- Conversational question answering (RAG-based)
- Multi-mode recursive summarization

This UI connects to the core RAG pipeline and summarization engine.
"""

import os
import streamlit as st

from core.summarize_full_pdf import summarize_pdf
from core.rag_pipeline import RAGPipeline
from core.config import PDF_DIR


# ======================================================
# Session State Initialization
# ======================================================

def initialize_session_state() -> None:
    """Initializes required Streamlit session state variables."""
    defaults = {
        "rag": RAGPipeline(),
        "current_pdf": None,
        "is_processing": False,
        "process_action": None,
        "summary_result": None,
        "chat_history": [],
        "selected_mode": "fast",
        "user_question": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


# ======================================================
# Page Configuration
# ======================================================

st.set_page_config(page_title="AI PDF Assistant", layout="wide")
st.title("🧠 PDFMIND ")


# ======================================================
# Sidebar: Upload & Summarization
# ======================================================

with st.sidebar:
    st.header("Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        disabled=st.session_state.is_processing
    )

    if uploaded_file:
        save_path = os.path.join(PDF_DIR, uploaded_file.name)
        os.makedirs(PDF_DIR, exist_ok=True)

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.session_state.current_pdf = save_path

        if st.button("Process PDF", disabled=st.session_state.is_processing):
            st.session_state.is_processing = True
            st.session_state.process_action = "process"
            st.rerun()

    st.markdown("---")
    st.header("Summarization")

    mode_label = st.radio(
        "Select Summary Mode:",
        ["⚡ Fast (Overview)", "🧠 Deep (Concept-Dense)"],
        disabled=st.session_state.is_processing
    )

    st.session_state.selected_mode = (
        "fast" if "Fast" in mode_label else "deep"
    )

    if st.button("Generate Summary", disabled=st.session_state.is_processing):
        if not st.session_state.current_pdf:
            st.warning("Upload and process a PDF first.")
        else:
            st.session_state.is_processing = True
            st.session_state.process_action = "summarize"
            st.rerun()

    # Display Summary
    if st.session_state.summary_result:
        st.markdown("---")
        st.subheader("📌 Summary")
        st.text_area(
            "Summary Output",
            st.session_state.summary_result,
            height=400
        )

    st.markdown("---")

    if st.button("New Conversation", disabled=st.session_state.is_processing):
        st.session_state.rag.reset_memory()
        st.session_state.summary_result = None
        st.session_state.chat_history = []
        st.rerun()


# ======================================================
# Main Chat Interface
# ======================================================

st.subheader("💬 Ask Questions About the PDF")

# Display chat history
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(chat["question"])
    with st.chat_message("assistant"):
        st.write(chat["answer"])


user_input = st.chat_input(
    "Ask a question...",
    disabled=st.session_state.is_processing
)

if user_input:
    st.session_state.user_question = user_input
    st.session_state.is_processing = True
    st.session_state.process_action = "qa"
    st.rerun()


# ======================================================
# Global Processing Block
# ======================================================

if st.session_state.is_processing:
    action = st.session_state.process_action

    if action == "process":
        with st.spinner("Processing PDF..."):
            message = st.session_state.rag.process_pdf(
                st.session_state.current_pdf
            )
            st.success(message)

    elif action == "summarize":
        with st.spinner("Generating summary..."):
            summary = summarize_pdf(
                st.session_state.current_pdf,
                st.session_state.selected_mode
            )
            st.session_state.summary_result = summary

    elif action == "qa":
        with st.spinner("Thinking..."):
            answer = st.session_state.rag.answer_question(
                st.session_state.user_question
            )

            st.session_state.chat_history.append(
                {
                    "question": st.session_state.user_question,
                    "answer": answer
                }
            )

    # Reset processing state
    st.session_state.is_processing = False
    st.session_state.process_action = None
    st.rerun()