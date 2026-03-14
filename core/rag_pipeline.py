"""
RAG Pipeline Module

Implements a Retrieval-Augmented Generation (RAG) pipeline for:
- PDF ingestion and vector indexing
- Context-based question answering
- Conversational memory management
    Short-Term Memory (STM) → Retrieval context
    Long-Term Memory (LTM) → Structured summary
- Embedding-driven document summarization

This pipeline integrates:
- SentenceTransformer embeddings
- ChromaDB vector storage
- Local LLM inference via Ollama (Mistral)
"""

import requests
from typing import List, Dict

from .pdf_loader import load_pdf_text
from .chunker import chunk_text
from .embeddings import get_embeddings
from .vector_store import create_collection, add_chunks_to_db, query_db
from .config import (
    OLLAMA_URL_CHAT, MODEL_NAME, TEMPERATURE,
    RAG_MAX_TOKENS, RAG_N_RESULTS
)


class RAGPipeline:

    def __init__(self, collection_name: str = "pdf_chat_session"):
        self.collection_name = collection_name
        self.collection = None
        self.chat_history: List[Dict[str, str]] = []
        self.long_term_memory: str = ""

    # ==================================================
    # RESET
    # ==================================================

    def reset_memory(self):
        self.chat_history.clear()
        self.long_term_memory = ""

    # ==================================================
    # PDF PROCESSING
    # ==================================================

    def process_pdf(self, pdf_path: str) -> str:
        pages_data = load_pdf_text(pdf_path)
        if not pages_data:
            return "Failed to read PDF."

        chunks, metadatas = chunk_text(pages_data)
        vectors = get_embeddings(chunks)

        self.collection = create_collection(self.collection_name)

        # Clear old session docs
        try:
            existing = self.collection.get().get("ids", [])
            if existing:
                self.collection.delete(existing)
        except Exception:
            pass

        add_chunks_to_db(self.collection, chunks, vectors, metadatas)

        return "PDF processed successfully!"

    # ==================================================
    # SHORT-TERM MEMORY (for retrieval only)
    # ==================================================

    def _build_stm_context(self) -> str:
        """
        Returns last 2 Q/A pairs for embedding augmentation.
        """
        recent = self.chat_history[-4:]  # last 2 Q/A
        return "\n".join(
            f"{m['role']}: {m['content']}"
            for m in recent
        )

    # ==================================================
    # LONG-TERM MEMORY (structured compression)
    # ==================================================

    def _update_long_term_memory(self):
        """
        Compresses conversation into structured summary.
        """
        if len(self.chat_history) < 6:
            return

        recent_turns = "\n".join(
            f"{m['role']}: {m['content']}"
            for m in self.chat_history[-8:]
        )

        prompt = f"""
        You are a structured memory compressor.

        Existing Memory:
        {self.long_term_memory}

        Recent Conversation:
        {recent_turns}

        Update structured memory in this format:

        - Main Topics:
        - Key Definitions:
        - Important Entities:
        - Important Numerical Values:
        - Laws / Rules Mentioned:
        - Open Questions:

        Rules:
        - Be domain independent.
        - Preserve exact numbers and terminology.
        - Remove redundancy.
        - Max 10 bullet lines.
        - Bullet format only.

        Updated Memory:
        """

        updated = self._call_ollama(prompt)
        if updated:
            self.long_term_memory = updated.strip()

    # ==================================================
    # QUESTION ANSWERING
    # ==================================================

    def answer_question(self, question: str) -> str:
        if not self.collection:
            return "Please upload a PDF first."

        # ---------- Short-Term Memory for retrieval ----------
        stm_context = self._build_stm_context()

        augmented_query = f"""
        Recent conversation:
        {stm_context}

        Current question:
        {question}
        """

        query_vec = get_embeddings([augmented_query])[0]
        results = query_db(self.collection, query_vec, RAG_N_RESULTS)

        if not results["documents"] or not results["documents"][0]:
            return "No relevant information found in the document."

        context = "\n".join(results["documents"][0])

        pages = sorted(
            set(m["page"] for m in results["metadatas"][0])
        )
        source_text = f"\n\n(Sources: Page {', '.join(map(str, pages))})"

        prompt = f"""
        You are an academic assistant answering questions strictly from the provided document.

        Long-Term Conversation Memory:
        {self.long_term_memory}

        Document Context:
        {context}

        Question:
        {question}

        Instructions:
        - Use ONLY the document context to answer.
        - Use memory only to resolve references like "it" or "that".
        - If explicitly stated, answer directly.
        - If logically inferable from definitions, state clearly.
        - Include numerical values if relevant.
        - Do NOT introduce information not present in context.
        - Keep answer concise (2–4 sentences).
        - If not found, respond exactly:
        "The answer is not found in the provided document."

        Answer:
        """

        answer = self._call_ollama(prompt)

        # Save conversation
        self.chat_history.append({"role": "user", "content": question})
        self.chat_history.append({"role": "assistant", "content": answer})

        # Update long-term memory periodically
        if len(self.chat_history) % 6 == 0:
            self._update_long_term_memory()

        return answer + source_text

    # ==================================================
    # LLM CALL
    # ==================================================

    def _call_ollama(self, user_prompt: str) -> str:
        system_message = (
            "You are a professional PDF assistant. "
            "Answer strictly from provided context. "
            "If unsure, say the answer is not found."
        )

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": TEMPERATURE,
                "num_predict": RAG_MAX_TOKENS,
            },
        }

        try:
            response = requests.post(OLLAMA_URL_CHAT, json=payload)
            return response.json().get("message", {}).get(
                "content", "Error: Empty response"
            )
        except Exception as e:
            return f"Ollama Error: {e}"