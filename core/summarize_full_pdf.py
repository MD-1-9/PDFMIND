"""
Recursive PDF Summarization Module

Implements a multi-stage document summarization pipeline:
1. Chunk-level summarization
2. Recursive summary reduction
3. Structured final summary generation

Uses a local LLM (Mistral via Ollama) for all stages.
"""

import requests
from typing import List

from .pdf_loader import load_pdf_text
from .chunker import chunk_text
from .config import (
    OLLAMA_URL_GENERATE, MODEL_NAME, TEMPERATURE,
    FAST_CHUNK_SIZE, FAST_OVERLAP,
    DEEP_CHUNK_SIZE, DEEP_OVERLAP,
    SUMMARIZE_FAST_MAX_TOKENS, SUMMARIZE_DEEP_MAX_TOKENS,
    CHUNK_SUMMARY_FAST_MAX_TOKENS, CHUNK_SUMMARY_DEEP_MAX_TOKENS,
    REDUCE_FAST_MAX_TOKENS, REDUCE_DEEP_MAX_TOKENS,
    REDUCE_FAST_BATCH_SIZE, REDUCE_DEEP_BATCH_SIZE
)

# ==============================
# Universal Prompts
# ==============================

FAST_PROMPT = """
You are summarizing an academic or technical document.

Provide a structured summary in this format:

Title:
Main Objective:
Key Themes:
Technologies or Methods (if applicable):
Applications or Impact:
Overall Significance:

Be concise and factual.
Do NOT invent information.

Content:
{content}
"""

DEEP_PROMPT = """
You are generating a comprehensive, concept-dense summary of a long document.

First determine the nature of the document.
Preserve intellectual structure while compressing length.

Produce a structured summary including:

1. Central Objective or Thesis
2. Major Sections or Themes
3. Core Concepts, Methods, or Arguments
4. Important Results or Principles
5. Logical Flow of Ideas
6. Key Takeaways

Guidelines:
- Do not assume document type.
- Preserve essential terminology.
- Avoid filler commentary.

Content:
{content}
"""


# ==============================
# LLM Interface
# ==============================

def call_llm(prompt: str, max_tokens: int) -> str:
    """
    Sends a prompt to the local Ollama LLM and returns the generated response.
    """
    try:
        response = requests.post(
            OLLAMA_URL_GENERATE,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": TEMPERATURE,
                    "num_predict": max_tokens,
                },
            },
        )
        return response.json().get("response", "Error: Empty response")
    except Exception as e:
        return f"Ollama Error: {e}"


# ==============================
# Stage 1: Chunk-Level Summaries
# ==============================

def summarize_chunks(chunks: List[str], mode: str) -> List[str]:
    """
    Generates summaries for each chunk of text.
    """
    summaries = []

    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i+1}/{len(chunks)}")

        if mode == "fast":
            prompt = f"""
            Summarize the following section briefly.
            Focus only on key ideas.

            Section:
            {chunk}
            """
            max_tokens =  CHUNK_SUMMARY_FAST_MAX_TOKENS
        else:
            prompt = f"""
            Summarize the following section carefully.
            Preserve important definitions and structure.

            Section:
            {chunk}
            """
            max_tokens = CHUNK_SUMMARY_DEEP_MAX_TOKENS

        summaries.append(call_llm(prompt, max_tokens))

    return summaries


# ==============================
# Stage 2: Recursive Reduction
# ==============================

def reduce_summaries(summaries: List[str], mode: str) -> str:
    """
    Recursively merges summaries until a single compressed summary remains.
    """
    batch_size = REDUCE_FAST_BATCH_SIZE if mode == "fast" else REDUCE_DEEP_BATCH_SIZE

    while len(summaries) > 1:
        new_summaries = []

        for i in range(0, len(summaries), batch_size):
            batch = summaries[i:i + batch_size]
            combined = "\n".join(batch)

            if mode == "fast":
                prompt = f"""
                Merge these summaries into a concise overview.
                Avoid repetition.

                {combined}
                """
                max_tokens = REDUCE_FAST_MAX_TOKENS
            else:
                prompt = f"""
                Merge these summaries into a structured, concept-dense summary.
                Preserve logical flow.

                {combined}
                """
                max_tokens = REDUCE_DEEP_MAX_TOKENS

            new_summaries.append(call_llm(prompt, max_tokens))

        summaries = new_summaries

    return summaries[0]


# ==============================
# Stage 3: Final Structured Summary
# ==============================

def generate_final_summary(compressed_text: str, mode: str) -> str:
    """
    Generates the final structured summary using a universal prompt.
    """
    if mode == "fast":
        prompt = FAST_PROMPT.format(content=compressed_text)
        max_tokens = SUMMARIZE_FAST_MAX_TOKENS
    else:
        prompt = DEEP_PROMPT.format(content=compressed_text)
        max_tokens = SUMMARIZE_DEEP_MAX_TOKENS

    return call_llm(prompt, max_tokens)


# ==============================
# Main Pipeline
# ==============================

def summarize_pdf(pdf_path: str, mode: str = "fast") -> str:
    """
    End-to-end recursive summarization pipeline.
    """
    print("Loading PDF...")
    pages_data = load_pdf_text(pdf_path)

    if not pages_data:
        return "Failed to load PDF."

    full_text = "\n\n".join(p["text"] for p in pages_data)
    combined_pages = [{"text": full_text, "page": "full_document"}]

    chunk_size = FAST_CHUNK_SIZE if mode == "fast" else DEEP_CHUNK_SIZE
    overlap = FAST_OVERLAP if mode == "fast" else DEEP_OVERLAP

    print("Chunking document...")
    chunks, _ = chunk_text(
        combined_pages,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    print(f"Total chunks: {len(chunks)}")

    print("Stage 1: Summarizing chunks...")
    first_level = summarize_chunks(chunks, mode)

    print("Stage 2: Reducing summaries...")
    compressed = reduce_summaries(first_level, mode)

    print("Stage 3: Generating final summary...")
    return generate_final_summary(compressed, mode)