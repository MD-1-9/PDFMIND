"""
Embedding Module

This module provides functionality for generating dense vector
embeddings from text chunks using a SentenceTransformer model.
The embeddings are used for semantic similarity search in the RAG pipeline.
"""

from sentence_transformers import SentenceTransformer
from typing import List
from .config import EMBEDDING_MODEL_NAME

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


def get_embeddings(chunks: List[str]) -> List[List[float]]:
    """
    Generates dense vector embeddings for a list of text chunks.

    Args:
        chunks (List[str]): List of text segments to encode.

    Returns:
        List[List[float]]: A list of embedding vectors (as Python lists).
                           Each inner list represents a single embedding.

    Notes:
        - Uses 'all-MiniLM-L6-v2' model (384-dimensional embeddings).
        - Optimized for local CPU/GPU usage.
        - Returns embeddings as lists for compatibility with ChromaDB.
    """
    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings.tolist()