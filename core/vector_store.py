"""
Vector Store Module

This module manages the persistent ChromaDB vector database used
for storing and retrieving document embeddings in the RAG pipeline.
"""

import chromadb
import os
from typing import List, Dict, Any
from .config import CHROMA_DB_DIR

DB_FOLDER = CHROMA_DB_DIR


def create_collection(collection_name: str):
    """
    Creates or retrieves a persistent ChromaDB collection.

    Args:
        collection_name (str): Name of the vector collection.

    Returns:
        chromadb.api.models.Collection.Collection:
            A ChromaDB collection instance.
    """
    client = chromadb.PersistentClient(path=DB_FOLDER)
    return client.get_or_create_collection(name=collection_name)


def add_chunks_to_db(
    collection,
    chunks: List[str],
    embeddings: List[List[float]],
    metadatas: List[Dict[str, Any]]
) -> None:
    """
    Adds text chunks and their embeddings into the vector database.

    Args:
        collection: ChromaDB collection instance.
        chunks (List[str]): Text segments to store.
        embeddings (List[List[float]]): Corresponding embedding vectors.
        metadatas (List[Dict[str, Any]]): Metadata for each chunk
                                          (e.g., page numbers).
    """
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )


def query_db(
    collection,
    query_embedding: List[float],
    n_results: int = 5
) -> Dict[str, Any]:
    """
    Performs similarity search on the vector database.

    Args:
        collection: ChromaDB collection instance.
        query_embedding (List[float]): Embedding vector of the query.
        n_results (int, optional): Number of top matches to retrieve.

    Returns:
        Dict[str, Any]: Retrieval results including:
            - documents
            - metadatas
            - distances
            - ids
    """
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )