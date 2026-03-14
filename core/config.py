"""
Central Configuration Module

All project-wide constants and settings are defined here.
Import from this module instead of hardcoding values in individual files.
"""

import os

# ==============================
# Paths
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
CHROMA_DB_DIR = os.path.join(DATA_DIR, "chroma_db")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ==============================
# Embedding Model
# ==============================

EMBEDDING_MODEL_NAME = "models/all-MiniLM-L6-v2"

# ==============================
# Ollama / LLM
# ==============================

OLLAMA_URL_CHAT = "http://localhost:11434/api/chat"
OLLAMA_URL_GENERATE = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"
TEMPERATURE = 0.2

# ==============================
# RAG Pipeline
# ==============================

RAG_MAX_TOKENS = 400
RAG_N_RESULTS = 6
RAG_COLLECTION_NAME = "pdf_chat_session"

# ==============================
# Chunking
# ==============================

DEFAULT_CHUNK_SIZE = 600
DEFAULT_OVERLAP = 100

FAST_CHUNK_SIZE = 28000
FAST_OVERLAP = 300

DEEP_CHUNK_SIZE = 18000
DEEP_OVERLAP = 500

# ==============================
# Summarization
# ==============================

SUMMARIZE_FAST_MAX_TOKENS = 450
SUMMARIZE_DEEP_MAX_TOKENS = 900
CHUNK_SUMMARY_FAST_MAX_TOKENS = 150
CHUNK_SUMMARY_DEEP_MAX_TOKENS = 250
REDUCE_FAST_MAX_TOKENS = 250
REDUCE_DEEP_MAX_TOKENS = 350
REDUCE_FAST_BATCH_SIZE = 8
REDUCE_DEEP_BATCH_SIZE = 5