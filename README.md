# 🧠 PDFMIND

> A local, privacy-first RAG (Retrieval-Augmented Generation) system for intelligent PDF question answering and multi-mode document summarization — powered by semantic embeddings, vector search, and a dual-memory conversational engine.

---

## 🚀 Demo

```
Upload any PDF → Ask questions in natural language → Get cited, context-aware answers
```

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?style=flat-square&logo=streamlit)
![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorStore-green?style=flat-square)
![Ollama](https://img.shields.io/badge/Ollama-Mistral-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## 🧠 What Makes This Different

Most PDF chatbots forget what you asked 3 messages ago. This one doesn't.

This project implements a **dual-memory architecture** — a design pattern borrowed from cognitive science:

| Memory Type | Role |
|---|---|
| **Short-Term Memory (STM)** | Last 2 Q&A pairs, used to augment retrieval queries |
| **Long-Term Memory (LTM)** | Structured compression of full conversation history |

The LTM is periodically updated using the LLM itself — compressing topics, key definitions, entities, numerical values, and open questions into a structured bullet format. This means the assistant maintains context across long conversations without bloating the prompt window.

---

## ✨ Features

- 📥 **PDF Ingestion** — Extracts clean, page-wise text using `unstructured`, filtering out headers, footers, and page numbers
- ✂️ **Smart Chunking** — Separator-aware overlapping chunker that respects paragraph and sentence boundaries
- 🔍 **Semantic Search** — Dense vector retrieval using `all-MiniLM-L6-v2` (384-dim embeddings) + ChromaDB
- 💬 **Conversational QA** — Context-aware Q&A with page-level source citations
- 🧠 **Dual Memory Engine** — STM for retrieval augmentation, LTM for structured conversation compression
- 📋 **Recursive Summarization** — 3-stage pipeline: chunk summaries → recursive reduction → structured final output
- ⚡ **Two Summary Modes** — Fast (overview) and Deep (concept-dense) summarization
- 🖥️ **Streamlit UI** — Clean, responsive chat interface with sidebar controls
- 🔒 **100% Local** — No data leaves your machine. No API keys required.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  Streamlit UI (app.py)                   │
│    (Session State, File Uploads, Chat, Mode Toggle)      │
└──────────────┬────────────────────────────┬──────────────┘
               │                            │
   Action: Ask Question / Process     Action: Summarize
               │                            │
┌──────────────▼─────────────┐    ┌─────────▼────────────────────┐
│        RAG ENGINE           │    │      SUMMARIZATION ENGINE    │
│      (rag_pipeline.py)      │    │   (summarize_full_pdf.py)    │
├─────────────────────────────┤    ├──────────────────────────────┤
│ • Short-Term Memory (STM)   │    │ • Recursive Reduction Logic  │
│ • Long-Term Memory (LTM)    │    │ • Fast vs. Deep Modes        │
│ • Metadata/Source Tracking  │    │ • Multi-Stage Pipeline       │
└──────────────┬──────────────┘    └─────────┬────────────────────┘
               │                             │
               └─────────────┬───────────────┘
                             │
               ┌─────────────▼─────────────────┐
               │          CORE UTILITIES        │
               ├────────────────────────────────┤
               │ PDF Loader   (unstructured)    │
               │ Chunker      (overlapping)     │
               │ Embeddings   (MiniLM-L6-v2)    │
               └─────────────┬──────────────────┘
                             │
           ┌─────────────────┴─────────────────┐
           │                                   │
   ┌───────▼───────┐                   ┌───────▼───────┐
   │   STORAGE     │                   │   INFERENCE   │
   │  (ChromaDB)   │                   │ (Ollama/LLM)  │
   ├───────────────┤                   ├───────────────┤
   │ Persistent    │                   │ Mistral Model │
   │ Vector Store  │                   │ Local REST API│
   └───────────────┘                   └───────────────┘
```

---

## 📁 Project Structure

```
PDFMIND/
│
├── app.py                    # Streamlit frontend
│
├── core/
│   ├── config.py 
│   ├── pdf_loader.py         # PDF extraction & cleaning
│   ├── chunker.py            # Overlapping text chunker
│   ├── embeddings.py         # SentenceTransformer embeddings
│   ├── vector_store.py       # ChromaDB vector store interface
│   ├── rag_pipeline.py       # Core RAG engine + memory system
│   └── summarize_full_pdf.py # Recursive summarization pipeline
│
├── data/
│   ├── pdfs/                 # Uploaded PDFs (auto-created)
│   └── chroma_db/            # Persistent vector database
│
├── models/
│   └── all-MiniLM-L6-v2/    # Local embedding model
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- Mistral model pulled via Ollama

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/pdf-research-assistant.git
cd pdf-research-assistant
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Pull the LLM Model

```bash
ollama pull mistral
```

### 4. Download the Embedding Model

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2').save('models/all-MiniLM-L6-v2')"
```

### 5. Run the App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---



## 💡 How It Works

### 1. PDF Processing
When you upload a PDF, the system:
- Extracts page-wise text using `unstructured`, stripping headers/footers
- Splits text into overlapping chunks (600 chars, 100 char overlap) with separator-aware boundaries
- Generates 384-dimensional dense embeddings for each chunk
- Stores chunks + embeddings in a persistent ChromaDB collection

### 2. Question Answering
When you ask a question:
1. Recent conversation (STM) is appended to your query for context-aware retrieval
2. The augmented query is embedded and used to fetch top-6 most relevant chunks
3. Retrieved chunks + LTM summary are injected into a strict RAG prompt
4. Mistral generates a 2–4 sentence answer grounded only in the document
5. Source page numbers are appended to every answer

### 3. Memory Updates
Every 6 conversation turns, the LTM is updated by prompting the LLM to compress the conversation into structured categories: topics, definitions, entities, numerical values, laws/rules, and open questions.

### 4. Summarization
Two modes available:
- **Fast** — Large chunks (28k chars), brief per-chunk summaries, quick structured overview
- **Deep** — Smaller chunks (18k chars), detailed summaries, concept-dense final output

---

## 🧪 Example Usage

```
User: What is the main contribution of this paper?
Assistant: The paper introduces a novel attention mechanism that reduces computational 
           complexity from O(n²) to O(n log n) while maintaining model accuracy on 
           standard NLP benchmarks. (Sources: Page 1, 2)

User: What datasets did they use?
Assistant: The authors evaluated their method on three benchmark datasets: GLUE, 
           SQuAD 2.0, and WMT-14 English-German translation task. (Sources: Page 4, 5)
```

---

## 🔮 Roadmap

- [ ] Sentence-aware semantic chunking
- [ ] CrossEncoder reranker for improved retrieval precision
- [ ] Retrieval evaluation metrics (hit rate, MRR)
- [ ] Multi-PDF support with document switching
- [ ] Export conversation as PDF report

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

## 📜 License

[MIT](LICENSE)

---

## 👤 Author

**Md Asif**  
B.Tech CSE   
[GitHub](https://github.com/yourusername) · [LinkedIn](https://linkedin.com/in/yourprofile)

---

> *"Built to understand documents the way humans do — with memory, context, and recall."*
