# Generative AI

A hands-on collection of Generative AI implementations built while learning the core building blocks of LLM application development — chat models, embeddings, RAG pipelines, tool-calling agents, and LangChain runnables.

## What's inside

| Folder | Description |
|---|---|
| `chatModels/` | Basic chat model integrations — cloud APIs (Groq, HuggingFace), local models, and a simple chatbot UI |
| `embeddingModels/` | Generating embeddings using HuggingFace and other embedding providers |
| `RAG/` | Retrieval-Augmented Generation pipeline — document loaders (PDF, web pages), vector stores (Chroma), retrieval strategies (MMR, multi-query, Arxiv retriever) |
| `runnables/` | LangChain Runnable patterns — sequential, parallel, and passthrough chains |
| `tools/` | Custom tool creation and tool-calling with LLMs, including a news summarizer using Tavily search |
| `aiAgents/` | Tool-using agents built with LangChain, including automated agent workflows with middleware |
| `movieSummarizer/` | A structured-output mini project that extracts movie details (name, genre, characters) using Pydantic parsers |

## Tech Stack

- **Frameworks:** LangChain, LangGraph
- **LLM Providers:** Groq, OpenAI, Google Gemini, Mistral AI, HuggingFace
- **Vector Store:** ChromaDB, FAISS
- **Embeddings:** Sentence Transformers, HuggingFace
- **Search/Tools:** Tavily
- **UI/Serving:** Streamlit, FastAPI, Uvicorn
- **Other:** PyPDF, python-dotenv, Transformers, Torch

## Setup

```bash
git clone <repo-url>
cd Generative-AI
pip install -r requirements.txt
```

Create a `.env` file in the root directory with the required API keys, e.g.:

```
GROQ_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
MISTRAL_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
```

## Purpose

This repo tracks my learning path through core Generative AI concepts — from basic chat model calls to full RAG pipelines and simple tool-calling agents — before moving into more advanced agentic AI architectures (see my separate [Agentic AI with LangGraph] repo for that work).

## Note

This is a learning/experimentation repo, not a production application — code favors clarity and concept coverage over polish.