# Generative AI

A hands-on collection of Generative AI implementations built while learning the core building blocks of LLM application development — chat models, embeddings, document loaders, RAG pipelines, retrievers, LangChain runnables, tool calling, and simple tool-using agents.

## What's inside

### `chatModels/`
Basic and UI-based chat model usage via LangChain's `init_chat_model` and provider-specific classes.
- `chat.py` — Compares responses from **Google Gemini**, **Groq**, and **Mistral** on the same prompt
- `chatbot.py` — A terminal-based chatbot (emoji-only assistant persona) using Mistral, with conversation history
- `UIchatbot.py` — Same chatbot logic wrapped in a **Streamlit** UI ("Bike Bot" — a bike-expert assistant) with session-state chat history
- `huggingface.py` — Chat via a **HuggingFace Inference Endpoint** (DeepSeek-R1)
- `localmodel.py` — Running a **local HuggingFace model** (TinyLlama-1.1B) via `HuggingFacePipeline`

### `embeddingModels/`
- `embeddings.py` — Generating embeddings with **OpenAI's** `text-embedding-3-large`
- `huggingface_embedding.py` — Generating embeddings with a **Sentence-Transformers** model (`all-MiniLM-L6-v2`) via HuggingFace

### `RAG/`
A Retrieval-Augmented Generation pipeline built incrementally:
- `document loaders/pdf.py`, `page.py`, `test.py` — Loading and chunking documents from PDFs (`PyPDFLoader`), web pages (`WebBaseLoader`), and plain text (`TextLoader`), using different text splitters
- `create_database.py` — Builds a persistent **ChromaDB** vector store from a PDF using HuggingFace embeddings
- `main.py` — Terminal RAG pipeline: retrieves from the Chroma store (MMR search) and answers questions using **Mistral**, strictly grounded in retrieved context
- `app.py` — The same RAG pipeline wrapped in a **Streamlit** "Chat with your PDF" app, supporting user-uploaded PDFs
- `retrievers/mmr.py` — Compares plain similarity search vs. **MMR (Maximal Marginal Relevance)** retrieval
- `retrievers/multiquery.py` — **Multi-query retrieval** using an LLM to generate query variants
- `retrievers/arxiv.py` — Retrieving papers directly from **Arxiv**
- `vector store/DB.py` — Building a Chroma vector store using **Mistral embeddings**

### `runnables/`
LangChain Expression Language (LCEL) patterns using the pipe (`|`) syntax:
- `sequencerunnables.py` — A simple sequential chain (prompt → model → parser)
- `parallelrunnables.py` — Running two chains (short vs. detailed explanation) in parallel with `RunnableParallel`
- `runnablepassthrough.py` — Using `RunnablePassthrough` to pass original input alongside a derived output (code generation + explanation)

### `tools/`
- `owntool.py` — Defining a custom tool with the `@tool` decorator
- `toolcalling.py` — Manual tool-calling loop: binding a tool to an LLM, executing the tool call, and feeding the result back
- `newssummarizer.py` — Uses **Tavily search** + Groq to fetch and summarize the latest AI news

### `aiAgents/`
- `Agents.py` — A weather + news assistant using manually defined tools (OpenWeather API, Tavily) with a Groq model
- `automateAgents.py` — The same idea rebuilt using LangChain's higher-level `create_agent` API with middleware (`wrap_tool_call`) instead of a manual tool-calling loop

### `movieSummarizer/`
A structured-output mini project:
- `core.py` — Extracts movie details (name, genre, characters, summary, theme, rating) from a text description using a Groq model and `PydanticOutputParser`
- `UIsummarizer.py` — Same logic wrapped in a **Streamlit** UI

## Tech Stack

- **Framework:** LangChain (core, community, classic)
- **LLM Providers:** Groq, OpenAI, Google Gemini, Mistral AI, HuggingFace (hosted + local)
- **Vector Store:** ChromaDB
- **Embeddings:** OpenAI, Mistral, Sentence-Transformers / HuggingFace
- **Search/Tools:** Tavily, OpenWeather API, Arxiv
- **UI:** Streamlit
- **Other:** PyPDF, python-dotenv, Pydantic, rich

> Note: `requirements.txt` also lists LangGraph, FAISS, FastAPI, and Uvicorn, but these aren't used by any script in the repo yet — likely staged for upcoming work.

## Setup

```bash
git clone <repo-url>
cd Generative-AI
pip install -r requirements.txt
```

Create a `.env` file in the root directory with the API keys relevant to what you're running, e.g.:

```
GROQ_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
MISTRAL_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
OPENWEATHER_API_KEY=your_key_here
HUGGINGFACEHUB_API_TOKEN=your_key_here
```

## Purpose

This repo tracks my learning path through core Generative AI concepts — chat models, embeddings, RAG, retrievers, LCEL runnables, and basic tool-calling agents — before moving into more advanced agentic AI architectures with LangGraph in a separate repo.

## Note

This is a learning/experimentation repo, not a production application — code favors clarity and concept coverage over polish. Some files (e.g. `RAG/main.py`) contain earlier commented-out approaches kept for reference.