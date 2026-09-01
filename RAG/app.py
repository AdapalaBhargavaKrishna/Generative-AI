import os
import tempfile
import shutil

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Chat with your PDF", page_icon="📄", layout="wide")

CHROMA_DIR = "RAG/chroma_db_streamlit"

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
""",
        ),
        (
            "human",
            """
Context:
{context}

Question:
{question}
""",
        ),
    ]
)


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return HuggingFaceEmbeddings()


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatMistralAI(model="mistral-small-latest")


def build_vectorstore(pdf_path: str):
    """Load the PDF, split it, embed it, and persist a fresh Chroma store."""
    # Wipe any previous store so re-uploads don't mix documents together
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embedding_model = get_embedding_model()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DIR,
    )
    return vectorstore, len(docs), len(chunks)


def get_retriever(vectorstore):
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5,  # note: fixed typo from original 'lambda_multi'
        },
    )


def answer_question(retriever, llm, query: str) -> str:
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)
    final_prompt = PROMPT.invoke({"context": context, "question": query})
    response = llm.invoke(final_prompt)
    return response.content


# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "book_name" not in st.session_state:
    st.session_state.book_name = None

# ---------- Sidebar: upload + build ----------
with st.sidebar:
    st.header("📚 Upload your book")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Build knowledge base", type="primary", use_container_width=True):
            with st.spinner("Reading and indexing the PDF... this can take a minute."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                try:
                    vectorstore, n_pages, n_chunks = build_vectorstore(tmp_path)
                    st.session_state.vectorstore = vectorstore
                    st.session_state.book_name = uploaded_file.name
                    st.session_state.messages = []
                    st.success(f"Indexed {n_pages} pages into {n_chunks} chunks.")
                finally:
                    os.remove(tmp_path)

    if st.session_state.book_name:
        st.info(f"Currently loaded: **{st.session_state.book_name}**")

    if st.session_state.vectorstore is not None:
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# ---------- Main: chat ----------
st.title("📄 Chat with your PDF")

if st.session_state.vectorstore is None:
    st.info("Upload a PDF and click **Build knowledge base** in the sidebar to get started.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Ask a question about the document...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                retriever = get_retriever(st.session_state.vectorstore)
                llm = get_llm()
                answer = answer_question(retriever, llm, query)
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})