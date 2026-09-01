from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

load_dotenv()

docs = [
    Document(
        page_content="Python is widely used in Artificial Intelligence.",
        metadata={"source": "python.txt"}
    ),
    Document(
        page_content="Pandas is used for data analysis in Python.",
        metadata={"source": "pandas.txt"}
    ),
    Document(
        page_content="Neural networks are used in deep learning.",
        metadata={"source": "deep_learning.txt"}
    ),
    Document(
        page_content="LangChain is used to build applications powered by language models.",
        metadata={"source": "langchain.txt"}
    ),
    Document(
        page_content="Retrieval Augmented Generation helps language models use external knowledge.",
        metadata={"source": "rag.txt"}
    )
]

embedding_model = MistralAIEmbeddings(model='mistral-embed')

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory='RAG/vector store/chroma-db'
)

result = vectorstore.similarity_search('what is used for data analytics?', k = 1)

print(result)