from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

# from langchain_community.document_loaders import PyPDFLoader
# from langchain.chat_models import init_chat_model
# from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

embedding_model = HuggingFaceEmbeddings()

vectorstore = Chroma(
    persist_directory='RAG/chroma_db',
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(
    search_type = 'mmr',
    search_kwargs = {
        'k' : 4,
        'fetch_k' : 10,
        'lambda_multi' : 0.5,

    }
)

llm = ChatMistralAI(model = 'mistral-small-latest')

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),
        (
            "human",
            """
Context:
{context}

Question:
{question}
"""
        )
    ]
)

print('RAG system created')

print('press 0 to exit')

while True:
    query = input('You : ')
    if query == '0':
        break

    docs = retriever.invoke(query)

    context = '\n\n'.join([doc.page_content for doc in docs])
    final_prompt = prompt.invoke({
        'context': context,
        'question': query
    })

    response = llm.invoke(final_prompt)
    print(f'\nAI : {response.content}')