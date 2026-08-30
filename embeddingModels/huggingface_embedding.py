from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()


embedding = HuggingFaceEmbeddings(
    model = 'sentence-transformers/all-MiniLM-L6-v2'
)

texts = [
        'Hello this is krishna',
        'Hello your name is YouTube',
        'And you all are very beautiful'
]

vector = embedding.embed_documents(texts)
print(vector)