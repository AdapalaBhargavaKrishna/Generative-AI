from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader('RAG/document loaders/deepLearning.pdf')

docs = data.load()

print(len(docs))