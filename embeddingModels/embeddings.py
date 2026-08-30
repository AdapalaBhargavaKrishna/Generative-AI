from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model='text-embedding-3-large',
    dimensions=64
)

vector = embeddings.embed_query('You are going to learn GenAI')