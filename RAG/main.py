from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

data = PyPDFLoader('RAG/document loaders/deepLearning.pdf')
docs = data.load()

template = ChatPromptTemplate.from_messages(
    [('system' , 'you are a AI that summarises the text'),
     ('human' , '{data}')]
)

model = init_chat_model(
    'mistralai:mistral-small-latest'
)

prompt = template.format_messages(data = docs[0].page_content)

result = model.invoke(prompt)
print(result.content)