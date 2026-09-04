from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = ChatPromptTemplate.from_template(
    'Explain {topic} in simple words'
)

model = ChatGroq(model = 'openai/gpt-oss-120b')
parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke('Machine Learning')

print(result)