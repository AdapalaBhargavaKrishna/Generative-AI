from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

load_dotenv()

model = ChatGroq(model = 'openai/gpt-oss-120b')
parser = StrOutputParser()

short_prompt = ChatPromptTemplate.from_template(
    'Explain {topic} in 1 - 2 lines'
)

detailed_prompt = ChatPromptTemplate.from_template(
    'Explain {topic} in detail'
)

topic = 'Machine Learning'

chain = RunnableParallel({
'short' : RunnableLambda(lambda x : x['short']) | short_prompt | model | parser,
'detailed' : RunnableLambda(lambda x : x['detailed']) | detailed_prompt | model | parser
})

result = chain.invoke({
    'short' : {'topic' : 'Machine Learning'},
    'detailed' : {'topic' : 'Deep Learning'}
})
print(result)