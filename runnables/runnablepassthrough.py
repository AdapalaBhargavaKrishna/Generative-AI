from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

load_dotenv()

model = ChatGroq(model = 'openai/gpt-oss-120b')
parser = StrOutputParser()

code_prompt = ChatPromptTemplate.from_messages([
    ('system' , 'You are a code generator'),
    ('human' , '{topic}')
])

explain_prompt = ChatPromptTemplate.from_messages([
    ('system' , 'You are helpful assistant who explains code in simple terms'),
    ('human' , 'Explain the following code in simple words: \n{code}')
])

seq = code_prompt | model | parser

seq2 = RunnableParallel({
    'code' : RunnablePassthrough(),
    'explanation' : explain_prompt | model | parser
})

chain = seq | seq2

result = chain.invoke({'topic' : 'please write a code of palindrome in python'})

print(result['code'])
print(result['explanation'])