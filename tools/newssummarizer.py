from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

search_tool = TavilySearchResults(max_result = 5)

llm = ChatGroq(model = 'openai/gpt-oss-120b')

prompt = ChatPromptTemplate.from_template(
    '''
    You are a helpful assistant
    summarizes the following news into clear bullet points
    {news}
    '''
)

chain = prompt | llm | StrOutputParser()

news_result = search_tool.run('Latest AI news of 2026')
result = chain.invoke({'news' : news_result})

print(result)