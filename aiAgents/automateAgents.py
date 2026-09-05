import os
import requests
from rich import print
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import HumanMessage, ToolMessage
from tavily import TavilyClient

load_dotenv()

# Weather Tool
@tool
def get_weather(city: str) -> str:
    '''Get current weather of a city'''
    API_KEY = os.getenv("OPENWEATHER_API_KEY")
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    response = requests.get(url)
    data = response.json()

    if response.status_code != 200:
        return f"Error: {data.get('message', 'Something went wrong')}"

    temp = data['main']['temp']
    desc = data['weather'][0]['description']

    return f'Weather in {city}: {desc}, {temp}°C'

# News Tool
tavily_client = TavilyClient()

@tool
def get_news(city: str) -> str:
    '''Get latest news about city'''

    response = tavily_client.search(
        query=f'latest news in {city}',
        search_depth='basic',
        max_results=3
    )

    results = response.get('results', [])

    if not results:
        return f'Non news found in {city}'

    news_list = []

    for r in results:
        title = r.get('title', 'No title')
        url = r.get('url', '')
        snippet = r.get('content', '')

        news_list.append(
            f'{title}\n  🔗 {url}\n  {snippet[:100]}...'
        )

    return f'Latest news in {city}:\n\n' + '\n\n'.join(news_list)

# LLM Setup
llm = ChatGroq(model='openai/gpt-oss-120b')

@wrap_tool_call
def human_approval(request, handler):
    '''Ask for human approval before every tool call.'''
    tool_name = request.tool_call['name']
    confirm = input(f'Agent wants to call {tool_name}. Approve ? (yes/no) : ')

    if confirm.lower() != 'yes':
        return ToolMessage(
            content = 'Tool call denied by user',
            tool_call_id = request.tool_call['id']
        )

    return handler(request)

agent = create_agent(
    llm,
    tools = [get_weather , get_news],
    system_prompt = 'you are a helpful city assistant.',
    middleware = [human_approval]
)

print('City Agent')

while True:
    user_input = input('You : ')

    result = agent.invoke({
        "messages": [{"role": "user", "content": user_input}],
    })

    print('bot : ' , result['messages'][-1].content)