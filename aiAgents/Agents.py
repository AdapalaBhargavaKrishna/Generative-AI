import os
import requests
from rich import print
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
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

# Tool Mapping
tools = {
    'get_weather': get_weather,
    'get_news': get_news
}

llm_with_tool = llm.bind_tools([get_weather, get_news])

# Conversation Loop
messages = []

print('City intelligence system')

while True:
    user_input = input('You : ')
    messages.append(HumanMessage(content=user_input))

    while True:
        # Get response from LLM
        result = llm_with_tool.invoke(messages)
        messages.append(result)

        print('Result : ', result)

        # Check if LLM wants to call a tool
        if result.tool_calls:
            for tool_call in result.tool_calls:
                tool_name = tool_call['name']

                print(tool_call)

                # Ask for permission before executing tool
                confirm = input(
                    f'Agent wants to call {tool_name} Approve (yes/no) : '
                )

                if confirm.lower() == 'no':
                    print('denied')
                    break

                # Execute the requested tool
                tool_result = tools[tool_name].invoke(tool_call)

                # Send tool result back to LLM
                messages.append(
                    ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_call['id']
                    )
                )

            continue

        else:
            print('Final Result :')
            print(result.content)
            break