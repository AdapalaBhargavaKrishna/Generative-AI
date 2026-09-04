from rich import print
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv()

@tool
def get_text_length(text : str) -> str:
    '''Returns the number of character in a given text'''
    return len(text)

tools = {
    'get_text_length' : get_text_length
}

llm = ChatGroq(model = 'openai/gpt-oss-120b')

llm_with_tool = llm.bind_tools([get_text_length])

message = []
query = HumanMessage("'Return the number of characters in the given text : 'Hello how are you'")
message.append(query)

result = llm_with_tool.invoke(message)
message.append(result)

if result.tool_calls:
    tool_name = result.tool_calls[0]['name']
    tool_message = tools[tool_name].invoke(result.tool_calls[0])
    message.append(tool_message)
    print(message)

result = llm_with_tool.invoke(message)
print(result.content)