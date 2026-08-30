from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage , SystemMessage , HumanMessage

load_dotenv()

model = init_chat_model("mistralai:mistral-small-latest")

messages = [
    SystemMessage(content='you are a emoji only AI assistant')
]
while True:
    prompt = input('You : ')
    messages.append(HumanMessage(content=prompt))

    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content))

    print('Bot : ', response.content)