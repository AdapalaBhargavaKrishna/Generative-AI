from langchain.tools import tool

@tool
def get_greeting(name : str) -> str:
    '''Generate a greeting message for a user'''
    return f'Hellow {name} , Welcome to the AI world'

result = get_greeting.invoke({'name' : 'Krishna'})

print(result)