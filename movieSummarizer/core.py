from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List,Optional
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-safeguard-20b")

class Movie(BaseModel):
    movie_name: str
    genre: List[str]
    main_characters: List[str]
    summary: str
    theme: str
    rating: Optional[str]

parser = PydanticOutputParser(pydantic_object=Movie)

prompt = ChatPromptTemplate.from_messages([
    (
        "system","""You are a movie summarization assistant.
        Analyze the movie description and return:

        Movie Name:
        Genre:
        Main Characters:
        Summary:
        Theme:

        Rules:
        - Keep the summary concise.
        - Do not use JSON.
        - If information is missing, write 'Not specified'.
        """
    ),
      ("human","{movie_description}")
])

prompt2 = ChatPromptTemplate.from_messages([
    ("system","""you are a movie summarization assistant. Analyze the movie description and return the following information in a structured format {format_instruction}:"""),
    ("human","{movie_description}")
])

para = input("Enter the movie description: ")

final_prompt = prompt2.invoke({
    "movie_description": para,
    "format_instruction": parser.get_format_instructions()
})

response = model.invoke(final_prompt)
movie_info = parser.parse(response.content)
print("\nMovie Summary:\n")
print(movie_info)
