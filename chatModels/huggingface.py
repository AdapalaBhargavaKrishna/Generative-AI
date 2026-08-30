from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    # temperature=0.7,
    # max_length=1024,
)
model = ChatHuggingFace(llm = llm)

response = model.invoke("Why do parrots talk?")
print(response.content)