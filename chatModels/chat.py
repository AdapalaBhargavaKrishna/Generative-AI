from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

# Google Gemini
google_model = init_chat_model(
    "google_genai:gemini-3.5-flash-lite"
)

# Groq
groq_model = init_chat_model(
    "groq:openai/gpt-oss-120b"
)

# Mistral
mistral_model = init_chat_model(
    "mistralai:mistral-small-latest"
)

# Test Google
response = google_model.invoke(
    "Why do parrots talk?"
)

print("GOOGLE:")
print(response.content)

# Test Groq
response = groq_model.invoke(
    "Why do parrots talk?"
)

print("\nGROQ:")
print(response.content)

# Test Mistral
response = mistral_model.invoke(
    "Why do parrots talk?"
)

print("\nMISTRAL:")
print(response.content)