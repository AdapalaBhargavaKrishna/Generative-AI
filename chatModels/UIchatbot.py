import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

load_dotenv()

st.set_page_config(page_title="Bike Bot", page_icon="😃")
st.title("😃 Bike Only AI Assistant")

# Cache the model so it's not re-initialized on every rerun
@st.cache_resource
def get_model():
    return init_chat_model("mistralai:mistral-small-latest")

model = get_model()

# Initialize session state for messages (same list structure as your original script)
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content='you are a Bike expert AI assistant')
    ]

# Render existing chat history (skip the SystemMessage)
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# Chat input (equivalent to input('You : '))
prompt = st.chat_input("You : ")

if prompt:
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    response = model.invoke(st.session_state.messages)
    st.session_state.messages.append(AIMessage(content=response.content))

    with st.chat_message("assistant"):
        st.markdown(response.content)