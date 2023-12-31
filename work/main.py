# %%
import os

import streamlit as st
from agent_test import agent_executor
from dotenv import load_dotenv
from langchain.callbacks.streamlit import StreamlitCallbackHandler

os.environ["LANGCHAIN_WANDB_TRACING"] = "true"
os.environ["WANDB_PROJECT"] = "langchain-testing-private"

base_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(base_dir, "../.env")
load_dotenv(dotenv_path=env_path)

# %%
if "agent_chain" not in st.session_state:
    st.session_state.agent_chain = agent_executor

st.title("Langchain_gemini_test")

if "message" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("What is up?")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        callback = StreamlitCallbackHandler(st.container())
        response = st.session_state.agent_chain(prompt, callbacks=[callback])
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
