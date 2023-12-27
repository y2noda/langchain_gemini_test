# %%
import os

import streamlit as st
from dotenv import load_dotenv
from google.cloud import aiplatform
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from langchain.chat_models import ChatVertexAI
from langchain.sql_database import SQLDatabase

os.environ["LANGCHAIN_WANDB_TRACING"] = "true"
os.environ["WANDB_PROJECT"] = "langchain-testing-private"

base_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(base_dir, "../.env")
load_dotenv(dotenv_path=env_path)

gcp_project_id = os.getenv("GCP_PROJECT_ID")
aiplatform.init(project=gcp_project_id)

gcp_dataset_id = os.getenv("GCP_DATASET_ID")
sqlalchemy_uri = f"bigquery://{gcp_project_id}/{gcp_dataset_id}"

db = SQLDatabase.from_uri(sqlalchemy_uri)
chat = ChatVertexAI(
    model="gemini-pro",
    max_output_tokens=2048,
    temperature=0,
    top_p=0.8,
    top_k=40,
    verbose=True,
)

toolkit = SQLDatabaseToolkit(db=db, llm=chat)
agent_executor = create_sql_agent(
    llm=chat, toolkit=toolkit, verbose=True, top_k=10, handle_parsing_errors=True
)

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
