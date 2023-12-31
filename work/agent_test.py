import os

from google.cloud import aiplatform
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.chat_models import ChatVertexAI
from langchain.sql_database import SQLDatabase

gcp_project_id = os.getenv("GCP_PROJECT_ID")
aiplatform.init(project=gcp_project_id)

gcp_dataset_id = os.getenv("GCP_DATASET_ID")
sqlalchemy_uri = f"bigquery://{gcp_project_id}/{gcp_dataset_id}"

db = SQLDatabase.from_uri(sqlalchemy_uri)
chat = ChatVertexAI(
    model="gemini-pro",
    max_output_tokens=2048,
    temperature=0,
    # top_p=0.7,
    # top_k=40,
    verbose=True,
)

toolkit = SQLDatabaseToolkit(db=db, llm=chat)
agent_executor = create_sql_agent(
    llm=chat,
    toolkit=toolkit,
    verbose=True,
    top_k=3,
    handle_parsing_errors=True,
    # return_direct=True,
)
