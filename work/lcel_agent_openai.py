# %%
import os
from operator import itemgetter

from google.cloud import aiplatform
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.agent import AgentFinish
from langchain.schema.output_parser import StrOutputParser
from langchain.tools.render import format_tool_to_openai_function
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.prompt import (
    SQL_FUNCTIONS_SUFFIX,
    SQL_PREFIX,
)
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.globals import set_debug
from langchain_core.runnables import RunnableLambda
from openai import BadRequestError

os.environ["LANGCHAIN_WANDB_TRACING"] = "true"
os.environ["WANDB_PROJECT"] = "langchain-testing-private"
# set_debug(True)

gcp_project_id = os.getenv("GCP_PROJECT_ID")
aiplatform.init(project=gcp_project_id)

gcp_dataset_id = os.getenv("GCP_DATASET_ID")
sqlalchemy_uri = f"bigquery://{gcp_project_id}/{gcp_dataset_id}"


# %%
class SQLAgent:
    def __init__(self):
        self.intermediate_steps = []

        # エージェントプロンプト
        prompt = ChatPromptTemplate.from_messages(
            [
                # ("system", "You are a useful assistant."),
                ("system", SQL_PREFIX),
                ("ai", SQL_FUNCTIONS_SUFFIX),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        # エラーメッセージを説明するためのプロンプト
        fallback_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are the AI that tells the user what the error is in plain Japanese. Since the error occurs at the end of the step, you must guess from the process flow and the error message, and communicate the error message to the user in an easy-to-understand manner.",
                ),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        # モデルを設定
        llm = ChatOpenAI(
            temperature=0,
            # model="gpt-4"
            model="gpt-4-0613",
        )

        db = SQLDatabase.from_uri(sqlalchemy_uri)
        self.tools = SQLDatabaseToolkit(db=db, llm=llm).get_tools()

        llm_with_tools = llm.bind(
            functions=[format_tool_to_openai_function(t) for t in self.tools]
        )

        # メモリを設定
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        agent_assigns = {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_functions(
                x["intermediate_steps"][:-1]
            ),
            "dialect": lambda x: x["dialect"],
            "top_k": lambda x: x["top_k"],
            "chat_history": RunnableLambda(self.memory.load_memory_variables)
            | itemgetter("chat_history"),
        }

        # エージェントを設定
        self.sql_agent = (
            agent_assigns | prompt | llm_with_tools | OpenAIFunctionsAgentOutputParser()
        )

        # エラーが発生した場合のLLMを設定
        self.fallback_chain = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_functions(
                    x["intermediate_steps"][:-1]
                ),
                "dialect": lambda x: x["dialect"],
                "top_k": lambda x: x["top_k"],
                "chat_history": RunnableLambda(self.memory.load_memory_variables)
                | itemgetter("chat_history"),
            }
            | fallback_prompt
            | llm
            | StrOutputParser()
        )

    def run(self, input_message):
        while True:
            try:
                output = self.agent.invoke(
                    {
                        "input": input_message,
                        "intermediate_steps": self.intermediate_steps,
                        "dialect": "bigquery",
                        "top_k": 10,
                    }
                )
            except BadRequestError as error:
                message = error.response.json()["error"]["message"]
                self.intermediate_steps[-1] = (self.intermediate_steps[-1][0], "")
                final_result = self.fallback_chain.invoke(
                    {
                        "input": message,
                        "intermediate_steps": self.intermediate_steps,
                        "dialect": "bigquery",
                        "top_k": 10,
                    }
                )
                yield final_result
                return

            if isinstance(output, AgentFinish):
                message = output.return_values["output"]
                self.memory.save_context({"input": input_message}, {"output": message})
                yield message
                return
            else:
                messages = ["== selected tool ==", output.tool, str(output.tool_input)]
                tool = {tool.name: tool for tool in self.tools}[output.tool]
                observation = tool.run(output.tool_input)
                messages += [
                    "== observation ==",
                    observation,
                ]
                self.intermediate_steps.append((output, observation))
                yield "\n".join(messages)


# %%
input_message = """\
## 命令
データベースにあるテーブル名を確認した後に、
アプリユーザーのデータを抽出するためのSQLクエリを生成してください。
SQLクエリを生成するための情報を適宜取得してください。

## 目的
クエリの目的は、プッシュ通知が配信された日から1週間以内にゲスト会員からアプリ本会員になったユーザーを見つけることです。

## 条件
- プッシュ配信日は、"2023-11-15"として新たに定義します。データ型はDATE型です。
"""
# agent = Agent(input_message)
agent = SQLAgent()
step_count = 0

# エージェントを実行
for step in agent.run(input_message):
    step_count += 1
    print(f"***** step {step_count} *****")
    print(step)
    print("\n")


# for step in agent.run("利用したSQLクエリそのもののみを教えてください"):
#     step_count += 1
#     print(f"***** step {step_count} *****")
#     print(step)
#     print("\n")

# %%
