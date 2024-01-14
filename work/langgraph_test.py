# %%
import os
from operator import itemgetter

from google.cloud import aiplatform
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.prompt import (
    SQL_FUNCTIONS_SUFFIX,
    SQL_PREFIX,
)
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.agents import AgentActionMessageLog, AgentFinish
from langchain_core.globals import set_debug
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langgraph.graph import END, Graph

os.environ["LANGCHAIN_WANDB_TRACING"] = "true"
os.environ["WANDB_PROJECT"] = "langchain-testing-private"
set_debug(True)

gcp_project_id = os.getenv("GCP_PROJECT_ID")
aiplatform.init(project=gcp_project_id)

gcp_dataset_id = os.getenv("GCP_DATASET_ID")
sqlalchemy_uri = f"bigquery://{gcp_project_id}/{gcp_dataset_id}"


# %%
intermediate_steps = []

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
tools = SQLDatabaseToolkit(db=db, llm=llm).get_tools()

list_sql_database_tool = tools[2]
info_sql_database_tool = tools[1]

llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])

# メモリを設定
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# %%
agent_assigns = {
    "input": lambda x: x["input"],
    "agent_scratchpad": lambda x: format_to_openai_functions(
        x["intermediate_steps"][:-1]
    ),
    "dialect": lambda x: x["dialect"],
    "top_k": lambda x: x["top_k"],
    "chat_history": RunnableLambda(memory.load_memory_variables)
    | itemgetter("chat_history"),
}

# エージェントを設定
agent_runnable = (
    agent_assigns | prompt | llm_with_tools | OpenAIFunctionsAgentOutputParser()
)

agent = RunnablePassthrough.assign(agent_outcome=agent_runnable)


# ツールを実行する関数
def execute_tools(data):
    agent_action = data.pop("agent_outcome")
    tool_to_use = {t.name: t for t in tools}[agent_action.tool]
    observation = tool_to_use.invoke(agent_action.tool_input)
    data["intermediate_steps"].append((agent_action, observation))
    return data


# 次に呼び出されるノードを決める関数
def should_continue(data):
    if isinstance(data["agent_outcome"], AgentFinish):
        return "exit"
    else:
        return "continue"


def first_agent(data):
    action = AgentActionMessageLog(
        tool="sql_db_list_tables",
        tool_input="",
        log="",
        message_log=[],
    )
    data["agent_outcome"] = action
    return data


def execute_first_tool(data):
    agent_action = data.pop("agent_outcome")
    tool_to_use = list_sql_database_tool
    observation = tool_to_use.invoke(agent_action.tool_input)
    data["intermediate_steps"].append((agent_action, observation))
    return data


# def second_agent(data):
#     action = AgentActionMessageLog(
#         tool="sql_db_schema",
#         # 一番最後の中間ステップの結果を使う
#         tool_input=data["intermediate_steps"][-1][1],
#         log="",
#         message_log=[],
#     )
#     data["agent_outcome"] = action
#     return data


# def execute_second_tool(data):
#     agent_action = data.pop("agent_outcome")
#     tool_to_use = info_sql_database_tool
#     observation = tool_to_use.invoke(agent_action.tool_input)
#     data["intermediate_steps"].append((agent_action, observation))
#     return data


# %%
# Graphオブジェクトの作成
workflow = Graph()

# nodeの追加
workflow.add_node("first_agent", first_agent)
workflow.add_node("first_tool", execute_first_tool)
# workflow.add_node("second_agent", second_agent)
# workflow.add_node("second_tool", execute_second_tool)
workflow.add_node("agent", agent)
workflow.add_node("tools", execute_tools)

# エントリーポイントの設定`
workflow.set_entry_point("first_agent")


# 条件付きエッジの追加
# agent -continue-> tools
#       |
#       -exit-> END
workflow.add_conditional_edges(
    "agent",  # start node
    should_continue,
    # should_continueの内容による条件分岐
    {"continue": "tools", "exit": END},
)

# tools -> agent
workflow.add_edge("first_agent", "first_tool")
# workflow.add_edge("first_tool", "second_agent")
workflow.add_edge("first_tool", "agent")
# workflow.add_edge("second_agent", "second_tool")
# workflow.add_edge("second_tool", "agent")
workflow.add_edge("tools", "agent")

chain = workflow.compile()

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

result = chain.invoke(
    {
        "input": input_message,
        "intermediate_steps": intermediate_steps,
        "dialect": "bigquery",
        "top_k": 10,
    }
)
output = result["agent_outcome"].return_values["output"]


# for output in chain.stream(
#     {
#         "input": input_message,
#         "intermediate_steps": intermediate_steps,
#         "dialect": "bigquery",
#         "top_k": 10,
#     }
# ):
#     # stream() yields dictionaries with output keyed by node name
#     for key, value in output.items():
#         print(f"Output from node '{key}':")
#         print("---")
#         print(value)
#     print("\n---\n")

# %%
