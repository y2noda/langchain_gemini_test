# %%
import os
from operator import itemgetter

from google.cloud import aiplatform
from langchain import hub
from langchain.agents import AgentExecutor, create_sql_agent, tool
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.globals import set_debug
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.agent import AgentFinish
from langchain.tools.render import format_tool_to_openai_function
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.prompt import (
    SQL_FUNCTIONS_SUFFIX,
    SQL_PREFIX,
)
from langchain_community.chat_models import ChatOpenAI, ChatVertexAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts.chat import (
    HumanMessagePromptTemplate,
)
from langchain_core.runnables import RunnableLambda
from langchain_experimental.sql import SQLDatabaseChain

# from langchain_google_genai import ChatGoogleGenerativeAI

# os.environ["GOOGLE_API_KEY"] = "YOUR_KEY"

# # Create the tool
# tools = [YouTubeSearchTool()]

# llm = ChatGoogleGenerativeAI(model="gemini-pro")


os.environ["LANGCHAIN_WANDB_TRACING"] = "true"
os.environ["WANDB_PROJECT"] = "langchain-testing-private"
set_debug(True)

gcp_project_id = os.getenv("GCP_PROJECT_ID")
aiplatform.init(project=gcp_project_id)

gcp_dataset_id = os.getenv("GCP_DATASET_ID")
sqlalchemy_uri = f"bigquery://{gcp_project_id}/{gcp_dataset_id}"

db = SQLDatabase.from_uri(sqlalchemy_uri)

# llm = ChatVertexAI(
#     model_name="gemini-pro",
#     max_output_tokens=2048,
#     temperature=0,
#     top_p=0.9,
#     # top_k=40,
#     verbose=True,
# )

# llm = ChatGoogleGenerativeAI(model_name="gemini-pro")


# 文字を数えるツールを定義
@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)


tools = [get_word_length]

llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0,
    verbose=True,
)
llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])

# %%

# toolkit = SQLDatabaseToolkit(db=db, llm=model)
# tools = toolkit.get_tools()

# messages = [
#     ("system", SQL_PREFIX),
#     ("ai", SQL_FUNCTIONS_SUFFIX),
#     ("human", "Chat History: {history} \n{input}"),
#     MessagesPlaceholder(variable_name="agent_scratchpad"),
# ]

# prompt = ChatPromptTemplate.from_messages(messages)

# memory = ConversationBufferMemory(return_messages=True)


# %%

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a useful assistant."),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIFunctionsAgentOutputParser()
)

intermediate_steps = []
# %%
while True:
    output = agent.invoke(
        {
            "input": "how many letters in the word 'apple'?",
            "intermediate_steps": intermediate_steps,
        }
    )
    if isinstance(output, AgentFinish):
        final_result = output.return_values["output"]
        break
    else:
        print(output.tool, output.tool_input)
        tool = {
            "get_word_length": get_word_length,
        }[output.tool]
        observation = tool.run(output.tool_input)
        intermediate_steps.append((output, observation))

print(final_result)


# input_prompt = """\
# ## 命令
# アプリユーザーのデータを抽出するためのSQLクエリを生成してください。
# SQLを生成する際には、SQL生成例を参考にしてください。
# 制約条件は必ずすべて守ってください。

# ## 目的
# クエリの目的は、プッシュ通知が配信された日から1週間以内にゲスト会員からアプリ本会員になったユーザーを見つけることです。

# ## 条件
# - DatasetName は marketing_sample_data とします。
# - ProjectName は langchain-gemini-test とします。
# - プッシュ配信日は、"2023-11-15"として新たに定義します。データ型はDATE型です。

# ## 制約条件
# - TIMESTAMP型のカラムは、DATE型に必ず変換して、使用してください。
# - 全てのテーブルの created_at のカラムのデータ型はTIMESTAMP型です。必ずDATE型に変換してください。
# - データ型に関するエラーが発生しないように、必ずデータ型を確認してください。
# - 異なるデータ型をWHERE句などの条件としてで使用する場合は、必ず変換してください。
# - TIMESTAMP型のカラムをWHERE句などの条件として使用する場合は、必ずDATE型に変換して使用してください。
# - エイリアスを定義した場合は、後のクエリで必ずエイリアスを使用してください。
# - CTEを利用して、複雑なクエリを分割して、読みやすく、管理しやすいクエリにしてください。
# - アプリユーザーには、customer_no に値を持つアプリ本会員と customer_no が null のゲスト会員がいます。
# - プッシュ通知日以前に会員になったユーザーを、ターゲットユーザーとして抽出するCTEを作成してください。
# - ターゲットユーザーの中から、プッシュ通知が配信された日から1週間以内にゲスト会員からアプリ本会員になったユーザーを抽出するCTEを作成してください。
# - 最終的に抽出するカラムは、カード番号、プッシュ配信日（2023-11-15）、本会員登録日です。
# - エイリアスは必ず英語にしてください。
# - テーブルは\`[プロジェクト名].[データセット名].[テーブル名]\`で指定すること。
# - クエリ生成時はBigQueryの記法に従ってください。
# - 異なるデータ型の場合、キャストをしてデータ型を揃えてください。
# - サブクエリがスコープ外にあるとき、直接参照しないでください。
# - SELECT句が3カラム以上の場合、並び替えは日付と名前で行ってください。
# - SELECT句に存在しないカラムのORDER BYは除外してください。
# - ORDER BYには、関数は含めないでください。
# - 命令に対して適切なカラムで並び替えを行うこと。
# - 集計関数を用いたカラムに対して必ずカラム名を付与すること。


# ## SQL生成例
# ```sql
# -- 複数のCTEを定義し、テーブルにエイリアスを使用
# WITH FirstCTE AS (
#   SELECT
#     FT.columnA,
#     FT.columnB
#   FROM
#     your_dataset.first_table AS FT
#   WHERE
#     FT.condition1 = 'value1'
# ),
# SecondCTE AS (
#   SELECT
#     ST.columnC,
#     ST.columnD
#   FROM
#     your_dataset.second_table AS ST
#   WHERE
#     ST.condition2 = 'value2'
# )

# -- CTEを組み合わせて使用
# SELECT
#   f.columnA,
#   f.columnB,
#   s.columnC,
#   s.columnD
# FROM
#   FirstCTE f
# JOIN
#   SecondCTE s ON f.columnA = s.columnC
# """

# %%
# result = agent_executor.invoke(
#     {"input": input_prompt, "dialect": db.dialect, "top_k": 10}
# )
# result = agent_executor(prompt)
# print(result["output"])


# %%
