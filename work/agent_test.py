# %%
import os
from typing import Any, Dict, List, Optional

from google.cloud import aiplatform
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatVertexAI
from langchain.globals import set_debug
from langchain.sql_database import SQLDatabase
from langchain_community.tools.sql_database.prompt import QUERY_CHECKER
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool,
)
from langchain_core.callbacks import (
    CallbackManagerForToolRun,
)
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import root_validator
from langchain_core.tools import BaseTool

os.environ["LANGCHAIN_WANDB_TRACING"] = "true"
os.environ["WANDB_PROJECT"] = "langchain-testing-private"
set_debug(True)

gcp_project_id = os.getenv("GCP_PROJECT_ID")
aiplatform.init(project=gcp_project_id)

gcp_dataset_id = os.getenv("GCP_DATASET_ID")
sqlalchemy_uri = f"bigquery://{gcp_project_id}/{gcp_dataset_id}"

db = SQLDatabase.from_uri(sqlalchemy_uri)

# %%
chat = ChatVertexAI(
    model="gemini-pro",
    max_output_tokens=2048,
    temperature=0,
    top_p=1,
    # top_k=40,
    verbose=True,
)

toolkit = SQLDatabaseToolkit(db=db, llm=chat)


# agent_executor = create_sql_agent(
#     llm=chat,
#     toolkit=toolkit,
#     verbose=True,
#     top_k=3,
#     handle_parsing_errors=True,
#     # return_direct=True,
# )


class CustomQuerySQLDataBaseTool(QuerySQLDataBaseTool):
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        # クエリのクリーニング
        cleaned_query = query.strip().removeprefix("```sql\n").removesuffix("\n```")
        # クリーニング後のクエリを実行
        return super()._run(cleaned_query, run_manager)


CUSTOM_QUERY_CHECKER = """\
{query}
Double check the {dialect} query above for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

Regarding table naming, please follow these guidelines:
- Ensure that tables are specified in the [ProjectName].[DatasetName].[TableName] format.
- Use aliases appropriately for tables to improve query readability and manageability.

Additionally, when utilizing the WITH clause in your query, please be mindful of these points:
- Each Common Table Expression (CTE) within the WITH clause should be defined only once.
- Use each CTE effectively within the query to avoid redundancy.
- Avoid creating overly complex CTEs, as they can lead to performance issues.
- Be aware of the sequence in which CTEs are defined; they should be established before being referenced in the main query.
- Importantly, the use of the WITH clause requires a main query to be present. Merely defining CTEs is not sufficient for a complete query."

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

Output the final SQL query only.

SQL Query: """


class CustomQuerySQLCheckerTool(QuerySQLCheckerTool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.template = CUSTOM_QUERY_CHECKER  # カスタムテンプレートの設定

    @root_validator(pre=True, allow_reuse=True)
    def initialize_llm_chain(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        # カスタムテンプレートが設定されている場合、それを使用
        if "llm_chain" not in values:
            from langchain.chains.llm import LLMChain

            values["llm_chain"] = LLMChain(
                llm=values.get("llm"),
                prompt=PromptTemplate(
                    template=values.get("template", QUERY_CHECKER),  # ここでカスタムテンプレートを使用
                    input_variables=["dialect", "query"],
                ),
            )

        if values["llm_chain"].prompt.input_variables != ["dialect", "query"]:
            raise ValueError(
                "LLM chain for QueryCheckerTool must have input variables ['query', 'dialect']"
            )

        return values


class CustomSQLDatabaseToolkit(SQLDatabaseToolkit):
    def get_tools(self) -> List[BaseTool]:
        # カスタムツールの定義
        list_sql_database_tool = ListSQLDatabaseTool(db=self.db)
        info_sql_database_tool_description = (
            "Input to this tool is a comma-separated list of tables, output is the "
            "schema and sample rows for those tables. "
            "Be sure that the tables actually exist by calling "
            f"{list_sql_database_tool.name} first! "
            "Example Input: table1, table2, table3"
        )
        info_sql_database_tool = InfoSQLDatabaseTool(
            db=self.db, description=info_sql_database_tool_description
        )
        query_sql_database_tool_description = (
            "Use this tool to query the database when you know the correct SQL query."
            "The input for this tool should be a detailed and correct SQL query, which has been double-checked for accuracy using the QuerySqlCheckerTool. "
            "The output will be a result from the database. If the query is not correct, an error message "
            "will be returned. If an error is returned, rewrite the query, check the "
            "query, and try again. If you encounter an issue with Unknown column "
            f"'xxxx' in 'field list', use {info_sql_database_tool.name} "
            "to query the correct table fields."
            "If you encounter a syntax error with 'Unexpected identifier', "
            "use the format [ProjectName].[DatasetName].[TableName] for specifying tables."
            "Also, to improve query readability, "
            "it is recommended to use aliases for these tables."
            "And, when using the WITH clause, please be mindful of these points:"
            "the use of the WITH clause requires a main query to be present. Merely defining CTEs is not sufficient for a complete query."
        )
        query_sql_database_tool = CustomQuerySQLDataBaseTool(
            db=self.db, description=query_sql_database_tool_description
        )
        query_sql_checker_tool_description = (
            "Use this tool to double check if your query is correct before executing "
            "it. Always use this tool before executing a query with "
            f"{query_sql_database_tool.name}!"
        )
        query_sql_checker_tool = CustomQuerySQLCheckerTool(
            db=self.db, llm=self.llm, description=query_sql_checker_tool_description
        )
        return [
            query_sql_database_tool,
            info_sql_database_tool,
            list_sql_database_tool,
            query_sql_checker_tool,
        ]


# カスタムツールキットの作成
custom_toolkit = CustomSQLDatabaseToolkit(db=db, llm=chat)

# カスタムエージェントの作成
agent_executor = create_sql_agent(
    llm=chat,
    toolkit=custom_toolkit,
    # toolkit=toolkit,
    verbose=True,
    top_k=3,
    handle_parsing_errors=True,
)

# %%
result = agent_executor.run(
    """\
## 命令
アプリユーザーのデータを抽出するためのSQLクエリを生成してください。
制約条件は必ずすべて守ってください。
SQLを生成する際には、SQL生成例を参考にしてください。

## クエリの目的
クエリの目的は、プッシュ通知が配信された日から1週間以内にゲスト会員からアプリ本会員になったユーザーを見つけることです。

## 条件
- DatasetName は marketing_sample_data とします。
- ProjectName は langchain-gemini-test とします。
- プッシュ配信日は、"2023-11-15"として新たに定義します。データ型はDATE型です。


## 制約条件
- TIMESTAMP型のカラムは、DATE型に必ず変換して、使用してください。
- 全てのテーブルの created_at のカラムのデータ型はTIMESTAMP型です。必ずDATE型に変換してください。
- データ型に関するエラーが発生しないように、必ずデータ型を確認してください。
- 異なるデータ型をWHERE句などの条件としてで使用する場合は、必ず変換してください。
- TIMESTAMP型のカラムをWHERE句などの条件として使用する場合は、必ずDATE型に変換して使用してください。
- エイリアスを定義した場合は、後のクエリで必ずエイリアスを使用してください。
- CTEを利用して、複雑なクエリを分割して、読みやすく、管理しやすいクエリにしてください。
- アプリユーザーには、customer_no に値を持つアプリ本会員と customer_no が null のゲスト会員がいます。
- プッシュ通知日以前に会員になったユーザーを、ターゲットユーザーとして抽出するCTEを作成してください。
- ターゲットユーザーの中から、プッシュ通知が配信された日から1週間以内にゲスト会員からアプリ本会員になったユーザーを抽出するCTEを作成してください。
- 最終的に抽出するカラムは、カード番号、プッシュ配信日（2023-11-15）、本会員登録日です。
- エイリアスは必ず英語にしてください。
- テーブルは\`[プロジェクト名].[データセット名].[テーブル名]\`で指定すること。
- クエリ生成時はBigQueryの記法に従ってください。
- 異なるデータ型の場合、キャストをしてデータ型を揃えてください。
- サブクエリがスコープ外にあるとき、直接参照しないでください。
- SELECT句が3カラム以上の場合、並び替えは日付と名前で行ってください。
- SELECT句に存在しないカラムのORDER BYは除外してください。
- ORDER BYには、関数は含めないでください。
- 命令に対して適切なカラムで並び替えを行うこと。
- 集計関数を用いたカラムに対して必ずカラム名を付与すること。


## SQL生成例
```sql
-- 複数のCTEを定義し、テーブルにエイリアスを使用
WITH FirstCTE AS (
  SELECT
    FT.columnA,
    FT.columnB
  FROM
    `your_dataset.first_table` AS FT
  WHERE
    FT.condition1 = 'value1'
),
SecondCTE AS (
  SELECT
    ST.columnC,
    ST.columnD
  FROM
    `your_dataset.second_table` AS ST
  WHERE
    ST.condition2 = 'value2'
)

-- CTEを組み合わせて使用
SELECT
  f.columnA,
  f.columnB,
  s.columnC,
  s.columnD
FROM
  FirstCTE f
JOIN
  SecondCTE s ON f.columnA = s.columnC
```
"""
)
print(result)

# %%
