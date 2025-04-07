"""
Classification Agent

This module defines a ClassificationAgent that retrieves information from the 'reports'
database based on a user's query. It's designed to function similarly to a RAG agent,
but with a specific focus on searching pre-generated reports.
"""

import typer
from dotenv import load_dotenv

import langroid.language_models as lm
from langroid.agent.task import Task
from langroid.utils.configuration import set_global, Settings
from langroid.agent.tool_message import ToolMessage
from langroid.agent.special.doc_chat_agent import DocChatAgent, DocChatAgentConfig, ChatDocument, ChatDocMetaData
from langroid.vector_store.qdrantdb import QdrantDB, QdrantDBConfig

app = typer.Typer()


class RetrieveReportTool(ToolMessage):
    """
    Tool for retrieving relevant reports from the 'reports' vector database.
    """

    request = "retrieve_report"
    purpose = "Retrieve relevant reports from the 'reports' vector database."
    query: str

    @classmethod
    def examples(cls) -> list["ToolMessage"]:
        return [cls(query="Why is DeepSeek cheap?")]

    @classmethod
    def instructions(cls) -> str:
        return "Use this tool to find existing research reports that answer the user's query."


class ClassificationAgent(DocChatAgent):
    """
    ClassificationAgent retrieves information from the 'reports' database.
    If relevant reports are found, it outputs them.
    If no relevant reports are found, it outputs a predefined signal.
    """

    def __init__(self, model: str, sys_msg: str):
        llm_config = lm.OpenAIGPTConfig(chat_model=model or lm.OpenAIChatModel.GPT4o)
        config = DocChatAgentConfig(  # Use DocChatAgentConfig instead of ChatAgentConfig
            system_message=sys_msg, llm=llm_config, doc_paths=[]  # Initialize doc_paths as an empty list
        )
        super().__init__(config)
        self.enable_message(RetrieveReportTool)

        print("[INFO] Connecting to Qdrant Cloud 'reports' collection...")
        qdrant_config = QdrantDBConfig(collection_name="reports", cloud=True, replace_collection=False)
        self.vecdb = QdrantDB(qdrant_config)

    def retrieve_report(self, msg: RetrieveReportTool) -> ChatDocument:
        """
        Retrieve relevant reports from the 'reports' vector database.
        If no relevant reports are found, return a ChatDocument with content "NO_RELEVANT_REPORTS".
        """

        print(f"[yellow] Checking 'reports' DB for: {msg.query} [/yellow]")
        _, extracts = self.get_relevant_extracts(msg.query)  # Assuming get_relevant_extracts is available or adjust as needed

        if not extracts:
            print("[yellow] No relevant reports found in 'reports' DB. [/yellow]")
            return ChatDocument(content="NO_RELEVANT_REPORTS", metadata=ChatDocMetaData(source="agent"))

        print("[green] Found relevant reports in 'reports' DB! [/green]")

        result_str = "=== Relevant Reports ===\n"
        for i, e in enumerate(extracts, start=1):
            content = e.content.strip()
            result_str += f"[{i}] CONTENT:\n{content}\n\n"  # Adjust formatting as needed
        return ChatDocument(content=result_str, metadata=ChatDocMetaData(source="agent"))


def make_classification_task(
    model: str = lm.OpenAIChatModel.GPT4o,
    sys_msg: str = """
    You are a classification agent. Your task is to retrieve information from a database of research reports.
    If a user asks a question that is answered by the reports, provide the relevant report.
    If there is no relevant report, respond with ONLY the string "NO_RELEVANT_REPORTS".
    """,
    restart: bool = False,
) -> Task:
    load_dotenv()
    set_global(Settings(debug=False, cache=True))
    agent = ClassificationAgent(model, sys_msg)
    return Task(
        agent,
        name="ClassificationAgent",
        llm_delegate=True,
        single_round=True,
        interactive=False,
        restart=restart,
    )


@app.command()
def main(
    query: str = typer.Argument(..., help="User query to retrieve reports"),
) -> None:
    """
    Command-line function to run the ClassificationAgent.
    """

    task = make_classification_task(restart=False)
    result = task.run(query)

    if result is not None and result.content == "NO_RELEVANT_REPORTS":
        print("NO_RELEVANT_REPORTS")
    elif result is not None:
        print("Classification Agent Result:", result.content)
    else:
        print("NO_RELEVANT_REPORTS")


if __name__ == "__main__":
    app()