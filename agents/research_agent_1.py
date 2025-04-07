"""
Research Agent 1 - Initial Information Retrieval

This agent is responsible for the 1st stage of research.

#### **Input Parameters:**
- `query (str)`: The original user question.
- `sub_question (str)`: A refined sub-question for more specific research.

#### **Key Responsibilities:**
- Performs web searches using external APIs or search engines.
- Retrieves raw content from online sources.
- Returns a summary with sources.
"""

import typer
from dotenv import load_dotenv
from agents.research_agent_template import ResearchAgentTemplate, RelevantExtractsTool, RelevantSearchExtractsTool

import langroid.language_models as lm
from langroid.agent.task import Task
from langroid.agent.special.doc_chat_agent import DocChatAgentConfig
from langroid.utils.configuration import set_global, Settings

app = typer.Typer()


class ResearchAgent1(ResearchAgentTemplate):
    """Research Agent 1 - Handles the first research sub-question."""


def make_research_task_1(
    original_query: str,
    sub_question: str,
    model: str = "",
    restart: bool = False
) -> Task:
    """
    Creates and configures a ResearchAgent1 task for retrieving and summarizing research data.

    Args:
        original_query (str): The original user research question.
        sub_question (str): The specific sub-question assigned to this agent.
        model (str): The LLM model to use (default is GPT-4o).
        restart (bool): Whether to restart the task each time it's run.

    Returns:
        Task: Configured Task instance for ResearchAgent1.
    """
    # Load environment variables
    load_dotenv()

    # Set global settings
    set_global(Settings(debug=False, cache=True))

    # Define LLM configuration
    llm_config = lm.OpenAIGPTConfig(
        chat_model=model or lm.OpenAIChatModel.GPT4o
    )

    system_message = f"""
    You are a research assistant helping to find relevant information.

    ## **Context of This Task**
    The user originally asked the following research question:

    **Original Research Question**:
    "{original_query}"

    To systematically address this broad topic, the question was **decomposed** into multiple sub-questions. 
    You are now responsible for researching **one specific sub-question**.

    ## **Your Task**
    1. Focus **ONLY** on the assigned sub-question below.
    2. First, check the vector database using `relevant_extracts` for related information.
    3. If no results are found, perform a web search using `relevant_search_extracts`.
    4. Provide a **concise, well-structured summary** with relevant sources.

    **IMPORTANT:**
    - Do **NOT** go off-topic—your response should stay directly related to the **Original Research Question**.
    - Do **NOT** generate new sub-questions or discuss unrelated topics.
    - Your job is to **research, extract, and summarize**.

    ## **Assigned Sub-Question**
    "{sub_question}"
    """

    # Create system message config
    config = DocChatAgentConfig(
        use_functions_api=True,
        use_tools=True,
        llm=llm_config,
        system_message=system_message,
    )

    # Create agent
    agent = ResearchAgent1(config)
    agent.enable_message(RelevantExtractsTool)
    agent.enable_message(RelevantSearchExtractsTool)

    # Create and return the task
    return Task(
        agent,
        name="ResearchAgent1",
        llm_delegate=True,
        single_round=True,  # One-time execution per call
        interactive=False,  # Prevents entering an interactive loop
        restart=restart,
    )


@app.command()
def main(
    original_query: str = typer.Argument(..., help="Original research question"),
    sub_question: str = typer.Argument(..., help="Assigned sub-question for this agent"),
) -> None:
    """
    Command-line function to run ResearchAgent1 via make_research_task_1().
    The user only needs to input the original query and sub-question.
    """

    task = make_research_task_1(original_query, sub_question, restart=False)
    response = task.run(sub_question)



if __name__ == "__main__":
    app()