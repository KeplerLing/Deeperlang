"""
Research Agent 3 - Initial Information Retrieval

This agent is responsible for the 3rd stage of research.

#### **Input Parameters:**
- `query (str)`: The original user question.
- `sub_question (str)`: A refined sub-question for more specific research.
- `context_1 (str)`: The results from research_agent_1
- `context_2 (str)`: The results from research_agent_2

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


class ResearchAgent3(ResearchAgentTemplate):
    """Research Agent 3 - Handles the third research sub-question."""


def make_research_task_3(
    original_query: str,
    sub_question: str,
    context_1: str,
    context_2: str,
    model: str = "",
    restart: bool = False
) -> Task:
    """
    Creates and configures a ResearchAgent3 task for retrieving and summarizing research data.

    Args:
        original_query (str): The original user research question.
        sub_question (str): The specific sub-question assigned to this agent.
        context_1 (str): The research findings from ResearchAgent1.
        context_2 (str): The research findings from ResearchAgent2.
        model (str): The LLM model to use (default is GPT-4o).
        restart (bool): Whether to restart the task each time it's run.

    Returns:
        Task: Configured Task instance for ResearchAgent3.
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

    ## **Previous Research Context (If Provided)**
    The following context has already been explored by previous research agents:

    **Context from Research Agent 1**:
    {context_1}

    **Context from Research Agent 2**:
    {context_2}

    Use these contexts to inform your research, ensuring that:
    - Your response **builds upon** previous research rather than repeating it.
    - You **continue exploring the sub-question** in alignment with the **Original Research Question**.
    - If certain aspects have already been covered, focus on gaps, limitations, or complementary insights.
    """

    # Create system message config
    config = DocChatAgentConfig(
        use_functions_api=True,
        use_tools=True,
        llm=llm_config,
        system_message=system_message,
    )

    # Create agent
    agent = ResearchAgent3(config)
    agent.enable_message(RelevantExtractsTool)
    agent.enable_message(RelevantSearchExtractsTool)

    # Create and return the task
    return Task(
        agent,
        name="ResearchAgent3",
        llm_delegate=True,
        single_round=True,  # One-time execution per call
        interactive=False,  # Prevents entering an interactive loop
        restart=restart,
    )


@app.command()
def main(
    original_query: str = typer.Argument(..., help="Original research question"),
    sub_question: str = typer.Argument(..., help="Assigned sub-question for this agent"),
    context_1: str = typer.Argument(..., help="Findings from ResearchAgent1"),
    context_2: str = typer.Argument(..., help="Findings from ResearchAgent2"),
) -> None:
    """
    Command-line function to run ResearchAgent3 via make_research_task_3().
    The user only needs to input the original query, sub-question, and previous research context.
    """

    task = make_research_task_3(original_query, sub_question, context_1, context_2, restart=False)
    response = task.run(sub_question)

if __name__ == "__main__":
    app()