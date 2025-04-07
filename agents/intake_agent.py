"""
User Intake Agent

This module handles the initial processing of user input.
It breaks down user queries into structured requests for further processing.

Key Components:
- Imports necessary dependencies.
- Defines an agent class responsible for handling user input.
- Implements preprocessing and query breakdown methods.
"""

import typer
from dotenv import load_dotenv

import langroid.language_models as lm
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.utils.configuration import set_global, Settings

app = typer.Typer()

class IntakeAgent(ChatAgent):
    """
    IntakeAgent is responsible for breaking down a research question into smaller sub-questions.
    """

    def __init__(self, model: str, sys_msg: str):
        # Define LLM configuration
        llm_config = lm.OpenAIGPTConfig(
            chat_model=model or lm.OpenAIChatModel.GPT4o
        )

        # Define system message
        config = ChatAgentConfig(
            system_message=sys_msg,
            llm=llm_config
        )

        super().__init__(config)

    def decompose_query(self, query: str) -> list[str]:
        """
        Uses LLM to break down the user's research question into sub-questions.

        Args:
            query (str): The user's original research question.

        Returns:
            list[str]: A list of sub-questions.
        """
        prompt = f"""
        Given the following research question, generate at most 3 sub-questions 
        that will help explore the topic systematically.

        Research Question: "{query}"

        Respond ONLY with a list of sub-questions, nothing else.
        """

        response = self.llm_response(prompt)
        sub_questions = response.content.split("\n") if response else []

        return sub_questions  # No explicit print, can be printed manually if needed

def make_intake_task(
    model: str = "", 
    sys_msg: str = """
    You are an assistant that helps break down a research question into smaller sub-questions.
    Given a user's query, your job is to generate **at most 3 sub-questions** that can help explore the topic step by step.
    Respond ONLY with a list of sub-questions, nothing else.
    """,
    restart: bool = True
) -> Task:
    """
    Creates and configures an IntakeAgent task for breaking down research questions.
    
    Args:
        model (str): The LLM model to use (default is GPT-4o).
        sys_msg (str): The system message for guiding the agent's behavior.
        restart (bool): Whether to restart the task each time it's run.

    Returns:
        Task: Configured Task instance for IntakeAgent.
    """
    # Load environment variables
    load_dotenv()
    
    # Set global settings
    set_global(Settings(debug=False, cache=True))

    # Create IntakeAgent instance
    agent = IntakeAgent(model, sys_msg)

    # Create and return the task
    return Task(
        agent,
        name="IntakeAgent",
        llm_delegate=True,
        single_round=True,  # only decomposing one question per call
        interactive=False,
        restart=restart,
    )

@app.command()
def main(
    query: str = typer.Argument(..., help="User research question"),
) -> None:
    """
    Command-line function to run the IntakeAgent via make_intake_task().
    The user only needs to input a query, while model and system message are set to defaults.
    """

    task = make_intake_task(restart=False)

    sub_questions = task.run(query)


if __name__ == "__main__":
    app()