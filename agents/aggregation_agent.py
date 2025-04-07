"""
Aggregation Agent

This module implements an AI agent responsible for aggregating data from multiple sources.
It processes and organizes collected data before passing it to the next stage.
"""

import typer
from dotenv import load_dotenv

import langroid.language_models as lm
from langroid.agent.task import Task
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig, ChatDocument
from langroid.utils.configuration import set_global, Settings
from langroid.agent.tool_message import ToolMessage

app = typer.Typer()

class GenerateReportTool(ToolMessage):
    """
    Tool for generating the final aggregated research report.
    """
    request = "generate_report_tool"
    purpose = "Aggregate research agent outputs into a final structured report."
    original_query: str
    sub_questions: list[str]
    agent_outputs: list[str]

    @classmethod
    def examples(cls) -> list["ToolMessage"]:
        return [cls(
            original_query="Why is deepseek cheap?",
            sub_questions=[
                "What factors contribute to the cost structure of DeepSeek's services or products?",
                "How does DeepSeek's pricing compare to its competitors in the market?",
                "Are there any specific business strategies or technologies that allow DeepSeek to reduce costs and offer lower prices?"
            ],
            agent_outputs=[
                "Output from Research Agent 1",
                "Output from Research Agent 2",
                "Output from Research Agent 3"
            ]
        )]

    @classmethod
    def instructions(cls) -> str:
        return (
            "Provide the original research question, a list of sub-questions, and the corresponding outputs from the research agents. "
            "The tool should then generate a well-structured, gallery-style research report with clickable markdown links for each source."
        )

class AggregationAgent(ChatAgent):
    """
    AggregationAgent is responsible for synthesizing the outputs of all research agents
    to generate a final structured report.
    """
    def __init__(self, model: str, sys_msg: str):
        # Define LLM configuration
        llm_config = lm.OpenAIGPTConfig(
            chat_model=model or lm.OpenAIChatModel.GPT4o
        )
        # Define system message
        config = ChatAgentConfig(
            system_message=sys_msg,
            llm=llm_config,
        )
        super().__init__(config)
        # enable the tool
        self.enable_message(GenerateReportTool)

    def generate_report(self, original_query: str, sub_questions: list[str], agent_outputs: list[str]) -> str:
        """
        Uses LLM to synthesize a structured report from the research agent outputs.
        """
        combined_text = ""
        for i, output_str in enumerate(agent_outputs, start=1):
            combined_text += f"\n=== Sub-Question {i} Findings ===\n"
            if not output_str.strip():
                combined_text += "(No valid or empty result)\n"
            else:
                combined_text += output_str.strip() + "\n"


        prompt = f"""
        You are an expert research synthesizer. Your task is to create a clear, coherent, and insightful research report based on the findings provided below. Your report should integrate the insights from the different sub-questions into a unified narrative and must include URL citations for each key insight. Each citation should be formatted as a clickable markdown link (e.g. [Source Title](https://example.com)). Use concise and clear language.

        Original Research Question: "{original_query}"

        Sub-questions:
        1) {sub_questions[0]}
        2) {sub_questions[1]}
        3) {sub_questions[2]}

        Combined Findings (content and source URLs):
        {combined_text}

        Please generate a final research report that:
        - Summarizes the key insights from the findings.
        - Clearly cites the source URLs for each insight (in clickable markdown link format).
        - Provides a coherent conclusion that answers the original research question.
        """

        response = self.llm_response(prompt)
        return response.content if response else "No valid report generated."

    def generate_report_tool(self, msg: GenerateReportTool) -> str:
        """
        Tool method to generate report from LLM invocation.
        """
        return self.generate_report(msg.original_query, msg.sub_questions, msg.agent_outputs)

def make_aggregation_task(
    model: str = "",
    sys_msg: str = """
    You are an expert research assistant tasked with aggregating and synthesizing research findings.
    Your job is to take multiple research agent outputs and generate a well-structured, insightful
    final report on the topic.
    """,
    restart: bool = False
) -> Task:
    """
    Creates and configures an AggregationAgent task for synthesizing research findings.

    Args:
        model (str): The LLM model to use (default is GPT-4o).
        sys_msg (str): The system message for guiding the agent's behavior.
        restart (bool): Whether to restart the task each time it's run.

    Returns:
        Task: Configured Task instance for AggregationAgent.
    """
    load_dotenv()
    set_global(Settings(debug=False, cache=True))
    agent = AggregationAgent(model, sys_msg)
    return Task(
        agent,
        name="AggregationAgent",
        llm_delegate=True,
        single_round=True,
        interactive=False,
        restart=restart,
    )

@app.command()
def main(
    original_query: str = typer.Argument(..., help="Original research question"),
    sub_q1: str = typer.Argument(..., help="First sub-question"),
    sub_q2: str = typer.Argument(..., help="Second sub-question"),
    sub_q3: str = typer.Argument(..., help="Third sub-question"),
    output1: str = typer.Argument(..., help="Output from research agent 1"),
    output2: str = typer.Argument(..., help="Output from research agent 2"),
    output3: str = typer.Argument(..., help="Output from research agent 3"),
) -> None:
    """
    Command-line function to run the AggregationAgent via make_aggregation_task().
    The user only needs to input the original query, sub-questions, and agent outputs.
    """
    task = make_aggregation_task(restart=False)
    sub_questions = [sub_q1, sub_q2, sub_q3]
    agent_outputs = [output1, output2, output3]
    final_report = task.run((original_query, sub_questions, agent_outputs))
    print(final_report)

if __name__ == "__main__":
    app()