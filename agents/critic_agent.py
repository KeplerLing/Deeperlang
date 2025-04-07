"""
Critic Agent

This module defines a CriticAgent that evaluates an aggregated report along with the original query and subquestions.
If the report quality is acceptable, it returns a signal to end the loop.
If the quality is not acceptable, it returns a signal to continue research along with refined subquestions.
"""

import json
import typer
from dotenv import load_dotenv

import langroid.language_models as lm
from langroid.agent.task import Task
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.utils.configuration import set_global, Settings
from langroid.agent.tool_message import ToolMessage

app = typer.Typer()

class EvaluateReportQualityTool(ToolMessage):
    """
    Tool for evaluating the quality of an aggregated report.
    
    It receives:
    - original_query: the original research question
    - report: the aggregated report content
    - subquestions: a list of subquestions used during the research
    """
    request = "evaluate_report_quality_tool"
    purpose = (
        "Evaluate the quality of an aggregated report based on strict criteria for relevance, clarity, completeness, and reliability. "
        "If the report is satisfactory, return ONLY the word 'end'. "
        "If unsatisfactory, return 'continue' followed by a newline and a JSON object with refined subquestions."
    )
    original_query: str
    report: str
    subquestions: list[str]

    @classmethod
    def examples(cls) -> list["ToolMessage"]:
        return [cls(
            original_query="How can we optimize machine learning algorithms?",
            report="The aggregated report offers some insights but lacks sufficient detail and clarity in addressing key challenges.",
            subquestions=[
                "What are the current limitations of machine learning algorithms?",
                "How can these limitations be addressed effectively?"
            ]
        )]

    @classmethod
    def instructions(cls) -> str:
        return (
            "You are a critical evaluator. Evaluate the aggregated report based on the following strict criteria:\n"
            "1. The report must be highly relevant to the original research question and extremely clear.\n"
            "2. It must provide detailed analysis with concrete examples and sufficient evidence, with a minimum of 500 words.\n"
            "3. It must include a clear structure with sections such as Introduction, Body, and Conclusion.\n"
            "4. It must include at least 2 distinct source citations formatted as clickable markdown links (e.g. [Source Title](https://example.com)).\n"
            "5. The narrative must be well-structured, logically coherent, and insightful.\n\n"
            "If the report meets ALL these criteria, reply with ONLY the word 'end'.\n"
            "If the report does not meet one or more of these criteria, reply with 'continue' on the first line, followed by a newline and a JSON object "
            "with a field \"refined_subquestions\" containing a list of at least 2 refined subquestions. For example:\n"
            "continue\n{\"refined_subquestions\": [\"refined subquestion 1\", \"refined subquestion 2\"]}\n"
            "Respond with nothing else."
        )

class CriticAgent(ChatAgent):
    """
    CriticAgent evaluates the aggregated report quality for a given research query.
    
    It uses the provided report, original_query, and subquestions to decide whether to end the research loop or to continue by refining subquestions.
    """
    def __init__(self, model: str, sys_msg: str):
        if hasattr(sys_msg, "content"):
            sys_msg = sys_msg.content
        elif not isinstance(sys_msg, str):
            sys_msg = str(sys_msg)

        llm_config = lm.OpenAIGPTConfig(
            chat_model=model or lm.OpenAIChatModel.GPT4o
        )
        config = ChatAgentConfig(
            system_message=sys_msg,
            llm=llm_config,
        )
        super().__init__(config)
        self.enable_message(EvaluateReportQualityTool)

    def evaluate_report_quality(self, original_query: str, report: str, subquestions: list[str]) -> str:
        """
        Evaluate the quality of the aggregated report based on strict criteria:
        - The report must be highly relevant and extremely clear.
        - It must provide detailed analysis with concrete examples and sufficient evidence (minimum 500 words).
        - It must include a clear structure (e.g., Introduction, Body, Conclusion).
        - It must include at least 2 distinct source citations, each formatted as a clickable markdown link (e.g. [Source Title](https://example.com)).
        - The narrative must be well-structured, logically coherent, and insightful.
        
        If all conditions are met, return "end".
        Otherwise, return "continue" on the first line, followed by a newline and a JSON object with refined subquestions.
        """
        formatted_subquestions = "\n".join(f"- {q}" for q in subquestions)
        prompt = f"""
You are an expert evaluator. Given the following:
- Original Research Question: {original_query}
- Aggregated Report: {report}
- Current Sub-Questions:
{formatted_subquestions}

Evaluate the quality of the aggregated report using these strict criteria:
1. The report must be highly relevant and extremely clear.
2. It must provide detailed analysis with concrete examples and sufficient evidence (minimum 500 words).
3. It must include a clear structure with sections (e.g., Introduction, Body, Conclusion).
4. It must include at least 2 distinct source citations, each formatted as a clickable markdown link (e.g. [Source Title](https://example.com)).
5. The narrative must be well-structured, logically coherent, and insightful.

If the report meets ALL these criteria, reply with ONLY the word "end".
If not, reply with "continue" on the first line, then a newline, followed by a JSON object with a field "refined_subquestions" that contains a list of at least 2 refined subquestions.
Respond with nothing else.
"""
        response = self.llm_response(prompt)
        answer = response.content.strip() if response else "No valid evaluation."
        return answer

    def evaluate_report_quality_tool(self, msg: EvaluateReportQualityTool) -> str:
        return self.evaluate_report_quality(msg.original_query, msg.report, msg.subquestions)

def make_critic_task(
    model: str = lm.OpenAIChatModel.GPT4o,
    sys_msg: str = """
You are a critical assistant evaluating an aggregated report for a research query.
Evaluate the report based on strict criteria for relevance, clarity, completeness, and reliability.
If the report meets all the criteria, reply with ONLY "end".
Otherwise, reply with "continue" on the first line, then a newline and a JSON object with refined subquestions.
""",
    restart: bool = False
) -> Task:
    load_dotenv()
    set_global(Settings(debug=False, cache=True))
    agent = CriticAgent(model, sys_msg)
    return Task(
        agent,
        name="CriticAgent",
        llm_delegate=True,
        single_round=True,
        interactive=False,
        restart=restart,
    )

@app.command()
def main(
    original_query: str = typer.Argument(..., help="Original research question"),
    report: str = typer.Argument(..., help="Aggregated report from research agents"),
    subquestion1: str = typer.Argument(..., help="Subquestion 1"),
    subquestion2: str = typer.Argument(..., help="Subquestion 2"),
    subquestion3: str = typer.Argument(..., help="Subquestion 3"),
) -> None:
    """
    Use command-line arguments to provide the original_query, report, and multiple subquestions.
    The CriticAgent evaluates the report quality and returns the evaluation result.
    """
    task = make_critic_task(restart=False)
    evaluation = task.run((
        original_query,
        report,
        [subquestion1, subquestion2, subquestion3],
    ))
    
    # Extract the content if evaluation is a ChatDocument-like object
    evaluation_str = evaluation.content if hasattr(evaluation, "content") else evaluation

    # Process the evaluation result
    if evaluation_str.strip() == "end":
        print("Evaluation Result:", evaluation_str.strip())
    elif evaluation_str.strip().startswith("continue"):
        # Remove "continue" prefix and extract JSON part after the first newline
        # Assume the output format is: "continue\n{...}"
        parts = evaluation_str.strip().split("\n", 1)
        if len(parts) > 1:
            json_part = parts[1].strip()
            try:
                result = json.loads(json_part)
                print("Evaluation Result: continue", result)
            except json.JSONDecodeError:
                print("Evaluation Result:", evaluation_str)
        else:
            print("Evaluation Result:", evaluation_str)
    else:
        print("Evaluation Result:", evaluation_str)

if __name__ == "__main__":
    app()