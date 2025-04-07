"""
Main Entry Point

This script serves as the entry point for the backend system. It initializes
necessary components and runs the agents in sequence, integrating UI elements
like a progress bar with Chainlit.
"""

import asyncio
import logging
import chainlit as cl
import json

# Import the agent creation functions
from agents.intake_agent import make_intake_task
from agents.research_agent_1 import make_research_task_1
from agents.research_agent_2 import make_research_task_2
from agents.research_agent_3 import make_research_task_3
from agents.aggregation_agent import make_aggregation_task
from agents.critic_agent import make_critic_task

logging.basicConfig(level=logging.WARNING)


async def update_message_content(msg: cl.Message, new_content: str):
    msg.content = new_content
    await msg.update()


def progress_bar_string(current: int, total: int, bar_length: int = 20):
    filled = int((current / total) * bar_length)
    bar = "■" * filled + "□" * (bar_length - filled)
    return f"[{bar}]"


def get_stage_label(round_num: int, max_rounds: int):
    percentage = (round_num / max_rounds) * 100
    if percentage < 33:
        return "Initial Research"
    elif percentage < 66:
        return "Intermediate Research"
    else:
        return "Finalizing"


async def simulate_progress(msg: cl.Message, round_num: int, max_rounds: int):
    bar = progress_bar_string(round_num, max_rounds)
    percentage = min(int((round_num / max_rounds) * 100), 100)
    stage = get_stage_label(round_num, max_rounds)
    await update_message_content(
        msg,
        f"**Research Pipeline Progress**\n\n{bar} {percentage}% – {stage}"
    )


async def research_pipeline(original_query: str, max_rounds: int = 3):
    progress_msg = cl.Message(
        content=f"**Starting Research Pipeline for:** {original_query}\n\n⏳"
    )
    await progress_msg.send()

    # Step 1: Intake Agent - Decompose query into sub-questions
    intake_task = make_intake_task(restart=False)
    sub_questions_doc = await asyncio.to_thread(intake_task.run, original_query)
    sub_questions = sub_questions_doc.content.split("\n") if sub_questions_doc else []
    sub_questions = [q.strip() for q in sub_questions if q.strip()]

    if len(sub_questions) < 3:
        await update_message_content(
            progress_msg,
            "❌ **Error:** Intake Agent did not generate enough sub-questions. Exiting."
        )
        return

    # Initialize with the first three sub-questions
    current_sub_questions = sub_questions[:3]
    final_report = None

    # Loop for up to max_rounds rounds
    for round_num in range(1, max_rounds + 1):
        await simulate_progress(progress_msg, round_num, max_rounds)
        await update_message_content(
            progress_msg,
            f"**Round {round_num} Processing**\nCurrent Sub-Questions: {current_sub_questions}"
        )

        # Step 2: Research Agents
        research_task_1 = make_research_task_1(original_query, current_sub_questions[0], restart=False)
        answer1 = await asyncio.to_thread(research_task_1.run, current_sub_questions[0])

        research_task_2 = make_research_task_2(original_query, current_sub_questions[1], answer1, restart=False)
        answer2 = await asyncio.to_thread(research_task_2.run, current_sub_questions[1])

        research_task_3 = make_research_task_3(original_query, current_sub_questions[2], answer1, answer2, restart=False)
        answer3 = await asyncio.to_thread(research_task_3.run, current_sub_questions[2])

        research_outputs = [answer1, answer2, answer3]

        # Step 3: Aggregation Agent - Generate aggregated report
        aggregation_task = make_aggregation_task(restart=False)
        agg_doc = await asyncio.to_thread(
            aggregation_task.run, (original_query, current_sub_questions, research_outputs)
        )
        report = agg_doc.content.strip() if agg_doc else "No valid report generated."

        await update_message_content(
            progress_msg,
            f"**Round {round_num} Aggregated Report**\n{report}"
        )

        # Step 4: Critic Agent - Evaluate aggregated report quality
        critic_task = make_critic_task(restart=False)
        evaluation = await asyncio.to_thread(
            critic_task.run, (original_query, report, current_sub_questions)
        )
        evaluation_str = evaluation.content if hasattr(evaluation, "content") else evaluation

        # Process the evaluation result
        if evaluation_str.strip() == "end":
            final_report = report
            await update_message_content(progress_msg, f"**Report Accepted at Round {round_num}**")
            break
        elif evaluation_str.strip().startswith("continue"):
            # Expect format: "continue" on first line, then newline, then JSON object.
            parts = evaluation_str.strip().split("\n", 1)
            if len(parts) > 1:
                try:
                    refined_obj = json.loads(parts[1].strip())
                    refined_list = refined_obj.get("refined_subquestions", [])
                    if isinstance(refined_list, list) and len(refined_list) > 0:
                        # If fewer than 3, pad with previous subquestions
                        if len(refined_list) < 3:
                            refined_list += current_sub_questions[len(refined_list):]
                        current_sub_questions = refined_list[:3]
                        await update_message_content(
                            progress_msg,
                            f"**Refined Sub-Questions Generated for Next Round**\n{current_sub_questions}"
                        )
                        continue  # Proceed to next round
                    else:
                        await update_message_content(
                            progress_msg,
                            "❌ **Error:** Refined subquestions not sufficient. Exiting."
                        )
                        break
                except json.JSONDecodeError:
                    await update_message_content(
                        progress_msg,
                        f"❌ **Error:** Failed to decode refined subquestions. Raw evaluation: {evaluation_str}"
                    )
                    break
            else:
                await update_message_content(
                    progress_msg,
                    "❌ **Error:** No refined subquestions provided after 'continue'. Exiting."
                )
                break
        else:
            await update_message_content(
                progress_msg,
                f"❌ **Error:** Unexpected evaluation result: {evaluation_str}"
            )
            break

    # If no round was accepted by critic, use the last generated report
    if final_report is None:
        final_report = report

    await update_message_content(
        progress_msg,
        f"**Research Complete ✅**\n\n**Final Aggregated Report for:** {original_query}\n\n{final_report}"
    )

    # ------------------ STORE FINAL REPORT ------------------
    from langroid.vector_store.qdrantdb import QdrantDB, QdrantDBConfig
    from langroid.agent.chat_document import ChatDocMetaData, ChatDocument
    from langroid.mytypes import Entity

    qcfg = QdrantDBConfig(collection_name="reports", cloud=True, replace_collection=False)
    db = QdrantDB(qcfg)
    doc = ChatDocument(
        content=final_report,
        metadata=ChatDocMetaData(source="", sender=Entity.USER)
    )
    db.add_documents([doc])
    print("[INFO] Final report stored in Qdrant 'reports' collection.")


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content=(
            "🚀 **Welcome to the Research Assistant!**\n\n"
            "Enter a research question, and I'll generate structured insights with verified sources."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    await research_pipeline(message.content)


if __name__ == "__main__":
    cl.run()