import asyncio
import logging
import chainlit as cl
import json
import re
from typing import Callable

from agents.intake_agent import make_intake_task
from agents.research_agent_1 import make_research_task_1
from agents.research_agent_2 import make_research_task_2
from agents.research_agent_3 import make_research_task_3
from agents.aggregation_agent import make_aggregation_task
from agents.classification_agent import make_classification_task

logging.basicConfig(level=logging.WARNING)


def remove_citations(text: str) -> str:
    return re.sub(r"\[\^(\d+)\]", "", text)


def progress_bar_string(percentage: int, bar_length: int = 20):
    filled = int((percentage / 100) * bar_length)
    bar = "█" * filled + "░" * (bar_length - filled)
    return f"`[{bar}]`"


def get_stage_label(percentage: int):
    if percentage < 33:
        return "🧠 Reasoning"
    elif percentage < 66:
        return "🔍 Searching"
    elif percentage < 99:
        return "🧷 Aggregating"
    else:
        return "📦 Finalizing"


async def update_message_content(msg: cl.Message, new_content: str):
    msg.content = new_content
    await msg.update()


async def fake_progress(msg: cl.Message, done_flag: asyncio.Event):
    percent_steps = [
        3, 7, 11, 15, 19, 23, 27, 31,
        35, 39, 43, 47, 51, 55, 59, 63,
        67, 71, 75, 79, 83, 87, 91, 94, 96, 98
    ]
    total_duration = 85  # total duration before 99%
    step_delay = total_duration / len(percent_steps)

    for pct in percent_steps:
        if done_flag.is_set():
            return
        bar = progress_bar_string(pct)
        stage = get_stage_label(pct)
        await update_message_content(
            msg,
            f"📊 **Research Progress**\n\n{bar} {pct}% – *{stage}*"
        )
        await asyncio.sleep(step_delay)

    # Final 99% pause
    if not done_flag.is_set():
        bar = progress_bar_string(99)
        await update_message_content(
            msg,
            f"📊 **Research Progress**\n\n{bar} 99% – *Finalizing*"
        )
        await asyncio.sleep(2)


async def research_pipeline(
    original_query: str,
    update_message_content_fn: Callable,
):
    progress_msg = cl.Message(content="🔍 Checking your database for previous reports...")
    await progress_msg.send()

    classification_task = make_classification_task(restart=False)
    classification_result = await asyncio.to_thread(classification_task.run, original_query)

    if classification_result and classification_result.content != "DO-NOT-KNOW":
        cleaned = remove_citations(classification_result.content)
        await update_message_content_fn(
            progress_msg,
            f"📄 **Found relevant insights:**\n\n{cleaned}"
        )
        return

    await update_message_content_fn(
        progress_msg,
        f"🚧 No existing report found.\n\n**Starting Research Pipeline for:** {original_query}\n\n⏳"
    )

    # ⏳ Launch fake progress
    progress_done = asyncio.Event()
    progress_task = asyncio.create_task(fake_progress(progress_msg, progress_done))

    intake_task = make_intake_task(restart=False)
    sub_questions_doc = await asyncio.to_thread(intake_task.run, original_query)
    sub_questions = sub_questions_doc.content.split("\n") if sub_questions_doc else []
    sub_questions = [q.strip() for q in sub_questions if q.strip()]

    if len(sub_questions) < 3:
        await update_message_content_fn(
            progress_msg,
            "❌ **Error:** Intake Agent did not generate enough sub-questions. Exiting."
        )
        return

    current_sub_questions = sub_questions[:3]

    # Simulate 1 research round
    research_task_1 = make_research_task_1(original_query, current_sub_questions[0], restart=False)
    answer1 = await asyncio.to_thread(research_task_1.run, current_sub_questions[0])

    research_task_2 = make_research_task_2(original_query, current_sub_questions[1], answer1, restart=False)
    answer2 = await asyncio.to_thread(research_task_2.run, current_sub_questions[1])

    research_task_3 = make_research_task_3(original_query, current_sub_questions[2], answer1, answer2, restart=False)
    answer3 = await asyncio.to_thread(research_task_3.run, current_sub_questions[2])

    research_outputs = [answer1, answer2, answer3]

    aggregation_task = make_aggregation_task(restart=False)
    agg_doc = await asyncio.to_thread(
        aggregation_task.run, (original_query, current_sub_questions, research_outputs)
    )
    final_report = agg_doc.content.strip() if agg_doc else "No valid report generated."

    # ✅ Stop the progress bar and reveal the final result
    progress_done.set()
    await progress_task

    await update_message_content_fn(
        progress_msg,
        f"{progress_bar_string(100)} 100% – *Done*\n\n🎉 **Research Complete!**\n\n**Final Report:**\n\n{final_report}"
    )

    # ✅ Store the final report
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
        content="🚀 **Welcome to the Research Assistant!**\n\nAsk me anything, and I’ll investigate it across the web and our knowledge base."
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    await research_pipeline(
        message.content,
        update_message_content_fn=update_message_content,
    )


if __name__ == "__main__":
    cl.run()