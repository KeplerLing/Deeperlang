"""
Research Agent Template

A base template for creating research agents. This module provides reusable functions and structure for building new research agents.

Key Components:
- Defines a base class or functions for research agents.
- Implements shared utilities that can be reused across different research agents.
"""


from langroid.agent.special.doc_chat_agent import DocChatAgent
from langroid.agent.tool_message import ToolMessage
from langroid.agent.chat_agent import ChatAgent, ChatDocument
from langroid.parsing.web_search import metaphor_search
from langroid.vector_store.qdrantdb import QdrantDB, QdrantDBConfig
from dotenv import load_dotenv


from rich import print
from typing import List, Any

load_dotenv()

class RelevantExtractsTool(ToolMessage):
    """Tool for retrieving relevant extracts from the vector database."""
    request = "relevant_extracts"
    purpose = "Retrieve relevant document extracts from the vector database."
    query: str

    @classmethod
    def examples(cls) -> List["ToolMessage"]:
        return [cls(query="What are the challenges of interpretability in LLMs?")]

    @classmethod
    def instructions(cls) -> str:
        return "IMPORTANT: Include an actual query in the `query` field."


class RelevantSearchExtractsTool(ToolMessage):
    """Tool for retrieving relevant extracts from web search."""
    request = "relevant_search_extracts"
    purpose = "Retrieve relevant document extracts from a web search."
    query: str
    num_results: int = 3

    @classmethod
    def examples(cls) -> List["ToolMessage"]:
        return [cls(query="What are the challenges of interpretability in LLMs?", num_results=3)]

    @classmethod
    def instructions(cls) -> str:
        return "IMPORTANT: Include an actual query in the `query` field."


class ResearchAgentTemplate(DocChatAgent):
    """Base template for Research Agents."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        print("[INFO] Connecting to Qdrant Cloud vector database...")
        qdrant_config = QdrantDBConfig(
            collection_name="qdrantdb",
            cloud=True,  
            replace_collection=False
        )

        self.vecdb = QdrantDB(qdrant_config)  # Each agent will connect to Qdrant independently**

    def llm_response(self, message: None | str | ChatDocument = None) -> ChatDocument | None:
        """
        Override default LLM response method.
        If vector DB and web search fail, fallback to LLM generation.
        """
        print(f"[yellow] Debug: LLM received message: {message} [/yellow]")
        response = ChatAgent.llm_response(self, message)
        print(f"[yellow] Debug: LLM generated response: {response} [/yellow]")
        return response

    def relevant_extracts(self, msg: RelevantExtractsTool) -> str:
        """Retrieve relevant extracts from the vector database."""

        # Debugging: Print vector database type and config
        print(f"[cyan] Debug: Using Vector DB of type {type(self.vecdb)} [/cyan]")
        if self.vecdb and hasattr(self.vecdb, "config"):
            print(f"[cyan] Debug: Vector DB Config -> {self.vecdb.config} [/cyan]")

        print(f"[yellow] Checking vector DB for: {msg.query} [/yellow]")
        _, extracts = self.get_relevant_extracts(msg.query)

        if not extracts:
            print("[yellow] No extracts found in vector DB. Switching to web search... [/yellow]")
            return self.relevant_search_extracts(RelevantSearchExtractsTool(query=msg.query, num_results=3))

        print("[green] Found extracts in vector DB! [/green]")

        result_str = "=== Relevant Extracts ===\n"
        for i, e in enumerate(extracts, start=1):
            content = e.content.strip()
            url = e.metadata.source if e.metadata and e.metadata.source else ""
            result_str += f"[{i}] CONTENT:\n{content}\nURL: {url}\n\n"
        return result_str

    def relevant_search_extracts(self, msg: RelevantSearchExtractsTool) -> str:
        """Fetch relevant extracts from a web search using custom scraper."""
        print(f"[blue]Performing web search for: {msg.query}[/blue]")

        # Search by Metaphor
        results = metaphor_search(msg.query, num_results=msg.num_results)

        if not results:
            print("[red] Web search returned no results! [/red]")
            return "No results found from web search."

        # Extract URL
        links = [r.link for r in results if hasattr(r, "link")]
        print(f"[green] Found {len(links)} results. Scraping full pages... [/green]")

        # Web Scraping
        self.ingest_doc_paths(links)

        # Retrieve relevant extracts from the vector database
        _, extracts = self.get_relevant_extracts(msg.query)


        result_str = "=== Web Search Extracts ===\n"
        for i, e in enumerate(extracts, start=1):
            content = e.content.strip()
            url = e.metadata.source if e.metadata and e.metadata.source else ""
            result_str += f"[{i}] CONTENT:\n{content}\nURL: {url}\n\n"
        return result_str