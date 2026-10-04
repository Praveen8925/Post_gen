"""
tools.py — LangChain tool wrappers around our MCP tools.

WHY THIS FILE EXISTS:
  Our MCP tools (in app/mcp/server.py) are plain Python functions.
  LangChain agents need tools wrapped with the @tool decorator so that:
    1. Tool name + docstring → sent to LLM as "available tools" list
    2. LLM can decide to call them by name
    3. LangChain parses the LLM's tool-call response and executes the function
    4. Result is returned back to the LLM for the next step

  Think of it as: plain function → @tool → LLM-callable tool

HOW @tool WORKS:
  @tool
  def my_tool(input: str) -> str:
      '''Description the LLM reads to understand WHEN to use this tool.'''
      return do_something(input)

  → LangChain reads the function name as tool name
  → LangChain reads the docstring as the tool description
  → LangChain reads the type hints as the input schema

USED BY: app/agent/agent.py
"""

from langchain_core.tools import tool

# Import the actual implementation from MCP/tool modules
from app.tools.web_scraper import web_scraper
from app.tools.summarizer import summarizer
from app.tools.web_search import web_search
from app.tools.post_generator import post_generator
from app.tools.rag_tool import rag_tool
from app.schemas.models import StructuredSummary
import json


# =============================================================================
# TOOL 1: Web Scraper
# =============================================================================
@tool
def scrape_url(url: str) -> str:
    """
    Scrapes a web URL and returns the extracted plain text content.
    Use this tool when the user input is a URL (web link to an article).
    Do NOT use for pasted text, topics, or uploaded documents.
    Returns extracted text ready to be passed to summarize_content().
    """
    return web_scraper.scrape(url)


# =============================================================================
# TOOL 2: Web Search
# =============================================================================
@tool
def search_web(topic: str) -> str:
    """
    Searches the web for a topic and returns aggregated content from results.
    Use this tool when the user provides a topic keyword or phrase instead of an article.
    Example topics: 'machine learning trends 2025', 'why RAG beats fine-tuning'.
    Returns search result text ready to be passed to summarize_content().
    """
    return web_search.search(topic)


# =============================================================================
# TOOL 3: Summarizer
# =============================================================================
@tool
def summarize_content(raw_text: str) -> str:
    """
    Analyzes raw text and extracts structured content for social media posts.
    Always call this after getting raw text (from scraping, search, or paste).
    Returns a JSON string with: topic, summary, key_points, hook, tone,
    target_audience, hashtags (linkedin + instagram), cta, and emojis.
    Pass this JSON to generate_post() as the next step.
    """
    summary_obj = summarizer.summarize(raw_text)
    return summary_obj.model_dump_json()


# =============================================================================
# TOOL 4: Post Generator
# =============================================================================
@tool
def generate_post(summary_json: str, platform: str) -> str:
    """
    Generates platform-specific social media posts from structured summary data.
    Always call summarize_content() first to get summary_json.
    platform must be one of: 'linkedin', 'instagram', or 'both'.
    Returns a JSON string with linkedin_post and/or instagram_post fields.
    """
    summary_dict = json.loads(summary_json)
    summary_obj = StructuredSummary(**summary_dict)
    result = post_generator.generate(summary_obj, platform)
    return json.dumps(result)


# =============================================================================
# TOOL 5: Document RAG
# =============================================================================
@tool
def process_document(file_path: str) -> str:
    """
    Extracts and retrieves relevant content from an uploaded PDF or DOCX file.
    Use this tool when the user uploads a document file.
    Uses RAG (embedding + cosine similarity) for large documents.
    Returns relevant text ready to be passed to summarize_content().
    """
    return rag_tool.process(file_path)


# ── Tool registry — all tools as a list (used by AgentExecutor) ───────────────
ALL_TOOLS = [
    scrape_url,
    search_web,
    summarize_content,
    generate_post,
    process_document,
]

# Tool name → function map (for rule-based lookup if needed)
TOOL_MAP = {t.name: t for t in ALL_TOOLS}
