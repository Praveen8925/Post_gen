"""
server.py — MCP (Model Context Protocol) server wrapping all tools.

WHAT IS MCP:
  MCP is a standard protocol (by Anthropic) that lets LLM agents discover
  and call tools in a structured way. Each tool has:
    - name:        unique identifier
    - description: tells the LLM WHEN and WHY to use it (critical!)
    - parameters:  what inputs the tool expects (with types)
    - return:      what the tool returns

WHY MCP OVER PLAIN FUNCTIONS:
  - Standard protocol → agent knows how to call tools automatically
  - Tool descriptions are part of the agent's context → better decisions
  - Easy to add/remove tools without changing agent code
  - Industry standard → works with Claude, OpenAI, LangChain agents

HOW IT FITS IN THE PIPELINE:
  Agent reads MCP tool descriptions
    → decides which tool to call (based on user input_type)
    → calls tool via MCP
    → gets result
    → passes to next tool in chain

TOOL CHAIN EXAMPLE (URL input):
  tool_scrape_url(url) → raw text
    → tool_summarize(raw_text) → structured JSON
      → tool_generate_post(summary_json, platform) → LinkedIn + Instagram post

USED BY: app/agent/agent.py
"""

import json
from mcp.server.fastmcp import FastMCP

# Import all tool singletons
from app.tools.web_scraper import web_scraper
from app.tools.summarizer import summarizer
from app.tools.web_search import web_search
from app.tools.post_generator import post_generator
from app.tools.rag_tool import rag_tool                 # Step 3
from app.schemas.models import StructuredSummary


# =============================================================================
# CREATE MCP SERVER INSTANCE
# "Content Generator Tools" = server name (shown in agent logs)
# =============================================================================
mcp_server = FastMCP("Content Generator Tools")


# =============================================================================
# TOOL 1: Scrape URL
# =============================================================================
@mcp_server.tool()
def tool_scrape_url(url: str) -> str:
    """
    Scrapes a web URL and returns the extracted plain text content.

    USE THIS TOOL WHEN: The user provides a URL as their input.
    DO NOT USE FOR: Pasted text, topics, or uploaded documents.

    Args:
        url: Full web URL to scrape (e.g. "https://ibm.com/think/topics/rag")

    Returns:
        Extracted plain text from the webpage (up to 8000 chars).
        Returns an error string if URL is unreachable.

    Next step: Pass the returned text to tool_summarize().
    """
    return web_scraper.scrape(url)


# =============================================================================
# TOOL 2: Summarize Text
# =============================================================================
@mcp_server.tool()
def tool_summarize(raw_text: str) -> str:
    """
    Analyzes raw text and extracts structured content for social media posts.
    Returns a JSON string with topic, key_points, hook, tone, hashtags, and CTA.

    USE THIS TOOL FOR ALL INPUT TYPES after getting raw text:
      - After tool_scrape_url() → pass the scraped text here
      - After tool_web_search() → pass the search results here
      - When user pastes text directly → pass it here immediately
      - After extracting text from a document → pass it here

    Args:
        raw_text: Any plain text content to analyze and structure.

    Returns:
        JSON string with keys: topic, summary, key_points, hook, tone,
        target_audience, hashtags, cta, emojis

    Next step: Pass the returned JSON to tool_generate_post().
    """
    # Call summarizer → get StructuredSummary Pydantic object
    summary_obj = summarizer.summarize(raw_text)
    # Convert Pydantic object to JSON string (for passing between tools)
    return summary_obj.model_dump_json()


# =============================================================================
# TOOL 3: Web Search
# =============================================================================
@mcp_server.tool()
def tool_web_search(topic: str) -> str:
    """
    Searches the web for a topic and returns aggregated content from results.

    USE THIS TOOL WHEN: The user provides a topic keyword instead of an article.
    Examples: "machine learning trends 2025", "why RAG matters for enterprise AI"

    Args:
        topic: Search query string (the topic the user wants to post about)

    Returns:
        Aggregated text from top 5 web search results.
        Returns error message if search fails (agent should fall back to LLM knowledge).

    Next step: Pass the returned text to tool_summarize().
    """
    return web_search.search(topic)


# =============================================================================
# TOOL 4: Generate Post
# =============================================================================
@mcp_server.tool()
def tool_generate_post(summary_json: str, platform: str) -> str:
    """
    Generates platform-specific social media posts from structured summary data.

    USE THIS TOOL WHEN: You have a structured summary and need the final post.
    Always call tool_summarize() first to get the summary_json.

    Args:
        summary_json: JSON string returned by tool_summarize()
        platform:     "linkedin" | "instagram" | "both"
                      - "linkedin"  → professional post with Unicode bold
                      - "instagram" → casual caption with emojis + hashtags
                      - "both"      → generates both

    Returns:
        JSON string with keys: linkedin_post (str or null), instagram_post (str or null)

    Example return:
        {
          "linkedin_post": "What if your AI always gave accurate answers?...",
          "instagram_post": "Stop relying on outdated AI!..."
        }
    """
    # Parse the summary JSON back into a dict
    summary_dict = json.loads(summary_json)

    # Rebuild StructuredSummary Pydantic object (for type safety)
    summary_obj = StructuredSummary(**summary_dict)

    # Generate the post(s)
    result = post_generator.generate(summary_obj, platform)

    # Return as JSON string
    return json.dumps(result)


# =============================================================================
# TOOL 5: Process Document (RAG)
# =============================================================================
@mcp_server.tool()
def tool_process_document(file_path: str) -> str:
    """
    Extracts and retrieves relevant content from an uploaded PDF or DOCX file.
    Uses RAG (chunking + embeddings + cosine similarity) for large documents.

    USE THIS TOOL WHEN: The user uploads a PDF or Word document.
    DO NOT USE FOR: URLs, pasted text, or topics.

    Args:
        file_path: Path to the saved upload file (e.g. "uploads/abc123.pdf")
                   This path is returned by the file upload handler.

    Returns:
        Extracted and relevant text from the document.
        For small docs: full text. For large docs: top-4 most relevant chunks.

    Next step: Pass the returned text to tool_summarize().
    """
    return rag_tool.process(file_path)
