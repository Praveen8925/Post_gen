"""
tools_routes.py — REST endpoints to test each tool individually (Step 2).

PURPOSE:
  Before the agent exists, you need a way to test each tool in isolation.
  These endpoints let you call tools directly via Postman or /docs.

  Think of this as a "tool testing panel" — not used in production,
  but essential for verifying each tool works before connecting them.

ENDPOINTS:
  POST /api/tools/scrape       → test web scraper
  POST /api/tools/summarize    → test summarizer
  POST /api/tools/search       → test web search
  POST /api/tools/generate     → test post generator (needs summary first)
  POST /api/tools/full-pipeline → test entire chain end-to-end (no agent yet)
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Literal, Optional
import json

from app.tools.web_scraper import web_scraper
from app.tools.summarizer import summarizer
from app.tools.web_search import web_search
from app.tools.post_generator import post_generator
from app.schemas.models import StructuredSummary

router = APIRouter(prefix="/tools", tags=["Step 2 - Tools"])


# ── Request models (only for this file, quick inline definitions) ─────────────
class ScrapeRequest(BaseModel):
    url: str

class SummarizeRequest(BaseModel):
    text: str

class SearchRequest(BaseModel):
    topic: str

class GenerateRequest(BaseModel):
    # Accepts the full StructuredSummary as input
    summary: StructuredSummary
    platform: Literal["linkedin", "instagram", "both"] = "both"

class PipelineRequest(BaseModel):
    """Test the full chain: input → scrape/search → summarize → generate post"""
    input_type: Literal["url", "text", "topic"]
    content: str
    platform: Literal["linkedin", "instagram", "both"] = "both"


# =============================================================================
# ENDPOINT 1: Test Web Scraper
# POST /api/tools/scrape
# =============================================================================
@router.post("/scrape", summary="Test: Scrape a URL and return extracted text")
def test_scrape(request: ScrapeRequest):
    """
    Calls web_scraper.scrape(url) directly.
    Use this to verify scraping works before connecting the summarizer.

    Test body:
        { "url": "https://www.ibm.com/think/topics/retrieval-augmented-generation" }
    """
    text = web_scraper.scrape(request.url)
    return {
        "url": request.url,
        "char_count": len(text),
        "extracted_text": text,
    }


# =============================================================================
# ENDPOINT 2: Test Summarizer
# POST /api/tools/summarize
# =============================================================================
@router.post("/summarize", summary="Test: Summarize text into structured content data")
def test_summarize(request: SummarizeRequest):
    """
    Calls summarizer.summarize(text) directly.
    Returns the full StructuredSummary as JSON.

    Test body:
        { "text": "Paste any article text here..." }
    """
    summary = summarizer.summarize(request.text)
    # .model_dump() converts Pydantic object to plain dict (JSON-serializable)
    return {
        "status": "ok",
        "summary": summary.model_dump(),
    }


# =============================================================================
# ENDPOINT 3: Test Web Search
# POST /api/tools/search
# =============================================================================
@router.post("/search", summary="Test: Search the web for a topic")
def test_search(request: SearchRequest):
    """
    Calls web_search.search(topic) directly.
    Returns aggregated text from DuckDuckGo results.

    Test body:
        { "topic": "retrieval augmented generation enterprise AI 2025" }
    """
    text = web_search.search(request.topic)
    return {
        "topic": request.topic,
        "char_count": len(text),
        "search_results_text": text,
    }


# =============================================================================
# ENDPOINT 4: Test Post Generator
# POST /api/tools/generate
# =============================================================================
@router.post("/generate", summary="Test: Generate LinkedIn/Instagram posts from a summary")
def test_generate(request: GenerateRequest):
    """
    Calls post_generator.generate(summary, platform) directly.
    Requires a full StructuredSummary — use /tools/summarize first to get one.

    Test body:
        {
            "summary": { ...output from /tools/summarize... },
            "platform": "both"
        }
    """
    result = post_generator.generate(request.summary, request.platform)
    return {
        "platform": request.platform,
        "linkedin_post": result.get("linkedin_post"),
        "instagram_post": result.get("instagram_post"),
    }


# =============================================================================
# ENDPOINT 5: Full Pipeline Test (no agent)
# POST /api/tools/full-pipeline
# =============================================================================
@router.post("/full-pipeline", summary="Test: Run the full tool chain end-to-end")
def test_full_pipeline(request: PipelineRequest):
    """
    Runs the complete tool chain without the agent.
    This lets you verify the pipeline before building the agent.

    Flow:
      input_type == "url"   → scrape → summarize → generate
      input_type == "text"  → summarize → generate
      input_type == "topic" → web_search → summarize → generate

    Test body (URL):
        {
            "input_type": "url",
            "content": "https://www.ibm.com/think/topics/retrieval-augmented-generation",
            "platform": "linkedin"
        }

    Test body (Topic):
        {
            "input_type": "topic",
            "content": "why RAG is better than fine-tuning LLMs",
            "platform": "both"
        }

    Test body (Text):
        {
            "input_type": "text",
            "content": "Paste 2-3 paragraphs of article text here...",
            "platform": "instagram"
        }
    """
    # ── Step 1: Get raw text based on input_type ──────────────────────────
    if request.input_type == "url":
        raw_text = web_scraper.scrape(request.content)
        source = f"scraped from {request.content}"

    elif request.input_type == "text":
        raw_text = request.content
        source = "pasted text"

    elif request.input_type == "topic":
        raw_text = web_search.search(request.content)
        source = f"web search for '{request.content}'"

    # ── Step 2: Summarize ─────────────────────────────────────────────────
    summary = summarizer.summarize(raw_text)

    # ── Step 3: Generate posts ────────────────────────────────────────────
    posts = post_generator.generate(summary, request.platform)

    # ── Return everything ─────────────────────────────────────────────────
    return {
        "input_type": request.input_type,
        "source": source,
        "platform": request.platform,
        "structured_summary": summary.model_dump(),
        "linkedin_post": posts.get("linkedin_post"),
        "instagram_post": posts.get("instagram_post"),
    }
