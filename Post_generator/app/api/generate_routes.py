"""
generate_routes.py — The final /api/generate endpoint (Step 4 — Agent).

THIS IS THE MAIN ENDPOINT users will call.
All previous steps (LLM, Tools, RAG) are integrated here through the agent.

ENDPOINTS:
  POST /api/generate              → text / url / topic input → agent → posts
  POST /api/generate/document     → file upload input → agent → posts

AGENT MODES (controlled by ?mode= query param):
  ?mode=pipeline  (default) → PipelineAgent: reliable, fast
  ?mode=llm                 → LLMAgent: autonomous tool selection by LLM

DIFFERENCE FROM tools_routes.py:
  tools_routes.py → tests individual tools directly (no agent)
  generate_routes.py → everything goes through the agent (production path)
"""

import os
import uuid
import shutil
from typing import Literal, Optional

from fastapi import APIRouter, UploadFile, File, Form, Query, HTTPException

from app.agent.agent import pipeline_agent, llm_agent
from app.core.config import settings

router = APIRouter(prefix="/generate", tags=["Step 4 - Agent"])

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc"}


# =============================================================================
# ENDPOINT 1: Generate from text / url / topic
# POST /api/generate
# =============================================================================
@router.get("", summary="Generate LinkedIn/Instagram post (text, URL, or topic)")
def generate(
    input_type: str = Query(..., description="url | text | topic"),
    content: str = Query(..., description="The URL, pasted text, or topic string"),
    platform: str = Query(default="both", description="linkedin | instagram | both"),
    mode: str = Query(default="pipeline", description="pipeline | llm"),
    media_mode: str = Query(default="no_media", description="no_media | ai_smart | manual"),
    media_type: Optional[str] = Query(default=None, description="INFOGRAPHIC | THEME_IMAGE | DIAGRAM | INFLUENCER | PDF_1PAGER | WORKFLOW_GIF | SHORT_VIDEO | INFO_GIF"),
):
    """
    Main generation endpoint. Accepts query parameters for easy testing.

    PIPELINE MODE (default):
      Agent uses rule-based routing → reliable, fast.
      LLM handles summarization and post generation.

    LLM MODE:
      Agent uses LangChain tool-calling loop.
      LLM autonomously decides which tools to call and in what order.
      Slower, but demonstrates true agentic behaviour.

    Examples:
      URL:    GET /api/generate?input_type=url&content=https://ibm.com/rag&platform=linkedin
      Text:   GET /api/generate?input_type=text&content=RAG is an AI technique...&platform=both
      Topic:  GET /api/generate?input_type=topic&content=LLM fine-tuning vs RAG&platform=instagram
    """
    # Validate inputs
    if input_type not in ("url", "text", "topic"):
        raise HTTPException(
            status_code=400,
            detail=f"input_type must be: url | text | topic  (for documents use /generate/document)"
        )
    if platform not in ("linkedin", "instagram", "both"):
        raise HTTPException(status_code=400, detail="platform must be: linkedin | instagram | both")

    # ── Choose agent mode ─────────────────────────────────────────────────────
    if mode == "llm":
        print(f"[ROUTE] Using LLM agent")
        result = llm_agent.run(input_type, content, platform)
    else:
        print(f"[ROUTE] Using Pipeline agent")
        result = pipeline_agent.run(
            input_type, content, platform,
            media_mode=media_mode,
            media_type=media_type,
        )

    # ── Handle errors from agent ──────────────────────────────────────────────
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    return result


# =============================================================================
# ENDPOINT 2: Generate from uploaded document (RAG)
# POST /api/generate/document
# =============================================================================
@router.post("/document", summary="Generate posts from uploaded PDF or DOCX")
async def generate_from_document(
    file: UploadFile = File(...),
    platform: str = Form(default="both"),
    mode: str = Form(default="pipeline"),
):
    """
    Uploads a PDF or DOCX file, runs RAG pipeline, generates posts.

    Steps:
      1. Save uploaded file to uploads/
      2. Agent calls rag_tool.process(file_path) → extracts relevant text
      3. Agent calls summarizer → structured summary
      4. Agent calls post_generator → LinkedIn + Instagram posts
      5. Clean up the uploaded file

    Test with curl:
        curl -X POST http://localhost:8000/api/generate/document
             -F "file=@article.pdf"
             -F "platform=both"
             -F "mode=pipeline"
    """
    if platform not in ("linkedin", "instagram", "both"):
        raise HTTPException(status_code=400, detail="platform must be: linkedin | instagram | both")

    # ── Validate file type ────────────────────────────────────────────────────
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Use: {ALLOWED_EXTENSIONS}"
        )

    # ── Save to disk ──────────────────────────────────────────────────────────
    unique_name = f"{uuid.uuid4()}{ext}"
    file_path = os.path.join(settings.UPLOAD_DIR, unique_name)

    with open(file_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)
    print(f"[ROUTE] Saved upload: {unique_name}")

    try:
        # ── Run agent with document input ─────────────────────────────────────
        result = pipeline_agent.run(
            input_type="document",
            content=file.filename,
            platform=platform,
            file_path=file_path,
        )

        if "error" in result:
            raise HTTPException(status_code=422, detail=result["error"])

        result["original_filename"] = file.filename
        return result

    finally:
        # ── Always clean up the uploaded file ─────────────────────────────────
        try:
            os.remove(file_path)
        except Exception:
            pass
