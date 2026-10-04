"""
media_routes.py — REST endpoints to test the Media Tool via Swagger UI.

WHY A SEPARATE ROUTE FILE:
  The media tool is embedded inside the full pipeline (/api/generate).
  These dedicated endpoints let you test media generation in isolation —
  without needing to run the entire content pipeline.

ENDPOINTS:
  GET  /api/media/types          → List all media types with descriptions
  POST /api/media/generate       → Generate image from custom prompt (raw test)
  POST /api/media/from-summary   → Generate image from structured summary JSON
  POST /api/media/smart          → Give a topic → LLM decides type → generates image

HOW TO TEST IN SWAGGER:
  1. Start server: uvicorn main:app --reload
  2. Open: http://localhost:8000/docs
  3. Find "Media Tool" section
  4. Click any endpoint → "Try it out" → fill fields → "Execute"
"""

import os
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.tools.media_tool import media_tool
from app.tools.summarizer import summarizer
from app.schemas.models import (
    StructuredSummary,
    MediaResult,
    MEDIA_TYPE_LIST,
    STATIC_IMAGE_TYPES,
    PHASE2_TYPES,
)

router = APIRouter(prefix="/media", tags=["Media Tool"])


# =============================================================================
# HELPER SCHEMAS — request bodies for Swagger UI
# =============================================================================

class MediaGenerateRequest(BaseModel):
    """Request to generate an image from a custom text prompt."""
    prompt: str = (
        "Minimalist professional infographic showing the key benefits of "
        "Retrieval-Augmented Generation in AI systems, flat design, white background"
    )
    media_type: str = "INFOGRAPHIC"

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Clean technical diagram showing RAG pipeline: PDF → chunks → embeddings → retrieval → LLM → answer",
                "media_type": "DIAGRAM",
            }
        }


class MediaFromTopicRequest(BaseModel):
    """Give a topic → summarizer builds StructuredSummary → AI picks type → generates image."""
    topic: str = "Why RAG is better than fine-tuning for enterprise AI"
    media_mode: str = "ai_smart"   # "ai_smart" | "manual"
    media_type: Optional[str] = None  # required only when media_mode == "manual"

    class Config:
        json_schema_extra = {
            "example": {
                "topic": "The future of agentic AI systems in 2025",
                "media_mode": "ai_smart",
                "media_type": None,
            }
        }


class MediaTypeInfo(BaseModel):
    """Info about one media type."""
    name: str
    description: str
    available_now: bool
    phase: str


# =============================================================================
# ENDPOINT 1: List all media types
# GET /api/media/types
# =============================================================================
@router.get("/types", summary="List all media types with descriptions")
def list_media_types() -> dict:
    """
    Returns all 8 media types with descriptions of when to use each.

    Use this to understand what each type produces before testing generation.

    No API key needed — this is purely informational.
    """
    type_info = {
        "INFOGRAPHIC": {
            "description": "Data-rich visual. Best for: stats, comparisons, '5 reasons why' content.",
            "available_now": True, "phase": "Phase 1",
        },
        "THEME_IMAGE": {
            "description": "Inspirational banner. Best for: thought leadership, abstract concepts.",
            "available_now": True, "phase": "Phase 1",
        },
        "DIAGRAM": {
            "description": "Technical flow/architecture. Best for: processes, how-things-work content.",
            "available_now": True, "phase": "Phase 1",
        },
        "INFLUENCER": {
            "description": "Professional portrait style. Best for: personal brand, leadership posts.",
            "available_now": True, "phase": "Phase 1",
        },
        "PDF_1PAGER": {
            "description": "Document layout style. Best for: guides, frameworks, step-by-step content.",
            "available_now": True, "phase": "Phase 1",
        },
        "WORKFLOW_GIF": {
            "description": "Animated workflow. Best for: process demonstrations. (HeyGen — Phase 2)",
            "available_now": False, "phase": "Phase 2 (HeyGen)",
        },
        "SHORT_VIDEO": {
            "description": "15-30s video. Best for: product demos, tutorials. (HeyGen — Phase 2)",
            "available_now": False, "phase": "Phase 2 (HeyGen)",
        },
        "INFO_GIF": {
            "description": "Animated infographic. Best for: quick data stories. (HeyGen — Phase 2)",
            "available_now": False, "phase": "Phase 2 (HeyGen)",
        },
    }

    return {
        "total_types": len(MEDIA_TYPE_LIST),
        "available_now": STATIC_IMAGE_TYPES,
        "phase_2": PHASE2_TYPES,
        "api_model": "NVIDIA NIM (swap to DALL-E later)",
        "types": type_info,
    }


# =============================================================================
# ENDPOINT 2: Generate image from custom prompt (raw API test)
# POST /api/media/generate
# =============================================================================
@router.post("/generate", summary="Generate image from a custom prompt (raw test)")
def generate_from_prompt(body: MediaGenerateRequest) -> dict:
    """
    Directly calls the NVIDIA image API with your custom prompt.

    Use this to:
    - Test that the NVIDIA API key works
    - Experiment with different prompt styles
    - Test a specific media_type without running the full pipeline

    Returns:
    - image_path: path to saved PNG file on server
    - prompt_used: the exact prompt sent to NVIDIA
    - message: status message

    After success, use GET /api/media/view/{filename} to see the image.
    """
    if body.media_type not in MEDIA_TYPE_LIST:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid media_type. Choose from: {MEDIA_TYPE_LIST}"
        )

    if body.media_type in PHASE2_TYPES:
        return {
            "status": "phase_2",
            "media_type": body.media_type,
            "message": f"'{body.media_type}' is a Phase 2 type (HeyGen). Not yet available.",
            "available_types": STATIC_IMAGE_TYPES,
        }

    print(f"[MEDIA ROUTE] Direct prompt test | type={body.media_type}")

    # Build a mock StructuredSummary to trigger _generate_static_image
    # but override the prompt by calling _call_nvidia_api directly
    try:
        image_bytes = media_tool._call_nvidia_api(body.prompt)
        image_path  = media_tool._save_image(image_bytes, body.media_type)

        return {
            "status": "success",
            "media_type": body.media_type,
            "image_path": image_path,
            "image_filename": os.path.basename(image_path),
            "prompt_used": body.prompt,
            "message": "Image generated. Use GET /api/media/view/{filename} to preview.",
        }

    except Exception as e:
        raise HTTPException(status_code=502, detail=f"NVIDIA API error: {str(e)}")


# =============================================================================
# ENDPOINT 3: AI Smart — topic → summarize → pick type → generate
# POST /api/media/smart
# =============================================================================
@router.post("/smart", summary="Give topic → LLM picks media type → generates image")
def generate_smart(body: MediaFromTopicRequest) -> dict:
    """
    Full AI-driven media generation in isolation:

    1. Takes your topic as input
    2. Creates a StructuredSummary using the summarizer
    3. If ai_smart: LLM decides the best media type automatically
    4. Generates the image via NVIDIA API
    5. Returns the image path + what type the LLM chose + why

    This is the same flow as the full /api/generate pipeline,
    but ONLY the media step — no post generation.

    Great for testing: "What type does the LLM choose for this content?"
    """
    if body.media_mode not in ("ai_smart", "manual"):
        raise HTTPException(
            status_code=400,
            detail="media_mode must be 'ai_smart' or 'manual'"
        )
    if body.media_mode == "manual" and not body.media_type:
        raise HTTPException(
            status_code=400,
            detail="media_type is required when media_mode is 'manual'"
        )
    if body.media_mode == "manual" and body.media_type not in MEDIA_TYPE_LIST:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid media_type. Choose from: {MEDIA_TYPE_LIST}"
        )

    print(f"[MEDIA ROUTE] Smart test | topic='{body.topic[:60]}' | mode={body.media_mode}")

    # Step 1: Build StructuredSummary from topic text
    print(f"[MEDIA ROUTE] Summarizing topic...")
    summary = summarizer.summarize(body.topic)
    print(f"[MEDIA ROUTE] Summary done: topic='{summary.topic}'")

    # Step 2: Generate media
    result = media_tool.generate(
        summary=summary,
        media_mode=body.media_mode,
        media_type=body.media_type,
    )

    # Build response
    resp = result.model_dump()
    resp["summary_topic"] = summary.topic
    resp["summary_tone"]  = summary.tone

    if result.image_path:
        resp["image_filename"] = os.path.basename(result.image_path)
        resp["preview_url"] = f"/api/media/view/{os.path.basename(result.image_path)}"

    return resp


# =============================================================================
# ENDPOINT 4: View/download the generated image
# GET /api/media/view/{filename}
# =============================================================================
@router.get("/view/{filename}", summary="View or download a generated image")
def view_image(filename: str):
    """
    Serves a generated image file from the media_output/ directory.

    Usage:
      After generating, copy the 'image_filename' from the response.
      Then call: GET /api/media/view/{image_filename}
      Swagger will display the image inline.

    Example:
      GET /api/media/view/INFOGRAPHIC_abc123.png
    """
    from app.core.config import settings

    # Security: strip any path traversal attempts
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(settings.MEDIA_OUTPUT_DIR, safe_filename)

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail=f"Image '{safe_filename}' not found in media_output/. Generate one first."
        )

    return FileResponse(
        path=file_path,
        media_type="image/png",
        filename=safe_filename,
    )


# =============================================================================
# ENDPOINT 5: List all generated images
# GET /api/media/list
# =============================================================================
@router.get("/list", summary="List all generated images in media_output/")
def list_generated_images() -> dict:
    """
    Lists all images currently saved in the media_output/ directory.

    Shows filename, size, and a preview URL for each.
    Use this to find filenames to pass to GET /api/media/view/{filename}.
    """
    from app.core.config import settings

    if not os.path.exists(settings.MEDIA_OUTPUT_DIR):
        return {"count": 0, "images": []}

    images = []
    for fname in os.listdir(settings.MEDIA_OUTPUT_DIR):
        if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            fpath = os.path.join(settings.MEDIA_OUTPUT_DIR, fname)
            size_kb = round(os.path.getsize(fpath) / 1024, 1)
            images.append({
                "filename": fname,
                "size_kb": size_kb,
                "preview_url": f"/api/media/view/{fname}",
                "media_type": fname.split("_")[0] if "_" in fname else "unknown",
            })

    images.sort(key=lambda x: x["filename"], reverse=True)

    return {
        "count": len(images),
        "directory": settings.MEDIA_OUTPUT_DIR,
        "images": images,
    }
