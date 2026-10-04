"""
models.py — Pydantic data models (shapes/contracts) for the entire project.

WHY PYDANTIC:
  - Validates data automatically (wrong type = instant error with clear message)
  - Converts JSON → Python object (no manual dict parsing)
  - Converts Python object → JSON (for HTTP responses)
  - Auto-generates API documentation (FastAPI uses these for /docs)

RULE: Every piece of data that crosses a file boundary should have a model.
"""

from pydantic import BaseModel, HttpUrl
from typing import Optional, Literal, List


# =============================================================================
# STEP 1 — LLM Test Models
# Used by: app/api/llm_routes.py
# =============================================================================

class LLMTestRequest(BaseModel):
    """
    What the user sends when testing the LLM directly.

    Example JSON body:
        {
            "system_prompt": "You are a helpful assistant.",
            "user_message": "What is RAG?"
        }
    """
    system_prompt: str = "You are a helpful assistant."   # has a default value
    user_message: str                                      # required — no default


class LLMTestResponse(BaseModel):
    """
    What we send back to the user after the LLM responds.

    Example JSON response:
        {
            "model": "llama3.2:3b",
            "response": "RAG stands for Retrieval-Augmented Generation..."
        }
    """
    model: str
    response: str


# =============================================================================
# STEP 2+ — Content Generation Models
# Used by: app/agent/agent.py, app/api/generate_routes.py
# =============================================================================

class GenerateRequest(BaseModel):
    """
    The main request from the user to generate a social media post.

    input_type choices:
      "url"      → user provides a web URL to scrape
      "text"     → user pastes 2-3 paragraphs directly
      "topic"    → user provides a topic keyword/phrase
      "document" → user uploads a PDF/Word file (separate endpoint)

    platform choices:
      "linkedin"  → generate LinkedIn post only
      "instagram" → generate Instagram caption only
      "both"      → generate both

    Example:
        {
            "input_type": "url",
            "content": "https://ibm.com/think/topics/rag",
            "platform": "both"
        }
    """
    input_type: Literal["url", "text", "topic"]
    content: str
    platform: Literal["linkedin", "instagram", "both"] = "both"


class StructuredSummary(BaseModel):
    """
    Intermediate output produced by the Summarizer tool.
    This structured JSON feeds BOTH the Post Generator and the Media tool.

    Why structured? So each downstream tool gets exactly what it needs:
      - post_generator needs: key_points, hook, tone, hashtags, cta
      - media_tool (DALL-E) needs: topic, summary, tone
    """
    topic: str
    summary: str
    key_points: List[str]
    hook: str              # Opening attention-grabbing sentence
    tone: str              # e.g. "educational", "inspirational", "informational"
    target_audience: str
    hashtags: dict         # {"linkedin": ["#AI", "#RAG"], "instagram": ["#ai", ...]}
    cta: dict              # {"linkedin": "Share your thoughts!", "instagram": "Save this!"}
    emojis: List[str]      # Suggested emojis: ["🤔", "✅", "🚀"]


class GenerateResponse(BaseModel):
    """
    The final response returned to the user with the generated posts.

    Optional fields because user may pick only one platform.
    """
    input_type_detected: str
    platform: str
    structured_summary: StructuredSummary
    linkedin_post: Optional[str] = None     # None if platform = "instagram"
    instagram_post: Optional[str] = None    # None if platform = "linkedin"


# =============================================================================
# MEDIA MODELS (Step 4 — Media Tool)
# =============================================================================

# All available media formats
MEDIA_TYPE_LIST = [
    "INFOGRAPHIC",
    "THEME_IMAGE",
    "DIAGRAM",
    "INFLUENCER",
    "WORKFLOW_GIF",   # Phase 2 — HeyGen
    "SHORT_VIDEO",    # Phase 2 — HeyGen
    "INFO_GIF",       # Phase 2 — HeyGen
    "PDF_1PAGER",     # Treated as static image for now
]

# Only these types generate a static image now (rest are Phase 2)
STATIC_IMAGE_TYPES = ["INFOGRAPHIC", "THEME_IMAGE", "DIAGRAM", "INFLUENCER", "PDF_1PAGER"]
PHASE2_TYPES = ["WORKFLOW_GIF", "SHORT_VIDEO", "INFO_GIF"]

MediaTypeLiteral = Literal[
    "INFOGRAPHIC", "THEME_IMAGE", "DIAGRAM", "INFLUENCER",
    "WORKFLOW_GIF", "SHORT_VIDEO", "INFO_GIF", "PDF_1PAGER"
]


class MediaResult(BaseModel):
    """
    Result from the media_tool.generate() call.

    media_mode:   "manual" | "no_media" | "ai_smart"
    media_type:   Which format was chosen (by user or by AI)
    image_path:   Local disk path to saved image (None if no_media or Phase 2 type)
    image_url:    Public URL of image (None until publishing feature added)
    prompt_used:  The image prompt sent to the API (for debugging / transparency)
    is_phase2:    True if type is GIF/Video (not yet generated, coming in Phase 2)
    message:      Human-readable status message
    """
    media_mode: str
    media_type: Optional[str] = None
    image_path: Optional[str] = None
    image_url: Optional[str] = None
    prompt_used: Optional[str] = None
    is_phase2: bool = False
    message: str = ""

