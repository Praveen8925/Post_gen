"""
media_tool.py — Generates media (images) for social media posts
using Azure OpenAI GPT Image models.

SUPPORTED MODES:
  1. manual    → user selects media type
  2. no_media  → skip generation
  3. ai_smart  → LLM automatically selects best media type

FLOW:
  summary
    → decide_media_type()
    → build_image_prompt()
    → _call_image_api()
    → save image
    → return MediaResult

USED BY:
  - app/agent/agent.py
  - app/api/generate_routes.py
"""

import os
import uuid
import base64
import requests

from typing import Optional

from app.core.config import settings
from app.llm.llm_client import llm_client
from app.schemas.models import (
    StructuredSummary,
    MediaResult,
    STATIC_IMAGE_TYPES,
    PHASE2_TYPES,
    MEDIA_TYPE_LIST,
)

# =============================================================================
# STYLE PROMPTS
# =============================================================================

MEDIA_STYLE_PREFIXES = {
    "INFOGRAPHIC": (
        "Minimalist professional infographic, flat design, clean white background, "
        "pastel accent colors, geometric icons, data visualization elements, "
        "modern sans-serif typography, LinkedIn-ready dimensions, "
    ),

    "THEME_IMAGE": (
        "Professional LinkedIn banner, inspiring abstract background, "
        "subtle gradient from deep blue to purple, "
        "minimal text overlay space, premium corporate aesthetic, "
    ),

    "DIAGRAM": (
        "Clean technical architecture diagram, white background, "
        "connected rectangular nodes with arrows, "
        "color-coded sections, labeled components, "
        "professional business presentation style, "
    ),

    "INFLUENCER": (
        "Professional business portrait style, warm studio lighting, "
        "confident expression, business casual attire, "
        "clean blurred background, LinkedIn profile quality, "
    ),

    "PDF_1PAGER": (
        "Clean one-page document layout, professional report design, "
        "white background with blue accent header, "
        "section dividers, bullet points, modern business template style, "
    ),
}

NEGATIVE_PROMPT = (
    "blurry, low quality, distorted faces, watermark, "
    "pixelated, amateur, cluttered, messy"
)

# =============================================================================
# AI SMART PROMPT
# =============================================================================

MEDIA_DECISION_PROMPT = """
You are a social media visual content strategist.

Based on the content summary below,
choose the BEST media type.

MEDIA TYPES:
- INFOGRAPHIC
- THEME_IMAGE
- DIAGRAM
- INFLUENCER
- PDF_1PAGER

WHEN TO USE:
- INFOGRAPHIC → statistics, comparisons, lists
- THEME_IMAGE → thought leadership, abstract ideas
- DIAGRAM → technical workflows/processes
- INFLUENCER → personal branding/career advice
- PDF_1PAGER → guides/frameworks/tutorials

RULES:
- Respond with EXACTLY ONE WORD
- No punctuation
- No explanation

Valid examples:
INFOGRAPHIC
DIAGRAM
THEME_IMAGE
"""

# =============================================================================
# MEDIA TOOL
# =============================================================================


class MediaTool:
    """
    Main media generation class.
    """

    def __init__(self):
        os.makedirs(settings.MEDIA_OUTPUT_DIR, exist_ok=True)

    # =========================================================================
    # MAIN ENTRY
    # =========================================================================

    def generate(
        self,
        summary: StructuredSummary,
        media_mode: str,
        media_type: Optional[str] = None,
    ) -> MediaResult:

        # ─────────────────────────────────────────────────────────────────────
        # MODE: no_media
        # ─────────────────────────────────────────────────────────────────────

        if media_mode == "no_media":
            print("[MEDIA] no_media selected")

            return MediaResult(
                media_mode="no_media",
                message="Media generation skipped by user."
            )

        # ─────────────────────────────────────────────────────────────────────
        # MODE: manual
        # ─────────────────────────────────────────────────────────────────────

        if media_mode == "manual":

            if not media_type:
                return MediaResult(
                    media_mode="manual",
                    message="media_type required for manual mode."
                )

            if media_type not in MEDIA_TYPE_LIST:
                return MediaResult(
                    media_mode="manual",
                    message=f"Invalid media_type: {media_type}"
                )

            chosen_type = media_type

            print(f"[MEDIA] Manual mode → {chosen_type}")

        # ─────────────────────────────────────────────────────────────────────
        # MODE: ai_smart
        # ─────────────────────────────────────────────────────────────────────

        elif media_mode == "ai_smart":

            chosen_type = self.decide_media_type(summary)

            print(f"[MEDIA] AI Smart selected → {chosen_type}")

        else:
            return MediaResult(
                media_mode=media_mode,
                message=(
                    "Unknown media_mode. "
                    "Use: manual | no_media | ai_smart"
                )
            )

        # ─────────────────────────────────────────────────────────────────────
        # PHASE 2 TYPES
        # ─────────────────────────────────────────────────────────────────────

        if chosen_type in PHASE2_TYPES:

            return MediaResult(
                media_mode=media_mode,
                media_type=chosen_type,
                is_phase2=True,
                message=(
                    f"{chosen_type} requires video/GIF generation. "
                    "Coming in Phase 2."
                )
            )

        # ─────────────────────────────────────────────────────────────────────
        # GENERATE IMAGE
        # ─────────────────────────────────────────────────────────────────────

        return self._generate_static_image(
            summary=summary,
            media_mode=media_mode,
            media_type=chosen_type,
        )

    # =========================================================================
    # AI SMART MEDIA DECISION
    # =========================================================================

    def decide_media_type(self, summary: StructuredSummary) -> str:
        """
        Uses the LLM to choose the best media type.
        """

        user_message = (
            f"Topic: {summary.topic}\n"
            f"Summary: {summary.summary}\n"
            f"Tone: {summary.tone}\n"
            f"Key points: {'; '.join(summary.key_points[:3])}"
        )

        response = llm_client.chat(
            system_prompt=MEDIA_DECISION_PROMPT,
            user_message=user_message,
        )

        chosen = (
            response
            .strip()
            .upper()
            .replace('"', '')
            .replace("'", "")
            .split()[0]
        )

        if chosen not in MEDIA_TYPE_LIST:
            print(f"[MEDIA] Invalid LLM output: {chosen}")
            return "THEME_IMAGE"

        return chosen

    # =========================================================================
    # BUILD IMAGE PROMPT
    # =========================================================================

    def build_image_prompt(
        self,
        summary: StructuredSummary,
        media_type: str
    ) -> str:
        """
        Creates a rich image prompt for GPT Image.
        """

        style_prefix = MEDIA_STYLE_PREFIXES.get(
            media_type,
            MEDIA_STYLE_PREFIXES["THEME_IMAGE"]
        )

        content_desc = (
            f"Create a professional visual representing: {summary.topic}. "
            f"Key concepts: {', '.join(summary.key_points[:3])}. "
            f"Tone: {summary.tone}. "
            f"Target audience: {summary.target_audience}. "
        )

        full_prompt = (
            style_prefix
            + content_desc
            + f"Avoid: {NEGATIVE_PROMPT}"
        )

        print(f"[MEDIA] Prompt length: {len(full_prompt)}")

        return full_prompt

    # =========================================================================
    # AZURE OPENAI IMAGE API
    # =========================================================================

    def _call_image_api(self, prompt: str) -> bytes:
        """
        Calls Azure OpenAI GPT Image API.

        Returns:
            PNG image bytes
        """

        url = (
            f"{settings.AZURE_OPENAI_ENDPOINT}"
            f"/openai/deployments/"
            f"{settings.AZURE_OPENAI_IMAGE_DEPLOYMENT}"
            f"/images/generations"
            f"?api-version={settings.AZURE_OPENAI_API_VERSION}"
        )

        headers = {
            "Content-Type": "application/json",

            # Azure usually uses api-key
            "api-key": settings.AZURE_OPENAI_API_KEY,
        }

        payload = {
            "prompt": prompt,
            "size": "1024x1024",
            "quality": "medium",
            "background": "auto",
            "moderation": "auto",
            "output_compression": 100,
            "output_format": "png",
            "n": 1,
        }

        print(f"[MEDIA] Calling Azure OpenAI API")
        print(f"[MEDIA] URL: {url}")

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=120,
        )

        print(f"[MEDIA] Status: {response.status_code}")

        if response.status_code != 200:

            print("[MEDIA] ERROR RESPONSE:")
            print(response.text)

            raise ValueError(
                f"Azure API error {response.status_code}: "
                f"{response.text[:500]}"
            )

        data = response.json()

        if "data" not in data or not data["data"]:
            raise ValueError(
                f"No image data returned. "
                f"Response keys: {list(data.keys())}"
            )

        image_b64 = data["data"][0].get("b64_json")

        if not image_b64:
            raise ValueError("Missing b64_json in response")

        return base64.b64decode(image_b64)

    # =========================================================================
    # SAVE IMAGE
    # =========================================================================

    def _save_image(
        self,
        image_bytes: bytes,
        media_type: str
    ) -> str:
        """
        Saves image to media_output directory.
        """

        filename = f"{media_type}_{uuid.uuid4()}.png"

        file_path = os.path.join(
            settings.MEDIA_OUTPUT_DIR,
            filename
        )

        with open(file_path, "wb") as f:
            f.write(image_bytes)

        print(f"[MEDIA] Saved image → {file_path}")

        return file_path

    # =========================================================================
    # GENERATE STATIC IMAGE
    # =========================================================================

    def _generate_static_image(
        self,
        summary: StructuredSummary,
        media_mode: str,
        media_type: str,
    ) -> MediaResult:
        """
        Full pipeline:
          prompt → generate → save → return
        """

        prompt = self.build_image_prompt(
            summary=summary,
            media_type=media_type,
        )

        try:

            image_bytes = self._call_image_api(prompt)

            image_path = self._save_image(
                image_bytes=image_bytes,
                media_type=media_type,
            )

            return MediaResult(
                media_mode=media_mode,
                media_type=media_type,
                image_path=image_path,
                prompt_used=prompt,
                is_phase2=False,
                message=f"Successfully generated {media_type}",
            )

        except Exception as e:

            print(f"[MEDIA] ERROR: {e}")

            return MediaResult(
                media_mode=media_mode,
                media_type=media_type,
                prompt_used=prompt,
                is_phase2=False,
                message=f"Image generation failed: {str(e)}",
            )


# =============================================================================
# SINGLETON
# =============================================================================

media_tool = MediaTool()