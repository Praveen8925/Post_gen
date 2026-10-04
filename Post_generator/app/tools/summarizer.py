import json
import re
from typing import Optional

from app.llm.llm_client import llm_client        # LLM singleton
from app.schemas.models import StructuredSummary  # Output shape/contract


SUMMARIZER_SYSTEM_PROMPT = """You are an expert AI content analyst and social media strategist with 10+ years of experience creating viral LinkedIn and Instagram content for top tech companies.

Your task: Analyze the provided text and extract key information to create compelling social media content.

CRITICAL RULES:
1. RESPOND WITH VALID JSON ONLY — no markdown, no code blocks (```), no explanation text
2. Start your response with { and end with }
3. Extract REAL information from the text — do not hallucinate facts
4. Hashtags must be relevant to the actual content

OUTPUT THIS EXACT JSON STRUCTURE:
{
  "topic": "Short clear topic title (max 8 words)",
  "summary": "2-3 sentence summary capturing the core message",
  "key_points": [
    "Key insight or fact #1 (one sentence)",
    "Key insight or fact #2 (one sentence)",
    "Key insight or fact #3 (one sentence)",
    "Key insight or fact #4 (one sentence)"
  ],
  "hook": "One powerful attention-grabbing opening sentence (creates curiosity or shock)",
  "tone": "educational",
  "target_audience": "Who would most benefit from this content (e.g., AI developers, marketing teams)",
  "hashtags": {
    "linkedin": ["#Hashtag1", "#Hashtag2", "#Hashtag3", "#Hashtag4", "#Hashtag5"],
    "instagram": ["#hashtag1", "#hashtag2", "#hashtag3", "#hashtag4", "#hashtag5",
                  "#hashtag6", "#hashtag7", "#hashtag8", "#hashtag9", "#hashtag10",
                  "#hashtag11", "#hashtag12", "#hashtag13", "#hashtag14", "#hashtag15"]
  },
  "cta": {
    "linkedin": "A professional question or call-to-action (e.g., What do you think? Share below!)",
    "instagram": "An engaging Instagram CTA (e.g., Save this for later! Tag someone who needs to see this!)"
  },
  "emojis": ["🤔", "💡", "🚀", "✅"]
}

tone must be ONE of: educational, inspirational, informational, controversial, motivational"""


# =============================================================================
# FALLBACK — Used when LLM output cannot be parsed as JSON
# =============================================================================
def _build_fallback_summary(raw_text: str) -> dict:
    """
    If the LLM fails to return valid JSON (can happen with small models),
    return a basic structure so the pipeline doesn't crash.
    """
    # Extract first 100 chars as a rough topic
    first_line = raw_text.strip().split("\n")[0][:80] if raw_text else "Unknown Topic"
    return {
        "topic": first_line,
        "summary": raw_text[:300] if raw_text else "Content could not be summarized.",
        "key_points": ["Content analysis in progress", "Please try again", "Or paste text directly"],
        "hook": "Here's what you need to know...",
        "tone": "informational",
        "target_audience": "General audience",
        "hashtags": {
            "linkedin": ["#AI", "#Technology", "#Innovation", "#Learning", "#Content"],
            "instagram": ["#ai", "#tech", "#innovation", "#learning", "#content",
                         "#digital", "#future", "#knowledge", "#growth", "#insights",
                         "#trending", "#viral", "#education", "#tips", "#socialmedia"]
        },
        "cta": {
            "linkedin": "What are your thoughts? Share in the comments below!",
            "instagram": "Save this post for later! Tag someone who needs to see this!"
        },
        "emojis": ["💡", "🚀", "✅", "🔥"]
    }


def _extract_json_from_response(text: str) -> Optional[dict]:
    """
    LLMs sometimes wrap JSON in markdown or add extra text.
    This function tries multiple strategies to extract valid JSON.

    Strategy order (most to least reliable):
      1. Direct json.loads() → works if LLM was perfect
      2. Strip markdown code blocks → handles ```json {...} ```
      3. Find raw {} block → handles extra text before/after JSON
    """
    # Strategy 1: Direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from markdown code blocks
    # Pattern: ```json { ... } ``` or ``` { ... } ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 3: Find outermost {} block
    # re.DOTALL makes . match newlines too
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # All strategies failed
    return None


class Summarizer:
    """
    Summarizes raw text into structured content using the LLM.

    Why a class?
      - Can cache recent summaries later (avoid re-calling LLM for same content)
      - Easy to swap LLM strategy (streaming vs non-streaming)
      - Consistent pattern with other tools
    """

    def summarize(self, raw_text: str) -> StructuredSummary:
        """
        Main method: takes raw text, returns StructuredSummary.

        Flow:
          raw_text → LLM (with summarizer prompt) → JSON string
          → _extract_json_from_response() → dict
          → StructuredSummary(**dict) → validated Pydantic object

        Args:
            raw_text: Any text (scraped article, pasted text, search results, doc extract)

        Returns:
            StructuredSummary Pydantic object (validated, typed)

        Example:
            summary = summarizer.summarize("RAG is an AI technique that...")
            print(summary.hook)        # "What if your AI always gave correct answers?"
            print(summary.key_points)  # ["RAG reduces hallucinations", ...]
            print(summary.hashtags)    # {"linkedin": [...], "instagram": [...]}
        """
        print(f"[SUMMARIZER] Processing {len(raw_text)} chars...")

        # ── Call LLM with the expert prompt ──────────────────────────────
        # system_prompt = SUMMARIZER_SYSTEM_PROMPT (the expert persona + JSON format)
        # user_message  = the actual text to analyze
        llm_response = llm_client.chat(
            system_prompt=SUMMARIZER_SYSTEM_PROMPT,
            user_message=f"Analyze and summarize this content:\n\n{raw_text[:6000]}",
            # [:6000] = don't send too much text, keep within model context window
        )

        print(f"[SUMMARIZER] LLM responded ({len(llm_response)} chars)")

        # ── Parse the JSON response ───────────────────────────────────────
        parsed = _extract_json_from_response(llm_response)

        if parsed is None:
            print("[SUMMARIZER] WARNING: Could not parse LLM response as JSON, using fallback")
            parsed = _build_fallback_summary(raw_text)

        # ── Validate with Pydantic ────────────────────────────────────────
        # StructuredSummary(**parsed) will:
        #   - Check all required fields exist
        #   - Check types are correct (list is list, str is str)
        #   - Raise ValidationError if something is wrong
        try:
            summary = StructuredSummary(**parsed)
            print(f"[SUMMARIZER] Done. Topic: '{summary.topic}'")
            return summary
        except Exception as e:
            print(f"[SUMMARIZER] Pydantic validation failed: {e}, using fallback")
            fallback = _build_fallback_summary(raw_text)
            return StructuredSummary(**fallback)


# ── Singleton ─────────────────────────────────────────────────────────────────
summarizer = Summarizer()
