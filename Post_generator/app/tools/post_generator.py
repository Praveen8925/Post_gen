"""
post_generator.py — Formats StructuredSummary into platform-specific posts.

JOB: StructuredSummary → LinkedIn post AND/OR Instagram caption.

WHY SEPARATE FROM SUMMARIZER:
  - Summarizer extracts facts/structure (analytical)
  - PostGenerator creates the final human-readable post (creative)
  - Splitting them = easy to swap post style without touching summarizer

PLATFORM DIFFERENCES (why two different prompts):
  LinkedIn                       Instagram
  ─────────────────────────────  ──────────────────────────────
  Professional, thought-leader   Casual, engaging, visual-first
  150-300 words                  80-150 words
  Bold key terms (Unicode)       Heavy emojis throughout
  Story arc structure            Hook → Value → CTA
  3-5 hashtags at end            15-20 hashtags at end
  "What do you think?" CTA       "Save this!" / "Tag a friend"

UNICODE BOLD: LinkedIn renders Unicode bold math letters as bold text.
  Normal: AI    Bold: 𝗔𝗜  (these are actual Unicode characters, not markdown)
  We tell the LLM to use them for key terms.

USED BY: app/mcp/server.py, app/agent/agent.py
INPUT:   StructuredSummary object + platform string
OUTPUT:  Formatted post string(s)
"""

from app.llm.llm_client import llm_client
from app.schemas.models import StructuredSummary


# =============================================================================
# LINKEDIN PROMPT ENGINEERING
# =============================================================================
LINKEDIN_SYSTEM_PROMPT = """You are a top-tier LinkedIn content strategist who has grown multiple accounts to 100K+ followers.

You write posts that feel HUMAN, not AI-generated. Your style:
- Opens with a bold hook that creates curiosity or shares a surprising insight
- Uses short paragraphs (2-3 lines max) for easy reading on mobile
- Bolds KEY TERMS using Unicode bold: 𝗹𝗶𝗸𝗲 𝘁𝗵𝗶𝘀 (not markdown **bold**)
- Uses bullet points for key takeaways
- Ends with a thought-provoking question or CTA
- Adds 3-5 relevant hashtags on the last line

POST STRUCTURE:
[Hook - 1 bold attention-grabbing sentence]

[Context - 2-3 sentences setting up the problem/idea]

[Key takeaways as bullet points with • symbol]
• 𝗣𝗼𝗶𝗻𝘁 𝟭: explanation
• 𝗣𝗼𝗶𝗻𝘁 𝟮: explanation
• 𝗣𝗼𝗶𝗻𝘁 𝟯: explanation

[Closing insight - 1-2 sentences]

[CTA question - engage the audience]

#Hashtag1 #Hashtag2 #Hashtag3 #Hashtag4 #Hashtag5

RULES:
- 150-300 words total
- NO markdown headers (##, **) — only Unicode bold for emphasis
- NO "In conclusion" or "To summarize" — sound natural
- Use emojis sparingly: 1-2 max, only where they add value
- Write in first person if possible ("I've seen...", "Here's what I've learned...")"""


LINKEDIN_USER_TEMPLATE = """Create a LinkedIn post using this content data:

Topic: {topic}
Hook: {hook}
Summary: {summary}
Key Points:
{key_points}
Tone: {tone}
Target Audience: {target_audience}
Hashtags to use: {hashtags}
CTA: {cta}
Emojis available: {emojis}

Write the complete LinkedIn post now:"""


# =============================================================================
# INSTAGRAM PROMPT ENGINEERING
# =============================================================================
INSTAGRAM_SYSTEM_PROMPT = """You are a viral Instagram content creator with 500K+ followers in the tech and business niche.

Your captions are PUNCHY, VISUAL, and SCROLL-STOPPING. Your style:
- First line is a HOOK that makes people stop scrolling (question, bold claim, or number)
- Uses emojis as bullet points and visual separators throughout
- Short sentences. Very punchy. Like this.
- Ends with a clear CTA ("Save this!", "Tag someone!", "Drop a 🔥 below")
- Followed by a line break and then ALL hashtags

POST STRUCTURE:
[HOOK - One bold/curious/shocking opening line]

[2-3 value lines with emojis as bullets]
✅ insight one
✅ insight two  
✅ insight three

[1-2 closing lines that feel warm and personal]

[CTA — direct and energetic]

.
.
.
[15-20 hashtags — mix of large (#ai), medium (#artificialintelligence), niche (#ragpipeline)]

RULES:
- 80-150 words (excluding hashtags)
- Heavy emoji use — at least 5-8 emojis in the post body
- NO formal business language — sound like a real person
- Hashtags must include: mix of popular and niche tags
- Line breaks matter for readability on Instagram"""


INSTAGRAM_USER_TEMPLATE = """Create an Instagram caption using this content data:

Topic: {topic}
Hook: {hook}
Summary: {summary}
Key Points:
{key_points}
Tone: {tone}
Hashtags to use: {hashtags}
CTA: {cta}
Emojis: {emojis}

Write the complete Instagram caption now:"""


# =============================================================================
# POST GENERATOR CLASS
# =============================================================================
class PostGenerator:
    """
    Generates platform-specific posts from a StructuredSummary.

    Methods:
      generate_linkedin(summary)  → LinkedIn post string
      generate_instagram(summary) → Instagram caption string
      generate(summary, platform) → dict with whichever posts are requested
    """

    def generate_linkedin(self, summary: StructuredSummary) -> str:
        """
        Generate a LinkedIn post from the structured summary.

        Fills LINKEDIN_USER_TEMPLATE with summary data, then calls LLM
        with LINKEDIN_SYSTEM_PROMPT as the expert persona.
        """
        print(f"[POST GEN] Generating LinkedIn post for: '{summary.topic}'")

        # Format key_points as a numbered list for the prompt
        key_points_text = "\n".join(
            f"{i+1}. {point}" for i, point in enumerate(summary.key_points)
        )

        # Fill the template with actual data
        # .format(**dict) replaces {topic}, {hook}, etc. with real values
        user_message = LINKEDIN_USER_TEMPLATE.format(
            topic=summary.topic,
            hook=summary.hook,
            summary=summary.summary,
            key_points=key_points_text,
            tone=summary.tone,
            target_audience=summary.target_audience,
            hashtags=" ".join(summary.hashtags.get("linkedin", [])),
            cta=summary.cta.get("linkedin", "What do you think? Share below!"),
            emojis=" ".join(summary.emojis),
        )

        post = llm_client.chat(
            system_prompt=LINKEDIN_SYSTEM_PROMPT,
            user_message=user_message,
        )

        print(f"[POST GEN] LinkedIn post: {len(post)} chars")
        return post.strip()

    # ─────────────────────────────────────────────────────────────────────────
    def generate_instagram(self, summary: StructuredSummary) -> str:
        """
        Generate an Instagram caption from the structured summary.

        Same pattern as LinkedIn but uses INSTAGRAM prompts.
        """
        print(f"[POST GEN] Generating Instagram caption for: '{summary.topic}'")

        key_points_text = "\n".join(
            f"• {point}" for point in summary.key_points
        )

        user_message = INSTAGRAM_USER_TEMPLATE.format(
            topic=summary.topic,
            hook=summary.hook,
            summary=summary.summary,
            key_points=key_points_text,
            tone=summary.tone,
            hashtags=" ".join(summary.hashtags.get("instagram", [])),
            cta=summary.cta.get("instagram", "Save this! Tag someone!"),
            emojis=" ".join(summary.emojis),
        )

        post = llm_client.chat(
            system_prompt=INSTAGRAM_SYSTEM_PROMPT,
            user_message=user_message,
        )

        print(f"[POST GEN] Instagram post: {len(post)} chars")
        return post.strip()

    # ─────────────────────────────────────────────────────────────────────────
    def generate(self, summary: StructuredSummary, platform: str) -> dict:
        """
        Generate posts for the requested platform(s).

        Args:
            summary:  StructuredSummary from the summarizer tool
            platform: "linkedin" | "instagram" | "both"

        Returns:
            dict with keys: linkedin_post, instagram_post (None if not requested)

        Example:
            result = post_generator.generate(summary, "both")
            result["linkedin_post"]   # full LinkedIn post string
            result["instagram_post"]  # full Instagram caption string
        """
        result = {"linkedin_post": None, "instagram_post": None}

        if platform in ("linkedin", "both"):
            result["linkedin_post"] = self.generate_linkedin(summary)

        if platform in ("instagram", "both"):
            result["instagram_post"] = self.generate_instagram(summary)

        return result


# ── Singleton ─────────────────────────────────────────────────────────────────
post_generator = PostGenerator()
