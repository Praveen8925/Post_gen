"""
agent.py — The orchestrator: routes input → calls tools → returns final posts.

TWO AGENT MODES (both exposed via the API):

1. PIPELINE AGENT (default, recommended for production):
   - Rule-based routing: input_type tells us WHICH tool to call first
   - Reliable, fast, predictable — doesn't waste LLM tokens on routing
   - LLM is still used inside the tools (summarize + generate)
   - This is how most production AI pipelines work

2. LLM AGENT (educational, shows LLM-MCP integration):
   - LangChain AgentExecutor: LLM reads tool descriptions and decides what to call
   - LLM autonomously plans: "I should scrape this URL, then summarize, then generate"
   - Shows the true "agentic" pattern but slower and less predictable with 3B models
   - Easy to switch to OpenAI GPT-4o later for better tool-calling reliability

HOW LLM AGENT WORKS:
  User prompt → LLM sees tool list (name + description)
    → LLM responds with: "I'll call tool X with args Y"
    → LangChain executes tool X(Y) → gets result
    → Result goes back to LLM
    → LLM decides next tool or final answer
    → Loop until LLM says "I'm done"

  This is called ReAct (Reason + Act) pattern:
    Reason: "I have a URL, so I need to scrape it first"
    Act:    call scrape_url("https://...")
    Observe: "Got 5000 chars of text"
    Reason: "Now I should summarize this"
    Act:    call summarize_content("5000 chars...")
    ... and so on
"""

import json
import time
from typing import Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.core.config import settings
from app.schemas.models import StructuredSummary, GenerateResponse
from app.agent.tools import (
    scrape_url,
    search_web,
    summarize_content,
    generate_post,
    process_document,
    ALL_TOOLS,
)
# Import raw tool functions for pipeline mode (direct call, no LangChain overhead)
from app.tools.web_scraper import web_scraper
from app.tools.summarizer import summarizer
from app.tools.web_search import web_search
from app.tools.post_generator import post_generator
from app.tools.rag_tool import rag_tool
from app.tools.media_tool import media_tool   # Step 4


# =============================================================================
# AGENT SYSTEM PROMPT (used by LLM Agent mode)
# This tells the LLM its role, what tools exist, and how to use them
# =============================================================================
AGENT_SYSTEM_PROMPT = """You are a social media content generation agent.
Your job is to generate high-quality LinkedIn and/or Instagram posts from user-provided content.

AVAILABLE TOOLS:
- scrape_url(url): Use when user gives a URL. Extracts article text.
- search_web(topic): Use when user gives a topic keyword. Finds web content.
- summarize_content(raw_text): ALWAYS call after getting raw text. Structures the content.
- generate_post(summary_json, platform): ALWAYS call last. Creates the final post(s).
- process_document(file_path): Use when user uploads a PDF or DOCX file.

WORKFLOW — always follow this sequence:
1. Get raw text: call the right input tool (scrape_url / search_web / process_document)
   OR if user pasted text directly, skip to step 2.
2. Summarize: call summarize_content(raw_text) → get summary_json
3. Generate: call generate_post(summary_json, platform) → get final post(s)

IMPORTANT:
- Always complete all 3 steps. Never stop after step 1 or 2.
- For pasted text input: skip step 1, call summarize_content(pasted_text) directly.
- platform must be exactly: 'linkedin', 'instagram', or 'both'
"""


# =============================================================================
# MODE 1: PIPELINE AGENT (rule-based routing + direct tool calls)
# =============================================================================
class PipelineAgent:
    """
    Reliable content generation agent using rule-based tool routing.

    Why rule-based routing:
      - input_type is KNOWN (user tells us explicitly)
      - Rule-based is 100% reliable; LLM routing can hallucinate
      - LLM is used where it adds value: summarization + post generation
      - Faster: no LLM tokens wasted on routing decisions

    This is the production-recommended pattern.
    """

    def run(
        self,
        input_type: str,
        content: str,
        platform: str,
        file_path: Optional[str] = None,
        media_mode: str = "no_media",        # "manual" | "no_media" | "ai_smart"
        media_type: Optional[str] = None,    # only for media_mode == "manual"
    ) -> dict:
        """
        Run the full content generation pipeline.

        Args:
            input_type: "url" | "text" | "topic" | "document"
            content:    URL / pasted text / topic string (not used for documents)
            platform:   "linkedin" | "instagram" | "both"
            file_path:  Path to uploaded file (only for input_type="document")

        Returns:
            dict with: input_type, platform, structured_summary, linkedin_post, instagram_post

        Pipeline:
            url      → scrape      → summarize → generate
            text     → (direct)    → summarize → generate
            topic    → web_search  → summarize → generate
            document → rag_process → summarize → generate
        """
        start = time.time()
        print(f"\n[AGENT] Starting pipeline | input_type={input_type} | platform={platform}")

        # ── STEP 1: Get raw text based on input_type ───────────────────────
        print(f"[AGENT] Step 1: Acquiring raw content...")

        if input_type == "url":
            raw_text = web_scraper.scrape(content)
            source = f"scraped: {content}"

        elif input_type == "text":
            raw_text = content
            source = "pasted text"

        elif input_type == "topic":
            raw_text = web_search.search(content)
            source = f"web search: '{content}'"

        elif input_type == "document":
            if not file_path:
                return {"error": "file_path required for input_type='document'"}
            raw_text = rag_tool.process(file_path)
            source = f"document: {file_path}"

        else:
            return {"error": f"Unknown input_type: '{input_type}'"}

        if raw_text.startswith("ERROR"):
            return {"error": raw_text}

        print(f"[AGENT] Step 1 done: {len(raw_text)} chars from {source}")

        # ── STEP 2: Summarize ─────────────────────────────────────────────
        print(f"[AGENT] Step 2: Summarizing content...")
        summary = summarizer.summarize(raw_text)
        print(f"[AGENT] Step 2 done: topic='{summary.topic}'")

        # ── STEP 3: Generate post(s) ──────────────────────────────────────
        print(f"[AGENT] Step 3: Generating {platform} post(s)...")
        posts = post_generator.generate(summary, platform)
        print(f"[AGENT] Step 3 done.")

        # ── STEP 4: Media generation ──────────────────────────────────────
        print(f"[AGENT] Step 4: Media mode='{media_mode}'...")
        media_result = media_tool.generate(
            summary=summary,
            media_mode=media_mode,
            media_type=media_type,
        )
        print(f"[AGENT] Step 4 done: {media_result.message}")

        elapsed = round(time.time() - start, 1)
        print(f"[AGENT] Pipeline complete in {elapsed}s")

        # ── Build response ────────────────────────────────────────────────
        return {
            "agent_mode": "pipeline",
            "input_type": input_type,
            "source": source,
            "platform": platform,
            "elapsed_seconds": elapsed,
            "structured_summary": summary.model_dump(),
            "linkedin_post": posts.get("linkedin_post"),
            "instagram_post": posts.get("instagram_post"),
            "media": media_result.model_dump(),
        }


# =============================================================================
# MODE 2: LLM AGENT (LangChain tool-calling — shows LLM-MCP integration)
# =============================================================================
class LLMAgent:
    """
    Agentic content generator — LLM autonomously decides tool sequence.

    HOW IT WORKS:
      1. We bind tools to the LLM: llm.bind_tools(ALL_TOOLS)
      2. We send user request + AGENT_SYSTEM_PROMPT to the LLM
      3. LLM responds with a tool_call (not text): { "name": "scrape_url", "args": {...} }
      4. We execute that tool, get the result
      5. We add the result to the conversation history
      6. We send the updated history back to LLM
      7. LLM decides the next tool or gives a final answer
      8. Repeat until LLM stops calling tools

    This is the true "agentic" loop: LLM → tool → LLM → tool → answer

    LIMITATION with llama3.2:3b:
      Small models can sometimes:
      - Call tools in wrong order
      - Stop too early (before generate_post)
      - Produce malformed tool call JSON
      We handle these cases with fallback logic.
    """

    def __init__(self):
        # Bind ALL_TOOLS to the LLM so it can call them
        # bind_tools() adds the tool schemas to every LLM call
        # The LLM uses these schemas to format its tool-call responses
        self.llm = ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            temperature=0.3,  # lower temp = more deterministic tool selection
        ).bind_tools(ALL_TOOLS)

        self.tools_by_name = {t.name: t for t in ALL_TOOLS}

    def run(
        self,
        input_type: str,
        content: str,
        platform: str,
    ) -> dict:
        """
        Run the LLM agent loop.

        The LLM sees: system prompt + user request
        The LLM calls: tools in the order it decides
        We execute each tool and return the result to the LLM
        Loop until: LLM gives a final response (no more tool calls)
        """
        start = time.time()
        print(f"\n[LLM AGENT] Starting | input_type={input_type} | platform={platform}")

        # ── Build initial message ─────────────────────────────────────────
        # Tell the LLM what we want in plain language
        user_message = self._build_user_message(input_type, content, platform)

        # Conversation history — starts with system + user
        messages = [
            SystemMessage(content=AGENT_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        final_result = None
        max_iterations = 6      # safety limit — prevent infinite loops
        iteration = 0

        # ── AGENT LOOP ────────────────────────────────────────────────────
        while iteration < max_iterations:
            iteration += 1
            print(f"[LLM AGENT] Iteration {iteration}: calling LLM...")

            # Send conversation to LLM → get response
            response = self.llm.invoke(messages)
            messages.append(response)   # add LLM response to history

            # ── Check if LLM wants to call a tool ────────────────────────
            if response.tool_calls:
                # LLM responded with tool calls (not plain text)
                print(f"[LLM AGENT] LLM called {len(response.tool_calls)} tool(s):")

                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_id   = tool_call["id"]

                    print(f"[LLM AGENT]   → {tool_name}({list(tool_args.keys())})")

                    # ── Execute the tool ──────────────────────────────────
                    if tool_name in self.tools_by_name:
                        tool_fn = self.tools_by_name[tool_name]
                        try:
                            # Call the LangChain tool with the LLM's args
                            tool_result = tool_fn.invoke(tool_args)
                            print(f"[LLM AGENT]   ← {tool_name} returned {len(str(tool_result))} chars")
                        except Exception as e:
                            tool_result = f"Tool error: {str(e)}"
                            print(f"[LLM AGENT]   ← {tool_name} ERROR: {e}")
                    else:
                        tool_result = f"Unknown tool: {tool_name}"

                    # ── Add tool result to conversation ───────────────────
                    # ToolMessage = the result of executing a tool
                    # The LLM reads this in the next iteration
                    messages.append(ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_id,
                    ))

                    # ── Check if generate_post was called ─────────────────
                    if tool_name == "generate_post":
                        try:
                            final_result = json.loads(tool_result)
                        except Exception:
                            pass   # LLM will still give a final answer

            else:
                # ── LLM gave a text response (no tool calls) ──────────────
                # This means the LLM is done with tool calls
                print(f"[LLM AGENT] LLM gave final text response")
                break

        # ── Build response ────────────────────────────────────────────────
        elapsed = round(time.time() - start, 1)
        print(f"[LLM AGENT] Done in {elapsed}s ({iteration} iterations)")

        if final_result:
            return {
                "agent_mode": "llm",
                "input_type": input_type,
                "platform": platform,
                "elapsed_seconds": elapsed,
                "iterations": iteration,
                "linkedin_post": final_result.get("linkedin_post"),
                "instagram_post": final_result.get("instagram_post"),
            }
        else:
            # Fallback: if LLM agent didn't produce structured output,
            # run pipeline agent as backup
            print("[LLM AGENT] No structured result from LLM agent, falling back to pipeline")
            pipeline = PipelineAgent()
            result = pipeline.run(input_type, content, platform)
            result["agent_mode"] = "llm_with_pipeline_fallback"
            result["iterations"] = iteration
            return result

    def _build_user_message(self, input_type: str, content: str, platform: str) -> str:
        """Build a clear task description for the LLM agent."""
        input_descriptions = {
            "url":      f"I have a URL to an article: {content}",
            "text":     f"I have pasted article text:\n\n{content[:2000]}",
            "topic":    f"I want a post about this topic: {content}",
            "document": f"I have uploaded a document at path: {content}",
        }
        desc = input_descriptions.get(input_type, content)
        return (
            f"{desc}\n\n"
            f"Please generate a social media post for platform: {platform}\n"
            f"Follow the workflow: get content → summarize → generate post."
        )


# =============================================================================
# SINGLETON INSTANCES — import these in routes
# =============================================================================
pipeline_agent = PipelineAgent()
llm_agent = LLMAgent()
