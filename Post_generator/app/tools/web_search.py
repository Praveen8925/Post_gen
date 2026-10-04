"""
web_search.py — Searches the web for a topic and returns aggregated text.

JOB: Topic keyword → web search results → plain text (ready for summarizer)

WHEN USED: When user's input_type is "topic"
  e.g. user says "write a post about RAG in 2025"
  → web_search.search("RAG retrieval augmented generation 2025")
  → returns text from top search results
  → passes to summarizer

WHY DuckDuckGo:
  - Free, no API key required (great for development)
  - Privacy-focused (no tracking, no bot detection issues)
  - DDGS().text() returns snippets directly (no need to scrape each result)

UPGRADE PATH (Phase 2):
  - Swap to Tavily API for richer, structured results
  - Just change the _search_duckduckgo() method

USED BY: app/mcp/server.py, app/agent/agent.py
"""

from duckduckgo_search import DDGS
from typing import List


class WebSearch:
    """
    Searches the web using DuckDuckGo and returns aggregated text.

    Result format from DDGS().text():
        [
            {"title": "...", "href": "...", "body": "snippet text..."},
            ...
        ]
    We extract the "body" (snippet) and "title" from top results.
    """

    def search(self, topic: str, max_results: int = 5) -> str:
        """
        Search for a topic and return aggregated text from top results.

        Args:
            topic:       Search query (e.g. "machine learning trends 2025")
            max_results: How many search results to use (default: 5)

        Returns:
            Aggregated plain text from search result snippets.
            Includes title + body for each result.

        Example:
            text = web_search.search("RAG retrieval augmented generation")
            # "Retrieval-Augmented Generation (RAG)\n
            #  RAG is an AI framework that combines...\n
            #  Why RAG Matters\n
            #  Organizations use RAG to..."
        """
        print(f"[WEB SEARCH] Searching: '{topic}'")

        try:
            results = self._search_duckduckgo(topic, max_results)
            combined = self._combine_results(results)
            print(f"[WEB SEARCH] Got {len(results)} results, {len(combined)} chars")
            return combined
        except Exception as e:
            print(f"[WEB SEARCH] ERROR: {e}")
            # Return error but don't crash — agent can fall back to LLM knowledge
            return f"Web search failed: {e}. Using LLM knowledge for topic: {topic}"

    # ─────────────────────────────────────────────────────────────────────────
    def _search_duckduckgo(self, query: str, max_results: int) -> List[dict]:
        """
        Internal: perform the actual DuckDuckGo search.
        Separated so we can easily swap search backends later.

        DDGS() = DuckDuckGo Search client (context manager)
        .text() = returns text search results (not images/news/maps)
        """
        with DDGS() as ddgs:
            results = list(ddgs.text(
                keywords=query,
                max_results=max_results,
                # region="wt-wt" = worldwide (no regional bias)
                region="wt-wt",
                # safesearch="off" = get all results (not just "safe" filtered)
                safesearch="off",
            ))
        return results

    def _combine_results(self, results: List[dict]) -> str:
        """
        Internal: combine search result snippets into a single text block.

        Each result looks like:
            {"title": "What is RAG?", "href": "https://...", "body": "RAG is..."}

        We format as:
            [Title]
            Snippet text...

            [Title 2]
            Snippet text 2...
        """
        if not results:
            return "No search results found."

        parts = []
        for result in results:
            title = result.get("title", "").strip()
            body = result.get("body", "").strip()
            href = result.get("href", "")

            if title and body:
                parts.append(f"{title}\n{body}\nSource: {href}")

        return "\n\n".join(parts)


# ── Singleton ─────────────────────────────────────────────────────────────────
web_search = WebSearch()
