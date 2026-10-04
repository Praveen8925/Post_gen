"""
web_scraper.py — Extracts clean text from a web URL.

JOB: URL in → plain text out. No LLM involved here.

HOW IT WORKS:
  1. requests.get(url)         → downloads the raw HTML
  2. BeautifulSoup(html)       → parses HTML into a navigable tree
  3. Remove noise              → strips scripts, styles, nav, footer tags
  4. Extract content tags      → grabs text from article, p, h1, h2, h3
  5. Clean whitespace          → strips extra spaces and blank lines
  6. Return plain text         → ready for the summarizer tool

WHY BeautifulSoup:
  - Handles messy/broken HTML (real websites are messy)
  - Easy to target specific tags
  - No JavaScript execution (Playwright needed for JS-heavy sites - Phase 2)

USED BY: app/mcp/server.py (tool_scrape_url)
         app/agent/agent.py  (when input_type == "url")
"""

import requests
from bs4 import BeautifulSoup
from typing import Optional


# Tags that contain the actual article content (not navigation/ads)
CONTENT_TAGS = ["article", "main", "section", "p", "h1", "h2", "h3", "h4", "li"]

# Tags that add noise — always remove these first
NOISE_TAGS = ["script", "style", "nav", "footer", "header", "aside",
              "advertisement", "iframe", "noscript"]

# Request headers — pretend to be a real browser
# Some websites block requests without a User-Agent header
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


class WebScraper:
    """
    Scrapes web pages and returns clean text content.

    Why a class?
      - Can hold session state later (cookies, rate limiting)
      - Easy to test with mock objects
      - Consistent with other tools in this project
    """

    def scrape(self, url: str, max_chars: int = 8000) -> str:
        """
        Scrape a URL and return clean extracted text.

        Args:
            url:       Full URL to scrape (e.g. "https://ibm.com/think/...")
            max_chars: Max characters to return (keeps summarizer prompt small)

        Returns:
            Clean plain text extracted from the page.
            Returns an error string if scraping fails (don't crash the whole pipeline).

        Example:
            text = web_scraper.scrape("https://ibm.com/think/topics/rag")
            # "Retrieval-Augmented Generation (RAG) is an AI framework..."
        """
        print(f"[SCRAPER] Fetching: {url}")

        # ── Step 1: Download HTML ─────────────────────────────────────────
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            response.raise_for_status()   # raises exception for 404, 500, etc.
        except requests.exceptions.Timeout:
            return f"ERROR: Request timed out for {url}"
        except requests.exceptions.ConnectionError:
            return f"ERROR: Could not connect to {url}"
        except requests.exceptions.HTTPError as e:
            return f"ERROR: HTTP {e.response.status_code} for {url}"
        except Exception as e:
            return f"ERROR: {str(e)}"

        # ── Step 2: Parse HTML with BeautifulSoup ─────────────────────────
        # "html.parser" = Python's built-in parser (no extra install needed)
        soup = BeautifulSoup(response.text, "html.parser")

        # ── Step 3: Remove noise tags ─────────────────────────────────────
        # .decompose() removes a tag AND its children from the tree
        for tag in soup(NOISE_TAGS):
            tag.decompose()

        # ── Step 4: Try to find main content container first ──────────────
        # Many sites wrap their article in <article> or <main>
        # This gives cleaner results than grabbing all <p> tags on the page
        content_container = (
            soup.find("article") or
            soup.find("main") or
            soup.find(class_=lambda c: c and any(
                word in str(c).lower()
                for word in ["article", "content", "post", "body"]
            )) or
            soup   # fallback: search entire page
        )

        # ── Step 5: Extract text from content tags ─────────────────────────
        texts = []
        for tag in content_container.find_all(CONTENT_TAGS):
            text = tag.get_text(separator=" ", strip=True)
            # Skip very short snippets (usually menu items, labels)
            if len(text) > 30:
                texts.append(text)

        # ── Step 6: Join and clean ─────────────────────────────────────────
        full_text = "\n".join(texts)

        # Remove duplicate blank lines
        lines = [line.strip() for line in full_text.splitlines() if line.strip()]
        cleaned = "\n".join(lines)

        # Truncate to max_chars (keeps summarizer prompt manageable)
        if len(cleaned) > max_chars:
            cleaned = cleaned[:max_chars] + "...[truncated]"

        print(f"[SCRAPER] Extracted {len(cleaned)} chars from {url}")
        return cleaned if cleaned else "ERROR: No readable content found on this page."


# ── Singleton ─────────────────────────────────────────────────────────────────
web_scraper = WebScraper()
