from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from ai.schema.model import Article

class Scraper:
    NOISE_TAGS = [
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "aside",
        "iframe",
        "noscript",
        "svg",
        "button",
        "form",
    ]

    NOISE_KEYWORDS = [
        "author",
        "bio",
        "related",
        "recommended",
        "newsletter",
        "subscribe",
        "footer",
        "social",
        "share",
        "advertisement",
        "promo",
        "cookie",
        "banner",
        "comment",
        "comments",
        "sidebar",
        "navigation",
    ]

    def _score_container(self, tag):
        text = tag.get_text(" ", strip=True)

        if len(text) < 300:
            return 0

        p_count = len(tag.find_all("p"))
        h_count = len(tag.find_all(["h2", "h3", "h4"]))
        li_count = len(tag.find_all("li"))

        score = (
            len(text)
            + (p_count * 500)
            + (h_count * 200)
            + (li_count * 100)
        )

        return score

    def scrape(self, url: str) -> Article:
        print(f"[SCRAPER] Fetching: {url}")

        with sync_playwright() as p:
            browser = p.chromium.launch(
                executable_path="/usr/bin/google-chrome",
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                ],
            )

            page = browser.new_page(
                user_agent=(
                    "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/137.0.0.0 Safari/537.36"
                )
            )

            page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=60000,
            )

            page.wait_for_timeout(3000)

            html = page.content()

            browser.close()

        soup = BeautifulSoup(html, "lxml")

        # Remove obvious junk globally
        for tag in soup.find_all(self.NOISE_TAGS):
            tag.decompose()

        # Extract title
        title = ""

        if soup.find("h1"):
            title = soup.find("h1").get_text(" ", strip=True)
        elif soup.title:
            title = soup.title.get_text(" ", strip=True)

        # Candidate containers
        candidates = []

        for tag in soup.find_all(
            ["article", "main", "section", "div"]
        ):
            score = self._score_container(tag)

            if score > 0:
                candidates.append((score, tag))

        if candidates:
            content_container = max(
                candidates,
                key=lambda x: x[0]
            )[1]

            print(
                "Container score:",
                max(candidates, key=lambda x: x[0])[0]
            )
        else:
            content_container = soup.body or soup

        # Remove author/share/related blocks
        remove_list = []

        for tag in content_container.find_all(True):
            classes = " ".join(
                tag.get("class", [])
            ).lower()

            ids = str(
                tag.get("id", "")
            ).lower()

            combined = f"{classes} {ids}"

            if any(
                keyword in combined
                for keyword in self.NOISE_KEYWORDS
            ):
                remove_list.append(tag)

        for tag in remove_list:
            tag.decompose()

        # Extract content
        allowed_tags = {
            "h1",
            "h2",
            "h3",
            "h4",
            "p",
            "li",
        }

        extracted = []
        seen = set()

        for element in content_container.find_all(
            allowed_tags
        ):
            text = element.get_text(
                " ",
                strip=True,
            )

            if not text:
                continue

            if len(text) < 15:
                continue

            if text in seen:
                continue

            seen.add(text)

            if element.name == "li":
                extracted.append(f"- {text}")
            else:
                extracted.append(text)

        content = "\n\n".join(extracted)

        return Article.model_validate( {
            "title": title,
            "content": content[:50000],
            } )

class ArticleExtractor:
    def extract(self, text: str) -> Article:
        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        if not lines:
            raise ValueError("Article is empty.")

        title = None

        # Find title
        for line in lines[:10]:
            if len(line) < 120 and len(line.split()) <= 12:
                title = line
                break

        # Fallback title
        if title is None:
            title = "Untitled"
            index = 0
        else:
            index = lines.index(title)

        # Extract remaining content
        content = "\n\n".join(lines[index + 1:])

        return Article.model_validate({
            "title": title,
            "content": content,
        })

    
        
web = Scraper()

web_scraper = web.scrape("https://en.wikipedia.org/wiki/Psycho-Cybernetics")  # Replace with the actual URL you want to scrape