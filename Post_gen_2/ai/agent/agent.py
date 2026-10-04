import time
from ai.agent.tool import scrap_url, generate_content, Article_extractor, summarize_content

class PipelineAgent:
    """
    Content generation pipeline.

    Supported inputs:
        1. URL
        2. Pasted article
        3. Topic / short idea

    Pipeline:

        URL
            ↓
        Scraper
            ↓
        Article

        Pasted Article
            ↓
        ArticleExtractor
            ↓
        Article

        Topic
            ↓
        ContentGenerator
            ↓
        Article

        Article
            ↓
        Summarizer
            ↓
        StructuredSummary
            ↓
        PostGenerator
    """

    def __init__(self):
        self.scraper = scrap_url()
        self.extractor = Article_extractor()
        self.generator = generate_content()
        self.summarizer = summarize_content()
        

    def run(
        self,
        input_type: str,
        content: str) -> dict:

        start = time.time()

        print(f"\n[AGENT] Input Type : {input_type}")
        print("[AGENT] Step 1 : Creating Article")

        # ---------------------------------------------------
        # STEP 1 : Produce Article
        # ---------------------------------------------------

        if input_type == "url":

            article = self.scraper.scrape(content)
            source = content

        elif input_type == "article":

            article = self.extractor.extract(content)
            source = "Pasted Article"

        elif input_type == "idea":

            article = self.generator.generate(content)
            source = "Generated from Idea"

        else:
            raise ValueError(
                "input_type must be one of: "
                "'url', 'article', 'idea'"
            )

        print("[AGENT] ✓ Article Ready")

        # ---------------------------------------------------
        # STEP 2 : Summarize
        # ---------------------------------------------------

        print("[AGENT] Step 2 : Summarizing")

        summary = self.summarizer.summarize(article)

        print(f"[AGENT] ✓ Topic : {summary.topic}")

        # ---------------------------------------------------
        # STEP 3 : Generate Posts
        # ---------------------------------------------------

        print("[AGENT] Step 3 : Generating Posts")

        posts = self.post_generator.generate(
            summary=summary,
            platform=platform,
        )

        elapsed = round(time.time() - start, 2)

        print(f"[AGENT] Finished in {elapsed}s")

        return {
            "input_type": input_type,
            "source": source,
            "platform": platform,
            "elapsed_seconds": elapsed,

            "article": article.model_dump(),

            "summary": summary.model_dump(),

            "linkedin_post": posts.get("linkedin_post"),

            "instagram_post": posts.get("instagram_post"),
        }