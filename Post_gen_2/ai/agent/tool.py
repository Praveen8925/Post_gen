from ai.tools.web_scraper import Scraper , ArticleExtractor
from ai.tools.content_generator import ContentGenerator
from ai.tools.summarizer import Summarizer
from langchain_core.tools import tool
from ai.schema.model import Article

@tool
def scrap_url(url: str) -> Article:
    """
    Scrape the content of a given URL and return the extracted text.
    
    Args:
        url (str): The URL to scrape. """
    
    return Scraper().scrape(url)

@tool
def generate_content(idea: str) -> Article:
    """
    Generate content based on a given idea using the ContentGenerator tool.
    
    Args:
        idea (str): The idea or topic to generate content for. """
    return ContentGenerator().generate(idea)

@tool
def Article_extractor(article_content: str) -> Article:
    """
    Extract the title and content from the given article content using the ArticleExtractor tool.
    
    Args:
        article_content (str): The content of the article to extract. """
    
    return ArticleExtractor().extract(article_content)

@tool
def summarize_content(article: str) -> Article:
    """
    Summarize the given article content using the Summarizer tool.
    
    Args:
        article_content (str): The content of the article to summarize. """
    return Summarizer().summarize(article)


    