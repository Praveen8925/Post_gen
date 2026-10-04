import json
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


from ai.schema.model import StructuredSummary , Article

from ai.tools.web_scraper import Scraper , ArticleExtractor
from ai.tools.content_generator import ContentGenerator

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI PPI KEY not found in environment variables")


class Summarizer:
    def __init__(self, model: str = "openai/gpt-oss-20b:free"):
        self.model = model
        self.client = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.2,
            
            
        )
        self.structured_llm = self.client.with_structured_output(
            StructuredSummary
        )

    def summarize(self, article: Article) -> StructuredSummary:
        system_prompt = """
You are an expert content analyst.

CRITICAL RULES:
1. Return VALID JSON ONLY.
2. Do not use markdown.
3. Do not use code blocks.
4. Extract information ONLY from the provided content.
5. Do not hallucinate facts.
6. Generate a concise topic (maximum 8 words) that accurately describes the main subject of the article.
7. Generate a 4–5 sentence summary using only facts from the provided content. Preserve the original meaning. Do not rewrite creatively, exaggerate, infer missing information, or introduce new facts.
8. Key_points must contain exactly 4 concise points.
9. Hook must be one compelling sentence.
10. Tone must be ONE of:
   - educational
   - inspirational
   - informational
   - controversial
   - motivational
11. Target_audience must be selected from:
   - Students
   - IT Professionals
   - Business Leaders
12. Generate exactly 5 relevant LinkedIn_hashtags.
13. Generate exactly 4 relevant emojis.

Return valid JSON only.
"""
        
        article_input = f""" article title: {article.title}\n\narticle content: {article.content}"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"CONTENT:\n{article_input}")
        ]

        try:
            result: StructuredSummary = self.structured_llm.invoke(messages)
            print("Summarized successfully")
            return result

        except Exception as e:
            raise RuntimeError(
                f"Failed to summarize article: {e}") from e
            
        
            

# Retrieval augmented generation, or RAG, is an architecture for optimizing the performance of an artificial intelligence (AI) model by connecting it with external knowledge bases. RAG helps large language models (LLMs) deliver more relevant responses at a higher quality.

# Generative AI (gen AI) models are trained on large datasets and refer to this information to generate outputs. However, training datasets are finite and limited to the information the AI developer can access—public domain works, internet articles, social media content and other publicly accessible data.

# RAG allows generative AI models to access additional external knowledge bases, such as internal organizational data, scholarly journals and specialized datasets. By integrating relevant information into the generation process, chatbots and other natural language processing (NLP) tools can create more accurate domain-specific content without needing further training.


# What is Retrieval-Augmented Generation (RAG)?
# What is Retrieval-Augmented Generation (RAG)? (6:32 min)
# What are the benefits of RAG?
# RAG empowers organizations to avoid high retraining costs when adapting generative AI models to domain-specific use cases. Enterprises can use RAG to complete gaps in a machine learning model’s knowledge base so it can provide better answers.

# The primary benefits of RAG include:

# Cost-efficient AI implementation and AI scaling
# Access to current domain-specific data
# Lower risk of AI hallucinations
# Increased user trust
# Expanded use cases
# Enhanced developer control and model maintenance
# Greater data security
# Cost-efficient AI implementation and AI scaling
# When implementing AI, most organizations first select a foundation model: the deep-learning models that serve as the basis for the development of more advanced versions. Foundation models typically have generalized knowledge bases populated with publicly available training data, such as internet content available at the time of training.

# Retraining a foundation model or fine-tuning it—where a foundation model is further trained on new data in a smaller, domain-specific dataset—is computationally expensive and resource-intensive. The model adjusts some or all of its parameters to adjust its performance to the new specialized data.

# With RAG, enterprises can use internal, authoritative data sources and gain similar model performance increases without retraining. Enterprises can scale their implementation of AI applications as needed while mitigating cost and resource requirement increases.

# Access to current and domain-specific data
# Generative AI models have a knowledge cutoff, the point at which their training data was last updated. As a model ages further past its knowledge cutoff, it loses relevance over time. RAG systems connect models with supplemental external data in real-time and incorporate up-to-date information into generated responses.

# Enterprises use RAG to equip models with specific information such as proprietary customer data, authoritative research and other relevant documents.

# RAG models can also connect to the internet with application programming interfaces (APIs) and gain access to real-time social media feeds and consumer reviews for a better understanding of market sentiment. Meanwhile, access to breaking news and search engines can lead to more accurate responses as models incorporate the retrieved information into the text-generation process.

# Lower risk of AI hallucinations
# Generative AI models such as OpenAI’s GPT work by detecting patterns in their data, then using those patterns to predict the most likely outcomes to user inputs. Sometimes models detect patterns that don’t exist. A hallucination or confabulation happens when models present incorrect or made-up information as though it is factual.

# RAG anchors LLMs in specific knowledge backed by factual, authoritative and current data. Compared to a generative model operating only on its training data, RAG models tend to provide more accurate answers within the contexts of their external data. While RAG can reduce the risk of hallucinations, it cannot make a model error-proof.

# Increased user trust
# Chatbots, a common generative AI implementation, answer questions posed by human users. For a chatbot such as ChatGPT to be successful, users need to view its output as trustworthy. RAG models can include citations to the knowledge sources in their external data as part of their responses.

# When RAG models cite their sources, human users can verify those outputs to confirm accuracy while consulting the cited works for follow-up clarification and additional information. Corporate data storage is often a complex and siloed maze. RAG responses with citations point users directly toward the materials they need.

# Expanded use cases
# Access to more data means that one model can handle a wider range of prompts. Enterprises can optimize models and gain more value from them by broadening their knowledge bases, in turn expanding the contexts in which those models generate reliable results.

# By combining generative AI with retrieval systems, RAG models can retrieve and integrate information from multiple data sources in response to complex queries.

# Enhanced developer control and model maintenance
# Modern organizations constantly process massive quantities of data, from order inputs to market projections to employee turnover and more. Effective data pipeline construction and data storage is paramount for strong RAG implementation.

# At the same time, developers and data scientists can tweak the data sources to which models have access at any time. Repositioning a model from one task to another becomes a task of adjusting its external knowledge sources as opposed to fine-tuning or retraining. If fine-tuning is needed, developers can prioritize that work instead of managing the model’s data sources.

# Greater data security
# Because RAG connects a model to external knowledge sources rather than incorporating that knowledge into the model’s training data, it maintains a divide between the model and that external knowledge. Enterprises can use RAG to preserve first-party data while simultaneously granting models access to it—access that can be revoked at any time.

# However, enterprises must be vigilant to maintain the security of the external databases themselves. RAG uses vector databases, which use embeddings to convert data points to numerical representations. If these databases are breached, attackers can reverse the vector embedding process and access the original data, especially if the vector database is unencrypted.""")
# Ari = web_scraper.scrape("https://aws.amazon.com/what-is/retrieval-augmented-generation/")

# summarizer = Summarizer()

# summary = summarizer.summarize(res)

# print("Summary:", summary)

# print("Title:", Ari['title'])

# print("Topic:", summary.topic)

# print("Summary:", summary.summary)
# key = summary.key_points

# for i, point in enumerate(key, start=1):
#     print(f"Key Point {i}:", point)


# print("Hook:", summary.hook)

# print("Tone:", summary.tone)

# print("Target Audience:", summary.target_audience)

# print("Hashtags:", summary.hashtags)

# print("CTA:", summary.cta)

# print("Emojis:", summary.emojis)

