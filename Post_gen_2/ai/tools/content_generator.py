import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from ai.schema.model import Article

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")


class ContentGenerator:
    def __init__(self, model: str = "google/gemma-4-31b-it:free"):
        self.model = model
        self.client = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.7,
            
        )
       
        
        
    def generate(self, idea: str) -> Article:
        system_prompt = """
You are an expert technical writer.

The user provides a topic or short idea.

Generate a high-quality educational article.

Requirements:
- Generate an engaging title.
- Write a well-structured article.
- 700–1200 words.
- Include headings and paragraphs.
- Be factually accurate.
- Do not use markdown.
- Do not include citations.
- Do not include introductory phrases like
  "Here is your article".
  
Return ONLY JSON.

{
  "title":"",
  "content":""
}

"""
        
        

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"idea :\n{idea}")
        ]
        
        response = self.client.invoke(messages)
        
        try:
            # AIMessage -> string
            raw_json = response.content   
            # string -> dict
            data = json.loads(raw_json)
            # dict -> Pydantic
            article = Article.model_validate(data)
            return article
        
        except json.JSONDecodeError as e:
            raise ValueError(
            f"Model returned invalid JSON:\n{response.content}"
        ) from e
        
        
        

        
# article = ContentGenerator().generate("explain about mcp  and its benefits in detail with examples")         
# print("Title:", article.title)
# print("Content:", article.content)    