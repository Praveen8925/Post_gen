from typing import List , Literal
from pydantic import BaseModel

from pydantic import BaseModel, Field, AliasChoices

class Article(BaseModel):
    title: str = Field(validation_alias=AliasChoices("title", "Title"))
    content: str = Field(validation_alias=AliasChoices("content", "Content"))
    


class CTA(BaseModel):
    linkedin: str
    instagram: str


class StructuredSummary(BaseModel):
    topic: str
    summary: str
    key_points: List[str]
    hook: str
    tone: Literal[
    "educational",
    "informational",
    "motivational",
    "inspirational",
    "controversial",
]
    target_audience: Literal[
    "Students",
    "IT Professionals",
    "Business Leaders"]
    hashtags: List[str]
    cta: CTA
    emojis: List[str]