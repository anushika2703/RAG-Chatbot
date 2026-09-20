from pydantic import BaseModel
from typing import List


class QueryRequest(BaseModel):
    question: str


class ContextChunk(BaseModel):
    content: str
    page: int | None = None
    score: float | None = None


class QueryResponse(BaseModel):
    answer: str
    context: List[ContextChunk]
    confidence: float