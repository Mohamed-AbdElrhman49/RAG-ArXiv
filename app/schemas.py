from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    top_k: int | None = Field(None, ge=1, le=10, description="عدد المصادر اللي تتبعت للـ LLM")
    hybrid: bool = True
    rerank: bool = True
    expand_parents: bool = False
    year_from: int | None = Field(None, ge=1990, le=2100)


class Source(BaseModel):
    n: int
    title: str
    arxiv_id: str
    page: int
    section: str
    chunk_id: str
    url: str
    snippet: str


class Latency(BaseModel):
    retrieval: float
    generation: float
    total: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    refused: bool
    citations: list[Source]
    invalid_citations: list[int]
    uncited: bool
    retrieved: list[Source]
    latency: Latency
    cached: bool = False


class Health(BaseModel):
    status: str
    ready: bool
    chunks: int = 0
