from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.retrieval.query_engine import ClinicalRAGEngine

_engine: ClinicalRAGEngine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    try:
        _engine = ClinicalRAGEngine()
    except Exception as exc:
        # Index not built yet — fail loudly and clearly at startup rather than
        # on the first request.
        raise RuntimeError(
            "Failed to load index. Run `python -m src.indexing.build_index` "
            f"first. Original error: {exc}"
        ) from exc
    yield


app = FastAPI(title="MedRAG Clinical Guideline API", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


class SourceItem(BaseModel):
    source_name: str
    score: float
    excerpt: str


class QueryResponse(BaseModel):
    answer: str
    abstained: bool
    top_score: float
    sources: list[SourceItem]
    warning: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized.")
    result = _engine.query(req.question)
    return QueryResponse(
        answer=result.answer,
        abstained=result.abstained,
        top_score=result.top_score,
        sources=result.sources,
        warning=result.warning,
    )
