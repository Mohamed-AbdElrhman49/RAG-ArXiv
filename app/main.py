import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from .cache import TTLCache
from .config import Settings, get_settings
from .index import ensure_index
from .rag.generator import GenerationError, GroqGenerator, RateLimited
from .rag.pipeline import RagPipeline
from .schemas import Health, QueryRequest, QueryResponse

log = logging.getLogger("rag")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
STATIC = Path(__file__).parent / "static"


def build_pipeline(s: Settings) -> RagPipeline:
    """بيحمّل كل حاجة تقيلة مرة واحدة وقت الـ startup (مش مع كل request)."""
    from .rag.retriever import HybridRetriever, load_embedder, load_reranker, load_rows, make_qdrant

    rows_path = s.data_dir / f"chunks_{s.strategy}.jsonl"
    parents_path = s.data_dir / f"parents_{s.strategy}.json"
    if not rows_path.exists():
        raise RuntimeError(f"{rows_path} مش موجود. راجع README (خطوة تجهيز الداتا)")

    log.info("loading chunks ...")
    rows = load_rows(rows_path)
    parents = json.loads(parents_path.read_text(encoding="utf-8")) if parents_path.exists() else {}

    qdrant = make_qdrant(s.qdrant_path)
    ensure_index(qdrant, s.collection, rows, s.data_dir / f"{s.collection}.npy")  # أول مرة بس

    log.info("loading models ...")
    embed = load_embedder(s.embed_model, s.embed_query_prefix)
    rerank = load_reranker(s.rerank_model) if s.use_reranker else None
    retriever = HybridRetriever(rows, parents, qdrant, s.collection, embed, rerank)
    generator = GroqGenerator(s.groq_api_key, s.groq_model, s.groq_temperature, s.groq_reasoning_effort)
    log.info("ready: %d chunks", len(retriever))
    return RagPipeline(retriever, generator, top_k=s.top_k, n_candidates=s.n_candidates,
                       max_new_tokens=s.max_new_tokens)


def create_app(pipeline: RagPipeline | None = None, settings: Settings | None = None) -> FastAPI:
    s = settings or get_settings()
    cache = TTLCache(s.cache_size, s.cache_ttl_seconds)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        owned = pipeline is None
        app.state.pipeline = build_pipeline(s) if owned else pipeline
        yield
        if owned:  # نقفل Qdrant المحلي عشان الفولدر يتفك (مهم لو التطبيق اتشغّل تاني في نفس الـ process)
            try:
                app.state.pipeline.retriever.qdrant.close()
            except Exception:
                log.exception("failed to close qdrant")

    app = FastAPI(title="arXiv RAG", version="1.0.0", lifespan=lifespan,
                  description="Advanced RAG over arXiv papers: hybrid search + reranking + cited answers")
    app.add_middleware(CORSMiddleware, allow_origins=s.cors_origins.split(","), allow_methods=["*"],
                       allow_headers=["*"])

    def auth(x_api_key: str | None = Header(None)):
        if s.service_api_key and x_api_key != s.service_api_key:
            raise HTTPException(401, "invalid or missing X-API-Key")

    def kwargs(req: QueryRequest) -> dict:
        return dict(top_k=req.top_k, hybrid=req.hybrid, rerank=req.rerank,
                    expand_parents=req.expand_parents, year_from=req.year_from)

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/health", response_model=Health)
    def health():
        p = getattr(app.state, "pipeline", None)
        return Health(status="ok", ready=p is not None, chunks=len(p.retriever) if p is not None else 0)

    @app.post("/query", response_model=QueryResponse, dependencies=[Depends(auth)])
    def query(req: QueryRequest):
        key = (req.question.strip().lower(), tuple(sorted(kwargs(req).items())))
        if (hit := cache.get(key)) is not None:
            return {**hit, "cached": True}
        try:
            out = app.state.pipeline.answer(req.question, **kwargs(req))
        except RateLimited as e:
            raise HTTPException(429, f"Groq rate limit. جرّب بعد شوية. ({e})")
        except GenerationError as e:
            raise HTTPException(502, f"generation failed: {e}")
        if not out["uncited"] and not out["invalid_citations"]:  # ما نحفظش إجابة فيها مشاكل
            cache.set(key, out)
        return out

    @app.post("/query/stream", dependencies=[Depends(auth)])
    def query_stream(req: QueryRequest):
        def events():
            try:
                for ev, data in app.state.pipeline.stream(req.question, **kwargs(req)):
                    yield f"event: {ev}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
            except RateLimited:
                yield f"event: error\ndata: {json.dumps({'detail': 'Groq rate limit'})}\n\n"
            except Exception as e:
                log.exception("stream failed")
                yield f"event: error\ndata: {json.dumps({'detail': str(e)[:300]})}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


def get_app() -> FastAPI:  # uvicorn app.main:get_app --factory
    return create_app()


app = None  # ما بنعملش instance وقت الـ import، عشان الاختبارات ما تحمّلش موديلات
