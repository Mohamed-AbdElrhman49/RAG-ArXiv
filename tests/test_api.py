import json

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.rag.generator import GenerationError, RateLimited
from app.rag.pipeline import RagPipeline


class Gen:
    calls = 0
    def __init__(self, text="It uses a reranker [1].", exc=None): self.text, self.exc = text, exc
    def chat(self, messages, max_tokens):
        Gen.calls += 1
        if self.exc: raise self.exc
        return self.text
    def stream(self, messages, max_tokens): yield from [self.text[:5], self.text[5:]]


def client(retriever, gen=None, **settings):
    p = RagPipeline(retriever, gen or Gen(), top_k=2)
    return TestClient(create_app(p, Settings(_env_file=None, **settings)))


def test_health(retriever):
    with client(retriever) as c:
        assert c.get("/health").json() == {"status": "ok", "ready": True, "chunks": 5}


def test_index_page(retriever):
    with client(retriever) as c:
        r = c.get("/")
        assert r.status_code == 200 and "arXiv RAG" in r.text


def test_query_ok_and_cached(retriever):
    Gen.calls = 0
    with client(retriever) as c:
        body = {"question": "cross-encoder reranker MRR"}
        a = c.post("/query", json=body).json()
        b = c.post("/query", json=body).json()
        assert a["cached"] is False and b["cached"] is True and Gen.calls == 1
        assert a["citations"][0]["n"] == 1 and a["latency"]["total"] >= 0


def test_uncited_answers_are_not_cached(retriever):
    Gen.calls = 0
    with client(retriever, Gen("no citation here")) as c:
        c.post("/query", json={"question": "cross-encoder reranker"})
        c.post("/query", json={"question": "cross-encoder reranker"})
        assert Gen.calls == 2


def test_validation_errors(retriever):
    with client(retriever) as c:
        assert c.post("/query", json={"question": "hi"}).status_code == 422
        assert c.post("/query", json={"question": "valid question", "top_k": 99}).status_code == 422


def test_rate_limit_maps_to_429_and_generic_to_502(retriever):
    with client(retriever, Gen(exc=RateLimited("slow down"))) as c:
        assert c.post("/query", json={"question": "reranker question"}).status_code == 429
    with client(retriever, Gen(exc=GenerationError("boom"))) as c:
        assert c.post("/query", json={"question": "reranker question"}).status_code == 502


def test_api_key_auth(retriever):
    with client(retriever, service_api_key="secret") as c:
        body = {"question": "reranker question"}
        assert c.post("/query", json=body).status_code == 401
        assert c.post("/query", json=body, headers={"X-API-Key": "nope"}).status_code == 401
        assert c.post("/query", json=body, headers={"X-API-Key": "secret"}).status_code == 200
        assert c.get("/health").status_code == 200  # health مفتوح


def test_stream_sse(retriever):
    with client(retriever) as c:
        r = c.post("/query/stream", json={"question": "cross-encoder reranker"})
        assert r.headers["content-type"].startswith("text/event-stream")
        events = [blk.split("\n")[0].removeprefix("event: ") for blk in r.text.strip().split("\n\n")]
        assert events[0] == "sources" and events[-1] == "done" and "token" in events
        done = r.text.strip().split("\n\n")[-1].split("data: ")[1]
        assert json.loads(done)["answer"] == "It uses a reranker [1]."


def test_stream_error_event(retriever):
    class Bad(Gen):
        def stream(self, messages, max_tokens):
            raise RateLimited("limit")
            yield
    with client(retriever, Bad()) as c:
        r = c.post("/query/stream", json={"question": "reranker question"})
        assert "event: error" in r.text
