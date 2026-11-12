import re
import zlib

import numpy as np
import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.rag.retriever import HybridRetriever

DIM = 128

DOCS = [
    ("a1", "Reranking paper", 2022, "We use a cross-encoder reranker to rescore candidates; MRR rose from 0.62 to 0.71.", "p1"),
    ("a2", "HyDE paper", 2023, "HyDE generates a hypothetical document from the query and embeds it for dense retrieval.", "p1"),
    ("a3", "Chunking paper", 2024, "Chunk size of 512 tokens with overlap gave the best recall for retrieval.", "p2"),
    ("a4", "Graph paper", 2025, "GraphRAG builds a knowledge graph over entities and communities.", "p3"),
    ("a5", "Cooking paper", 2021, "Bake the bread at 220 degrees for thirty minutes.", "p3"),
]


def fake_embed(text: str) -> list[float]:
    """Bag-of-words hashing: كلمات مشتركة = vectors قريبة. بديل رخيص للموديل الحقيقي في الاختبارات."""
    v = np.zeros(DIM)
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        v[zlib.crc32(w.encode()) % DIM] += 1
    n = np.linalg.norm(v)
    return (v / n if n else v).tolist()


def fake_rerank(query: str, texts: list[str]) -> list[float]:
    q = set(re.findall(r"[a-z0-9]+", query.lower()))
    return [len(q & set(re.findall(r"[a-z0-9]+", t.lower()))) for t in texts]


@pytest.fixture
def rows():
    return [{
        "chunk_id": f"c{i}", "arxiv_id": a, "title": t, "page": 1, "section": "Method",
        "published": f"{y}-01-01T00:00:00", "raw_text": txt, "text": f"{t} | Method\n{txt}",
        "parent_id": "P1" if i < 3 else "P2",
    } for i, (a, t, y, txt, _) in enumerate(DOCS)]


@pytest.fixture
def retriever(rows):
    q = QdrantClient(":memory:")
    q.create_collection("test", vectors_config=VectorParams(size=DIM, distance=Distance.COSINE))
    q.upsert("test", points=[
        PointStruct(id=i, vector=fake_embed(r["text"]),
                    payload={"chunk_id": r["chunk_id"], "year": int(r["published"][:4])})
        for i, r in enumerate(rows)])
    parents = {"P1": "PARENT ONE full text", "P2": "PARENT TWO full text"}
    return HybridRetriever(rows, parents, q, "test", fake_embed, fake_rerank)
