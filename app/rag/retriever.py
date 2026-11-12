"""Hybrid retrieval: Dense (Qdrant) + BM25 -> RRF -> Cross-encoder rerank -> (اختياري) parent expansion.

الموديلات بتتحقن كدوال (embed_query, rerank_fn) عشان نقدر نختبر الكود من غير ما ننزل موديلات.
"""
import json
import re
from pathlib import Path
from typing import Callable

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, Range
from rank_bm25 import BM25Okapi

STOP = set("a an the of in on for to and or is are was were be by with as at from that this it its we our".split())
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """بنحافظ على المصطلحات المركبة زي bge-m3 و self-rag"""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOP]


def rrf(rank_lists: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: بيدمج بالترتيب مش بالـ score، لأن مقاييس BM25 و cosine مختلفة."""
    scores: dict[int, float] = {}
    for lst in rank_lists:
        for rank, idx in enumerate(lst):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def load_rows(path: Path) -> list[dict]:
    """الترتيب هنا = id الـ point في Qdrant = الصف في ملف الـ .npy. ممنوع يتغيّر."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def make_qdrant(path: str) -> QdrantClient:
    """Qdrant بيشتغل كمكتبة جوه التطبيق وبيخزّن في فولدر (من غير سيرفر)."""
    return QdrantClient(path=path)


class HybridRetriever:
    def __init__(
        self,
        rows: list[dict],
        parents: dict[str, str],
        qdrant: QdrantClient,
        collection: str,
        embed_query: Callable[[str], list[float]],
        rerank_fn: Callable[[str, list[str]], list[float]] | None = None,
    ):
        self.rows, self.parents = rows, parents
        self.qdrant, self.collection = qdrant, collection
        self.embed_query, self.rerank_fn = embed_query, rerank_fn
        self.years = np.array([int(r["published"][:4]) for r in rows])
        self.bm25 = BM25Okapi([tokenize(r.get("text") or r["raw_text"]) for r in rows])

    def __len__(self) -> int:
        return len(self.rows)

    # ---------- Dense ----------
    def dense(self, query: str, n: int, year_from: int | None = None) -> list[int]:
        flt = None
        if year_from:
            flt = Filter(must=[FieldCondition(key="year", range=Range(gte=year_from))])
        res = self.qdrant.query_points(
            self.collection, query=self.embed_query(query), limit=n, query_filter=flt
        )
        return [int(p.id) for p in res.points]

    # ---------- Sparse ----------
    def sparse(self, query: str, n: int, year_from: int | None = None) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        if year_from:
            scores = np.where(self.years >= year_from, scores, -1)
        top = np.argsort(-scores)[:n]
        return [int(i) for i in top if scores[i] > 0]

    # ---------- Rerank ----------
    def rerank(self, query: str, ids: list[int], top_k: int) -> list[tuple[int, float | None]]:
        if not self.rerank_fn or not ids:
            return [(i, None) for i in ids[:top_k]]
        texts = [
            f"{self.rows[i]['title']} | {self.rows[i]['section']}\n{self.rows[i]['raw_text']}" for i in ids
        ]
        scores = self.rerank_fn(query, texts)
        order = np.argsort(-np.asarray(scores))[:top_k]
        return [(ids[j], float(scores[j])) for j in order]

    # ---------- Pipeline ----------
    def search(
        self,
        query: str,
        top_k: int = 3,
        n_candidates: int = 30,
        hybrid: bool = True,
        rerank: bool = True,
        expand_parents: bool = False,
        year_from: int | None = None,
    ) -> list[dict]:
        lists = [self.dense(query, n_candidates, year_from)]
        if hybrid:
            lists.append(self.sparse(query, n_candidates, year_from))
        fused = [i for i, _ in rrf(lists)][:n_candidates]

        if rerank:
            ranked = self.rerank(query, fused, top_k)
        else:
            ranked = [(i, None) for i in fused[:top_k]]

        results = []
        for i, score in ranked:
            r = self.rows[i]
            results.append({
                "idx": i, "chunk_id": r["chunk_id"], "arxiv_id": r["arxiv_id"], "title": r["title"],
                "page": r["page"], "section": r.get("section", ""), "text": r["raw_text"],
                "parent_id": r.get("parent_id"), "score": score,
            })

        if expand_parents:  # small-to-big: نبحث بالصغير ونبعت الـ parent
            seen, expanded = set(), []
            for r in results:
                pid = r["parent_id"]
                if pid in seen:
                    continue
                seen.add(pid)
                expanded.append({**r, "text": self.parents.get(pid, r["text"])})
            results = expanded
        return results


# ---------- Loaders للموديلات الحقيقية (lazy imports عشان الاختبارات ما تحتاجهاش) ----------
def load_embedder(model_name: str, query_prefix: str, device: str | None = None) -> Callable[[str], list[float]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = 512

    def embed_query(q: str) -> list[float]:
        return model.encode(query_prefix + q, normalize_embeddings=True).tolist()

    return embed_query


def load_reranker(model_name: str, device: str | None = None) -> Callable[[str, list[str]], list[float]]:
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name, device=device, max_length=512)

    def rerank_fn(query: str, texts: list[str]) -> list[float]:
        return model.predict([(query, t) for t in texts], batch_size=16, show_progress_bar=False).tolist()

    return rerank_fn
