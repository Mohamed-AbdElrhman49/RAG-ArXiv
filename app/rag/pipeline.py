import re
import time
from typing import Iterator, Protocol

NO_ANSWER = "I could not find the answer in the provided documents."

SYSTEM_PROMPT = f"""You are a research assistant that answers questions about scientific papers.
Rules:
1. Use ONLY the numbered sources provided. Never use outside knowledge.
2. Every sentence MUST end with a citation like [1] or [2][3]. Only use numbers that exist in the sources.
3. Copy numbers exactly as written. Never compute or infer numbers. If unsure, omit the number.
4. If the sources do not contain the answer, reply exactly: "{NO_ANSWER}"

Example:
Sources: [1] ... "Adding a reranker raised MRR from 0.62 to 0.71."
Question: Does reranking help?
Answer: Yes, adding a reranker raised MRR from 0.62 to 0.71 [1]."""

_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


class Generator(Protocol):
    def chat(self, messages: list[dict], max_tokens: int) -> str: ...
    def stream(self, messages: list[dict], max_tokens: int) -> Iterator[str]: ...


def build_sources(results: list[dict], snippet_chars: int = 300) -> list[dict]:
    return [{
        "n": n, "title": r["title"], "arxiv_id": r["arxiv_id"], "page": r["page"],
        "section": r["section"] or "", "chunk_id": r["chunk_id"],
        "url": f"https://arxiv.org/abs/{r['arxiv_id']}",
        "snippet": " ".join(r["text"].split())[:snippet_chars],
    } for n, r in enumerate(results, 1)]


def build_messages(question: str, results: list[dict], max_chars_per_source: int = 3000) -> list[dict]:
    blocks = []
    for n, r in enumerate(results, 1):
        sec = f", section: {r['section']}" if r["section"] else ""
        blocks.append(f"[{n}] {r['title']} (arXiv:{r['arxiv_id']}, p.{r['page']}{sec})\n{r['text'][:max_chars_per_source]}")
    user = "Sources:\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\n\nAnswer:"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def check_citations(text: str, n_sources: int) -> tuple[list[int], list[int]]:
    cited = sorted({int(x) for m in _CITE.findall(text) for x in re.split(r"\s*,\s*", m)})
    valid = [n for n in cited if 1 <= n <= n_sources]
    return valid, [n for n in cited if n not in valid]


def is_refusal(text: str) -> bool:
    return NO_ANSWER.lower().rstrip(".") in text.lower()


class RagPipeline:
    def __init__(self, retriever, generator: Generator, *, top_k: int = 3, n_candidates: int = 30,
                 max_new_tokens: int = 1500):
        self.retriever, self.generator = retriever, generator
        self.top_k, self.n_candidates, self.max_new_tokens = top_k, n_candidates, max_new_tokens

    def _retrieve(self, question, top_k, hybrid, rerank, expand_parents, year_from):
        t0 = time.time()
        results = self.retriever.search(
            question, top_k=top_k or self.top_k, n_candidates=self.n_candidates, hybrid=hybrid,
            rerank=rerank, expand_parents=expand_parents, year_from=year_from)
        return results, time.time() - t0

    def _package(self, question, text, sources, t_ret, t_gen, t_total) -> dict:
        valid, invalid = check_citations(text, len(sources))
        refused = is_refusal(text)
        return {
            "question": question, "answer": text, "refused": refused,
            "citations": [s for s in sources if s["n"] in valid],
            "invalid_citations": invalid,
            "uncited": (not refused) and not valid,
            "retrieved": sources,
            "latency": {"retrieval": round(t_ret, 2), "generation": round(t_gen, 2), "total": round(t_total, 2)},
        }

    def answer(self, question: str, top_k: int | None = None, hybrid: bool = True, rerank: bool = True,
               expand_parents: bool = False, year_from: int | None = None) -> dict:
        t0 = time.time()
        results, t_ret = self._retrieve(question, top_k, hybrid, rerank, expand_parents, year_from)
        sources = build_sources(results)
        t1 = time.time()
        text = self.generator.chat(build_messages(question, results), self.max_new_tokens)
        return self._package(question, text, sources, t_ret, time.time() - t1, time.time() - t0)

    def stream(self, question: str, top_k: int | None = None, hybrid: bool = True, rerank: bool = True,
               expand_parents: bool = False, year_from: int | None = None) -> Iterator[tuple[str, dict]]:
        """بيطلّع events: ('sources', ...) ثم ('token', ...) كتير ثم ('done', النتيجة النهائية بعد التحقق من الـ citations)"""
        t0 = time.time()
        results, t_ret = self._retrieve(question, top_k, hybrid, rerank, expand_parents, year_from)
        sources = build_sources(results)
        yield "sources", {"retrieved": sources, "retrieval_latency": round(t_ret, 2)}
        t1, parts = time.time(), []
        for piece in self.generator.stream(build_messages(question, results), self.max_new_tokens):
            parts.append(piece)
            yield "token", {"text": piece}
        text = "".join(parts).strip()
        if not text:
            raise RuntimeError("empty answer from the model")
        yield "done", self._package(question, text, sources, t_ret, time.time() - t1, time.time() - t0)
