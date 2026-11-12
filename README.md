# arXiv RAG

Advanced RAG over arXiv papers on retrieval-augmented generation, served with FastAPI.

```
question ─► Dense (bge-base, Qdrant) ┐
                                     ├─► RRF fusion ─► cross-encoder rerank ─► Groq LLM ─► answer + validated citations
            BM25 (in-memory) ────────┘
```

- **Hybrid retrieval**: dense + BM25 merged with Reciprocal Rank Fusion
- **Reranking**: `BAAI/bge-reranker-base` cross-encoder
- **Cited answers**: every answer is checked in code for missing / invalid citations; out-of-scope questions get a fixed refusal sentence
- **Streaming** over SSE, TTL cache for repeated questions, optional API key, tests
## Run (3 steps)

**1) Install**

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env                                   # then put your GROQ_API_KEY inside .env
```

**2) Copy the prepared data from Drive (`arxiv-rag/`) into `data/`:**

| From | To |
|---|---|
| `data/chunks/chunks_structure.jsonl` | `data/chunks_structure.jsonl` |
| `data/chunks/parents_structure.json` | `data/parents_structure.json` |
| `data/embeddings/structure__bge-base.npy` | `data/structure__bge-base.npy` |

**3) Start**

```bash
python run.py
```

Open http://127.0.0.1:8000 (demo page) or http://127.0.0.1:8000/docs (Swagger).
The first start downloads the models and builds the vector index in `data/qdrant` (one time only). Later starts are fast.

```bash
curl -s 127.0.0.1:8000/query -H 'content-type: application/json' \
  -d '{"question":"How does a cross-encoder reranker improve retrieval quality?"}' | python -m json.tool
```

## API

| Endpoint | Description |
|---|---|
| `POST /query` | answer + sources + citation checks + latency |
| `POST /query/stream` | SSE events: `sources`, `token`..., `done` (or `error`) |
| `GET /health` | readiness |

Request fields: `question`, `top_k`, `hybrid`, `rerank`, `expand_parents`, `year_from`.
Errors: `422` bad input, `429` Groq rate limit, `502` generation failed.

## Notes

- If you change the chunks file, delete `data/qdrant` (or just restart: a mismatch triggers an automatic rebuild). The chunks file and the `.npy` must come from the same run.
- Groq free tier is rate-limited per organization (see your console limits page), so repeated questions are cached.
- Reranking runs on CPU: 30 candidates by default (`N_CANDIDATES`). Lower it if latency matters.
- To share it on your network use `HOST=0.0.0.0 python run.py`.
