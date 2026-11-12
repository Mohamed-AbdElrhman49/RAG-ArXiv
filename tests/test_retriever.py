from app.rag.retriever import rrf, tokenize


def test_tokenize_keeps_compound_terms_and_drops_stopwords():
    assert tokenize("The bge-m3 and Self-RAG of 0.71") == ["bge-m3", "self-rag", "0.71"]


def test_rrf_rewards_agreement():
    fused = [i for i, _ in rrf([[1, 2, 3], [3, 1, 4]])]
    assert fused[:2] == [1, 3]          # ظاهرين في الليستين
    assert set(fused) == {1, 2, 3, 4}


def test_rrf_empty():
    assert rrf([[], []]) == []


def test_dense_finds_relevant_chunk(retriever):
    assert retriever.dense("hypothetical document query embeds", 3)[0] == 1


def test_sparse_exact_term(retriever):
    assert retriever.sparse("GraphRAG", 3) == [3]


def test_sparse_no_match_returns_empty(retriever):
    assert retriever.sparse("zzzzqqq", 3) == []


def test_search_hybrid_rerank_top1(retriever):
    res = retriever.search("cross-encoder reranker MRR", top_k=2, hybrid=True, rerank=True)
    assert res[0]["chunk_id"] == "c0"
    assert res[0]["score"] is not None and len(res) == 2


def test_search_without_rerank_has_no_score(retriever):
    res = retriever.search("chunk size overlap recall", top_k=2, rerank=False)
    assert res[0]["chunk_id"] == "c2" and res[0]["score"] is None


def test_year_filter_applies_to_dense_and_sparse(retriever):
    res = retriever.search("retrieval", top_k=5, rerank=False, year_from=2024)
    assert res and all(int(retriever.rows[r["idx"]]["published"][:4]) >= 2024 for r in res)


def test_parent_expansion_dedupes_same_parent(retriever):
    res = retriever.search("retrieval reranker hypothetical chunk", top_k=3, expand_parents=True, rerank=False)
    parents = [r["parent_id"] for r in res]
    assert len(parents) == len(set(parents))
    assert any("PARENT" in r["text"] for r in res)
