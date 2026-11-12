import numpy as np
import pytest
from qdrant_client import QdrantClient

from app.index import ensure_index
from tests.conftest import fake_embed


def _emb(rows, path):
    np.save(path, np.array([fake_embed(r["text"]) for r in rows], dtype="float32"))
    return path


def test_builds_once_then_reuses(rows, tmp_path):
    q, p = QdrantClient(":memory:"), _emb(rows, tmp_path / "e.npy")
    assert ensure_index(q, "c", rows, p) is True
    assert q.count("c").count == len(rows)
    assert ensure_index(q, "c", rows, p) is False       # جاهز، مفيش إعادة بناء


def test_point_id_is_row_number(rows, tmp_path):
    q = QdrantClient(":memory:")
    ensure_index(q, "c", rows, _emb(rows, tmp_path / "e.npy"))
    pts = q.retrieve("c", ids=[2])
    assert pts[0].payload["chunk_id"] == rows[2]["chunk_id"]


def test_rebuilds_when_chunks_changed(rows, tmp_path):
    q, p = QdrantClient(":memory:"), _emb(rows, tmp_path / "e.npy")
    ensure_index(q, "c", rows, p)
    assert ensure_index(q, "c", rows[:3], _emb(rows[:3], tmp_path / "e3.npy")) is True
    assert q.count("c").count == 3


def test_missing_or_mismatched_embeddings_fail_clearly(rows, tmp_path):
    with pytest.raises(RuntimeError, match="مش موجود"):
        ensure_index(QdrantClient(":memory:"), "c", rows, tmp_path / "nope.npy")
    with pytest.raises(RuntimeError, match="مختلف"):
        ensure_index(QdrantClient(":memory:"), "c", rows, _emb(rows[:2], tmp_path / "e2.npy"))
