"""بيبني فهرس Qdrant من ملف الـ embeddings أوتوماتيك أول مرة التطبيق يشتغل.

الـ id لكل point = رقم الصف في ملف الـ chunks (الـ retriever بيعتمد على ده).
الفهرس بيتخزّن في data/qdrant، فالبناء بيحصل مرة واحدة بس.
"""
import logging
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

log = logging.getLogger("rag")


def ensure_index(qdrant: QdrantClient, collection: str, rows: list[dict], emb_path: Path, batch: int = 512) -> bool:
    """بيرجّع True لو بنى الفهرس، False لو كان جاهز ومطابق."""
    if qdrant.collection_exists(collection):
        if qdrant.count(collection).count == len(rows):
            return False
        log.warning("الفهرس مش مطابق لملف الـ chunks، هيتبني من جديد")
        qdrant.delete_collection(collection)

    if not emb_path.exists():
        raise RuntimeError(f"{emb_path} مش موجود. انسخه من Drive لمجلد data/ (راجع README)")
    emb = np.load(emb_path)
    if len(emb) != len(rows):
        raise RuntimeError(
            f"عدد الـ chunks ({len(rows)}) مختلف عن عدد الـ embeddings ({len(emb)}). "
            "الـ embeddings اتحسبت على نسخة تانية من الـ chunks، لازم الملفين من نفس التشغيل")

    log.info("building Qdrant index (%d vectors) ...", len(rows))
    qdrant.create_collection(collection, vectors_config=VectorParams(size=emb.shape[1], distance=Distance.COSINE))
    for i in range(0, len(rows), batch):
        qdrant.upsert(collection, points=[
            PointStruct(id=j, vector=emb[j].tolist(),
                        payload={"chunk_id": rows[j]["chunk_id"], "arxiv_id": rows[j]["arxiv_id"],
                                 "page": rows[j]["page"], "year": int(rows[j]["published"][:4])})
            for j in range(i, min(i + batch, len(rows)))])
    return True
