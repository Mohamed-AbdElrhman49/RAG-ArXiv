from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """كل الإعدادات بتتقرا من environment variables أو من ملف .env"""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Generation (Groq) ---
    groq_api_key: str = ""
    groq_model: str = "qwen/qwen3.8-27b"
    groq_temperature: float = 0.3
    # اتركه فاضي لو مش عايز تبعته. القيم المسموحة بتختلف من موديل لموديل، راجع docs بتاعة Groq
    groq_reasoning_effort: str | None = None
    max_new_tokens: int = 1500

    # --- Retrieval ---
    embed_model: str = "BAAI/bge-base-en-v1.5"
    embed_query_prefix: str = "Represent this sentence for searching relevant passages: "
    rerank_model: str = "BAAI/bge-reranker-base"
    use_reranker: bool = True
    top_k: int = 3
    n_candidates: int = 30  # على CPU الـ reranker بطيء، 30 أخف من 50

    # --- Data ---
    data_dir: Path = Path("data")
    strategy: str = "structure"
    collection: str = "structure__bge-base"
    qdrant_path: str = "data/qdrant"

    # --- Service ---
    service_api_key: str | None = None  # لو اتحطت، الـ endpoints بتطلب header: X-API-Key
    cache_size: int = 256
    cache_ttl_seconds: int = 3600
    cors_origins: str = "*"


@lru_cache
def get_settings() -> Settings:
    return Settings()
