"""StockMem module configuration."""

from shared.config.base_config import BaseAppConfig


class StockMemConfig(BaseAppConfig):
    """Configuration specific to the StockMem module."""

    vector_backend: str = "memory"  # "pgvector" | "faiss" | "memory"
    embedding_dimension: int = 128

    model_config = {"env_prefix": "STOCKMEM_", "extra": "ignore"}
