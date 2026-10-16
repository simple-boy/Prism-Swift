"""Explicit deployment limits; no model credential is used by this service."""
from __future__ import annotations

from dataclasses import dataclass, field
import os


def _integer(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _boolean(name: str, default: bool) -> bool:
    value = os.getenv(name, "1" if default else "0").strip().lower()
    if value not in {"1", "0", "true", "false"}:
        raise ValueError(f"{name} must be 1 or 0")
    return value in {"1", "true"}


@dataclass(frozen=True)
class Settings:
    db_path: str = "data/swift.sqlite3"
    api_key: str = field(default="", repr=False)
    strategy: str = "adaptive"
    max_add_concurrency: int = 2
    max_search_concurrency: int = 2
    max_request_bytes: int = 64 * 1024**2
    max_pixels: int = 16_000_000
    max_image_bytes: int = 10 * 1024**2
    max_media_bytes: int = 30 * 1024**2
    max_response_bytes: int = 64 * 1024**2
    max_results: int = 100
    min_free_bytes: int = 128 * 1024**2
    read_timeout_seconds: int = 120
    production: bool = True
    version: str = "0.1.0"
    commit: str = "unfixed"
    host: str = "0.0.0.0"
    port: int = 8000
    visual_reranker: str = "none"
    reranker_model_path: str = "models/tinyclip"
    reranker_threads: int = 2
    reranker_candidates: int = 32

    def __post_init__(self) -> None:
        if self.strategy not in {"raw", "sparse", "adaptive"}:
            raise ValueError("STRATEGY must be raw, sparse or adaptive")
        if self.visual_reranker not in {"none","tinyclip"}:
            raise ValueError("VISUAL_RERANKER must be none or tinyclip")
        if not 1 <= self.reranker_threads <= 8 or not 1 <= self.reranker_candidates <= 100:
            raise ValueError("Invalid local reranker capacity")
        for name in ("max_add_concurrency", "max_search_concurrency", "max_request_bytes",
                     "max_pixels", "max_image_bytes", "max_media_bytes", "max_response_bytes",
                     "max_results", "read_timeout_seconds", "port"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_free_bytes < 0 or self.port > 65535:
            raise ValueError("Invalid disk reserve or port")
        if self.max_image_bytes > 10 * 1024**2 or self.max_media_bytes > 30 * 1024**2:
            raise ValueError("Media limits cannot exceed the official decoded-byte limits")
        if self.max_results > 100:
            raise ValueError("max_results cannot exceed the declared service maximum 100")
        if self.production and not self.api_key.strip():
            raise ValueError("PRODUCTION=1 requires a non-empty MEMORY_API_KEY")
        if self.api_key != self.api_key.strip() or any(c in self.api_key for c in "\r\n"):
            raise ValueError("MEMORY_API_KEY cannot have surrounding whitespace or newlines")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            db_path=os.getenv("MEMORY_DB_PATH", "data/swift.sqlite3"),
            api_key=os.getenv("MEMORY_API_KEY", ""),
            strategy=os.getenv("STRATEGY", "adaptive"),
            max_add_concurrency=_integer("MAX_ADD_CONCURRENCY", 2),
            max_search_concurrency=_integer("MAX_SEARCH_CONCURRENCY", 2),
            max_request_bytes=_integer("MAX_REQUEST_BYTES", 64 * 1024**2),
            max_pixels=_integer("MAX_IMAGE_PIXELS", 16_000_000),
            max_image_bytes=_integer("MAX_IMAGE_BYTES", 10 * 1024**2),
            max_media_bytes=_integer("MAX_MEDIA_BYTES", 30 * 1024**2),
            max_response_bytes=_integer("MAX_RESPONSE_BYTES", 64 * 1024**2),
            max_results=_integer("MAX_RESULTS", 100),
            min_free_bytes=_integer("MIN_FREE_BYTES", 128 * 1024**2),
            read_timeout_seconds=_integer("READ_TIMEOUT_SECONDS", 120),
            production=_boolean("PRODUCTION", True),
            version=os.getenv("SYSTEM_VERSION", "0.1.0"),
            commit=os.getenv("SOURCE_COMMIT", "unfixed"),
            host=os.getenv("HOST", "0.0.0.0"),
            port=_integer("PORT", 8000),
            visual_reranker=os.getenv("VISUAL_RERANKER","none"),
            reranker_model_path=os.getenv("RERANKER_MODEL_PATH","models/tinyclip"),
            reranker_threads=_integer("RERANKER_THREADS",2),
            reranker_candidates=_integer("RERANKER_CANDIDATES",32),
        )
