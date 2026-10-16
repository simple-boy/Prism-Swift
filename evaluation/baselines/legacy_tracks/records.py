"""Shared processor interface. Content is evidence; index text is never an answer."""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PreparedEvidence:
    content: str | list[dict[str, Any]]
    index_text: str
    features: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreparedQuery:
    text: str
    features: dict[str, Any] = field(default_factory=dict)


class ProviderError(RuntimeError):
    """An unavailable model must fail the request, rather than acknowledge lost work."""


class PayloadTooLarge(ValueError):
    pass
