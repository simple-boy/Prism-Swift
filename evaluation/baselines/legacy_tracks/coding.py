"""Source-preserving coding memory preparation and query-conditioned ranking.

This processor parses only supplied text. It never runs code, fetches a repository,
or calls a model during Search. Optional model hints are retrieval indexes, not
returned evidence.
"""

from __future__ import annotations

import re
from typing import Any

from records import PreparedEvidence, PreparedQuery, ProviderError


_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_WORDS_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:[A-Za-z]:[\\/]|\.{1,2}[\\/]|/)?"
    r"(?:[A-Za-z0-9_.@+~-]+[\\/])+[A-Za-z0-9_.@+~-]+(?::\d+(?::\d+)?)?"
    r"|(?<![A-Za-z0-9_])(?:[A-Za-z0-9_@+~-]+\.)+"
    r"(?:pyi?|ipynb|js|jsx|ts|tsx|java|kt|kts|c|cc|cpp|cxx|h|hpp|cs|go|rs|rb|php|"
    r"swift|scala|sh|ps1|bat|cmd|sql|json|ya?ml|toml|ini|cfg|xml|html?|css|scss|md|"
    r"rst|txt|lock|proto|vue|svelte)(?::\d+(?::\d+)?)?\b",
    re.IGNORECASE,
)
_ERROR_RE = re.compile(
    r"\b[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception|Failure)\b"
    r"|\b(?:E[A-Z][A-Z0-9_]{2,}|ERR_[A-Z0-9_]+|TS\d{3,}|CS\d{3,}|"
    r"SIG(?:SEGV|ABRT|BUS|ILL|FPE))\b"
)
_TRACE_RE = re.compile(
    r"Traceback\s*\(|stack\s*trace|stacktrace|\bFile\s+[\"'].+?[\"'],\s*line\s+\d+"
    r"|^\s*at\s+\S+\s*\(.+?:\d+", re.IGNORECASE | re.MULTILINE,
)
_SUCCESS_RE = re.compile(
    r"\b\d+\s+passed\b|\b(?:all\s+)?tests?\s+(?:passed|pass|successful)\b"
    r"|\b(?:build|compilation)\s+(?:succeeded|successful)\b"
    r"|\b(?:exit|return)\s+(?:code|status)\s*[:=]?\s*0\b|\b0\s+failed\b",
    re.IGNORECASE,
)
_FAILURE_RE = re.compile(
    r"\b[1-9]\d*\s+failed\b|\b(?:tests?|build|compilation)\s+(?:failed|failing)\b"
    r"|\btests?\s+(?:did\s+not|do\s+not|never)\s+pass\b"
    r"|\b(?:exit|return)\s+(?:code|status)\s*[:=]?\s*[1-9]\d*\b",
    re.IGNORECASE,
)
_BOILERPLATE_RE = re.compile(
    r"^\s*(?:\[[^\]\r\n]+\]\s*)?(?:INFO\s+|DEBUG\s+)?"
    r"(?:heartbeat|keep[- ]?alive|poll(?:ing)?\b|downloading\b|uploading\b|"
    r"fetching\b|progress\b|\d{1,3}%\s*[|\[])"
    r"|\b\d{1,3}%\s*\|[=\-# >]+\|", re.IGNORECASE,
)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "can", "could",
    "did", "do", "does", "for", "from", "how", "i", "in", "is", "it", "its",
    "me", "my", "of", "on", "or", "our", "please", "show", "that", "the", "this",
    "to", "was", "we", "were", "what", "where", "which", "who", "why", "with",
    "would", "you", "your", "find", "history", "previous", "memory", "memories",
    "的", "了", "是", "吗", "请", "我", "哪", "个", "么", "什", "如", "何",
}
_CALL_KEYWORDS = {
    "if", "for", "while", "switch", "catch", "return", "print", "with", "assert",
    "sizeof", "typeof", "super", "require", "import",
}
_LANGUAGE_PATTERNS = {
    "cplusplus": re.compile(r"(?<!\w)c\+\+(?:\d{2})?(?![\w+])|\b(?:cplusplus|cpp|cxx)\b", re.IGNORECASE),
    "csharp": re.compile(r"(?<!\w)c#(?![\w#])|\bcsharp\b", re.IGNORECASE),
    # Preserve a separate token for the standalone uppercase language name. A
    # drive prefix or a component of a path must not be treated as the C language.
    "clanguage": re.compile(r"(?<![\w./\\])C(?![\w:/\\+#])|\bc\s+(?:language|compiler|programming)\b"),
}


def _split_identifier(identifier: str) -> list[str]:
    return [
        part.casefold()
        for segment in identifier.split("_")
        for part in _CAMEL_BOUNDARY_RE.split(segment)
        if part
    ]


def _terms(text: str) -> set[str]:
    values = set(_WORDS_RE.findall(text.casefold()))
    for identifier in _IDENTIFIER_RE.findall(text):
        values.update(_split_identifier(identifier))
    return values - _STOPWORDS


def _paths(text: str) -> set[str]:
    values: set[str] = set()
    for match in _PATH_RE.finditer(text):
        # A URL is not a local source-code path. No network lookup is performed.
        if text[max(0, match.start() - 3):match.start()] == "://":
            continue
        value = re.sub(r":\d+(?::\d+)?$", "", match.group(0))
        value = value.replace("\\", "/").rstrip(".").casefold()
        while value.startswith("./"):
            value = value[2:]
        if value:
            values.add(value)
    return values


def _symbols(text: str) -> set[str]:
    values = {
        identifier.casefold() for identifier in _IDENTIFIER_RE.findall(text)
        if ("_" in identifier and len(identifier.strip("_")) > 1)
        or bool(re.search(r"[a-z0-9][A-Z]|[A-Z]{2}[a-z]", identifier))
    }
    for pattern in (
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        r"\b(?:def|function|class|interface|struct|enum|fn|func|trait|const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)",
        r"\b[A-Za-z_][A-Za-z0-9_]*\.([A-Za-z_][A-Za-z0-9_]*)\b",
        r"`([A-Za-z_][A-Za-z0-9_]*)`",
    ):
        values.update(value.casefold() for value in re.findall(pattern, text)
                      if value.casefold() not in _CALL_KEYWORDS)
    return values


def _success_cues(text: str) -> list[str]:
    values: list[str] = []
    for match in _SUCCESS_RE.finditer(text):
        prefix = text[max(0, match.start() - 30):match.start()]
        if re.search(r"\b(?:not|never|no|without)\s+(?:all\s+)?$", prefix, re.IGNORECASE):
            continue
        values.append(match.group(0))
    return values


def _noise_features(text: str) -> tuple[float, list[str]]:
    """Describe repeated/progress text; irrelevance is decided only with a query."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0, []
    boilerplate = sum(bool(_BOILERPLATE_RE.search(line)) for line in lines)
    repetition = max(0, len(lines) - len(set(lines))) / len(lines)
    noise = min(1.0, 0.65 * boilerplate / len(lines) + 0.35 * repetition)
    cues = []
    if boilerplate:
        cues.append("progress_or_heartbeat")
    if repetition:
        cues.append("repeated_lines")
    return noise, cues


def _features(text: str) -> dict[str, Any]:
    noise, noise_cues = _noise_features(text)
    languages = {name: [match.group(0) for match in pattern.finditer(text)]
                 for name, pattern in _LANGUAGE_PATTERNS.items()}
    language_aliases = sorted(name for name, literals in languages.items() if literals)
    return {
        "terms": sorted(_terms(text) | set(language_aliases)),
        "languages": language_aliases,
        "language_literals": [literal for literals in languages.values() for literal in literals],
        "paths": sorted(_paths(text)),
        "symbols": sorted(_symbols(text)),
        "errors": sorted({value.casefold() for value in _ERROR_RE.findall(text)
                          if value not in {"ERROR", "ERRORS"}}),
        "stacktrace": bool(_TRACE_RE.search(text)),
        # These are observed textual cues, not claims that this processor ran tests.
        "success_cues": _success_cues(text),
        "failure_cues": [match.group(0) for match in _FAILURE_RE.finditer(text)],
        "noise": noise,
        "noise_cues": noise_cues,
    }


def _expanded_index(text: str, features: dict[str, Any], derived: str = "") -> str:
    # Keep exact terms and split identifier/path terms for tokenizer-based search.
    pieces = set(features["terms"])
    for value in features["paths"] + features["symbols"] + features["errors"]:
        pieces.add(value)
        pieces.update(_terms(value))
    additions = " ".join(sorted(pieces))
    if not text.strip() and not additions and not derived:
        # This marks a preserved source span, not a semantic property of the text.
        additions = "_source_whitespace_"
    return text + "\n" + additions + ("\n" + derived if derived else "")


class CodingProcessor:
    name = "coding"

    def __init__(self, model: Any = None, chunk_chars: int = 8000) -> None:
        if isinstance(chunk_chars, bool) or not isinstance(chunk_chars, int) or chunk_chars < 1:
            raise ValueError("chunk_chars must be a positive integer")
        self.model = model
        self.chunk_chars = chunk_chars

    def config_dict(self) -> dict[str, Any]:
        return {"name": self.name, "version": 2, "chunk_chars": self.chunk_chars,
                "model_index_enabled": self.model is not None}

    def _chunks(self, content: str):
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Coding content must be a non-empty string")
        start = 0
        while start < len(content):
            end = min(len(content), start + self.chunk_chars)
            if end < len(content):
                # Prefer a nearby line boundary without overlap or dropped whitespace.
                boundary = content.rfind("\n", start + self.chunk_chars * 3 // 4, end)
                if boundary >= start:
                    end = boundary + 1
            chunk = content[start:end]
            yield start, end, chunk
            start = end

    def planned_calls(self, content: str) -> int:
        return sum(bool(chunk.strip()) for _, _, chunk in self._chunks(content))

    def prepare(self, content: str) -> list[PreparedEvidence]:
        records: list[PreparedEvidence] = []
        for start, end, chunk in self._chunks(content):
            features = _features(chunk)
            features.update({"source_start": start, "source_end": end,
                             "chunk_index": len(records), "derived_terms": []})
            derived = ""
            if self.model is not None and chunk.strip():
                derived = self.model.index_code(chunk)
                if not isinstance(derived, str):
                    raise ProviderError("index_code must return a string of index hints")
                features["derived_terms"] = sorted(_terms(derived))
            records.append(PreparedEvidence(chunk, _expanded_index(chunk, features, derived), features))
        return records

    def query(self, query: str, options: list[str] | None = None) -> PreparedQuery:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Coding query must be a non-empty string")
        if options is not None and (not isinstance(options, list)
                                    or any(not isinstance(value, str) or not value.strip()
                                           for value in options)):
            raise ValueError("options must be a list of non-empty strings or None")
        features = _features(query)
        features["option_terms"] = sorted(_terms("\n".join(options or [])))
        features["validation_intent"] = bool(re.search(
            r"\b(?:test|tests|verify|verification|validate|validation|regression|passed)\b"
            r"|验证|测试|回归", query, re.IGNORECASE))
        features["debug_intent"] = bool(features["errors"] or re.search(
            r"\b(?:debug|failure|failed|traceback|exception|crash|error)\b|报错|异常|失败",
            query, re.IGNORECASE))
        index = _expanded_index(query, features)
        if options:
            index += "\n" + "\n".join(options) + "\n" + " ".join(features["option_terms"])
        return PreparedQuery(index, features)

    def score(self, query_features: dict, record_features: dict) -> float:
        q_terms = set(query_features.get("terms", []))
        r_terms = set(record_features.get("terms", []))
        if not q_terms:
            return 0.0
        coverage = len(q_terms & r_terms) / len(q_terms)
        symbol_queries = set(query_features.get("symbols", []))
        error_queries = set(query_features.get("errors", []))
        symbol_overlap = len(symbol_queries & set(record_features.get("symbols", []))) / max(1, len(symbol_queries))
        error_overlap = len(error_queries & set(record_features.get("errors", []))) / max(1, len(error_queries))
        path_queries = set(query_features.get("paths", []))
        record_paths = set(record_features.get("paths", []))
        path_matches = []
        for path in path_queries:
            if path in record_paths:
                path_matches.append(1.0)
            elif any(path.endswith("/" + item) or item.endswith("/" + path) for item in record_paths):
                path_matches.append(0.75)
            elif any(path.rsplit("/", 1)[-1] == item.rsplit("/", 1)[-1] for item in record_paths):
                path_matches.append(0.25)
            else:
                path_matches.append(0.0)
        path_overlap = sum(path_matches) / max(1, len(path_matches))
        anchor_overlap = 4.0 * symbol_overlap + 3.0 * error_overlap + 3.0 * path_overlap
        score = 3.0 * coverage + anchor_overlap
        language_queries = set(query_features.get("languages", []))
        language_overlap = len(language_queries & set(record_features.get("languages", []))) / max(1, len(language_queries))
        score += 0.75 * language_overlap
        derived_overlap = len(q_terms & set(record_features.get("derived_terms", []))) / len(q_terms)
        score += 0.5 * derived_overlap
        option_terms = set(query_features.get("option_terms", []))
        score += 0.25 * len(option_terms & r_terms) / max(1, len(option_terms))
        # Failed attempts and traces are useful evidence. Only query-weak boilerplate
        # is discounted; a matching error/symbol/path protects diagnostic logs.
        if not anchor_overlap:
            noise = max(0.0, min(1.0, float(record_features.get("noise", 0.0))))
            score *= 1.0 - 0.7 * noise * (1.0 - min(1.0, coverage * 2.0))
        if score > 0 and query_features.get("validation_intent") and record_features.get("success_cues"):
            score += 0.15
        if score > 0 and query_features.get("debug_intent") and record_features.get("stacktrace"):
            score += 0.1
        return float(score)

    def media_bytes(self, content: Any) -> int:
        return 0
