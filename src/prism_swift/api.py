"""Bounded, authenticated synchronous Add/Search contract over FastAPI."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hmac
import hashlib
import json
import logging
import sqlite3
import threading
import time
import uuid
from pathlib import Path

import anyio
from fastapi import FastAPI, Request
from starlette.requests import ClientDisconnect
from starlette.responses import Response

from .config import Settings
from .storage import MemoryStore
from .validation import ContractError

LOGGER = logging.getLogger("prism_swift")


def _json_bytes(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _error(status: int, reason: str, operation_id: str = "", retry_after: int | None = None) -> Response:
    headers = {"Cache-Control": "no-store"}
    if operation_id:
        headers["X-Chronicle-Request-ID"] = operation_id
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    return Response(_json_bytes({"detail": {"reason": reason}}), status_code=status,
                    media_type="application/json", headers=headers)


class Counters:
    """Aggregate operational statistics only; never persist bodies or identities."""
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.started_at = time.monotonic()
        self.completed: dict[str, int] = {}
        self.duration_ms = {"add": 0.0, "search": 0.0}
        self.active = {"add": 0, "search": 0}
        self.received_bytes = 0

    def begin(self, operation: str) -> None:
        with self.lock:
            self.active[operation] += 1

    def finish(self, operation: str, status: int, elapsed_ms: float, received_bytes: int) -> None:
        with self.lock:
            self.active[operation] -= 1
            label = f"{operation}_{status}"
            self.completed[label] = self.completed.get(label, 0) + 1
            self.duration_ms[operation] += elapsed_ms
            self.received_bytes += received_bytes

    def reject(self, operation: str, status: int) -> None:
        with self.lock:
            label = f"{operation}_{status}"
            self.completed[label] = self.completed.get(label, 0) + 1

    def snapshot(self) -> dict:
        with self.lock:
            return {"uptime_seconds": round(time.monotonic() - self.started_at, 3),
                    "completed": dict(self.completed), "active": dict(self.active),
                    "duration_ms_sum": {k: round(v, 3) for k, v in self.duration_ms.items()},
                    "received_bytes": self.received_bytes}


def _authorized(request: Request, key: str) -> bool:
    if not key:
        return True  # Explicitly available only through development Settings.
    values = [request.headers.get("x-api-key", "")]
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() in {"token", "bearer"}:
        values.append(token)
    encoded = key.encode("utf-8")
    return any(hmac.compare_digest(value.encode("utf-8"), encoded) for value in values)


def _unique_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("non_finite_json")


async def _read_json(request: Request, settings: Settings) -> tuple[dict, int]:
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type != "application/json":
        raise ContractError("content_type_must_be_application_json", status=415)
    encoding = request.headers.get("content-encoding", "identity").strip().lower()
    if encoding not in {"", "identity"}:
        raise ContractError("unsupported_content_encoding", status=415)
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            length = int(declared)
        except ValueError:
            raise ContractError("invalid_content_length", status=400) from None
        if length < 0:
            raise ContractError("invalid_content_length", status=400)
        if length > settings.max_request_bytes:
            raise ContractError("request_body_too_large", status=413)
    else:
        length = None
    body = bytearray()

    async def consume() -> None:
        async for chunk in request.stream():
            if len(body) + len(chunk) > settings.max_request_bytes:
                raise ContractError("request_body_too_large", status=413)
            body.extend(chunk)

    try:
        await asyncio.wait_for(consume(), timeout=settings.read_timeout_seconds)
    except asyncio.TimeoutError:
        raise ContractError("request_read_timeout", status=408) from None
    if length is not None and length != len(body):
        raise ContractError("content_length_mismatch", status=400)
    if not body:
        raise ContractError("json_object_required", status=422)
    try:
        payload = json.loads(body, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (ValueError, UnicodeError, RecursionError):
        raise ContractError("invalid_json", status=400) from None
    if not isinstance(payload, dict):
        raise ContractError("json_object_required", status=422)
    return payload, len(body)


def _safe_error(exc: Exception) -> tuple[int, str]:
    if isinstance(exc, ContractError):
        return exc.status, exc.reason
    if isinstance(exc, UnicodeError):
        return 422, "invalid_unicode"
    if isinstance(exc, sqlite3.OperationalError):
        code = getattr(exc, "sqlite_errorcode", 0)
        primary = code & 0xFF if isinstance(code, int) else 0
        if primary in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED, sqlite3.SQLITE_FULL,
                       sqlite3.SQLITE_IOERR, sqlite3.SQLITE_CANTOPEN, sqlite3.SQLITE_READONLY}:
            return 503, "storage_unavailable"
    if isinstance(exc, OSError):
        return 503, "storage_unavailable"
    return 500, "internal_error"


def create_app(settings: Settings | None = None, store: MemoryStore | None = None) -> FastAPI:
    if not LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    settings = settings or Settings.from_env()
    runtime_hash = hashlib.sha256()
    package_root = Path(__file__).resolve().parent
    for path in sorted(package_root.glob("*.py")):
        runtime_hash.update(path.name.encode()+b"\0"+hashlib.sha256(path.read_bytes()).digest())
    source_fingerprint = runtime_hash.hexdigest()
    reranker = None
    if store is None and settings.visual_reranker == "tinyclip":
        from .visual_reranker import TinyCLIPReranker
        reranker = TinyCLIPReranker(settings.reranker_model_path,threads=settings.reranker_threads,
                                   candidates=settings.reranker_candidates)
    store = store or MemoryStore(
        settings.db_path, strategy=settings.strategy, max_pixels=settings.max_pixels,
        max_image_bytes=settings.max_image_bytes, max_media_bytes=settings.max_media_bytes,
        response_bytes=settings.max_response_bytes, max_results=settings.max_results,
        min_free_bytes=settings.min_free_bytes,reranker=reranker)
    slots = {"add": threading.BoundedSemaphore(settings.max_add_concurrency),
             "search": threading.BoundedSemaphore(settings.max_search_concurrency)}
    counters = Counters()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        store.close()

    app = FastAPI(title="Prism-Swift", version=settings.version, lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store, app.state.settings, app.state.counters = store, settings, counters

    @app.get("/health")
    async def health() -> Response:
        def check() -> tuple[int, bytes]:
            try:
                status = store.health()
                return 200, _json_bytes({**status, "service": "prism-swift", "track": "multimodal",
                                         "version": settings.version, "commit": settings.commit,
                                         "source_sha256": source_fingerprint,
                                         "strategy": settings.strategy, "model_calls": 0,
                                         "submission_ready": False})
            except Exception:
                return 503, _json_bytes({"status": "unavailable", "service": "prism-swift"})
            finally:
                try:
                    store.close()
                except Exception:
                    LOGGER.error('{"event":"health_cleanup_error"}')
        status, body = await anyio.to_thread.run_sync(check)
        return Response(body, status_code=status, media_type="application/json", headers={"Cache-Control": "no-store"})

    @app.get("/metrics")
    async def metrics(request: Request) -> Response:
        if not _authorized(request, settings.api_key):
            return _error(401, "unauthorized")
        def read_metrics() -> dict:
            try:
                return store.metrics() if hasattr(store, "metrics") else {}
            finally:
                store.close()
        try:
            storage_metrics = await anyio.to_thread.run_sync(read_metrics)
        except Exception:
            return _error(503, "storage_unavailable", retry_after=1)
        return Response(_json_bytes({**counters.snapshot(), "storage": storage_metrics}), media_type="application/json",
                        headers={"Cache-Control": "no-store"})

    async def operate(request: Request, operation: str) -> Response:
        operation_id = uuid.uuid4().hex
        started = time.monotonic()
        if not _authorized(request, settings.api_key):
            counters.reject(operation, 401)
            LOGGER.info(json.dumps({"event": "request_rejected", "operation_id": operation_id,
                                    "operation": operation, "status": 401, "reason": "unauthorized"}))
            return _error(401, "unauthorized", operation_id)
        if not slots[operation].acquire(blocking=False):
            counters.reject(operation, 429)
            LOGGER.info(json.dumps({"event": "request_rejected", "operation_id": operation_id,
                                    "operation": operation, "status": 429, "reason": "concurrency_limit"}))
            return _error(429, "concurrency_limit", operation_id, retry_after=1)
        counters.begin(operation)
        received_bytes = 0
        executed = False
        status = 500
        reason = "internal_error"

        def execute(payload: dict) -> tuple[int, bytes, str]:
            try:
                result = store.add(payload) if operation == "add" else store.search(payload)
                body = _json_bytes(result)
                if len(body) > settings.max_response_bytes:
                    return 503, _json_bytes({"detail": {"reason": "response_body_limit"}}), "response_body_limit"
                return 200, body, "ok"
            except Exception as exc:
                result_status, result_reason = _safe_error(exc)
                return result_status, _json_bytes({"detail": {"reason": result_reason}}), result_reason
            finally:
                try:
                    store.close()  # Same worker thread as the operation.
                except Exception:
                    LOGGER.error(json.dumps({"event": "cleanup_error", "operation_id": operation_id,
                                             "operation": operation}, separators=(",", ":")))
                finally:
                    slots[operation].release()  # Released before ASGI can send a response.

        try:
            payload, received_bytes = await _read_json(request, settings)
            with anyio.CancelScope(shield=True):
                executed = True
                status, body, reason = await anyio.to_thread.run_sync(execute, payload)
            return Response(body, status_code=status, media_type="application/json", headers={
                "Cache-Control": "no-store", "X-Chronicle-Request-ID": operation_id,
                **({"Retry-After": "1"} if status == 503 else {})})
        except ClientDisconnect:
            status, reason = 408, "client_disconnected"
            return _error(status, reason, operation_id)
        except Exception as exc:
            status, reason = _safe_error(exc)
            return _error(status, reason, operation_id)
        finally:
            if not executed:
                slots[operation].release()
            elapsed_ms = (time.monotonic() - started) * 1000
            counters.finish(operation, status, elapsed_ms, received_bytes)
            LOGGER.info(json.dumps({"event": "request_completed", "operation_id": operation_id,
                                    "operation": operation, "status": status, "reason": reason,
                                    "elapsed_ms": round(elapsed_ms, 3), "request_bytes": received_bytes},
                                   separators=(",", ":")))

    @app.post("/add")
    async def add(request: Request) -> Response:
        return await operate(request, "add")

    @app.post("/search")
    async def search(request: Request) -> Response:
        return await operate(request, "search")

    return app


def main() -> None:
    import uvicorn
    settings = Settings.from_env()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, workers=1,
                access_log=False, timeout_keep_alive=5, timeout_graceful_shutdown=1800)


if __name__ == "__main__":
    main()
