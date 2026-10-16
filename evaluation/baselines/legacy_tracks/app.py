"""Participant-hosted Coding/Multimodal Add/Search. One frozen track per service."""
from __future__ import annotations

import hmac
import json
import os
import re
import signal
import socket
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from coding import CodingProcessor
from models import ModelClient
from multimodal import MultimodalProcessor
from records import PayloadTooLarge, ProviderError
from store import TrackStore


def _emit(event, **fields):
    """Operational metadata only: never include exception text, source lines or bodies."""
    try:
        print(json.dumps({"event": event, **fields}, ensure_ascii=True), flush=True)
    except Exception:
        pass  # Logging failure must not affect a durable write.


def _error_metadata(exc):
    fields = {"error_type": type(exc).__name__, "frames": [
        {"file": os.path.basename(frame.f_code.co_filename), "line": line,
         "function": frame.f_code.co_name}
        for frame, line in traceback.walk_tb(exc.__traceback__)
    ][-4:]}
    name = getattr(exc, "sqlite_errorname", None)
    if isinstance(name, str) and re.fullmatch(r"SQLITE_[A-Z0-9_]+", name):
        fields["sqlite_errorname"] = name
    if isinstance(exc, ProviderError):
        match = re.fullmatch(r"Index model HTTP (\d{3}); no write acknowledged", str(exc))
        if match:
            fields["provider_http_status"] = int(match[1])
        reasons = {
            "Index model connection unavailable": "model_connection_unavailable",
            "Add model time budget exhausted; no write acknowledged": "model_time_budget",
            "Add model-call budget exhausted; no write acknowledged": "model_call_budget",
            "Model returned incomplete index text": "model_incomplete_index",
            "Invalid index-model response": "model_invalid_response",
        }
        if str(exc) in reasons:
            fields["reason"] = reasons[str(exc)]
    return fields


def create_server(host="127.0.0.1", port=8000, *, track="coding", db_path="data/coding.sqlite3",
                  api_key="", model=None, max_request_bytes=64 * 1024 * 1024,
                  add_concurrency=1, search_concurrency=4, temporal=True):
    if track not in {"coding", "multimodal"}:
        raise ValueError("MEMORY_TRACK must be coding or multimodal")
    if min(max_request_bytes, add_concurrency, search_concurrency) < 1:
        raise ValueError("Request byte limit and concurrency must be positive")
    processor = CodingProcessor(model=model) if track == "coding" else MultimodalProcessor(model=model)
    store = TrackStore(db_path, processor, temporal=temporal)
    add_slots = threading.BoundedSemaphore(add_concurrency)
    search_slots = threading.BoundedSemaphore(search_concurrency)

    class Handler(BaseHTTPRequestHandler):
        server_version = "ChronicleTracks/1.0"

        def log_message(self, fmt, *args):
            # Request bodies, queries, identities and credentials are never logged.
            return

        def event(self, name, **fields):
            if not hasattr(self, "_operation_id"):
                return
            _emit(name, operation_id=self._operation_id, track=track,
                  path=self._safe_path, phase=self._phase,
                  elapsed_ms=round((time.monotonic() - self._started) * 1000, 2), **fields)

        def set_phase(self, phase):
            self._phase = phase
            self.event("request_phase")

        def setup(self):
            super().setup()
            self.connection.settimeout(120)

        def send_json(self, status, body, retry_after=None):
            data = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                if hasattr(self, "_operation_id"):
                    self.send_header("X-Chronicle-Request-ID", self._operation_id)
                if retry_after is not None:
                    self.send_header("Retry-After", str(retry_after))
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError) as exc:
                self.close_connection = True
                self.event("request_finished", status=status, response_sent=False,
                           outcome="client_disconnected", error_type=type(exc).__name__)
                return False
            self.event("request_finished", status=status, response_sent=True, outcome="sent")
            return True

        def do_GET(self):
            if self.path.split("?", 1)[0] == "/health":
                self.send_json(200, {"status": "ok", "service": "chronicle-memory-tracks", "track": track})
            else:
                self.send_json(404, {"detail": {"reason": "not_found"}})

        def authorized(self):
            if not api_key:
                return True
            values = [self.headers.get("X-Api-Key", "")]
            auth = self.headers.get("Authorization", "")
            if auth.lower().startswith(("token ", "bearer ")):
                values.append(auth.split(" ", 1)[1])
            return any(hmac.compare_digest(value.encode("utf-8"), api_key.encode("utf-8")) for value in values)

        def do_POST(self):
            self._operation_id = uuid.uuid4().hex
            self._started = time.monotonic()
            self._phase = "authentication"
            path = self.path.split("?", 1)[0]
            self._safe_path = path if path in {"/add", "/search"} else "other"
            self.event("request_started")
            if not self.authorized():
                self.send_json(401, {"detail": {"reason": "unauthorized"}})
                return
            if path not in {"/add", "/search"}:
                self.send_json(404, {"detail": {"reason": "not_found"}})
                return
            slots = add_slots if path == "/add" else search_slots
            if not slots.acquire(blocking=False):
                self._phase = "admission"
                self.send_json(429, {"detail": {"reason": "concurrency_limit"}}, retry_after=1)
                return
            try:
                self.set_phase("reading_request")
                length = int(self.headers.get("Content-Length", "0"))
                if length > max_request_bytes:
                    raise PayloadTooLarge("Request body exceeds declared byte limit")
                if length <= 0:
                    raise ValueError("A JSON request body is required")
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("Incomplete request body")
                body = json.loads(raw)
                if not isinstance(body, dict):
                    raise ValueError("JSON object required")
                if path == "/add":
                    self.set_phase("validating_add")
                    for field in ("request_id", "user_id", "session_id"):
                        if not isinstance(body.get(field), str) or not body[field].strip():
                            raise ValueError(f"{field} is required")
                    store.add_messages(request_id=body["request_id"], user_id=body["user_id"],
                                       session_id=body["session_id"], messages=body.get("messages"),
                                       on_progress=self.set_phase)
                    result = {"success": True, **{k: body[k] for k in ("request_id", "user_id", "session_id")}}
                else:
                    self.set_phase("searching")
                    result = {"data": store.search(user_id=body.get("user_id"), query=body.get("query"),
                              top_k=body.get("top_k"), options=body.get("options"))}
                    self._phase = "searched"
            except PayloadTooLarge as exc:
                self.event("request_error", status=413, **_error_metadata(exc))
                status, result = 413, {"detail": {"reason": str(exc)}}
            except (ValueError, UnicodeError) as exc:
                status = 409 if "request_id was reused" in str(exc) else 422
                self.event("request_error", status=status, **_error_metadata(exc))
                result = {"detail": {"reason": str(exc)}}
            except ProviderError as exc:
                self.event("request_error", status=503, **_error_metadata(exc))
                status, result = 503, {"detail": {"reason": str(exc)}}
            except (TimeoutError, socket.timeout) as exc:
                self.event("request_error", status=408, **_error_metadata(exc))
                status, result = 408, {"detail": {"reason": "request_read_timeout"}}
            except Exception as exc:
                self.event("request_error", status=500, **_error_metadata(exc))
                status, result = 500, {"detail": {"reason": "internal_error"}}
            else:
                status = 200
            # Complete database cleanup and release processing capacity before
            # the caller can receive a response and issue its next request.
            try:
                store.close()  # Close only this request-thread connection.
            except Exception as exc:
                self.event("cleanup_error", **_error_metadata(exc))
            finally:
                slots.release()
            self.send_json(status, result)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = False  # server_close waits for active requests within the host's grace period.
    server.store = store
    return server


def main():
    track = os.getenv("MEMORY_TRACK", "coding")
    api_key = os.getenv("MEMORY_API_KEY", "").strip()
    model = ModelClient.from_env()
    if os.getenv("PRODUCTION", "0") == "1" and (not api_key or not model.enabled):
        raise ValueError("PRODUCTION=1 requires MEMORY_API_KEY and MODEL_PROVIDER=openai")
    server = create_server(host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")),
            track=track, db_path=os.getenv("MEMORY_DB_PATH", f"data/{track}.sqlite3"), api_key=api_key,
            model=model if model.enabled else None,
            max_request_bytes=int(os.getenv("MAX_REQUEST_BYTES", str(64 * 1024 * 1024))),
            add_concurrency=int(os.getenv("MAX_ADD_CONCURRENCY", "1")),
            search_concurrency=int(os.getenv("MAX_SEARCH_CONCURRENCY", "4")),
            temporal=os.getenv("TEMPORAL_RETRIEVAL", "1") == "1")
    print(f"Chronicle {track} listening on port {server.server_port}; model={model.mode}", flush=True)
    def stop_service(signum, frame):
        # shutdown must run outside serve_forever's thread. The host controls
        # its grace period; unfinished Add is never acknowledged as durable.
        threading.Thread(target=server.shutdown, daemon=True).start()
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, stop_service)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        server.store.close()


if __name__ == "__main__":
    main()
