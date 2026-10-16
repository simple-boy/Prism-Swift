"""Public API contract, bounded admission, privacy, durability and image round-trips."""
from __future__ import annotations

import asyncio
import base64
from dataclasses import replace
import io
import json
import logging
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest import mock

import httpx
from PIL import Image

from prism_swift.api import create_app
from prism_swift.config import Settings
from prism_swift.storage import MemoryStore


def image_uri(fmt="PNG", color=(33, 77, 199), size=(32, 32)):
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, format=fmt)
    mime = {"PNG": "png", "JPEG": "jpeg", "WEBP": "webp"}[fmt]
    return "data:image/" + mime + ";base64," + base64.b64encode(output.getvalue()).decode()


def add_payload(request_id="request-a", user_id="user-a", content=None):
    return {"request_id": request_id, "user_id": user_id, "session_id": "session-a",
            "messages": [{"role": "user", "timestamp": 1704067200000,
                          "content": content or "Alice received the purple notebook."}]}


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.directory.name) / "api.sqlite3")
        self.settings = Settings(db_path=self.db_path, api_key="test-private-key", production=True,
                                 min_free_bytes=0, commit="a" * 40)
        self.app = create_app(self.settings)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test")
        self.headers = {"Authorization": "Token test-private-key"}

    async def asyncTearDown(self):
        await self.client.aclose()
        self.app.state.store.close()
        self.directory.cleanup()

    async def post(self, path, payload):
        return await self.client.post(path, json=payload, headers=self.headers)

    async def test_health_is_unauthenticated_safe_and_not_official_readiness(self):
        response = await self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        health = response.json()
        self.assertEqual(health["service"], "prism-swift")
        self.assertEqual(health["commit"], "a" * 40)
        self.assertFalse(health["submission_ready"])
        self.assertEqual(health["model_calls"], 0)
        self.assertNotIn(self.db_path, response.text)
        self.assertNotIn("test-private-key", response.text)
        self.assertEqual((await self.client.get("/metrics")).status_code, 401)

    async def test_all_three_authentication_schemes_and_unauthorized_body(self):
        self.assertEqual((await self.client.post("/add", content=b"bad-json")).status_code, 401)
        for headers in ({"Authorization": "Token test-private-key"},
                        {"Authorization": "Bearer test-private-key"}, {"X-Api-Key": "test-private-key"}):
            response = await self.client.post("/search", json={"user_id": "empty", "query": "notebook", "top_k": 1}, headers=headers)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"data": []})
        response = await self.client.post("/add", json=add_payload(), headers={"Authorization": "Token wrong"})
        self.assertEqual(response.status_code, 401)

    async def test_add_ack_and_immediate_cross_session_search(self):
        payload = add_payload()
        response = await self.post("/add", payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"success": True, "request_id": "request-a", "user_id": "user-a", "session_id": "session-a"})
        response = await self.post("/search", {"user_id": "user-a", "query": "purple notebook", "top_k": 100})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["data"]), 1)
        self.assertEqual(response.json()["data"][0]["content"], payload["messages"][0]["content"])
        second = add_payload("request-b", content="Bob borrowed a silver pen.")
        second["session_id"] = "session-b"
        self.assertEqual((await self.post("/add", second)).status_code, 200)
        self.assertTrue((await self.post("/search", {"user_id": "user-a", "query": "notebook", "top_k": 1})).json()["data"])

    async def test_ordered_multimodal_round_trip_and_image_query(self):
        for fmt in ("PNG", "JPEG", "WEBP"):
            uri = image_uri(fmt)
            parts = [{"type": "text", "text": "bluecard-" + fmt},
                     {"type": "image_url", "image_url": {"url": uri}},
                     {"type": "text", "text": "after the image"}]
            payload = add_payload("image-" + fmt, user_id="user-" + fmt, content=parts)
            self.assertEqual((await self.post("/add", payload)).status_code, 200)
            response = await self.post("/search", {"user_id": "user-" + fmt,
                                                   "query": [{"type": "image_url", "image_url": {"url": uri}}], "top_k": 1})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["data"][0]["content"], parts)

    async def test_idempotent_receipt_and_conflicting_reuse(self):
        payload = add_payload()
        first = await self.post("/add", payload)
        repeat = await self.post("/add", payload)
        self.assertEqual(first.json(), repeat.json())
        self.assertEqual(repeat.status_code, 200)
        conflict = await self.post("/add", add_payload(content="A different notebook."))
        self.assertEqual(conflict.status_code, 409)
        records = (await self.post("/search", {"user_id": "user-a", "query": "notebook", "top_k": 100})).json()["data"]
        self.assertEqual(len(records), 1)

    async def test_user_isolation_and_top_k(self):
        for i in range(4):
            self.assertEqual((await self.post("/add", add_payload("r" + str(i), content=f"Alice notebook version {i}."))).status_code, 200)
        self.assertEqual((await self.post("/add", add_payload("other", user_id="other-user", content="Alice notebook private evidence."))).status_code, 200)
        response = await self.post("/search", {"user_id": "user-a", "query": "notebook", "options": ["notebook", "pen"], "top_k": 2})
        self.assertEqual(len(response.json()["data"]), 2)
        self.assertNotIn("private evidence", response.text)
        self.assertEqual((await self.post("/search", {"user_id": "absent-user", "query": "notebook", "top_k": 100})).json(), {"data": []})

    async def test_semantic_validation_rejects_roles_timestamp_ids_and_top_k(self):
        payloads = []
        for role in ("system", "tool", None, [], {}):
            p = add_payload(); p["messages"][0]["role"] = role; payloads.append(p)
        for stamp in (None, True, 1.5, "1704067200000", 10**30):
            p = add_payload(); p["messages"][0]["timestamp"] = stamp; payloads.append(p)
        for field in ("request_id", "user_id", "session_id"):
            p = add_payload(); p[field] = " "; payloads.append(p)
        for p in payloads:
            self.assertEqual((await self.post("/add", p)).status_code, 422)
        for value in (0, -1, 101, True, 1.5, "1"):
            self.assertEqual((await self.post("/search", {"query": "x", "user_id": "u", "top_k": value})).status_code, 422)
        for options in (None, {}, "A", 1, [""], [1]):
            self.assertEqual((await self.post("/search", {"query": "x", "user_id": "u", "top_k": 1, "options": options})).status_code, 422)

    async def test_malformed_json_nonfinite_duplicate_keys_and_media(self):
        headers = {**self.headers, "Content-Type": "application/json"}
        for raw in (b"{", b'{"x":NaN}', b'{"user_id":"a","user_id":"b"}', b"\xff"):
            self.assertEqual((await self.client.post("/add", content=raw, headers=headers)).status_code, 400)
        self.assertEqual((await self.post("/add", [])).status_code, 422)
        self.assertEqual((await self.client.post("/add", content=b"{}", headers=self.headers)).status_code, 415)
        for uri in ("https://example.com/image.png", "data:image/png;base64,AAAA", "data:image/gif;base64,AAAA"):
            payload = add_payload(content=[{"type": "image_url", "image_url": {"url": uri}}])
            self.assertEqual((await self.post("/add", payload)).status_code, 422)
        raw = b'{"request_id":"surrogate","user_id":"u","session_id":"s","messages":[{"role":"user","content":"\\ud800"}]}'
        self.assertEqual((await self.client.post("/add", content=raw, headers=headers)).status_code, 422)

    async def test_declared_and_streaming_body_limit_and_slots_recovery(self):
        app = create_app(replace(self.settings, max_request_bytes=256))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            payload = add_payload(content="notebook " * 200)
            self.assertEqual((await client.post("/add", json=payload, headers=self.headers)).status_code, 413)
            async def chunks():
                yield b" " * 200
                yield b" " * 200
            self.assertEqual((await client.post("/add", content=chunks(), headers={**self.headers, "Content-Type": "application/json"})).status_code, 413)
            response = await client.post("/search", json={"user_id": "none", "query": "x", "top_k": 1}, headers=self.headers)
            self.assertEqual(response.status_code, 200)
        app.state.store.close()

    async def test_image_pixel_and_decoded_capacity_errors(self):
        app = create_app(replace(self.settings, max_pixels=100))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            payload = add_payload(content=[{"type": "image_url", "image_url": {"url": image_uri(size=(32, 32))}}])
            self.assertEqual((await client.post("/add", json=payload, headers=self.headers)).status_code, 413)
        app.state.store.close()

    async def test_read_timeout_releases_admission_capacity(self):
        app = create_app(replace(self.settings, read_timeout_seconds=1, max_add_concurrency=1))
        async def stalled_body():
            yield b'{"x":'
            await asyncio.sleep(2)
            yield b"1}"
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/add", content=stalled_body(), headers={**self.headers, "Content-Type": "application/json"})
            self.assertEqual(response.status_code, 408)
            self.assertEqual((await client.post("/add", json=add_payload(), headers=self.headers)).status_code, 200)
        app.state.store.close()

    async def test_restart_reopens_committed_image_and_stable_id(self):
        payload = add_payload(content=[{"type": "text", "text": "persistent purple notebook"},
                                       {"type": "image_url", "image_url": {"url": image_uri()}}])
        self.assertEqual((await self.post("/add", payload)).status_code, 200)
        search = {"user_id": "user-a", "query": "notebook", "top_k": 1}
        before = (await self.post("/search", search)).json()
        new_app = create_app(self.settings)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=new_app), base_url="http://test") as client:
            response = await client.post("/search", json=search, headers=self.headers)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), before)
        new_app.state.store.close()

    async def test_repeated_serial_requests_never_hit_admission_race(self):
        app = create_app(replace(self.settings, max_add_concurrency=1, max_search_concurrency=1))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            for i in range(15):
                response = await client.post("/add", json=add_payload("serial-" + str(i)), headers=self.headers)
                self.assertEqual(response.status_code, 200)
                response = await client.post("/search", json={"user_id": "user-a", "query": "notebook", "top_k": 1}, headers=self.headers)
                self.assertEqual(response.status_code, 200)
        app.state.store.close()

    async def test_structured_log_has_no_body_identity_or_credentials(self):
        logger = logging.getLogger("prism_swift")
        with self.assertLogs(logger, level="INFO") as logs:
            await self.post("/add", add_payload(content="secret-evidence-never-log"))
        encoded = "\n".join(logs.output)
        for sensitive in ("secret-evidence-never-log", "test-private-key", "user-a", "request-a", "session-a"):
            self.assertNotIn(sensitive, encoded)
        self.assertIn("request_completed", encoded)


class StubStore:
    def __init__(self):
        self.closed_threads = []
        self.worker_threads = []
        self.exception = None
        self.block = False
        self.started = threading.Event()
        self.release = threading.Event()

    def add(self, payload):
        self.worker_threads.append(threading.get_ident())
        if self.block:
            self.started.set()
            if not self.release.wait(5):
                raise RuntimeError("test wait timeout")
        if self.exception:
            raise self.exception
        return {"success": True, **{k: payload[k] for k in ("request_id", "user_id", "session_id")}}

    def search(self, payload):
        self.worker_threads.append(threading.get_ident())
        return {"data": []}

    def health(self):
        return {"status": "ok"}

    def close(self):
        self.closed_threads.append(threading.get_ident())


class AdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_true_concurrency_429_retry_after_and_search_independent(self):
        store = StubStore(); store.block = True
        settings = Settings(api_key="test", min_free_bytes=0, max_add_concurrency=1)
        app = create_app(settings, store)
        headers = {"Authorization": "Token test"}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            task = asyncio.create_task(client.post("/add", json=add_payload(), headers=headers))
            await asyncio.wait_for(asyncio.to_thread(store.started.wait, 3), timeout=4)
            rejected = await client.post("/add", json=add_payload("busy"), headers=headers)
            self.assertEqual(rejected.status_code, 429)
            self.assertEqual(rejected.headers["Retry-After"], "1")
            search = await client.post("/search", json={"user_id": "u", "query": "x", "top_k": 1}, headers=headers)
            self.assertEqual(search.status_code, 200)
            store.release.set()
            self.assertEqual((await task).status_code, 200)
            self.assertEqual((await client.post("/add", json=add_payload("after"), headers=headers)).status_code, 200)
        self.assertEqual(sorted(store.worker_threads), sorted(store.closed_threads))

    async def test_exception_redaction_cleanup_and_recovery(self):
        store = StubStore()
        store.exception = RuntimeError("PRIVATE-BODY KEY=secret")
        app = create_app(Settings(api_key="test", min_free_bytes=0), store)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            headers = {"Authorization": "Token test"}
            response = await client.post("/add", json=add_payload(), headers=headers)
            self.assertEqual(response.status_code, 500)
            self.assertEqual(response.json(), {"detail": {"reason": "internal_error"}})
            self.assertNotIn("PRIVATE", response.text)
            store.exception = None
            self.assertEqual((await client.post("/add", json=add_payload(), headers=headers)).status_code, 200)
        self.assertEqual(store.worker_threads, store.closed_threads)

    async def test_storage_unavailable_returns_retryable_safe_error(self):
        store = StubStore(); store.exception = OSError("SECRET DISK PATH")
        app = create_app(Settings(api_key="test", min_free_bytes=0), store)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/add", json=add_payload(), headers={"Authorization": "Token test"})
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.headers["Retry-After"], "1")
            self.assertNotIn("SECRET", response.text)

    async def test_response_body_limit_is_explicit_no_silent_truncation(self):
        store = StubStore()
        store.search = lambda payload: {"data": [{"id": "source", "content": "x" * 2000}]}
        app = create_app(Settings(api_key="test", min_free_bytes=0, max_response_bytes=1024), store)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/search", json={"user_id": "u", "query": "x", "top_k": 1}, headers={"Authorization": "Token test"})
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json(), {"detail": {"reason": "response_body_limit"}})


class SettingsTests(unittest.TestCase):
    def test_production_requires_memory_key_but_no_remote_model_key(self):
        with self.assertRaises(ValueError):
            Settings(production=True)
        settings = Settings(api_key="private", production=True)
        self.assertEqual(settings.strategy, "adaptive")
        self.assertNotIn("private", repr(settings))
        self.assertFalse(hasattr(settings, "openai_api_key"))
        self.assertEqual(Settings(production=False).api_key, "")

    def test_environment_limits_and_frozen_validation(self):
        with mock.patch.dict("os.environ", {"MEMORY_API_KEY": "private", "MAX_ADD_CONCURRENCY": "2", "STRATEGY": "sparse", "SOURCE_COMMIT": "b" * 40}, clear=True):
            settings = Settings.from_env()
            self.assertEqual(settings.strategy, "sparse")
            self.assertEqual(settings.max_add_concurrency, 2)
            self.assertEqual(settings.commit, "b" * 40)
        for changes in ({"max_add_concurrency": 0}, {"max_results": 101}, {"max_image_bytes": 11 * 1024**2}, {"api_key": " space "}):
            with self.assertRaises(ValueError):
                replace(Settings(api_key="valid"), **changes)


if __name__ == "__main__":
    unittest.main()
