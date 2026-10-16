#!/usr/bin/env python3
"""Public-contract self-test using synthetic evidence only; never runs AML Full."""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from PIL import Image


def request(base_url, path, *, key="", method="GET", payload=None, auth="token", timeout=120):
    headers = {"Accept": "application/json"}
    if key:
        if auth == "x-api-key":
            headers["X-Api-Key"] = key
        else:
            headers["Authorization"] = ("Bearer " if auth == "bearer" else "Token ") + key
    data = None if payload is None else json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base_url.rstrip("/") + path, data=data, headers=headers, method=method)
    started = time.perf_counter()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as response:
            status, raw = response.status, response.read(64 * 1024**2 + 1)
    except urllib.error.HTTPError as exc:
        status, raw = exc.code, exc.read(64 * 1024**2 + 1)
    if len(raw) > 64 * 1024**2:
        raise RuntimeError("response_exceeds_local_check_capacity")
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError):
        raise RuntimeError(f"non_json_response_http_{status}") from None
    return status, value, time.perf_counter() - started


def sample_image():
    image = Image.new("RGB", (64, 64), (77, 33, 188))
    image.putpixel((5, 5), (255, 222, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def new_state(base_url):
    suffix = uuid.uuid4().hex
    marker = "swift-contract-" + suffix
    parts = [{"type": "text", "text": marker + " Alice received a purple notebook."},
             {"type": "image_url", "image_url": {"url": sample_image()}},
             {"type": "text", "text": "Original image follows the first text, before this text."}]
    payload = {"request_id": "selfcheck-request-" + suffix, "user_id": "selfcheck-user-" + suffix,
               "session_id": "selfcheck-session-" + suffix,
               "messages": [{"role": "user", "timestamp": 1704067200000, "content": parts}]}
    return {"base_url": base_url.rstrip("/"), "marker": marker, "payload": payload}


def run_checks(base_url, key, *, state=None, phase="all", auth="token", timeout=120):
    state = state or new_state(base_url)
    if state.get("base_url") != base_url.rstrip("/"):
        raise RuntimeError("state_base_url_mismatch")
    payload = state["payload"]
    status, health, elapsed = request(base_url, "/health", timeout=timeout)
    if status != 200 or health.get("service") != "prism-swift":
        raise RuntimeError(f"health_failed_http_{status}")
    summary = {"health": "passed", "health_seconds": round(elapsed, 6),
               "server_version": health.get("version"), "server_commit": health.get("commit"),
               "real_model_calls": 0, "official_smoke": "not_run", "official_full": "not_run"}
    if key:
        status, _, _ = request(base_url, "/search", method="POST", payload={
            "query": "synthetic authorization check", "user_id": payload["user_id"], "top_k": 1}, timeout=timeout)
        if status != 401:
            raise RuntimeError("unauthenticated_request_not_rejected")
        summary["authentication"] = "passed"
    else:
        summary["authentication"] = "not_checked_no_key"
    if phase in {"all", "write"}:
        status, ack, elapsed = request(base_url, "/add", key=key, auth=auth, method="POST", payload=payload, timeout=timeout)
        expected = {"success": True, **{name: payload[name] for name in ("request_id", "user_id", "session_id")}}
        if status != 200 or ack != expected:
            raise RuntimeError(f"add_contract_failed_http_{status}")
        summary["add"] = "passed"
        summary["add_seconds"] = round(elapsed, 6)
        # Same logical write and exact payload: tests durable receipt semantics.
        status, retry_ack, _ = request(base_url, "/add", key=key, auth=auth, method="POST", payload=payload, timeout=timeout)
        if status != 200 or retry_ack != expected:
            raise RuntimeError("idempotent_retry_failed")
        summary["idempotency"] = "passed"
        conflict = {**payload, "messages": [{"role": "user", "content": "Different synthetic content."}]}
        status, _, _ = request(base_url, "/add", key=key, auth=auth, method="POST", payload=conflict, timeout=timeout)
        if status != 409:
            raise RuntimeError("changed_payload_idempotency_conflict_not_rejected")
        summary["changed_payload_conflict"] = "passed"
    if phase in {"all", "read", "write"}:
        search = {"user_id": payload["user_id"], "query": state["marker"], "top_k": 1,
                  "options": ["a purple notebook", "a green bicycle"]}
        status, response, elapsed = request(base_url, "/search", key=key, auth=auth, method="POST", payload=search, timeout=timeout)
        rows = response.get("data")
        if status != 200 or not isinstance(rows, list) or len(rows) != 1:
            raise RuntimeError(f"immediate_search_failed_http_{status}")
        evidence = rows[0]
        if not isinstance(evidence.get("id"), str) or not evidence["id"]:
            raise RuntimeError("stable_memory_id_missing")
        if state.get("memory_id") and evidence["id"] != state["memory_id"]:
            raise RuntimeError("memory_id_changed_since_previous_phase")
        state["memory_id"] = evidence["id"]
        if evidence.get("content") != payload["messages"][0]["content"]:
            raise RuntimeError("source_parts_image_or_order_not_preserved")
        status, repeat, _ = request(base_url, "/search", key=key, auth=auth, method="POST", payload=search, timeout=timeout)
        if status != 200 or repeat.get("data", [{}])[0].get("id") != evidence["id"]:
            raise RuntimeError("memory_id_unstable")
        status, isolated, _ = request(base_url, "/search", key=key, auth=auth, method="POST", payload={
            **search, "user_id": "absent-selfcheck-" + uuid.uuid4().hex}, timeout=timeout)
        if status != 200 or isolated != {"data": []}:
            raise RuntimeError("user_isolation_failed")
        summary.update({"search": "passed", "search_seconds": round(elapsed, 6),
                        "original_image_and_order": "passed", "stable_id": "passed", "user_isolation": "passed"})
    return summary, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--phase", choices=("all", "write", "read"), default="all")
    parser.add_argument("--state", type=Path, default=Path("state-selfcheck.json"))
    parser.add_argument("--auth", choices=("token", "bearer", "x-api-key"), default="token")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    try:
        saved = json.loads(args.state.read_text(encoding="utf-8")) if args.state.exists() else None
        if args.phase == "read" and saved is None:
            raise RuntimeError("read_phase_requires_existing_state_file")
        result, state = run_checks(args.base_url, os.getenv("MEMORY_API_KEY", ""), state=saved,
                                  phase=args.phase, auth=args.auth, timeout=args.timeout)
        if args.phase != "read":
            args.state.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        # Never print upstream bodies, headers, URLs with credentials, or exception strings.
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__,
                          "reason": str(exc) if isinstance(exc, RuntimeError) else "selfcheck_transport_or_configuration_error"}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
