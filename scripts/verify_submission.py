#!/usr/bin/env python3
"""Synthetic API capacity checks plus explicit, auditable operator readiness attestations."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import ipaddress
import json
import os
from pathlib import Path
import re
import socket
import statistics
import sys
import urllib.parse

from smoke_test import new_state, request, run_checks


def public_endpoint(base_url):
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        return False
    addresses = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
    return bool(addresses) and all(ipaddress.ip_address(item[4][0]).is_global for item in addresses)


def capacity_check(base_url, key, concurrency, requests, timeout):
    def once(_):
        state = new_state(base_url)
        status, ack, elapsed = request(base_url, "/add", key=key, method="POST", payload=state["payload"], timeout=timeout)
        if status != 200 or ack.get("success") is not True:
            return {"ok": False, "http_status": status, "seconds": elapsed}
        status, value, _ = request(base_url, "/search", key=key, method="POST", payload={
            "query": state["marker"], "user_id": state["payload"]["user_id"], "top_k": 1}, timeout=timeout)
        rows = value.get("data", [])
        return {"ok": status == 200 and len(rows) == 1 and rows[0].get("content") == state["payload"]["messages"][0]["content"],
                "http_status": status, "seconds": elapsed}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(once, range(requests)))
    latencies = sorted(item["seconds"] for item in results)
    return {"requests": requests, "concurrency": concurrency, "successes": sum(item["ok"] for item in results),
            "errors": sum(not item["ok"] for item in results),
            "add_p50_seconds": statistics.median(latencies), "add_p95_seconds": latencies[min(len(latencies) - 1, int(.95 * len(latencies)))],
            "workload": "64x64 synthetic PNG + text per Add; not a large-payload capacity benchmark"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--write", action="store_true", help="Authorize synthetic writes; no deletion is performed")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--commit", default="")
    parser.add_argument("--source-sha256",default="",help="Expected runtime fingerprint from the fixed source release")
    parser.add_argument("--division", choices=("commercial", "open-source"))
    parser.add_argument("--eligibility-evidence", default="", help="Organizer approval reference or documented applicable division policy")
    parser.add_argument("--confirm-disk", action="store_true", help="Operator checked volume persistence and sufficient free disk for estimated Full")
    parser.add_argument("--confirm-version", action="store_true", help="Operator froze public repository commit and matching deployed image/source")
    parser.add_argument("--confirm-no-private-data", action="store_true", help="Operator audited repository/package for keys and evaluation data")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = {"ready_for_submission": False, "official_smoke": "not_run", "official_full": "not_run",
              "model_calls": 0, "checks": {}, "operator_attestations": {}}
    try:
        if args.concurrency < 1 or args.requests < args.concurrency:
            raise RuntimeError("requests_must_be_at_least_positive_concurrency")
        key = os.getenv("MEMORY_API_KEY", "")
        status, health, _ = request(args.base_url, "/health", timeout=args.timeout)
        report["checks"]["health"] = status == 200 and health.get("service") == "prism-swift"
        report["checks"]["public_dns_and_address"] = public_endpoint(args.base_url)
        report["checks"]["key_configured"] = bool(key)
        report["checks"]["commit_matches_health"] = bool(re.fullmatch(r"[0-9a-fA-F]{40}", args.commit)) and health.get("commit") == args.commit
        report["checks"]["runtime_source_matches"] = bool(re.fullmatch(r"[0-9a-fA-F]{64}",args.source_sha256)) and health.get("source_sha256")==args.source_sha256
        if args.write:
            smoke, _ = run_checks(args.base_url, key, timeout=args.timeout)
            report["synthetic_smoke"] = smoke
            report["checks"]["synthetic_smoke"] = bool(key) and smoke.get("search") == "passed" and smoke.get("authentication") == "passed"
            capacity = capacity_check(args.base_url, key, args.concurrency, args.requests, args.timeout)
            report["capacity"] = capacity
            report["checks"]["declared_small_request_capacity"] = capacity["errors"] == 0
        else:
            report["checks"]["synthetic_smoke"] = False
            report["checks"]["declared_small_request_capacity"] = False
        report["operator_attestations"] = {
            "division": args.division, "eligibility_evidence": args.eligibility_evidence,
            "disk_and_persistence": args.confirm_disk, "version_fixed": args.confirm_version,
            "no_private_data": args.confirm_no_private_data,
            "independently_verified_by_script": False}
        attestations = bool(args.division and args.eligibility_evidence.strip() and args.confirm_disk
                            and args.confirm_version and args.confirm_no_private_data)
        report["ready_for_submission"] = all(report["checks"].values()) and attestations
        report["meaning"] = "Technical self-check and operator declarations only; never official approval or a Smoke/Full result."
    except Exception as exc:
        report["failure"] = {"error_type": type(exc).__name__,
                             "reason": str(exc) if isinstance(exc, RuntimeError) else "verification_transport_or_configuration_error"}
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["ready_for_submission"] else 1


if __name__ == "__main__":
    sys.exit(main())
