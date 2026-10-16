#!/usr/bin/env python3
"""Build and test an isolated localhost-only Docker service with synthetic evidence.

Never reads an existing service, credential file, or evaluation database. The new
stopped test container and its uniquely named test volume are retained for audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import time
import uuid
import urllib.parse

from smoke_test import request, run_checks
from verify_submission import capacity_check
from release import runtime_hash

ROOT = Path(__file__).resolve().parents[1]


def docker(*args, timeout=300):
    completed = subprocess.run(["docker", *args], check=False, capture_output=True, text=True, timeout=timeout)
    if completed.returncode:
        # Docker may echo a source line. Keep full diagnostics only in a synthetic build log.
        raise RuntimeError("docker_command_failed_" + args[0])
    return completed.stdout.strip()


def ready(base_url, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            status, health, _ = request(base_url, "/health", timeout=3)
            if status == 200:
                return health
        except Exception:
            pass
        time.sleep(.5)
    raise RuntimeError("isolated_container_health_timeout")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="prism-swift:local-test")
    parser.add_argument("--base-image", default="", help="Explicit public base mirror; requires Dockerfile PYTHON_IMAGE build argument")
    parser.add_argument("--package-index", default="", help="Optional public HTTPS PyPI mirror; no credentials or query parameters")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "container_check.json")
    args = parser.parse_args()
    if args.package_index:
        parsed = urllib.parse.urlsplit(args.package_index)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            parser.error("package index must be a public HTTPS URL without credentials or query parameters")
    suffix = uuid.uuid4().hex[:12]
    name = "swift-local-verification-" + suffix
    volume = "swift-local-verification-" + suffix
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "failed", "cloud_deployment": "not_run", "official_smoke": "not_run",
              "official_full": "not_run", "model_calls": 0, "synthetic_only": True,
              "container": name, "volume": volume, "image_tag": args.image,
              "base_image_override": args.base_image or None,
              "package_index_override": args.package_index or None,
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted((ROOT / "src").rglob("*.py"))}}
    created = False
    try:
        report["docker_version"] = docker("version", "--format", "{{.Server.Version}}")
        if not args.skip_build:
            log_path = ROOT / "artifacts" / "private" / (args.output.stem + ".build.log")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as log:
                build = ["docker", "build", "-t", args.image]
                if args.base_image:
                    build.extend(["--build-arg", "PYTHON_IMAGE=" + args.base_image])
                if args.package_index:
                    build.extend(["--build-arg", "SWIFT_PIP_INDEX=" + args.package_index])
                completed = subprocess.run([*build, str(ROOT)], stdout=log,
                                           stderr=subprocess.STDOUT, timeout=600, check=False)
            if completed.returncode:
                raise RuntimeError("docker_build_failed_see_synthetic_build_log")
        report["image_id"] = docker("image", "inspect", args.image, "--format", "{{.Id}}")
        docker("volume", "create", volume)
        with tempfile.TemporaryDirectory(prefix="swift-docker-check-") as temporary:
            env_path = Path(temporary) / "private-test.env"
            key = secrets.token_urlsafe(32)
            env_path.write_text("MEMORY_API_KEY=" + key + "\nPRODUCTION=1\nMIN_FREE_BYTES=0\n"
                                "MAX_ADD_CONCURRENCY=2\nMAX_SEARCH_CONCURRENCY=2\nSTRATEGY=adaptive\n", encoding="utf-8")
            os.chmod(env_path, 0o600)
            docker("run", "-d", "--name", name, "--env-file", str(env_path),
                   "-p", "127.0.0.1::8000", "--mount", "type=volume,source=" + volume + ",target=/app/data",
                   "--memory", "512m", "--cpus", "2", "--pids-limit", "128", "--read-only",
                   "--tmpfs", "/tmp:size=33554432,mode=1777", "--cap-drop", "ALL", "--cap-add", "CHOWN",
                   "--cap-add", "SETUID", "--cap-add", "SETGID", "--security-opt", "no-new-privileges:true", args.image)
            created = True
            binding = json.loads(docker("inspect", name, "--format", '{{json (index .NetworkSettings.Ports "8000/tcp")}}'))[0]
            if binding["HostIp"] != "127.0.0.1":
                raise RuntimeError("test_container_not_loopback_bound")
            base_url = "http://127.0.0.1:" + binding["HostPort"]
            report["health"] = ready(base_url)
            report["runtime_source_matches"] = report["health"].get("source_sha256") == runtime_hash()
            if not report["runtime_source_matches"]:
                raise RuntimeError("deployed_runtime_source_mismatch")
            uid_code = "from pathlib import Path; import json; rows=[]\nfor p in Path('/proc').iterdir():\n if p.name.isdigit():\n  try:\n   parts=(p/'cmdline').read_bytes().split(b'\\0')\n   if parts and parts[0].rsplit(b'/',1)[-1].startswith(b'python') and b'uvicorn' in parts:\n    rows.append(int(next(x for x in (p/'status').read_text().splitlines() if x.startswith('Uid:')).split()[1]))\n  except (OSError,StopIteration,ValueError): pass\nprint(json.dumps(rows))"
            report["application_uids"] = json.loads(docker("exec", name, "python", "-c", uid_code))
            if not report["application_uids"] or any(uid == 0 for uid in report["application_uids"]):
                raise RuntimeError("application_running_as_root")
            if report["health"].get("synchronous") != 2 or report["health"].get("journal_mode") != "wal":
                raise RuntimeError("sqlite_not_wal_full")
            report["write_and_immediate_read"], state = run_checks(base_url, key)
            report["capacity"] = capacity_check(base_url, key, 2, 8, 120)
            if report["capacity"]["errors"]:
                raise RuntimeError("synthetic_container_capacity_failed")
            status, metrics, _ = request(base_url, "/metrics", key=key)
            if status != 200:
                raise RuntimeError("authenticated_metrics_unavailable")
            report["metrics_before_restart"] = metrics
            docker("restart", "-t", "30", name, timeout=60)
            binding = json.loads(docker("inspect", name, "--format", '{{json (index .NetworkSettings.Ports "8000/tcp")}}'))[0]
            base_url = "http://127.0.0.1:" + binding["HostPort"]
            ready(base_url)
            state = {**state, "base_url":base_url}
            report["read_after_restart"], _ = run_checks(base_url, key, state=state, phase="read")
            if report["read_after_restart"].get("stable_id") != "passed":
                raise RuntimeError("restart_source_or_id_not_preserved")
            docker("kill", "--signal", "KILL", name, timeout=30)
            docker("start", name)
            binding = json.loads(docker("inspect", name, "--format", '{{json (index .NetworkSettings.Ports "8000/tcp")}}'))[0]
            base_url = "http://127.0.0.1:" + binding["HostPort"]
            ready(base_url)
            state = {**state, "base_url":base_url}
            report["read_after_sigkill"], _ = run_checks(base_url, key, state=state, phase="read")
            if report["read_after_sigkill"].get("stable_id") != "passed":
                raise RuntimeError("crash_recovery_source_or_id_not_preserved")
            # Compose syntax check uses only this private synthetic test key.
            env = os.environ.copy()
            env.update(MEMORY_API_KEY=key, PUBLIC_PORT="18803")
            compose = subprocess.run(["docker", "compose", "--env-file", str(env_path), "-f", str(ROOT / "compose.yaml"), "config", "--no-env-resolution", "--quiet"],
                                     check=False, capture_output=True, text=True, timeout=30, env=env)
            report["compose_config"] = "passed" if compose.returncode == 0 else "failed"
            if compose.returncode:
                raise RuntimeError("compose_syntax_validation_failed")
            after_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted((ROOT / "src").rglob("*.py"))}
            report["source_snapshot_unchanged"] = after_hashes == report["source_sha256"]
            if not report["source_snapshot_unchanged"]:
                raise RuntimeError("source_changed_during_container_verification_rebuild_required")
            report["status"] = "passed"
    except Exception as exc:
        report["failure"] = {"error_type": type(exc).__name__,
                             "reason": str(exc) if isinstance(exc, RuntimeError) else "isolated_container_check_failed"}
    finally:
        if created:
            try:
                docker("stop", "-t", "30", name, timeout=60)
                report["retained_container_state"] = "stopped"
            except Exception:
                report["retained_container_state"] = "check_failed"
        encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        args.output.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
