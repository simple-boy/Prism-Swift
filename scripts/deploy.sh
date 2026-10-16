#!/bin/sh
set -eu

if [ "${1:-}" = "--container-start" ]; then
    shift
    exec python -c '
import os, pathlib, sys
directory = pathlib.Path("/app/data")
if directory.is_symlink():
    raise SystemExit("Refusing symlink data directory")
directory.mkdir(parents=True, exist_ok=True)
database = pathlib.Path(os.environ.get("MEMORY_DB_PATH", "/app/data/swift.sqlite3"))
if not database.is_absolute() or database.parent != directory:
    raise SystemExit("Container MEMORY_DB_PATH must be directly under /app/data")
if os.geteuid() == 0:
    os.chown(directory, 10001, 10001)
    os.setgroups([])
    os.setgid(10001)
    os.setuid(10001)
os.chmod(directory, 0o700)
if not os.access(directory, os.W_OK):
    raise SystemExit("Data directory is not writable by service user")
os.execvp(sys.argv[1], sys.argv[1:])
' "$@"
fi

cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
if [ ! -f .env ]; then
    printf '%s\n' 'Create .env from .env.example and set MEMORY_API_KEY first.' >&2
    exit 1
fi
if ! command -v docker >/dev/null 2>&1; then
    printf '%s\n' 'Docker is not installed. Follow docs/deployment_guide.md.' >&2
    exit 1
fi
docker compose version >/dev/null
# Do not print `docker compose config`: it can expand and disclose the API key.
docker compose config --quiet
docker compose up -d --build
docker compose ps
