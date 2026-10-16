"""Verify source ZIP hashes, frozen baselines and absence of runtime/private assets."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    args=parser.parse_args()
    with zipfile.ZipFile(args.archive) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)): raise ValueError("Duplicate ZIP entries")
        manifest=json.loads(archive.read("SOURCE_MANIFEST.json"))
        if set(names)!=(set(manifest["sha256"])|{"SOURCE_MANIFEST.json"}):
            raise ValueError("Source manifest membership mismatch")
        for name in names:
            path=Path(name)
            if path.is_absolute() or ".." in path.parts: raise ValueError("Unsafe archive path")
            if name=="SOURCE_MANIFEST.json": continue
            if any(part in {".git","artifacts","data","models","__pycache__"} or part.startswith(".venv") for part in path.parts):
                raise ValueError("Private/runtime directory in source release")
            if path.name.startswith(".env") and path.name!=".env.example":
                raise ValueError("Private environment file in source release")
            if path.suffix in {".db",".sqlite3",".pyc",".log",".private",".safetensors",".bin"}:
                raise ValueError("Runtime asset in source release")
            payload=archive.read(name)
            if hashlib.sha256(payload).hexdigest()!=manifest["sha256"][name]:
                raise ValueError("Source hash mismatch")
            if path.suffix in {".py",".md",".txt",".yaml",".json",".example"}:
                text=payload.decode("utf-8")
                if re.search(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|\bsk-[A-Za-z0-9_-]{24,}",text):
                    raise ValueError("Credential-like secret in source release")
        frozen=json.loads(archive.read("evaluation/frozen_20261015/SNAPSHOT_MANIFEST.json"))
        for name,digest in frozen["files_sha256"].items():
            if hashlib.sha256(archive.read("evaluation/frozen_20261015/"+name)).hexdigest()!=digest:
                raise ValueError("Frozen experiment source changed")
        legacy=json.loads(archive.read("evaluation/baselines/legacy_tracks/SOURCE_MANIFEST.json"))
        for name,digest in legacy["sha256"].items():
            if hashlib.sha256(archive.read("evaluation/baselines/legacy_tracks/"+name)).hexdigest()!=digest:
                raise ValueError("Legacy comparison source changed")
        runtime=hashlib.sha256()
        for name in sorted(n for n in names if n.startswith("src/prism_swift/") and n.endswith(".py")):
            runtime.update(Path(name).name.encode()+b"\0"+hashlib.sha256(archive.read(name)).digest())
        if runtime.hexdigest()!=manifest["runtime_sha256"]: raise ValueError("Runtime fingerprint mismatch")
    print(json.dumps({"integrity":"passed","files":len(names),"runtime_sha256":runtime.hexdigest(),
                      "git_commit":manifest.get("git_commit"),"git_dirty":manifest.get("git_dirty"),
                      "zip_sha256":hashlib.sha256(args.archive.read_bytes()).hexdigest(),
                      "scope":"Source-package audit only; no public API or official eligibility claim."}))


if __name__=="__main__": main()
