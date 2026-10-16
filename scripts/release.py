"""Create a source-only release with hashes; exclude all private data and model weights."""
import argparse
import hashlib
import json
from pathlib import Path
import stat
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def runtime_hash():
    digest=hashlib.sha256()
    for path in sorted((ROOT/"src/prism_swift").glob("*.py")):
        digest.update(path.name.encode()+b"\0"+hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-hash",action="store_true")
    parser.add_argument("--output",type=Path,default=ROOT/"dist/prism-swift-0.1.0.zip")
    args=parser.parse_args()
    if args.source_hash:
        print(runtime_hash()); return
    files=[]
    for path in ROOT.rglob("*"):
        if not path.is_file(): continue
        relative=path.relative_to(ROOT)
        if any(part in {".git","artifacts","data","models","dist","__pycache__"} or part.startswith(".venv") for part in relative.parts): continue
        if path.name.startswith(".env") and path.name!=".env.example": continue
        if path.suffix in {".sqlite3",".db",".pyc",".log",".private",".zip"}: continue
        if path.name.startswith("state") and path.suffix==".json": continue
        if path.name.endswith("-draft.png"): continue
        if path.is_symlink() or ROOT not in path.resolve().parents: raise ValueError("Unsafe source path")
        files.append(path)
    commit=subprocess.run(["git","-C",str(ROOT),"rev-parse","HEAD"],capture_output=True,text=True)
    status=subprocess.run(["git","-C",str(ROOT),"status","--porcelain"],capture_output=True,text=True)
    manifest={"kind":"source_snapshot","runtime_sha256":runtime_hash(),
              "git_commit":commit.stdout.strip() if commit.returncode==0 else None,
              "git_dirty":bool(status.stdout.strip()) if status.returncode==0 else None,"sha256":{
        path.relative_to(ROOT).as_posix():hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(files)}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,"w",compression=zipfile.ZIP_DEFLATED) as output:
        for path in sorted(files):
            name=path.relative_to(ROOT).as_posix()
            info=zipfile.ZipInfo(name,date_time=(2026,10,16,0,0,0))
            info.create_system=3
            info.external_attr=(stat.S_IFREG | (0o755 if path.suffix==".sh" else 0o644))<<16
            output.writestr(info,path.read_bytes(),compress_type=zipfile.ZIP_DEFLATED)
        output.writestr("SOURCE_MANIFEST.json",json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"zip":str(args.output),"files":len(files),"runtime_sha256":manifest["runtime_sha256"],
                      "zip_sha256":hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__=="__main__": main()
