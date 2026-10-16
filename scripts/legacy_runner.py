"""Isolated no-model legacy HTTP baseline. Never points at a deployed database."""
import argparse
import os
from pathlib import Path
import sys

parser=argparse.ArgumentParser()
parser.add_argument("--source-root",required=True)
args=parser.parse_args()
sys.path.insert(0,args.source_root)
from app import create_server
from models import ModelClient

server=create_server(os.environ["HOST"],int(os.environ["PORT"]),track="multimodal",
                     db_path=os.environ["MEMORY_DB_PATH"],api_key=os.environ["MEMORY_API_KEY"],
                     model=ModelClient(mode="none"),add_concurrency=2,search_concurrency=2)
original=server.RequestHandlerClass
class Handler(original):
    def do_GET(self):
        if self.path!="/metrics":
            return super().do_GET()
        if not self.authorized():
            self.send_json(401,{"detail":{"reason":"unauthorized"}})
            return
        conn=server.store._connection()
        count=conn.execute("SELECT count(*) FROM raw_messages").fetchone()[0]
        size=conn.execute("SELECT coalesce(sum(length(content_json)),0) FROM source_records").fetchone()[0]
        path=Path(os.environ["MEMORY_DB_PATH"])
        self.send_json(200,{"storage":{"messages":count,"source_bytes":size,
            "database_bytes":sum(p.stat().st_size for p in (path,Path(str(path)+"-wal")) if p.exists()),
            "model_calls":0,"stage_timing":"not_instrumented_legacy"}})
        server.store.close()
server.RequestHandlerClass=Handler
server.serve_forever()
