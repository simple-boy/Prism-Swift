"""Generated 10k-source persistence/retrieval load; never uses official input."""
import argparse
import hashlib
import io
import base64
import json
import os
from pathlib import Path
import random
import statistics
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from PIL import Image
import psutil
from prism_swift.storage import MemoryStore


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages",type=int,default=10000)
    parser.add_argument("--output",type=Path,default=ROOT/"results/scale-20261016.json")
    args=parser.parse_args()
    if args.messages<100: parser.error("At least 100 messages")
    if hasattr(os,"sched_getaffinity"): os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    rng=random.Random(919)
    images=[]
    for _ in range(16):
        with Image.frombytes("RGB",(512,512),rng.randbytes(512*512*3)) as image:
            buffer=io.BytesIO();image.save(buffer,"JPEG",quality=90)
            images.append("data:image/jpeg;base64,"+base64.b64encode(buffer.getvalue()).decode())
    cache=Path.home()/".cache/prism-swift-runs"
    cache.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache,prefix="scale-") as folder:
        store=MemoryStore(Path(folder)/"memory.sqlite3")
        latencies=[]
        rss=0
        process=psutil.Process()
        io_before=process.io_counters()
        started=time.perf_counter()
        try:
            for offset in range(0,args.messages,100):
                messages=[]
                for index in range(offset,min(offset+100,args.messages)):
                    text=f"Person{index%100} placed document number {index} in room {index%30}."
                    content=text if index%4 else [{"type":"text","text":text},
                        {"type":"image_url","image_url":{"url":images[index%16]}}]
                    messages.append({"role":"user","content":content,"timestamp":1700000000000+index*60000})
                moment=time.perf_counter()
                assert store.add({"user_id":"scale-user","request_id":str(offset),"session_id":str(offset//1000),"messages":messages})["success"]
                latencies.append(time.perf_counter()-moment)
                rss=max(rss,process.memory_info().rss)
            elapsed=time.perf_counter()-started
            actual=store._connection().execute("SELECT count(*) FROM messages").fetchone()[0]
            assert actual==args.messages
            store.close()
            query_times=[]
            for index in range(30):
                moment=time.perf_counter()
                result=store.search({"user_id":"scale-user","query":f"Where did Person{index} place the document?","top_k":10})
                assert result["data"]
                query_times.append(time.perf_counter()-moment)
            stats=store.metrics()
            ordered=sorted(query_times)
            io_after=process.io_counters()
            report={"kind":"generated_large_scope_load_not_official_score","sources":actual,"image_occurrences":(actual+3)//4,
                "unique_blob_rows":store._connection().execute("SELECT count(*) FROM blobs").fetchone()[0],
                "batch_messages":100,"add_requests":len(latencies),"add_seconds":elapsed,"messages_per_second":actual/elapsed,
                "add_requests_per_second":len(latencies)/elapsed,"add_batch_p50_seconds":statistics.median(latencies),
                "search_p50_ms":statistics.median(query_times)*1000,"search_p95_ms":ordered[int(.95*(len(ordered)-1))]*1000,
                "rss_max_sampled_bytes":rss,"os_write_bytes":io_after.write_bytes-io_before.write_bytes,
                "metrics":stats,"errors":0,"remote_model_calls":0,"gpu_used":False,
                "cpu_affinity":sorted(os.sched_getaffinity(0)),"source_resolution":[512,512],
                "limits":"Local ext4, 2 logical CPU affinity, no RAM quota; repeated images within one user. Direct Store, no HTTP/Answer/Eval. Not cloud capacity.",
                "source_sha256":{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/"src").rglob("*.py")}}
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2)+"\n")
            print(json.dumps({k:report[k] for k in ("sources","messages_per_second","search_p95_ms","errors")}))
        finally: store.close()


if __name__=="__main__": main()
