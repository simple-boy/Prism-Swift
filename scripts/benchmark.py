"""Reproducible synthetic workload, real HTTP/SQLite and measured resource use.

Never reads official evaluation data. No network except this script's localhost API.
"""
from __future__ import annotations
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from urllib.request import build_opener, ProxyHandler

ROOT=Path(__file__).resolve().parents[1]
RUNS=Path.home()/".cache/prism-swift-runs"
RUNS.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/"src"))
from PIL import Image, ImageDraw
import psutil
from prism_swift.storage import MemoryStore
LOCAL_HTTP=build_opener(ProxyHandler({})).open


def percentile(values,p):
    if not values: return None
    ordered=sorted(values)
    index=(len(ordered)-1)*p
    left=int(index)
    return ordered[left]+(ordered[min(left+1,len(ordered)-1)]-ordered[left])*(index-left)


def picture(seed,size=512,format="JPEG",noise=True):
    rng=random.Random(seed)
    raw=rng.randbytes(size*size*3) if noise else bytes([rng.randrange(256) for _ in range(3)])*(size*size)
    with Image.frombytes("RGB",(size,size),raw) as image:
        output=io.BytesIO()
        image.save(output,format,quality=90)
    return "data:image/"+{"PNG":"png","JPEG":"jpeg"}[format]+";base64,"+base64.b64encode(output.getvalue()).decode()


def item(text,uris):
    return [{"type":"text","text":text}]+[{"type":"image_url","image_url":{"url":uri}} for uri in uris]


def payloads(profile,count,seed):
    definitions={"small_text":(0,0,64),"large_text":(0,0,256*1024),
                 "single_image_512":(1,512,256),"single_image_2048":(1,2048,256),
                 "four_images_512":(4,512,256),"mixed_eight_messages":(1,512,512),
                 "concurrent_images":(2,512,256),"sustained":(1,512,256)}
    n,size,length=definitions[profile]
    # Fixed pool with both repetitions and unique images; no generation in timed region.
    pool=[picture(seed*100+i,size) for i in range(max(8,n))] if n else []
    output=[]
    for number in range(count):
        text=("source event engineering gallery Alice blue receipt "*((length//52)+1))[:length]
        uris=[pool[(number*n+i)%len(pool)] for i in range(n)] if pool else []
        messages=[{"role":"user","content":item(text,uris) if n else text,"timestamp":1700000000000+number}]
        if profile=="mixed_eight_messages":
            messages=[{"role":"user","content":item(text,[pool[(number+i)%len(pool)]]),"timestamp":1700000000000+number*8+i}
                      for i in range(8)]
        output.append({"user_id":"load-user-"+str(number%4),"request_id":"load-"+str(number),"session_id":"s", "messages":messages})
    return output


class Resources:
    def __init__(self,pid):
        self.process=psutil.Process(pid)
        self.stop=threading.Event()
        self.rss=self.process.memory_info().rss
        self.samples=0
        self.cpu=[]
    def sample(self):
        self.start_cpu=self.process.cpu_times()
        self.start_io=self.process.io_counters()
        self.process.cpu_percent(None)
        while not self.stop.wait(.01):
            try:
                self.rss=max(self.rss,self.process.memory_info().rss)
                self.cpu.append(self.process.cpu_percent(None))
                self.samples+=1
            except psutil.NoSuchProcess:
                break
    def finish(self,seconds):
        self.stop.set()
        end_cpu=self.process.cpu_times()
        io=self.process.io_counters()
        return {"rss_peak_bytes":self.rss,"sampling_interval_seconds":.01,"resource_samples":self.samples,
                "process_cpu_seconds":end_cpu.user+end_cpu.system-self.start_cpu.user-self.start_cpu.system,
                "mean_cpu_percent_one_core_100":statistics.mean(self.cpu) if self.cpu else None,
                "os_write_bytes":io.write_bytes-self.start_io.write_bytes,"os_read_bytes":io.read_bytes-self.start_io.read_bytes,
                "os_write_bytes_per_second":(io.write_bytes-self.start_io.write_bytes)/seconds,
                "gpu_used":False,"gpu_utilization":None}


def post(base,path,payload):
    raw=json.dumps(payload,separators=(",", ":")).encode()
    request=Request(base+path,data=raw,method="POST",headers={"Authorization":"Token synthetic-benchmark-key","Content-Type":"application/json"})
    started=time.perf_counter()
    try:
        with LOCAL_HTTP(request,timeout=180) as response:
            value=json.load(response)
    except HTTPError as error:
        value={"benchmark_http_error":error.code}
        error.close()
    return time.perf_counter()-started,value,len(raw)


def performance(strategy,profile,args):
    count=max(args.requests,60) if profile=="sustained" else args.requests
    concurrency=2 if profile in {"concurrent_images","sustained"} else 1
    values=payloads(profile,count,args.seed)
    with tempfile.TemporaryDirectory(prefix="swift-perf-",dir=RUNS) as directory:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1",0)); port=sock.getsockname()[1]
        env={**os.environ,"PYTHONPATH":str(ROOT/"src"),"MEMORY_DB_PATH":directory+"/memory.sqlite3",
             "MEMORY_API_KEY":"synthetic-benchmark-key","STRATEGY":strategy,"MIN_FREE_BYTES":"0",
             "MAX_ADD_CONCURRENCY":"2","MAX_SEARCH_CONCURRENCY":"2","PORT":str(port),"HOST":"127.0.0.1"}
        with open(directory+"/server.log","w") as log:
            startup=time.perf_counter()
            command=([sys.executable,"-B",str(ROOT/"scripts/legacy_runner.py"),"--source-root",str(args.legacy_source)]
                     if strategy=="legacy_none" else [sys.executable,"-B","-m","prism_swift.api"])
            process=subprocess.Popen(command,env=env,stdout=log,stderr=log)
            base="http://127.0.0.1:"+str(port)
            try:
                for _ in range(200):
                    if process.poll() is not None:
                        raise RuntimeError(Path(directory+"/server.log").read_text())
                    try:
                        with LOCAL_HTTP(base+"/health",timeout=1) as response: json.load(response)
                        break
                    except OSError: time.sleep(.05)
                else: raise RuntimeError("Local benchmark API did not start")
                startup_seconds=time.perf_counter()-startup
                monitor=Resources(process.pid)
                thread=threading.Thread(target=monitor.sample)
                thread.start()
                # Wait only for the sampler baseline, not an artificial service warm-up.
                while not hasattr(monitor,"start_io"): time.sleep(.001)
                started=time.perf_counter()
                if profile=="sustained":
                    deadline=started+args.sustained_seconds
                    def loop(worker):
                        completed=[]
                        number=0
                        while time.perf_counter()<deadline:
                            value={**values[number%len(values)],"request_id":f"sustained-{worker}-{number}"}
                            completed.append((post(base,"/add",value),value))
                            number+=1
                        return completed
                    with ThreadPoolExecutor(max_workers=concurrency) as pool:
                        completed=[pair for group in pool.map(loop,range(concurrency)) for pair in group]
                    writes=[pair[0] for pair in completed]
                    values=[pair[1] for pair in completed]
                    count=len(values)
                else:
                    with ThreadPoolExecutor(max_workers=concurrency) as pool:
                        writes=list(pool.map(lambda value: post(base,"/add",value),values))
                elapsed=time.perf_counter()-started
                resources=monitor.finish(elapsed); thread.join()
                search=[]
                for index in range(min(12,count)):
                    search.append(post(base,"/search",{"user_id":"load-user-"+str(index%4),"query":"Alice blue receipt","top_k":5})[0])
                with LOCAL_HTTP(Request(base+"/metrics",headers={"Authorization":"Token synthetic-benchmark-key"})) as response:
                    metrics=json.load(response)
                sizes=[sum(len(base64.b64decode(part["image_url"]["url"].split(",",1)[1])) for msg in value["messages"]
                           for part in msg["content"] if part["type"]=="image_url") if isinstance(value["messages"][0]["content"],list) else 0 for value in values]
                messages=sum(len(value["messages"]) for value in values)
                images=sum(sum(part["type"]=="image_url" for part in msg["content"]) for value in values for msg in value["messages"] if isinstance(msg["content"],list))
                latencies=[entry[0] for entry in writes]
                successes=sum(entry[1].get("success") is True for entry in writes)
                return {"strategy":strategy,"profile":profile,"requests":count,"successful_requests":successes,"concurrency":concurrency,
                        "service_start_seconds":startup_seconds,
                        "duration_seconds":elapsed,"add_p50_ms":percentile(latencies,.5)*1000,
                        "add_p95_ms":percentile(latencies,.95)*1000,"add_p99_ms":percentile(latencies,.99)*1000,
                        "requests_per_second":successes/elapsed,"messages_per_second":metrics["storage"]["messages"]/elapsed,
                        "decoded_image_bytes_per_second":sum(sizes)/elapsed,"images_per_second":images/elapsed,
                        "request_wire_bytes_min":min(entry[2] for entry in writes),"request_wire_bytes_max":max(entry[2] for entry in writes),
                        "successful_source_bytes_per_second":metrics["storage"]["source_bytes"]/elapsed,
                        "search_p50_ms":percentile(search,.5)*1000,"search_p95_ms":percentile(search,.95)*1000,
                        "error_rate":1-successes/count,"retry_count":0,"metrics":metrics["storage"],**resources}
            finally:
                process.terminate()
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: process.kill(); process.wait()


def retrieval_corpus(seed):
    """Synthetic developer diagnostic; source names, options and queries are not labels in writer."""
    rng=random.Random(seed)
    messages=[]; queries=[]
    names=["Alice","Bob","Caroline","Diego","Evan","Fatima","Grace","Hector","Iris","Jonah","Kira","Leo"]
    for index,name in enumerate(names):
        before=len(messages)
        messages.append({"role":"user","content":name+" ate breakfast at the hotel.","timestamp":1704067200000+index*1000000})
        anchor=len(messages)
        messages.append({"role":"user","content":name+" boarded an airplane at the airport.","timestamp":1704067300000+index*1000000})
        current=len(messages)
        messages.append({"role":"user","content":name+" changed the favorite drink from coffee to tea.","timestamp":1704067400000+index*1000000})
        queries += [{"query":"What did "+name+" do before boarding the airplane?","targets":[before,anchor],"group":name,"type":"two_event_order"},
                    {"query":"What is "+name+" latest favorite drink?","targets":[current],"group":name,"type":"latest"},
                    {"query":"What happened to "+name+" at the airport?","targets":[anchor],"group":name,"type":"explicit"}]
    for index in range(36):
        # Uncaptioned visual sources test the real limitation of sparse-only retrieval.
        color=(rng.randrange(255),rng.randrange(255),rng.randrange(255))
        with Image.new("RGB",(256,256),color) as image:
            draw=ImageDraw.Draw(image); draw.rectangle((index*3%120,30,150,90+index),fill="white")
            output=io.BytesIO(); image.save(output,"PNG")
            uri="data:image/png;base64,"+base64.b64encode(output.getvalue()).decode()
            target=len(messages)
            parts=[{"type":"image_url","image_url":{"url":uri}}]
            messages.append({"role":"user","content":parts})
            # Query is transcoded JPEG: exact file hashes cannot solve this category.
            query_buffer=io.BytesIO(); image.save(query_buffer,"JPEG",quality=88)
            query_uri="data:image/jpeg;base64,"+base64.b64encode(query_buffer.getvalue()).decode()
            queries.append({"query":[{"type":"image_url","image_url":{"url":query_uri}}],"targets":[target],"group":"visual-"+str(index),"type":"transcoded_visual"})
    for index in range(24):
        name=names[index%len(names)]
        messages.append({"role":"user","content":name+" discussed unrelated budget plans and airport architecture."})
    return messages,queries


def retrieval(strategy,seed,top_k,ablation="none"):
    messages,queries=retrieval_corpus(seed)
    with tempfile.TemporaryDirectory(prefix="swift-retrieval-",dir=RUNS) as directory:
        store=MemoryStore(directory+"/memory.sqlite3",strategy=strategy,ablation=ablation)
        try:
            started=time.perf_counter()
            for index,message in enumerate(messages):
                store.add({"user_id":"synthetic-eval","request_id":"source-"+str(index),"session_id":"source-history", "messages":[message]})
            build_seconds=time.perf_counter()-started
            ids={index:"swift_"+hashlib.sha256(json.dumps(["synthetic-eval","source-"+str(index),0],ensure_ascii=False,sort_keys=True,separators=(",", ":")).encode()).hexdigest()[:32] for index in range(len(messages))}
            results=[]
            for query in queries:
                started=time.perf_counter()
                found=store.search({"user_id":"synthetic-eval","query":query["query"],"top_k":top_k})["data"]
                latency=time.perf_counter()-started
                wanted={ids[index] for index in query["targets"]}
                ranked=[row["id"] for row in found]
                hits=[1 if mid in wanted else 0 for mid in ranked]
                recall={str(k):len(wanted.intersection(ranked[:k]))/len(wanted) for k in (1,5,10)}
                dcg=sum(hit/math.log2(i+2) for i,hit in enumerate(hits))
                ideal=sum(1/math.log2(i+2) for i in range(min(len(wanted),top_k)))
                complete=all(row["content"]==messages[next(index for index,mid in ids.items() if mid==row["id"])]["content"] for row in found)
                results.append({"group":query["group"],"type":query["type"],"recall":recall,
                                "mrr":next((1/(i+1) for i,hit in enumerate(hits) if hit),0),
                                "ndcg":dcg/ideal,"source_complete":complete,"search_ms":latency*1000})
            return {"strategy":strategy,"ablation":ablation,"seed":seed,"source_count":len(messages),"query_count":len(queries),
                    "build_seconds":build_seconds,"top_k":top_k,"queries":results,
                    "recall":{str(k):statistics.mean(row["recall"][str(k)] for row in results) for k in (1,5,10)},
                    "mrr":statistics.mean(row["mrr"] for row in results),"ndcg":statistics.mean(row["ndcg"] for row in results),
                    "source_completeness":sum(row["source_complete"] for row in results)/len(results),
                    "search_p50_ms":percentile([row["search_ms"] for row in results],.5),
                    "search_p95_ms":percentile([row["search_ms"] for row in results],.95)}
        finally: store.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("performance","retrieval","all"),default="all")
    parser.add_argument("--requests",type=int,default=12)
    parser.add_argument("--seed",type=int,default=41)
    parser.add_argument("--cores",type=int,default=2)
    parser.add_argument("--sustained-seconds",type=float,default=30)
    parser.add_argument("--ablations",action="store_true")
    parser.add_argument("--legacy-zip",type=Path)
    parser.add_argument("--legacy-dir",type=Path,default=ROOT/"evaluation/baselines/legacy_tracks",
                        help="Verified bundled source; --legacy-zip can supply the original archive")
    parser.add_argument("--strategies",nargs="+",default=["raw","sparse","adaptive"])
    parser.add_argument("--profiles",nargs="+",default=["small_text","large_text","single_image_512","single_image_2048","four_images_512","mixed_eight_messages","concurrent_images","sustained"])
    parser.add_argument("--output",type=Path,default=ROOT/"results/benchmark.json")
    args=parser.parse_args()
    if args.requests<2: parser.error("At least two requests required")
    if hasattr(os,"sched_getaffinity"):
        allowed=sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0,allowed[:args.cores])
    args.legacy_source=None
    legacy_info=None
    if "legacy_none" in args.strategies:
        expected="f838f314aab67e03a2a27411a3a9912dd55e566d8c2116eaf9ba27d6c82acc0e"
        names=("app.py","store.py","models.py","coding.py","multimodal.py","memory_core.py","records.py")
        if args.legacy_zip:
            legacy_digest=hashlib.sha256(args.legacy_zip.read_bytes()).hexdigest()
            if legacy_digest!=expected: parser.error("Unexpected legacy source snapshot SHA256")
            args.legacy_source=RUNS/("legacy-source-"+legacy_digest[:12])
            args.legacy_source.mkdir(exist_ok=True)
            with zipfile.ZipFile(args.legacy_zip) as archive:
                manifest=json.loads(archive.read("SOURCE_MANIFEST.json"))["sha256"]
                for name in names:
                    data=archive.read(name)
                    if hashlib.sha256(data).hexdigest()!=manifest[name]: raise ValueError("Legacy manifest mismatch")
                    (args.legacy_source/name).write_bytes(data)
        else:
            args.legacy_source=args.legacy_dir.resolve()
            manifest=json.loads((args.legacy_source/"SOURCE_MANIFEST.json").read_text())
            if manifest["origin_zip_sha256"]!=expected: raise ValueError("Legacy provenance mismatch")
            for name in names:
                if hashlib.sha256((args.legacy_source/name).read_bytes()).hexdigest()!=manifest["sha256"][name]:
                    raise ValueError("Bundled legacy source mismatch")
        legacy_info={"source_zip_sha256":expected,"provider":"none","real_model_calls":0,
                     "limitation":"No-model legacy floor, not live Quanzil service or official-run diagnosis"}
    report={"kind":"synthetic_developer_diagnostic_not_official_score","date":"2026-10-15","real_model_calls":0,
            "platform":platform.platform(),"python":sys.version,"cpu_affinity":sorted(os.sched_getaffinity(0)),
            "ram_limit_applied":False,"database_directory":str(RUNS),"database_filesystem":"WSL persistent cache, WAL FULL; see mount probe; not cloud disk",
            "configuration":{key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
            "legacy":legacy_info,"performance":[],"retrieval":[]}
    if args.mode in {"performance","all"}:
        for strategy in args.strategies:
            for profile in args.profiles:
                print("Measuring",strategy,profile,flush=True)
                report["performance"].append(performance(strategy,profile,args))
                args.output.parent.mkdir(parents=True,exist_ok=True)
                args.output.write_text(json.dumps(report,indent=2)+"\n")
    if args.mode in {"retrieval","all"}:
        for strategy in args.strategies:
            for seed in (args.seed,args.seed+1,args.seed+2):
                print("Retrieval",strategy,seed,flush=True)
                report["retrieval"].append(retrieval(strategy,seed,10))
            if args.ablations and strategy=="adaptive":
                for ablation in ("no_visual","no_neighbours","no_coverage"):
                    for seed in (args.seed,args.seed+1,args.seed+2):
                        print("Ablation",ablation,seed,flush=True)
                        report["retrieval"].append(retrieval(strategy,seed,10,ablation))
    report["source_sha256"]={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted((ROOT/"src").rglob("*.py"))}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"output":str(args.output),"performance_runs":len(report["performance"]),"retrieval_runs":len(report["retrieval"]),"real_model_calls":0}))


if __name__=="__main__": main()
