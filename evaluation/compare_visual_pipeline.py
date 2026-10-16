"""Actual synchronous source/FTS/TinyCLIP pipelines on generated multimodal data.

B is an experimental adapter, not the default public API. Its vectors are
persisted in the same Add transaction outside content. No gold, query, or label
is supplied to any writer. No real API/official evaluation data is used.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from PIL import Image, ImageDraw
from prism_swift.storage import MemoryStore
from prism_swift.validation import canonical, restore

VARIANTS = ("raw", "A_sparse", "B_tinyclip", "C_adaptive", "C_no_visual", "C_no_neighbours", "C_no_coverage")
COLORS = {"red": "#d91f26", "green": "#159b43", "blue": "#185bd9",
          "yellow": "#e4d719", "purple": "#8625a7", "orange": "#e98a21"}
SHAPES = ("square", "circle", "triangle")
MODEL_REVISION = "a2a8c6eaa2549ad66eb7c31b85022bf58273a26c"
WEIGHTS_SHA = "9339ee3d736344d0ddcaa6c03edc9f89688f08caaea5401220885233da726fcc"
SOURCE_FILES = ("src/prism_swift/storage.py", "src/prism_swift/retrieval.py",
                "src/prism_swift/validation.py", "evaluation/compare_visual_pipeline.py")


def source_hashes():
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def cpu_limit():
    result = {"requested_cpus": 2, "affinity_applied": False}
    try:
        if hasattr(os, "sched_getaffinity"):
            cpus = sorted(os.sched_getaffinity(0))[:2]
            os.sched_setaffinity(0, cpus)
            result.update(affinity_applied=True, cpus=cpus)
        elif os.name == "nt":
            import ctypes
            from ctypes import wintypes
            fn = ctypes.windll.kernel32.GetCurrentProcess
            fn.restype = wintypes.HANDLE
            process = fn()
            allowed, system = ctypes.c_size_t(), ctypes.c_size_t()
            get = ctypes.windll.kernel32.GetProcessAffinityMask
            get.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
            if not get(process, ctypes.byref(allowed), ctypes.byref(system)):
                raise OSError("affinity_read_failed")
            cpus = [i for i in range(allowed.value.bit_length()) if allowed.value & (1 << i)][:2]
            set_mask = ctypes.windll.kernel32.SetProcessAffinityMask
            set_mask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
            if not set_mask(process, sum(1 << i for i in cpus)):
                raise OSError("affinity_set_failed")
            result.update(affinity_applied=True, cpus=cpus)
    except (OSError, AttributeError) as error:
        result["error_type"] = type(error).__name__
    return result


def memory():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        class Counter(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD),
                        *[(name, ctypes.c_size_t) for name in ("peak", "rss", "a", "b", "c", "d", "e", "f")]]
        value = Counter()
        value.cb = ctypes.sizeof(value)
        current = ctypes.windll.kernel32.GetCurrentProcess
        current.restype = wintypes.HANDLE
        function = ctypes.windll.psapi.GetProcessMemoryInfo
        function.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counter), wintypes.DWORD]
        if function(current(), ctypes.byref(value), value.cb):
            return {"rss_bytes": value.rss, "peak_rss_bytes": value.peak}
    try:
        import resource
        return {"rss_bytes": None, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    except ImportError:
        return {"rss_bytes": None, "peak_rss_bytes": None}


def quantile(values, p):
    ordered = sorted(values)
    location = (len(ordered) - 1) * p
    low = int(location)
    return ordered[low] + (ordered[min(low + 1, len(ordered)-1)] - ordered[low]) * (location-low)


def picture(rng, color, shape):
    image = Image.new("RGB", (512, 512), "white")
    draw = ImageDraw.Draw(image)
    half = rng.randint(125, 170)
    x, y = rng.randint(210, 290), rng.randint(210, 290)
    box = (x-half, y-half, x+half, y+half)
    if shape == "square":
        draw.rectangle(box, fill=COLORS[color])
    elif shape == "circle":
        draw.ellipse(box, fill=COLORS[color])
    else:
        draw.polygon([(x,y-half), (x-half,y+half), (x+half,y+half)], fill=COLORS[color])
    output = io.BytesIO()
    image.save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def source_id(user, request):
    return "swift_" + hashlib.sha256(canonical([user, request, 0]).encode()).hexdigest()[:32]


def corpus(seed):
    rng = random.Random(seed)
    writes, truth, groups = [], {}, {}
    # Gold categories exist only here in the evaluator; content is neutral.
    for user_number in range(2):
        user = f"synthetic-user-{user_number}"
        descriptors = [(color, shape, repeat) for color in COLORS for shape in SHAPES for repeat in range(4)]
        rng.shuffle(descriptors)
        for number, (color, shape, repeat) in enumerate(descriptors):
            request = f"source-{number}"
            code = hashlib.sha256(f"{seed}:{user}:{number}".encode()).hexdigest()[:12]
            uri = picture(rng, color, shape)
            content = [{"type":"text", "text":f"Archive record {code}. This is the image I shared during our gallery visit."},
                       {"type":"image_url", "image_url":{"url":uri}}]
            body = {"request_id":request, "user_id":user, "session_id":f"session-{number//12}",
                    "messages":[{"role":"user", "timestamp":1700000000000 + number*60000, "content":content}]}
            mid = source_id(user,request)
            writes.append(body)
            truth[mid] = {"content":content, "user":user, "code":code, "uri":uri}
            groups.setdefault((user,color,shape),[]).append(mid)
    rng.shuffle(writes)
    queries = []
    for (user,color,shape), relevant in groups.items():
        queries.append({"kind":"visual_text", "body":{"user_id":user,
            "query":f"Find the photograph showing a {color} {shape} on a white background.", "top_k":10},
            "gold": relevant})
    for user_number in range(2):
        user = f"synthetic-user-{user_number}"
        ids = [mid for mid, row in truth.items() if row["user"] == user]
        rng.shuffle(ids)
        for mid in ids[:6]:
            queries.append({"kind":"image_identity", "body":{"user_id":user, "top_k":10,
                "query":[{"type":"text","text":"Find the earlier record for this attached photograph."},
                         {"type":"image_url","image_url":{"url":truth[mid]["uri"]}}]}, "gold":[mid]})
        for mid in ids[6:12]:
            queries.append({"kind":"lexical", "body":{"user_id":user, "top_k":10,
                "query":f"Recall archive record {truth[mid]['code']}."}, "gold":[mid]})
    return writes, queries, truth


class TinyCLIPPipeline:
    """Experimental source+FTS+visual adapter. Vectors commit with source+receipt."""
    def __init__(self, path, model_dir):
        import torch
        from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizerFast
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        self.torch = torch
        if hashlib.sha256((model_dir/"model.safetensors").read_bytes()).hexdigest() != WEIGHTS_SHA:
            raise ValueError("Pinned TinyCLIP weight SHA256 mismatch")
        self.model = CLIPModel.from_pretrained(str(model_dir), local_files_only=True).to("cpu").eval()
        self.processor = CLIPImageProcessor.from_pretrained(str(model_dir),local_files_only=True)
        self.tokenizer = CLIPTokenizerFast(tokenizer_file=str(model_dir/"tokenizer.json"))
        self.store = MemoryStore(path,strategy="sparse")
        self.image_encodes = self.text_encodes = 0

    def encode_image(self, uri):
        image = Image.open(io.BytesIO(base64.b64decode(uri.split(",",1)[1]))).convert("RGB")
        inputs = self.processor(images=image,return_tensors="pt")
        with self.torch.inference_mode():
            output = self.model.get_image_features(**inputs)
        self.image_encodes += 1
        return (output/output.norm(dim=-1,keepdim=True))[0].tolist()

    def add(self, body):
        messages = []
        for message in body["messages"]:
            uris = [part["image_url"]["url"] for part in message["content"] if part["type"] == "image_url"]
            vectors = [self.encode_image(uri) for uri in uris]
            # Extra envelope field, not content. Stored by the same MemoryStore.add
            # transaction; never writes vector after an acknowledged source commit.
            messages.append({**message,"_experimental_tinyclip_vectors":vectors})
        return self.store.add({**body,"messages":messages})

    def search(self, body):
        query = body["query"]
        if isinstance(query,str):
            if not any(shape in query for shape in SHAPES):
                return self.store.search(body)
            inputs = self.tokenizer([query],return_tensors="pt",padding=True,truncation=True,max_length=77)
            with self.torch.inference_mode():
                vector = self.model.get_text_features(**inputs)
            vector = vector/vector.norm(dim=-1,keepdim=True)
            self.text_encodes += 1
        else:
            uris = [part["image_url"]["url"] for part in query if part["type"] == "image_url"]
            vector = self.torch.tensor([self.encode_image(uris[0])])
        conn = self.store._connection()
        conn.execute("BEGIN")
        try:
            records = []
            for row in conn.execute("SELECT id,source_json,media_bytes,created_at FROM messages WHERE user_id=?",(body["user_id"],)):
                source = json.loads(row[1])
                for item in source.get("_experimental_tinyclip_vectors",[]):
                    records.append((row[0],source,row[2],row[3],item))
            if not records:
                conn.commit()
                return {"data":[]}
            matrix = self.torch.tensor([row[4] for row in records])
            scores = (matrix@vector.T).flatten().tolist()
            order = sorted(range(len(records)),key=lambda i:(-scores[i],records[i][0]))
            output, media, wire = [],0,12
            used = set()
            for index in order:
                mid,source,size,created,_ = records[index]
                if mid in used:
                    continue
                def blob(digest):
                    value = conn.execute("SELECT data FROM blobs WHERE user_id=? AND digest=?",(body["user_id"],digest)).fetchone()
                    if value is None:
                        raise ValueError("Missing scoped blob")
                    return value[0]
                item = {"id":mid,"content":restore(source["content"],blob),"score":scores[index],"created_at":created}
                length = len(canonical(item).encode())+1
                if media+size>self.store.max_media_bytes or wire+length>self.store.response_bytes:
                    break
                output.append(item)
                used.add(mid)
                wire += length
                media += size
                if len(output) >= body["top_k"]:
                    break
            conn.commit()
            return {"data":output}
        except BaseException:
            conn.rollback()
            raise


def worker(args):
    hashes = source_hashes()
    affinity = cpu_limit()
    writes, queries, truth = corpus(args.seed)
    digest = hashlib.sha256(canonical(writes).encode()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="swift-visual-pipeline-") as folder:
        path = Path(folder)/"memory.sqlite3"
        setup_start = time.perf_counter()
        if args.worker == "B_tinyclip":
            pipeline = TinyCLIPPipeline(path,args.model_dir)
            store = pipeline.store
        else:
            strategy = "raw" if args.worker == "raw" else "sparse" if args.worker == "A_sparse" else "adaptive"
            ablation = args.worker.replace("C_","") if args.worker.startswith("C_no_") else "none"
            pipeline = store = MemoryStore(path,strategy=strategy,ablation=ablation)
        setup_seconds = time.perf_counter()-setup_start
        add_times, search_times, returned = [],[],[]
        start_cpu,start = time.process_time(),time.perf_counter()
        for body in writes:
            tick = time.perf_counter()
            # Include JSON serialization/parsing plus actual sync persistence.
            result = pipeline.add(json.loads(canonical(body)))
            if result.get("success") is not True:
                raise AssertionError("Add did not succeed")
            canonical(result)
            add_times.append(time.perf_counter()-tick)
        add_wall = time.perf_counter()-start
        add_cpu = time.process_time()-start_cpu
        quality = {}
        evidence_total = exact_evidence = 0
        for query in queries:
            tick = time.perf_counter()
            response = pipeline.search(json.loads(canonical(query["body"])))
            canonical(response)
            search_times.append(time.perf_counter()-tick)
            ids = [item["id"] for item in response["data"]]
            if len(ids) > query["body"]["top_k"] or len(set(ids)) != len(ids):
                raise AssertionError("Top K or duplicate-source violation")
            for item in response["data"]:
                evidence_total += 1
                if truth[item["id"]]["user"] != query["body"]["user_id"]:
                    raise AssertionError("Cross-user leak")
                exact_evidence += canonical(item["content"]) == canonical(truth[item["id"]]["content"])
            gold = set(query["gold"])
            ranks = [ids.index(mid)+1 for mid in gold if mid in ids]
            row = {f"recall_at_{k}":len(gold.intersection(ids[:k]))/len(gold) for k in (1,5,10)}
            row["mrr"] = 1/min(ranks) if ranks else 0
            row["hit_at_1"] = float(bool(ids) and ids[0] in gold)
            row["ndcg_at_10"] = sum(1/math.log2(i+2) for i,mid in enumerate(ids[:10]) if mid in gold)/sum(1/math.log2(i+2) for i in range(min(10,len(gold))))
            quality.setdefault(query["kind"],[]).append(row)
        stored_vector_bytes = sum(len(canonical(json.loads(row[0]).get("_experimental_tinyclip_vectors",[])).encode())
                                  for row in store._connection().execute("SELECT source_json FROM messages")) if args.worker == "B_tinyclip" else 0
        footprint_open = sum(p.stat().st_size for p in Path(folder).glob("memory.sqlite3*") if p.is_file())
        count = store._connection().execute("SELECT count(*) FROM messages").fetchone()[0]
        if count != 144:
            raise AssertionError("Wrong durable source count")
        # Close and reopen the committed database. Reuse only the loaded encoder,
        # never an in-memory vector index: B loads persisted vectors at Search.
        store.close()
        reopened = MemoryStore(path,strategy=store.strategy,ablation=store.ablation)
        if args.worker == "B_tinyclip":
            pipeline.store = reopened
        else:
            pipeline = reopened
        repeat = queries[-1]
        restart_response = pipeline.search(repeat["body"])
        if not set(repeat["gold"]).intersection(item["id"] for item in restart_response["data"]):
            if repeat["kind"] == "lexical" and args.worker != "raw":
                raise AssertionError("Persisted lexical source unavailable after reopen")
        reopened.close()
        footprint_closed = path.stat().st_size
        resources = memory()
        summarized = {kind:{metric:statistics.mean(row[metric] for row in rows) for metric in rows[0]}
                      for kind,rows in quality.items()}
        for kind,rows in quality.items():
            summarized[kind]["queries"] = len(rows)
        source_wire_bytes = sum(len(canonical(body).encode()) for body in writes)
        if source_hashes() != hashes:
            raise RuntimeError("Source changed during measurement; discard and rerun the affected comparison")
        return {"variant":args.worker,"seed":args.seed,"source_manifest_sha256":digest,
                "source_sha256":hashes,
                "sources":144,"images":144,"queries":len(queries),"source_resolution":[512,512],
                "source_wire_bytes":source_wire_bytes,"setup_seconds":setup_seconds,
                "add_seconds":add_wall,"add_requests_per_second":144/add_wall,
                "add_messages_per_second":144/add_wall,"add_images_per_second":144/add_wall,
                "add_source_wire_bytes_per_second":source_wire_bytes/add_wall,
                "add_cpu_seconds":add_cpu,"add_cpu_percent_one_core_scale":100*add_cpu/add_wall,
                "add_p50_ms":1000*statistics.median(add_times),"add_p95_ms":1000*quantile(add_times,.95),
                "add_p99_ms":1000*quantile(add_times,.99),
                "search_p50_ms":1000*statistics.median(search_times),"search_p95_ms":1000*quantile(search_times,.95),
                "disk_bytes_open_including_wal_shm":footprint_open,"disk_bytes_after_close":footprint_closed,
                "stored_vector_json_bytes":stored_vector_bytes,"evidence_integrity":exact_evidence/max(1,evidence_total),
                "returned_evidence_count":evidence_total,"quality":summarized,"resources":resources,
                "cpu_affinity":affinity,"journal_mode":"wal","synchronous":"FULL",
                "real_api_calls":0,"gpu_used":False,"error_count":0,
                "image_encodes":getattr(pipeline,"image_encodes",0),"text_encodes":getattr(pipeline,"text_encodes",0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker",choices=VARIANTS)
    parser.add_argument("--seed",type=int,default=11)
    parser.add_argument("--seeds",type=int,nargs="+",default=[11,23,37])
    parser.add_argument("--variants",choices=VARIANTS,nargs="+",default=list(VARIANTS))
    parser.add_argument("--model-dir",type=Path,default=ROOT/"artifacts/models/tinyclip")
    parser.add_argument("--output",type=Path,default=ROOT/"docs/visual_pipeline_comparison.json")
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args),ensure_ascii=True))
        return
    runs = []
    for index,seed in enumerate(args.seeds):
        variants = list(args.variants)
        variants = variants[index:]+variants[:index]
        for variant in variants:
            print(f"Running independent pipeline: seed={seed} variant={variant}",file=sys.stderr,flush=True)
            result = subprocess.run([sys.executable,"-B",str(Path(__file__).resolve()),"--worker",variant,
                "--seed",str(seed),"--model-dir",str(args.model_dir)],capture_output=True,text=True,check=False)
            if result.returncode:
                raise RuntimeError(f"Pipeline failed {variant} seed={seed}: {result.stderr[-4000:]}")
            runs.append(json.loads(result.stdout))
    for seed in args.seeds:
        if len({row["source_manifest_sha256"] for row in runs if row["seed"]==seed}) != 1:
            raise AssertionError("Variants received unequal source corpora")
    if len({canonical(row["source_sha256"]) for row in runs}) != 1:
        raise RuntimeError("Measured source changed between pipeline runs; do not aggregate")
    summary = {}
    metrics = ("add_requests_per_second","add_p50_ms","add_p95_ms","add_p99_ms","search_p50_ms","search_p95_ms",
               "disk_bytes_after_close","stored_vector_json_bytes","evidence_integrity","setup_seconds")
    for variant in args.variants:
        values = [row for row in runs if row["variant"]==variant]
        summary[variant] = {metric:{"mean":statistics.mean(row[metric] for row in values),
                                    "std_across_generated_seeds":statistics.stdev(row[metric] for row in values) if len(values)>1 else None}
                            for metric in metrics}
        summary[variant]["quality"] = {kind:{metric:statistics.mean(row["quality"][kind][metric] for row in values)
                                            for metric in values[0]["quality"][kind]}
                                       for kind in values[0]["quality"]}
    output = {"kind":"actual_sqlite_full_pipeline_synthetic_comparison_not_aml_score", "python":sys.version,
        "platform":platform.platform(),"logical_cpus":os.cpu_count(),"seeds":args.seeds,
        "run_order":"rotating variant order by seed; independent child process per variant/seed",
        "method":"Same generated sources, CPU affinity requested 2, batch size 1, sequential Add. Includes validation, model encoding when B, FULL transaction, JSON processing and evidence reconstruction; excludes HTTP/network.",
        "B_scope":"Experimental Industrial adapter only; image vectors stored as extra source envelope fields in the same transaction as blobs/FTS/receipt. No metadata label is passed to writer. Search scans persisted vectors scoped by user (O(N)); no cross-user cache.",
        "limitations":["144 easy synthetic color/shape source messages per seed; not natural-image/official QA accuracy.",
            "B is a complete experimental store adapter, not the default public API and not claimed Academic-eligible.",
            "All source PNGs are small/compressible; high-entropy/large requests are covered by the main HTTP benchmark, not this diagnostic.",
            "Source manifests are generated independently by seed; no model training or hyperparameter search.",
            "Only B imports torch, so its fresh-child RSS includes optional model dependencies.",
            "A/B/C source resolution, exact input, FULL durability, Top K=10 and response limits are identical; B's derived vector is extra stored cost.",
            "No cloud run, formal Smoke/Full, retrieval Answer model, API inference, or GPU is used."],
        "model":{"name":"wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M","revision":MODEL_REVISION,"weights_sha256":WEIGHTS_SHA,
                 "device":"cpu","threads":2,"license":"MIT model card"},
        "source_sha256":runs[0]["source_sha256"],
        "runs":runs,"summary":summary}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({variant:{"add_rps":round(value["add_requests_per_second"]["mean"],2),
                              "visual_text_r1":value["quality"]["visual_text"]["recall_at_1"]}
                      for variant,value in summary.items()}))


if __name__ == "__main__":
    main()
