"""Real query-only visual reranking on synthetic sources, no training/private data."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"evaluation"))
from compare_visual_pipeline import corpus, cpu_limit, memory, quantile
from prism_swift.storage import MemoryStore
from prism_swift.visual_reranker import TinyCLIPReranker


def worker(args):
    affinity=cpu_limit()
    load=time.perf_counter()
    reranker=TinyCLIPReranker(args.model_dir,threads=2,candidates=args.worker)
    setup=time.perf_counter()-load
    writes,all_queries,truth=corpus(args.seed)
    queries=[]
    rng=random.Random(args.seed+9101)
    for kind in ("visual_text","image_identity","lexical"):
        group=[query for query in all_queries if query["kind"]==kind]
        queries.extend(rng.sample(group,min(args.queries_per_kind,len(group))))
    with tempfile.TemporaryDirectory(prefix="prism-query-rerank-") as directory:
        stores=[MemoryStore(str(Path(directory)/"baseline.sqlite3")),
                MemoryStore(str(Path(directory)/"reranker.sqlite3"),reranker=reranker)]
        summaries=[]
        for store in stores:
            started=time.perf_counter()
            for value in writes: store.add(value)
            add_seconds=time.perf_counter()-started
            store.close()
            rows=[]
            for query in queries:
                actual=query["body"]
                started=time.perf_counter()
                result=store.search(actual)["data"]
                latency=time.perf_counter()-started
                ids=[row["id"] for row in result]
                wanted=set(query["gold"])
                hit=bool(ids and ids[0] in wanted)
                recall5=len(wanted.intersection(ids[:5]))/len(wanted)
                recall10=len(wanted.intersection(ids[:10]))/len(wanted)
                complete=all(row["content"]==truth[row["id"]]["content"] for row in result)
                rows.append({"kind":query["kind"],"hit_at_1":hit,"recall_at_5":recall5,"recall_at_10":recall10,
                             "source_complete":complete,"seconds":latency,"returned":len(ids)})
            summary={kind:{"queries":len(subset:=[row for row in rows if row["kind"]==kind]),
                          "hit_at_1":statistics.mean(row["hit_at_1"] for row in subset),
                          "recall_at_5":statistics.mean(row["recall_at_5"] for row in subset),
                          "recall_at_10":statistics.mean(row["recall_at_10"] for row in subset),
                          "search_p50_ms":1000*statistics.median(row["seconds"] for row in subset),
                          "search_p95_ms":1000*quantile([row["seconds"] for row in subset],.95)}
                     for kind in ("visual_text","image_identity","lexical")}
            summaries.append({"reranker":store.reranker is not None,"add_seconds":add_seconds,
                              "add_requests_per_second":len(writes)/add_seconds,"quality":summary,
                              "source_completeness":sum(row["source_complete"] for row in rows)/len(rows),"queries":rows})
            store.close()
        return {"seed":args.seed,"candidates":args.worker,"cpu_affinity":affinity,"model_setup_seconds":setup,
                "sources":len(writes),"summaries":summaries,"resources":memory(),"remote_model_calls":0,
                "local_model":reranker.config_dict()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker",type=int)
    parser.add_argument("--seed",type=int,default=11)
    parser.add_argument("--queries-per-kind",type=int,default=6)
    parser.add_argument("--model-dir",type=Path,default=ROOT/"artifacts/models/tinyclip")
    parser.add_argument("--output",type=Path,default=ROOT/"results/query-reranker-20261016.json")
    args=parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args)));return
    runs=[]
    for seed in (11,23,37):
        for count in (32,64):
            print("Query reranker",seed,count,file=sys.stderr,flush=True)
            result=subprocess.run([sys.executable,"-B",str(Path(__file__).resolve()),"--worker",str(count),
                "--seed",str(seed),"--queries-per-kind",str(args.queries_per_kind),"--model-dir",str(args.model_dir)],
                capture_output=True,text=True,timeout=600)
            if result.returncode: raise RuntimeError(result.stderr[-3000:])
            runs.append(json.loads(result.stdout))
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps({"kind":"query_only_real_local_encoder_synthetic_diagnostic_not_aml",
                "platform":platform.platform(),"python":sys.version,"runs":runs,
                "source_sha256":{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/"src").rglob("*.py")},
                "limitations":["Developer-generated shape/color sources, not natural-image quality or official scores.",
                    "Same Windows FS and two CPU affinity for both stores; excludes HTTP/upload/platform stages.",
                    "No encoder is invoked during Add; optional local model is preloaded in the process for both paired stores.",
                    "Candidate prefix may miss relevant evidence; visual model cannot recover a source outside it.",
                    "Runs are independently generated seeds, not a trained model evaluation."]},indent=2)+"\n")
    print(json.dumps({"runs":len(runs),"output":str(args.output),"remote_model_calls":0}))


if __name__=="__main__": main()
