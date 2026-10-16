"""Optional TinyCLIP CPU diagnostic; separate from the submission runtime.

Only downloads the pinned public MIT model when --download is explicitly set.
Uses synthetic images, no provider key, no inference API, and no formal-run data.
Install its optional torch/transformers/Pillow dependencies in a separate env.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import time
import urllib.request

REPO = "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M"
REVISION = "a2a8c6eaa2549ad66eb7c31b85022bf58273a26c"
FILES = ("config.json", "preprocessor_config.json", "tokenizer.json", "model.safetensors")
MODEL_BYTES = 93_812_468
MODEL_SHA256 = "9339ee3d736344d0ddcaa6c03edc9f89688f08caaea5401220885233da726fcc"


def download_weights(folder: Path, proxy_from_git: bool = False) -> dict:
    opener = urllib.request.build_opener()
    if proxy_from_git:
        proxy = subprocess.run(["git", "config", "--get", "http.proxy"],
                               capture_output=True, text=True, check=False).stdout.strip()
        if not proxy:
            raise RuntimeError("No Git HTTP proxy configured")
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    folder.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    downloaded = 0
    for name in FILES:
        target = folder / name
        if target.is_file():
            continue
        partial = target.with_suffix(target.suffix + ".partial")
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}?download=true"
        size = 0
        with partial.open("wb") as stream:
            if name == "model.safetensors":
                # Some proxies buffer a full large GET. Bounded validated ranges
                # keep this optional public download observable and reproducible.
                chunk = 2 * 1024 * 1024
                ranges = [(left, min(MODEL_BYTES - 1, left + chunk - 1))
                          for left in range(0, MODEL_BYTES, chunk)]
                def get_range(pair):
                    left, right = pair
                    req = urllib.request.Request(url, headers={"Range": f"bytes={left}-{right}"})
                    with opener.open(req, timeout=30) as response:
                        expected = f"bytes {left}-{right}/{MODEL_BYTES}"
                        if response.status != 206 or response.headers.get("Content-Range") != expected:
                            raise ValueError("Public download did not honor the exact bounded range")
                        block = response.read(right - left + 2)
                        if len(block) != right - left + 1:
                            raise ValueError("Incomplete public model range")
                        return block
                with ThreadPoolExecutor(max_workers=4) as pool:
                    for block in pool.map(get_range, ranges):
                        size += len(block)
                        stream.write(block)
            else:
                with opener.open(url, timeout=30) as response:
                    while block := response.read(1024 * 1024):
                        size += len(block)
                        if size > 110 * 1024 * 1024:
                            raise ValueError("Public model asset exceeds the 110 MiB per-file bound")
                        stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
        if name == "model.safetensors" and hashlib.sha256(partial.read_bytes()).hexdigest() != MODEL_SHA256:
            raise ValueError("Downloaded weights do not match the pinned public LFS SHA256")
        partial.replace(target)
        downloaded += size
    return {"downloaded_bytes": downloaded, "download_seconds": time.perf_counter() - start}


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction)))]


def memory_usage():
    try:
        import psutil
        return {"rss_bytes": psutil.Process().memory_info().rss, "peak_rss_bytes": None}
    except ImportError:
        if os.name != "nt":
            return {"rss_bytes": None, "peak_rss_bytes": None}
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        *[(name, ctypes.c_size_t) for name in (
                            "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                            "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                            "PagefileUsage", "PeakPagefileUsage")]]
        value = Counters()
        value.cb = ctypes.sizeof(value)
        get_process = ctypes.windll.kernel32.GetCurrentProcess
        get_process.restype = wintypes.HANDLE
        fn = ctypes.windll.psapi.GetProcessMemoryInfo
        fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not fn(get_process(), ctypes.byref(value), value.cb):
            return {"rss_bytes": None, "peak_rss_bytes": None}
        return {"rss_bytes": value.WorkingSetSize, "peak_rss_bytes": value.PeakWorkingSetSize}


def run(folder: Path, repeats: int, download_info: dict) -> dict:
    started = time.perf_counter()
    cpu_started = time.process_time()
    print("encoder_probe: importing optional CPU dependencies", file=sys.stderr, flush=True)
    import torch
    import transformers
    from PIL import Image, ImageDraw
    from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizerFast
    dependency_import_seconds = time.perf_counter() - started

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    load_start = time.perf_counter()
    print("encoder_probe: loading pinned local weights", file=sys.stderr, flush=True)
    model = CLIPModel.from_pretrained(str(folder), local_files_only=True).to("cpu").eval()
    image_processor = CLIPImageProcessor.from_pretrained(str(folder), local_files_only=True)
    tokenizer = CLIPTokenizerFast(tokenizer_file=str(folder / "tokenizer.json"))
    load_seconds = time.perf_counter() - load_start
    print("encoder_probe: measuring synthetic encoding and retrieval", file=sys.stderr, flush=True)
    pngs, labels = [], []
    colors = {"red": "#d91f26", "green": "#159b43", "blue": "#185bd9",
              "yellow": "#e4d719", "purple": "#8625a7", "orange": "#e98a21"}
    for color, rgb in colors.items():
        for shape in ("square", "circle", "triangle"):
            image = Image.new("RGB", (512, 512), "white")
            draw = ImageDraw.Draw(image)
            if shape == "square":
                draw.rectangle((96, 96, 416, 416), fill=rgb)
            elif shape == "circle":
                draw.ellipse((96, 96, 416, 416), fill=rgb)
            else:
                draw.polygon([(256, 80), (80, 432), (432, 432)], fill=rgb)
            stream = io.BytesIO()
            image.save(stream, format="PNG")
            pngs.append(stream.getvalue())
            labels.append(f"a {color} {shape} on a white background")

    def encode_one(raw):
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        inputs = image_processor(images=image, return_tensors="pt")
        with torch.inference_mode():
            vector = model.get_image_features(**inputs)
        return vector / vector.norm(dim=-1, keepdim=True)

    with torch.inference_mode():
        text_inputs = tokenizer(labels, padding=True, truncation=True, max_length=77, return_tensors="pt")
        text_start = time.perf_counter()
        text_vectors = model.get_text_features(**text_inputs)
        text_vectors = text_vectors / text_vectors.norm(dim=-1, keepdim=True)
        text_seconds = time.perf_counter() - text_start
    cold_start = time.perf_counter()
    encode_one(pngs[0])
    first_seconds = time.perf_counter() - cold_start
    start_to_first_image_seconds = time.perf_counter() - started
    image_seconds, vectors = [], None
    for _ in range(repeats):
        round_vectors = []
        for raw in pngs:
            start = time.perf_counter()
            round_vectors.append(encode_one(raw))
            image_seconds.append(time.perf_counter() - start)
        vectors = torch.cat(round_vectors)
    similarity = text_vectors @ vectors.T
    ranks = []
    for index, row in enumerate(similarity):
        ranking = row.argsort(descending=True).tolist()
        ranks.append(ranking.index(index) + 1)
    resources = memory_usage()
    assets = {name: {"bytes": (folder / name).stat().st_size,
                     "sha256": hashlib.sha256((folder / name).read_bytes()).hexdigest()} for name in FILES}
    return {
        "kind": "optional_encoder_probe_not_aml_score",
        "model": REPO, "revision": REVISION, "license": "MIT (model card)",
        "academic_leaderboard_eligible": None,
        "eligibility_note": "Non-standard embedding model: do not enable for Academic submission without organizer approval; Industrial comparison only.",
        "source_url": f"https://huggingface.co/{REPO}/tree/{REVISION}",
        "real_inference_api_calls": 0, "download": download_info,
        "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "device": "cpu", "torch_threads": 2, "torch_interop_threads": 1,
        "host_logical_cpus": os.cpu_count(), "gpu_used": False,
        "source_resolution": [512, 512], "processed_resolution": [224, 224],
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "assets": assets, "model_load_seconds": load_seconds,
        "dependency_import_seconds": dependency_import_seconds,
        "process_run_to_first_image_seconds": start_to_first_image_seconds,
        "first_image_preprocess_and_encode_seconds": first_seconds,
        "warm_image_repeats": repeats, "warm_images_total": len(image_seconds),
        "warm_image_seconds_p50": statistics.median(image_seconds),
        "warm_image_seconds_p95": percentile(image_seconds, .95),
        "warm_image_seconds_p99": percentile(image_seconds, .99),
        "warm_images_per_second": len(image_seconds) / sum(image_seconds),
        "text_encode_18_labels_seconds": text_seconds,
        "observed_process_rss_bytes": resources["rss_bytes"],
        "observed_peak_rss_bytes": resources["peak_rss_bytes"],
        "probe_process_cpu_seconds": time.process_time() - cpu_started,
        "probe_process_wall_seconds": time.perf_counter() - started,
        "synthetic_shape_color_text_to_image": {
            "queries": len(ranks), "recall_at_1": sum(r <= 1 for r in ranks) / len(ranks),
            "recall_at_5": sum(r <= 5 for r in ranks) / len(ranks),
            "mrr": sum(1 / r for r in ranks) / len(ranks), "ranks": ranks,
        },
        "limitations": ["Synthetic shape/color diagnostic; no natural-image or official QA inference.",
                        "CPU thread count is 2 on a 24-logical-CPU Windows host, not the user's cloud VM.",
                        "No DB, HTTP request, durable commit, upload, queue, or Answer/Eval time included.",
                        "CPU set/processor contention not isolated; no deployment-memory guarantee.",
                        "All measured repeats are deterministic timing repeats, not independent accuracy seeds."]
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/models/tinyclip"))
    parser.add_argument("--output", type=Path, default=Path("docs/encoder_probe.json"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--proxy-from-git", action="store_true", help="Use Git's configured proxy without printing it")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    download_info = download_weights(args.model_dir, args.proxy_from_git) if args.download else {
        "downloaded_bytes": 0, "download_seconds": 0}
    if not all((args.model_dir / name).is_file() for name in FILES):
        parser.error("Pinned public weights unavailable; use --download or supply a complete model directory")
    result = run(args.model_dir, args.repeats, download_info)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": args.output.as_posix(), "p50_ms": result["warm_image_seconds_p50"] * 1000,
                      "synthetic_recall_at_1": result["synthetic_shape_color_text_to_image"]["recall_at_1"]}))


if __name__ == "__main__":
    main()
