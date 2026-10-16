"""Optional, local-only TinyCLIP evidence reranker (Industrial configuration).

No network, source rewrite, answer generation, or cross-query embedding cache.
Only already-retrievable candidate images are encoded. Computation budgets do
not remove an image from the authoritative source or change Add semantics.
"""
from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path
import threading


MODEL = "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M"
REVISION = "a2a8c6eaa2549ad66eb7c31b85022bf58273a26c"
ASSETS = {
    "model.safetensors": "9339ee3d736344d0ddcaa6c03edc9f89688f08caaea5401220885233da726fcc",
    "config.json": "c586d856dbd5589b67525db9e791ee0917402405672c7773aead373efb7d7bca",
    "preprocessor_config.json": "910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd",
    "tokenizer.json": "b556ac8c99757ffb677208af34bc8c6721572114111a6e0aaf5fa69ff0b8d842",
}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024*1024):
            digest.update(block)
    return digest.hexdigest()


class TinyCLIPReranker:
    def __init__(self, model_path, threads=2, candidates=32, max_images=64):
        if isinstance(threads,bool) or not isinstance(threads,int) or not 1 <= threads <= 8:
            raise ValueError("Reranker threads must be an integer from 1 to 8")
        if isinstance(candidates,bool) or not isinstance(candidates,int) or not 1 <= candidates <= 100:
            raise ValueError("Reranker candidates must be an integer from 1 to 100")
        if isinstance(max_images,bool) or not isinstance(max_images,int) or not 1 <= max_images <= 64:
            raise ValueError("Reranker max_images must be an integer from 1 to 64")
        folder = Path(model_path)
        for name, expected in ASSETS.items():
            path = folder/name
            if not path.is_file() or _sha256(path) != expected:
                raise ValueError("Pinned local reranker asset missing or SHA256 mismatch: " + name)
        # Imports intentionally live here, so the default no-model service does
        # not need torch/transformers or import their CUDA-capable builds.
        import torch
        from PIL import Image
        from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizerFast
        torch.set_num_threads(threads)
        self.torch, self.Image = torch, Image
        self.threads, self.candidates, self.max_images = threads, candidates, max_images
        self.model = CLIPModel.from_pretrained(str(folder),local_files_only=True,
                                             use_safetensors=True).to("cpu").eval()
        self.processor = CLIPImageProcessor.from_pretrained(str(folder),local_files_only=True)
        self.tokenizer = CLIPTokenizerFast(tokenizer_file=str(folder/"tokenizer.json"))
        self._lock = threading.Lock()

    def config_dict(self):
        return {"model":MODEL,"revision":REVISION,"weights_sha256":ASSETS["model.safetensors"],
                "threads":self.threads,"candidates":self.candidates,"max_images":self.max_images}

    def _encode_images(self, values):
        images = []
        try:
            for raw in values:
                if not isinstance(raw,bytes) or not 0 < len(raw) <= 10*1024**2:
                    raise ValueError("Reranker image bytes must be nonempty and at most 10 MiB")
                with self.Image.open(io.BytesIO(raw)) as image:
                    if image.format not in {"PNG","JPEG","WEBP"} or image.width*image.height > 16_000_000:
                        raise ValueError("Reranker image exceeds supported format/pixel capacity")
                    images.append(image.convert("RGB"))
            inputs = self.processor(images=images,return_tensors="pt")
            with self.torch.inference_mode():
                vectors = self.model.get_image_features(**inputs)
                return vectors/vectors.norm(dim=-1,keepdim=True).clamp_min(1e-12)
        finally:
            for image in images:
                image.close()

    def scores(self, query_text, query_images, candidates):
        """Return cosine evidence scores for evaluated IDs; never a final answer.

        Candidate records are considered in supplied primary-retrieval order.
        Only the first ``candidates`` records and first ``max_images`` image
        occurrences are evaluated. Total candidate media also has a 30 MiB
        prefix budget. Limits affect rerank computation, never source content.
        Duplicate image bytes are encoded once within this call only.
        """
        if not isinstance(query_text,str) or not isinstance(query_images,list) or not isinstance(candidates,list):
            raise ValueError("Invalid reranker query/candidate types")
        if any(not isinstance(raw,bytes) or not raw for raw in query_images):
            raise ValueError("Reranker query image must be nonempty bytes")
        if len(query_images) > 64 or sum(len(value) for value in query_images) > 30*1024**2:
            raise ValueError("Reranker query images exceed declared capacity")
        if not query_text.strip() and not query_images:
            return {}
        selected, occurrences, total = [],0,0
        for row in candidates[:self.candidates]:
            if not isinstance(row,dict) or not isinstance(row.get("id"),str) or not row["id"]:
                raise ValueError("Reranker candidate requires a nonempty ID")
            if not isinstance(row.get("image_bytes"),list):
                raise ValueError("Reranker candidate requires an image_bytes array")
            values = []
            for raw in row["image_bytes"]:
                if not isinstance(raw,bytes) or not raw:
                    raise ValueError("Reranker candidate image must be bytes")
                if occurrences >= self.max_images or total+len(raw) > 30*1024**2:
                    break
                values.append(raw)
                occurrences += 1
                total += len(raw)
            if values:
                selected.append((row["id"],values))
            if occurrences >= self.max_images or total >= 30*1024**2:
                break
        if not selected:
            return {}
        with self._lock:
            cache, unique = {},[]
            for raw in query_images + [raw for _,values in selected for raw in values]:
                if not isinstance(raw,bytes):
                    raise ValueError("Reranker query image must be bytes")
                key = hashlib.sha256(raw).digest()
                if key not in cache:
                    cache[key] = None
                    unique.append((key,raw))
            for start in range(0,len(unique),4):
                batch = unique[start:start+4]
                vectors = self._encode_images([raw for _,raw in batch])
                for (key,_),vector in zip(batch,vectors):
                    cache[key] = vector
            text_vector = None
            if query_text.strip():
                inputs = self.tokenizer([query_text],return_tensors="pt",padding=True,
                                        truncation=True,max_length=77)
                with self.torch.inference_mode():
                    text_vector = self.model.get_text_features(**inputs)[0]
                    text_vector = text_vector/text_vector.norm().clamp_min(1e-12)
            query_vectors = [cache[hashlib.sha256(raw).digest()] for raw in query_images]
            result = {}
            for mid,values in selected:
                matrix = self.torch.stack([cache[hashlib.sha256(raw).digest()] for raw in values])
                text_score = (matrix@text_vector).max().item() if text_vector is not None else None
                image_score = sum((matrix@vector).max().item() for vector in query_vectors)/len(query_vectors) if query_vectors else None
                if image_score is not None and text_score is not None:
                    score = .75*image_score + .25*text_score
                else:
                    score = image_score if image_score is not None else text_score
                if score is not None and math.isfinite(score):
                    result[mid] = max(-1.,min(1.,float(score)))
            return result
