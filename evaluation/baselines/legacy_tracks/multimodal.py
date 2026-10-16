"""Ordered image/text evidence with local visual retrieval and optional Add indexing.

Base64-decoded image files are limited to 10 MiB by the AML contract. The
16-million-pixel default is an additional, configurable service capacity limit.
Captions are derived index text: Search always receives the original evidence.
"""

from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import io
import re
import warnings
from typing import Any

from PIL import Image, UnidentifiedImageError

from records import PayloadTooLarge, PreparedEvidence, PreparedQuery, ProviderError


MAX_IMAGE_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_PIXELS = 16_000_000
IMAGE_CANDIDATE_MARKER = "_image_evidence_"
_URI_PATTERN = re.compile(
    r"\Adata:(image/(?:jpeg|png|webp));base64,([A-Za-z0-9+/]+={0,2})\Z",
    re.IGNORECASE,
)
_MIME_FORMATS = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}


class MultimodalProcessor:
    """Prepare one evidence record per source message; never answer a query."""

    name = "multimodal"

    def __init__(self, model: Any = None, image_features: bool = True,
                 max_pixels: int = DEFAULT_MAX_PIXELS) -> None:
        if isinstance(max_pixels, bool) or not isinstance(max_pixels, int) or max_pixels < 1:
            raise ValueError("max_pixels must be a positive integer")
        self.model = model
        self.image_features = bool(image_features)
        self.max_pixels = max_pixels

    @property
    def _model_enabled(self) -> bool:
        return self.model is not None and bool(getattr(self.model, "enabled", True))

    def config_dict(self) -> dict[str, Any]:
        """JSON-serializable parameters to freeze with the memory database."""
        return {
            "processor": self.name,
            "feature_algorithm": "sha256-dhash64-rgbhist8-v1",
            "image_features": self.image_features,
            "model_enabled": self._model_enabled,
            "max_image_bytes": MAX_IMAGE_BYTES,
            "max_image_pixels": self.max_pixels,
            "feature_frame": "first",
        }

    def prepare(self, content: str | list[dict[str, Any]]) -> list[PreparedEvidence]:
        original, source_text, images, image_uris = self._parse(content)
        # Validate every part before any model request; malformed payloads must
        # not spend an otherwise valid first image's indexing call.
        index_parts = [source_text] if source_text else []
        if images:
            index_parts.append(IMAGE_CANDIDATE_MARKER)
            if self._model_enabled:
                describe = getattr(self.model, "describe_image", None)
                if not callable(describe):
                    raise ProviderError("The indexing provider does not implement describe_image")
                for uri in image_uris:
                    caption = describe(uri)
                    if not isinstance(caption, str) or not caption.strip():
                        raise ProviderError("Image indexing returned an empty or invalid caption")
                    index_parts.append(caption)
        elif self._model_enabled:
            # The provider's generic text alias is preferred. index_code is a
            # compatible fallback for clients exposing the shared text routine.
            indexer = getattr(self.model, "index_text", None)
            if not callable(indexer):
                indexer = getattr(self.model, "index_code", None)
            if not callable(indexer):
                raise ProviderError("The indexing provider does not implement text indexing")
            derived = indexer(source_text)
            if isinstance(derived, str) and derived.strip():
                index_parts.append(derived)
            elif isinstance(derived, list) and all(isinstance(item, str) for item in derived):
                index_parts.extend(item for item in derived if item.strip())
            else:
                raise ProviderError("Text indexing returned an invalid result")
        return [PreparedEvidence(original, "\n".join(index_parts), {"images": images})]

    def query(self, query: str | list[dict[str, Any]],
              options: list[str] | None = None) -> PreparedQuery:
        _, source_text, images, _ = self._parse(query)
        if options is not None and (
            not isinstance(options, list)
            or any(not isinstance(option, str) or not option.strip() for option in options)
        ):
            raise ValueError("options must be an array of non-empty strings")
        query_parts = [source_text] if source_text else []
        query_parts.extend(options or [])
        if images:
            query_parts.append(IMAGE_CANDIDATE_MARKER)
        # Query parsing deliberately has no access to describe_image/index_text.
        return PreparedQuery("\n".join(query_parts), {"images": images})

    def media_bytes(self, content: str | list[dict[str, Any]]) -> int:
        """Count Base64-decoded bytes, not URI characters or decoded pixels."""
        _, _, images, _ = self._parse(content, compute_features=False)
        return sum(image["bytes"] for image in images)

    def planned_calls(self, content: str | list[dict[str, Any]]) -> int:
        _, _, images, _ = self._parse(content, compute_features=False)
        return len(images) or 1

    def score(self, query_features: dict[str, Any], record_features: dict[str, Any]) -> float:
        """Return visual similarity in [0, 1], using only persisted features.

        For multiple query images, each image gets its best record-image match;
        their mean prevents one exact image from masking other missing images.
        This is a retrieval heuristic, not a calibrated relevance probability.
        """
        if not self.image_features:
            return 0.0
        query_images = query_features.get("images", [])
        record_images = record_features.get("images", [])
        if not query_images or not record_images:
            return 0.0
        best = [max(self._image_score(query, record) for record in record_images)
                for query in query_images]
        return sum(best) / len(best)

    @staticmethod
    def _image_score(query: dict[str, Any], record: dict[str, Any]) -> float:
        if query.get("sha256") and query["sha256"] == record.get("sha256"):
            return 1.0
        try:
            left_hash, right_hash = int(query["dhash"], 16), int(record["dhash"], 16)
            hash_similarity = 1.0 - (left_hash ^ right_hash).bit_count() / 64.0
            left_hist, right_hist = query["color_hist"], record["color_hist"]
            if len(left_hist) != 24 or len(right_hist) != 24:
                return 0.0
            color_similarity = sum(min(left, right) for left, right in zip(left_hist, right_hist))
            return max(0.0, min(1.0, 0.35 * hash_similarity + 0.65 * color_similarity))
        except (KeyError, TypeError, ValueError):
            return 0.0

    def _parse(self, content: str | list[dict[str, Any]], *,
               compute_features: bool | None = None
               ) -> tuple[str | list[dict[str, Any]], str, list[dict[str, Any]], list[str]]:
        if isinstance(content, str):
            if not content.strip():
                raise ValueError("content must be non-empty")
            return content, content, [], []
        if not isinstance(content, list) or not content:
            raise ValueError("content must be a non-empty string or ordered content-parts array")
        text_parts: list[str] = []
        images: list[dict[str, Any]] = []
        image_uris: list[str] = []
        make_features = self.image_features if compute_features is None else compute_features
        for part in content:
            if not isinstance(part, dict):
                raise ValueError("every content part must be an object")
            if part.get("type") == "text":
                value = part.get("text")
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("text parts require non-empty text")
                text_parts.append(value)
            elif part.get("type") == "image_url":
                image_url = part.get("image_url")
                uri = image_url.get("url") if isinstance(image_url, dict) else None
                if not isinstance(uri, str):
                    raise ValueError("image_url parts require image_url.url")
                images.append(self._inspect_image(uri, make_features))
                image_uris.append(uri)
            else:
                raise ValueError("content part type must be text or image_url")
        return copy.deepcopy(content), "\n".join(text_parts), images, image_uris

    def _inspect_image(self, uri: str, compute_features: bool) -> dict[str, Any]:
        match = _URI_PATTERN.fullmatch(uri)
        if match is None:
            raise ValueError("images require inline Base64 JPEG, PNG, or WebP Data URIs")
        mime, encoded = match.groups()
        # Reject a large valid-length Base64 payload before allocating decoded
        # bytes. MIME/decoder validation remains separate from this byte limit.
        padding = len(encoded) - len(encoded.rstrip("="))
        if len(encoded) > ((MAX_IMAGE_BYTES + 2) // 3) * 4:
            raise PayloadTooLarge("image exceeds the 10 MiB Base64-decoded byte limit")
        if len(encoded) % 4 == 0 and len(encoded) // 4 * 3 - padding > MAX_IMAGE_BYTES:
            raise PayloadTooLarge("image exceeds the 10 MiB Base64-decoded byte limit")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("image Data URI contains malformed Base64") from exc
        if not data:
            raise ValueError("image Data URI must not be empty")
        if len(data) > MAX_IMAGE_BYTES:
            raise PayloadTooLarge("image exceeds the 10 MiB Base64-decoded byte limit")
        expected_format = _MIME_FORMATS[mime.lower()]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(data)) as image:
                    if image.format != expected_format:
                        raise ValueError("image MIME type does not match its actual file format")
                    width, height = image.size
                    if width * height > self.max_pixels:
                        raise PayloadTooLarge(f"image exceeds the service limit of {self.max_pixels} pixels")
                    image.verify()
                # verify() invalidates the decoder; reopening and loading also
                # rejects truncated data before captioning or acknowledgement.
                with Image.open(io.BytesIO(data)) as image:
                    image.load()
                    features: dict[str, Any] = {
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "bytes": len(data), "width": width, "height": height,
                        "format": expected_format,
                    }
                    if compute_features:
                        features.update(self._visual_features(image))
                    return features
        except PayloadTooLarge:
            raise
        except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise PayloadTooLarge("image exceeds the safe image-decoder pixel limit") from exc
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
            raise ValueError(f"invalid image: {exc}") from exc

    @staticmethod
    def _visual_features(image: Image.Image) -> dict[str, Any]:
        # Fixed tiny rasters avoid full-resolution RGB/grayscale working copies.
        rgb = image.resize((64, 64), Image.Resampling.BILINEAR).convert("RGB")
        gray = rgb.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
        pixels = gray.tobytes()
        dhash = 0
        for row in range(8):
            for column in range(8):
                offset = row * 9 + column
                dhash = (dhash << 1) | int(pixels[offset] > pixels[offset + 1])
        histogram = rgb.histogram()
        coarse = [sum(histogram[channel * 256 + bucket * 32:channel * 256 + (bucket + 1) * 32])
                  for channel in range(3) for bucket in range(8)]
        normalizer = 3.0 * rgb.width * rgb.height
        return {"dhash": f"{dhash:016x}", "color_hist": [value / normalizer for value in coarse]}
