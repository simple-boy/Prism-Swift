"""Validate untrusted AML data without network access or silent truncation."""
from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
import hashlib
import io
import json
import re
from datetime import datetime, timezone

from PIL import Image, UnidentifiedImageError


class ContractError(ValueError):
    def __init__(self, reason: str, status: int = 422):
        super().__init__(reason)
        self.reason, self.status = reason, status


def canonical(value):
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError):
        raise ContractError("invalid_json_value") from None


def identifier(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ContractError(field + "_required")
    if len(value.encode("utf-8")) > 4096:
        raise ContractError("identifier_exceeds_capacity", 413)
    return value


def timestamp(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError("timestamp_must_be_unix_milliseconds")
    try:
        datetime.fromtimestamp(value / 1000, timezone.utc)
    except (ValueError, OverflowError, OSError):
        raise ContractError("timestamp_out_of_range") from None
    return value


URI = re.compile(r"\A(data:(image/(?:png|jpeg|webp));base64,)([A-Za-z0-9+/]+={0,2})\Z", re.I)
FORMATS = {"image/png": "PNG", "image/jpeg": "JPEG", "image/webp": "WEBP"}


@dataclass
class ImageData:
    digest: str
    data: bytes
    width: int
    height: int
    features: dict


def decode_image(uri, *, max_image_bytes, max_pixels, features=True):
    if not isinstance(uri, str):
        raise ContractError("image_url_must_be_data_uri")
    # Bound before regex/group allocation and base64 decoding.
    if len(uri) > 4 * ((max_image_bytes + 2) // 3) + 128:
        raise ContractError("image_exceeds_10mib_or_declared_capacity", 413)
    match = URI.fullmatch(uri)
    if not match:
        raise ContractError("only_inline_jpeg_png_webp_base64_supported")
    prefix, mime, encoded = match.groups()
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise ContractError("invalid_image_base64") from None
    if not data or len(data) > max_image_bytes:
        raise ContractError("image_size_out_of_range", 413)
    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            if width * height > max_pixels:
                raise ContractError("image_pixels_exceed_declared_capacity", 413)
            if image.format != FORMATS[mime.lower()]:
                raise ContractError("image_mime_format_mismatch")
            image.verify()
        visual = {}
        with Image.open(io.BytesIO(data)) as image:
            # verify() alone does not detect truncated JPEG pixel streams.
            # All baselines pay the same required validation cost.
            image.load()
            if features:
                image.draft("RGB", (64, 64))
                image.thumbnail((64, 64))
                rgb = image.convert("RGB")
                gray = rgb.convert("L").resize((9, 8))
                pixels = gray.tobytes()
                bits = 0
                for y in range(8):
                    for x in range(8):
                        bits = (bits << 1) | (pixels[y * 9 + x] > pixels[y * 9 + x + 1])
                hist = rgb.histogram()
                total = rgb.width * rgb.height * 3
                bins = [sum(hist[channel * 256 + b * 32:channel * 256 + (b + 1) * 32]) / total
                        for channel in range(3) for b in range(8)]
                mean = [sum(i * hist[channel * 256 + i] for i in range(256)) /
                        (rgb.width * rgb.height * 255) for channel in range(3)]
                visual = {"dhash": f"{bits:016x}", "hist": bins, "mean": mean}
    except ContractError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise ContractError("invalid_or_unsafe_image") from None
    digest = hashlib.sha256(data).hexdigest()
    # Standard input is stored only as binary. Preserve rare noncanonical pad bits verbatim.
    override = encoded if base64.b64encode(data).decode("ascii") != encoded else None
    return ImageData(digest, data, width, height, visual), prefix, override


def parse_content(content, *, max_image_bytes, max_pixels, features=True, cache=None):
    """Return source template, text, image occurrences. Cache exists within one request only."""
    if isinstance(content, str):
        if not content.strip():
            raise ContractError("content_must_be_nonempty")
        return content, content, []
    if not isinstance(content, list) or not content:
        raise ContractError("content_requires_text_or_ordered_parts")
    template, text, images = [], [], []
    cache = {} if cache is None else cache
    for part in content:
        if not isinstance(part, dict):
            raise ContractError("content_part_must_be_object")
        if part.get("type") == "text":
            if not isinstance(part.get("text"), str) or not part["text"].strip():
                raise ContractError("text_part_must_be_nonempty")
            template.append(part.copy())
            text.append(part["text"])
        elif part.get("type") == "image_url":
            value = part.get("image_url")
            if not isinstance(value, dict) or not isinstance(value.get("url"), str):
                raise ContractError("image_url_requires_url")
            uri = value["url"]
            # Hashing a URI is much cheaper than decoding the same source again.
            key = hashlib.sha256(uri.encode()).digest()
            if key not in cache:
                cache[key] = decode_image(uri, max_image_bytes=max_image_bytes, max_pixels=max_pixels,
                                          features=features)
            image, prefix, override = cache[key]
            source = {**part, "image_url": {**value}}
            source["image_url"]["url"] = {"blob": image.digest, "prefix": prefix}
            if override is not None:
                source["image_url"]["url"]["base64_override"] = override
            template.append(source)
            images.append(image)
        else:
            raise ContractError("unsupported_content_part_type")
    return template, "\n".join(text), images


def restore(template, get_blob):
    if isinstance(template, str):
        return template
    result = []
    for part in template:
        if part["type"] == "text":
            result.append(part.copy())
        else:
            source = {**part, "image_url": {**part["image_url"]}}
            reference = source["image_url"]["url"]
            encoded = reference.get("base64_override")
            if encoded is None:
                encoded = base64.b64encode(get_blob(reference["blob"])).decode("ascii")
            source["image_url"]["url"] = reference["prefix"] + encoded
            result.append(source)
    return result
