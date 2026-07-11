# SPDX-License-Identifier: Apache-2.0
"""
Image processing utilities for VLM (Vision-Language Model) support.

This module provides functions for loading images from base64 data URIs,
extracting images from OpenAI-format messages, and computing image
hashes for prefix cache deduplication.
"""

import base64
import binascii
import hashlib
import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image, ImageOps

from ..exceptions import InvalidRequestError

logger = logging.getLogger(__name__)

# Decoded payload cap for data:video/... URIs (~100 MiB).
VIDEO_DECODED_SIZE_CAP = 100 * 1024 * 1024

_VIDEO_MIME_SUFFIX = {
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
    "video/x-matroska": ".mkv",
}


class TempVideoPath(str):
    """Path to a decoded data-URL video written to a temporary file."""


VideoReference = Union[str, bytes, Path, TempVideoPath]


_IMAGE_INPUT_ERROR = (
    "Image inputs must be base64 data URIs "
    "(data:image/...;base64,...). Remote URLs and local file paths are not supported."
)
_AUDIO_INPUT_ERROR = (
    "input_audio.data must be a base64 string or base64 data URI. "
    "Local file paths are not supported."
)


def _decode_base64_data_uri(value: str, *, field: str) -> bytes:
    """Decode a base64 data URI, mapping malformed input to a request error."""
    if not isinstance(value, str):
        raise InvalidRequestError(_IMAGE_INPUT_ERROR, field=field)

    stripped = value.strip()
    if not stripped.startswith("data:"):
        raise InvalidRequestError(_IMAGE_INPUT_ERROR, field=field)

    prefix, separator, encoded = stripped.partition(",")
    prefix_lower = prefix.lower()
    if (
        separator != ","
        or not prefix_lower.startswith("data:image/")
        or ";base64" not in prefix_lower
    ):
        raise InvalidRequestError(
            f"{field} must use a base64 image data URI.",
            field=field,
        )

    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidRequestError(
            f"{field} contains invalid base64 data.",
            field=field,
        ) from exc


def _decode_input_audio_data(data: str, *, field: str = "input_audio.data") -> bytes:
    """Decode input_audio.data without falling back to filesystem paths."""
    stripped = data.strip()
    if stripped.startswith("data:"):
        prefix, separator, encoded = stripped.partition(",")
        if separator != "," or ";base64" not in prefix.lower():
            raise InvalidRequestError(
                f"{field} must use a base64 data URI.",
                field=field,
            )
    else:
        encoded = stripped

    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidRequestError(_AUDIO_INPUT_ERROR, field=field) from exc


def validate_image_data_uri(value: str, *, field: str = "image") -> str:
    """Validate that a request-facing image reference is an inline data URI."""
    _decode_base64_data_uri(value, field=field)
    return value


def load_image(url_or_base64: str, *, field: str = "image_url") -> Image.Image:
    """
    Load an image from a base64 data URI.

    Supports:
    - Data URIs: "data:image/jpeg;base64,..." format

    Args:
        url_or_base64: Image base64 data URI string

    Returns:
        PIL Image object

    Raises:
        InvalidRequestError: If the input is not a valid image data URI
    """
    img_bytes = _decode_base64_data_uri(url_or_base64, field=field)
    try:
        img = Image.open(io.BytesIO(img_bytes))
    except Exception as exc:
        raise InvalidRequestError(
            f"{field} does not contain a decodable image.",
            field=field,
        ) from exc

    # Apply EXIF orientation (phone photos etc.) before processing.
    # Matches mlx-vlm's load_image which calls ImageOps.exif_transpose().
    img = ImageOps.exif_transpose(img)
    # Ensure RGB format (RGBA/P/L etc. cause broadcast errors in vision processors)
    return img.convert("RGB")


def load_video_reference(url_or_data: str) -> VideoReference:
    """Load a video reference acceptable by ``mlx_vlm.utils.prepare_inputs``.

    HTTP(S) URLs and local file paths are returned unchanged. Data URIs are
    decoded, size-checked, and written to a temporary file whose path is
    returned as :class:`TempVideoPath` for later cleanup.

    Raises:
        ValueError: On invalid base64 or oversize decoded payloads.
    """
    if url_or_data.startswith("data:"):
        try:
            header, data_part = url_or_data.split(",", 1)
        except ValueError as exc:
            raise ValueError(
                f"Invalid video data URI format: {url_or_data[:50]}..."
            ) from exc
        if ";base64" not in header:
            raise ValueError(
                f"Unsupported video data URI (expected base64): {header}"
            )
        try:
            video_bytes = base64.b64decode(data_part, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(
                f"Invalid base64 in video data URI: {exc}"
            ) from exc
        if len(video_bytes) > VIDEO_DECODED_SIZE_CAP:
            cap_mib = VIDEO_DECODED_SIZE_CAP // (1024 * 1024)
            raise ValueError(
                f"Decoded video data too large: {len(video_bytes)} bytes "
                f"exceeds {cap_mib} MiB cap"
            )
        mime = header[5:].split(";", 1)[0].strip().lower()
        suffix = _VIDEO_MIME_SUFFIX.get(mime, ".mp4")
        fd, path = tempfile.mkstemp(prefix="omlx_video_", suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(video_bytes)
        except Exception:
            os.unlink(path)
            raise
        return TempVideoPath(path)

    if url_or_data.startswith(("http://", "https://")):
        return url_or_data

    return url_or_data


def cleanup_temp_video_paths(videos: List[VideoReference]) -> None:
    """Delete temporary files created for decoded video data URIs."""
    for video in videos:
        if isinstance(video, TempVideoPath):
            try:
                os.unlink(video)
            except OSError:
                logger.debug("Failed to delete temp video file %s", video, exc_info=True)


def extract_media_from_messages(
    messages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Image.Image], List, List[VideoReference]]:
    """
    Extract images, audio, and videos from OpenAI-format messages.

    Processes messages containing content arrays with image_url, video_url,
    or input_audio parts, loads the media, and returns cleaned text-only
    messages alongside the loaded media references.

    Args:
        messages: List of OpenAI-format chat messages. Each message may have
            content as a string or a list of content parts
            (text/image_url/video_url/input_audio).

    Returns:
        Tuple of (text_messages, images, audio, videos):
        - text_messages: Messages with media parts removed, text parts joined
        - images: List of loaded PIL Image objects in order of appearance
        - audio: List of BytesIO/str audio references for load_audio()
        - videos: List of file paths/URLs for ``prepare_inputs(videos=...)``
    """
    text_messages = []
    images = []
    audio = []
    videos: List[VideoReference] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if not isinstance(content, list):
            # Simple string content — pass through
            text_messages.append({"role": role, "content": content or ""})
            # Preserve extra fields (tool_calls, tool_call_id, etc.)
            for key in msg:
                if key not in ("role", "content"):
                    text_messages[-1][key] = msg[key]
            continue

        # Content array with text, image_url, and/or input_audio parts
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                part_type = part.get("type")
            else:
                # Pydantic model (ContentPart)
                part_type = getattr(part, "type", None)

            if part_type == "text":
                text = (
                    part.get("text")
                    if isinstance(part, dict)
                    else getattr(part, "text", None)
                )
                if text:
                    text_parts.append(text)

            elif part_type in ("image_url", "input_image"):
                # OpenAI chat format: {"type":"image_url","image_url":{"url":"..."}}
                # Responses-style format: {"type":"input_image","image_url":"..."}
                image_url_obj = (
                    part.get("image_url")
                    if isinstance(part, dict)
                    else getattr(part, "image_url", None)
                )
                if image_url_obj is None and isinstance(part, dict):
                    image_url_obj = part.get("input_image")

                url = None
                if isinstance(image_url_obj, str):
                    url = image_url_obj
                elif isinstance(image_url_obj, dict):
                    url = image_url_obj.get("url")
                elif image_url_obj is not None:
                    url = getattr(image_url_obj, "url", None)

                if url:
                    images.append(load_image(url, field="image_url"))

            elif part_type == "input_audio":
                # OpenAI audio format: {"type":"input_audio","input_audio":{"data":"...","format":"wav"}}
                input_audio = (
                    part.get("input_audio")
                    if isinstance(part, dict)
                    else getattr(part, "input_audio", None)
                )
                if input_audio and isinstance(input_audio, dict):
                    data = input_audio.get("data", "")
                    if isinstance(data, str):
                        audio.append(io.BytesIO(_decode_input_audio_data(data)))
                    elif isinstance(data, bytes):
                        audio.append(io.BytesIO(data))
                    else:
                        audio.append(data)

            elif part_type == "video_url":
                video_url_obj = (
                    part.get("video_url") if isinstance(part, dict)
                    else getattr(part, "video_url", None)
                )
                url = None
                if isinstance(video_url_obj, str):
                    url = video_url_obj
                elif isinstance(video_url_obj, dict):
                    url = video_url_obj.get("url")
                elif video_url_obj is not None:
                    url = getattr(video_url_obj, "url", None)

                if url:
                    videos.append(load_video_reference(url))

        new_msg = {"role": role, "content": "\n".join(text_parts) if text_parts else ""}
        # Preserve extra fields
        for key in msg:
            if key not in ("role", "content"):
                new_msg[key] = msg[key]
        text_messages.append(new_msg)

    return text_messages, images, audio, videos


def extract_images_from_messages(
    messages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Image.Image], List]:
    """
    Extract images and audio from OpenAI-format messages.

    Processes messages containing content arrays with image_url or input_audio
    parts, loads the media, and returns cleaned text-only messages alongside
    the loaded images and audio files. Video parts are stripped from text
    messages but not returned (see :func:`extract_media_from_messages`).

    Args:
        messages: List of OpenAI-format chat messages. Each message may have
            content as a string or a list of content parts
            (text/image_url/input_audio).

    Returns:
        Tuple of (text_messages, images, audio):
        - text_messages: Messages with media parts removed, text parts joined
        - images: List of loaded PIL Image objects in order of appearance
        - audio: List of BytesIO/str audio references for load_audio()
    """
    text_messages, images, audio, _videos = extract_media_from_messages(messages)
    return text_messages, images, audio


def compute_image_hash(images: List[Image.Image]) -> Optional[str]:
    """
    Compute a SHA256 hash from a list of images for prefix cache deduplication.

    Uses image size and raw pixel data to produce a deterministic hash.
    Returns None if images list is empty.

    Args:
        images: List of PIL Image objects

    Returns:
        Hex-encoded SHA256 hash string, or None if no images
    """
    if not images:
        return None

    hasher = hashlib.sha256()
    for img in images:
        # Include image dimensions
        hasher.update(f"{img.size[0]}x{img.size[1]}".encode())
        # Include raw pixel data (convert to RGB for consistency)
        rgb_img = img.convert("RGB")
        hasher.update(rgb_img.tobytes())

    return hasher.hexdigest()


def compute_per_image_hashes(images: List[Image.Image]) -> List[str]:
    """Compute individual SHA256 hashes for each image.

    Returns a list of hex-encoded hash strings, one per image.
    """
    hashes = []
    for img in images:
        hasher = hashlib.sha256()
        hasher.update(f"{img.size[0]}x{img.size[1]}".encode())
        rgb_img = img.convert("RGB")
        hasher.update(rgb_img.tobytes())
        hashes.append(hasher.hexdigest())
    return hashes
