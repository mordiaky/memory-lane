"""Local media storage for photos and audio attached to memories.

Files live in a media directory (default ~/.memory-lane/media/), one
subdirectory per patient. The Memory.photo_path column stores a
relative path (e.g. "<patient_id>/<filename>") so the DB stays
portable and the files can be moved underneath it.

We intentionally keep media handling dead-simple: local filesystem,
whitelist a few safe mime types, reject anything else. No thumbnails,
no transcoding, no cloud. Phase-1 scope.
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

from .models import Memory

# Photo types we accept. Everything else is rejected — not because it
# can't work, but because we don't want to silently accept executables
# or exotic image formats that aren't universally renderable.
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".gif"}
ALLOWED_AUDIO_SUFFIXES = {".mp3", ".m4a", ".wav", ".ogg", ".flac"}

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]")


class UnsupportedMediaType(ValueError):  # noqa: N818 - name deliberately mirrors HTTP 415 status.
    pass


def media_root() -> Path:
    """Where media files live on disk. Override with MEMORY_LANE_MEDIA_DIR."""
    override = os.environ.get("MEMORY_LANE_MEDIA_DIR")
    if override:
        root = Path(override)
    else:
        root = Path.home() / ".memory-lane" / "media"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_filename(original: str) -> str:
    """Strip the filename down to a safe form, preserving the extension.

    Adds a short UUID prefix so two uploads of the same filename don't
    collide and so we don't leak user-supplied names verbatim.
    """
    name = Path(original).name  # strip any directory traversal
    name = _UNSAFE_CHARS.sub("_", name)
    # Keep total length reasonable.
    name = name[:80]
    return f"{uuid.uuid4().hex[:8]}_{name}"


def _kind_for_suffix(suffix: str) -> str:
    suffix = suffix.lower()
    if suffix in ALLOWED_IMAGE_SUFFIXES:
        return "photo"
    if suffix in ALLOWED_AUDIO_SUFFIXES:
        return "audio"
    raise UnsupportedMediaType(
        f"Unsupported file extension: {suffix}. "
        f"Supported images: {', '.join(sorted(ALLOWED_IMAGE_SUFFIXES))}. "
        f"Supported audio: {', '.join(sorted(ALLOWED_AUDIO_SUFFIXES))}."
    )


def save_media_bytes(
    patient_id: str,
    original_filename: str,
    data: bytes,
) -> tuple[str, str]:
    """Persist a binary payload to the media directory.

    Returns (kind, relative_path) where kind is 'photo' or 'audio'
    and relative_path is what to store on the Memory row.
    """
    suffix = Path(original_filename).suffix
    kind = _kind_for_suffix(suffix)
    safe = _safe_filename(original_filename)

    per_patient = media_root() / patient_id
    per_patient.mkdir(parents=True, exist_ok=True)
    target = per_patient / safe
    target.write_bytes(data)

    relative = f"{patient_id}/{safe}"
    return kind, relative


def resolve_media_path(relative_path: str) -> Path:
    """Resolve a stored relative path to an absolute path under media_root.

    Guards against path-traversal: the resolved path must live inside
    media_root or we refuse to serve it.
    """
    root = media_root().resolve()
    full = (root / relative_path).resolve()
    try:
        full.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "Refusing to resolve media path outside media root."
        ) from exc
    if not full.exists():
        raise FileNotFoundError(str(full))
    return full


def attach_media(memory: Memory, kind: str, relative_path: str) -> None:
    """Mutate the Memory row to reference the stored file. Caller commits."""
    if kind == "photo":
        memory.photo_path = relative_path
    elif kind == "audio":
        memory.audio_path = relative_path
    else:  # pragma: no cover - guarded above
        raise ValueError(f"Unknown media kind: {kind}")
