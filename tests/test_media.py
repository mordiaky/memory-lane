"""Photo/audio attachment tests.

Verifies:
  - Upload stores under the media root with a safe, namespaced filename.
  - Unsupported extensions raise UnsupportedMediaType.
  - Path traversal via the relative_path getter is refused.
  - Roundtrip: attach -> read bytes back through resolve_media_path.
  - Memory.photo_path / audio_path get populated correctly.
"""

from __future__ import annotations

import pytest

from memory_lane import media, service


@pytest.fixture(autouse=True)
def _isolate_media_root(tmp_path, monkeypatch):
    """Every test gets a fresh media directory under tmp_path."""
    monkeypatch.setenv("MEMORY_LANE_MEDIA_DIR", str(tmp_path / "media"))
    yield


def test_media_root_created(tmp_path) -> None:
    root = media.media_root()
    assert root.exists()
    assert root.is_dir()
    assert str(tmp_path) in str(root)


def test_save_and_read_photo_bytes(db) -> None:
    patient = service.create_patient(db, "Eleanor")
    memory = service.add_memory(
        db,
        patient_id=patient.id,
        title="Roses",
        description="Her summer garden.",
    )

    fake_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"stub-photo-bytes"
    memory_after, kind, relative = service.attach_media_to_memory(
        db, memory.id, "roses.jpg", fake_jpeg
    )
    assert kind == "photo"
    assert memory_after.photo_path == relative
    assert memory_after.audio_path is None
    # Roundtrip read.
    resolved = media.resolve_media_path(relative)
    assert resolved.read_bytes() == fake_jpeg


def test_save_audio(db) -> None:
    patient = service.create_patient(db, "Eleanor")
    memory = service.add_memory(
        db, patient_id=patient.id, title="Lullaby", description="Her mother's song.",
    )
    memory_after, kind, relative = service.attach_media_to_memory(
        db, memory.id, "lullaby.mp3", b"ID3stubmp3"
    )
    assert kind == "audio"
    assert memory_after.audio_path == relative
    assert memory_after.photo_path is None


def test_unsupported_extension_rejected(db) -> None:
    patient = service.create_patient(db, "Eleanor")
    memory = service.add_memory(
        db, patient_id=patient.id, title="Doc", description="...",
    )
    with pytest.raises(media.UnsupportedMediaType):
        service.attach_media_to_memory(db, memory.id, "malware.exe", b"MZ")


def test_path_traversal_refused(db) -> None:
    # Even if someone stores a bad relative_path, the resolver refuses
    # to return anything outside the media root.
    with pytest.raises(ValueError):
        media.resolve_media_path("../../../etc/passwd")


def test_filenames_are_namespaced_and_sanitized(db) -> None:
    patient = service.create_patient(db, "Eleanor")
    memory = service.add_memory(
        db, patient_id=patient.id, title="Trip", description="...",
    )
    _, _, relative = service.attach_media_to_memory(
        db, memory.id, "my weird file name!!.jpg", b"stub"
    )
    # Relative path is under the patient id directory
    assert relative.startswith(patient.id + "/")
    # The unsafe characters got stripped
    assert "!!" not in relative
    assert " " not in relative


def test_attach_to_unknown_memory_raises(db) -> None:
    with pytest.raises(ValueError):
        service.attach_media_to_memory(
            db,
            "00000000-0000-0000-0000-000000000000",
            "a.jpg",
            b"stub",
        )
