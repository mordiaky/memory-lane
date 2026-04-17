"""Exporter tests — roundtripping is the critical invariant."""

from __future__ import annotations

import json

from memory_lane import exporters, importers, service
from memory_lane.lmd_bridge import LMDBridge
from memory_lane.models import EmotionalTone, ReactionKind


def test_json_export_shape(db) -> None:
    patient = service.create_patient(db, "Eleanor", birth_year=1942)
    service.add_memory(
        db,
        patient_id=patient.id,
        title="Wedding",
        description="Sunny afternoon.",
        tone=EmotionalTone.JOYFUL,
        approximate_year=1965,
    )
    visit = service.start_session(db, patient.id, caregiver_name="Daughter")
    service.end_session(db, visit.id, summary="Good visit.")

    archive = exporters.export_patient_json(db, patient.id)
    assert archive["schema"] == "memory-lane-export/v1"
    assert archive["patient"]["display_name"] == "Eleanor"
    assert archive["patient"]["birth_year"] == 1942
    assert len(archive["memories"]) == 1
    assert archive["memories"][0]["title"] == "Wedding"
    assert len(archive["sessions"]) == 1


def test_json_export_string_is_valid_json(db) -> None:
    patient = service.create_patient(db, "Eleanor")
    service.add_memory(
        db, patient_id=patient.id, title="A", description="...",
    )
    payload = exporters.export_patient_json_string(db, patient.id)
    # Valid JSON, parses back to the same shape.
    parsed = json.loads(payload)
    assert parsed["patient"]["display_name"] == "Eleanor"
    assert len(parsed["memories"]) == 1


def test_json_export_includes_session_reactions(db) -> None:
    patient = service.create_patient(db, "Eleanor")
    memory = service.add_memory(
        db, patient_id=patient.id, title="Roses", description="...",
        tone=EmotionalTone.JOYFUL,
    )
    visit = service.start_session(db, patient.id)
    service.log_reaction(
        db,
        session_id=visit.id,
        memory_id=memory.id,
        kind=ReactionKind.RECOGNIZED_POSITIVE,
        bridge=LMDBridge(use_language_grounding=False),
    )
    archive = exporters.export_patient_json(db, patient.id)
    assert len(archive["sessions"]) == 1
    assert len(archive["sessions"][0]["reactions"]) == 1
    assert archive["sessions"][0]["reactions"][0]["kind"] == "recognized_positive"


def test_csv_roundtrip_preserves_memories(db, tmp_path) -> None:
    patient = service.create_patient(db, "Eleanor")
    original = service.add_memory(
        db,
        patient_id=patient.id,
        title="Wedding",
        description="Sunny afternoon.",
        tone=EmotionalTone.JOYFUL,
        approximate_year=1965,
        era_label="1960s",
        valence_start=0.5,
        valence_peak=0.9,
        valence_end=0.7,
    )
    assert original.id  # sanity

    # Export then import into a fresh patient.
    csv_str = exporters.export_memories_csv_string(db, patient.id)
    csv_path = tmp_path / "out.csv"
    csv_path.write_text(csv_str, encoding="utf-8")

    new_patient = service.create_patient(db, "Fresh")
    report = importers.import_csv(db, new_patient.id, csv_path)
    assert report.imported == 1
    assert report.skipped == 0

    imported_memories = service.list_memories(db, new_patient.id)
    assert len(imported_memories) == 1
    m = imported_memories[0]
    assert m.title == "Wedding"
    assert m.description == "Sunny afternoon."
    assert m.tone == EmotionalTone.JOYFUL
    assert m.approximate_year == 1965
    assert m.era_label == "1960s"
    assert m.valence_peak == 0.9


def test_export_raises_for_unknown_patient(db) -> None:
    import pytest

    with pytest.raises(ValueError):
        exporters.export_patient_json(db, "00000000-0000-0000-0000-000000000000")
