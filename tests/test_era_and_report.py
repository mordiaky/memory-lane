"""Tests for era-based browsing and visit-report generation."""

from __future__ import annotations

from memory_lane import service
from memory_lane.lmd_bridge import LMDBridge
from memory_lane.models import EmotionalTone, MemoryStatus, ReactionKind


def _bridge() -> LMDBridge:
    return LMDBridge(use_language_grounding=False)


def test_era_overview_groups_by_decade_and_explicit_label(db) -> None:
    patient = service.create_patient(db, "Eleanor")

    service.add_memory(
        db,
        patient_id=patient.id,
        title="Wedding",
        description="...",
        tone=EmotionalTone.JOYFUL,
        approximate_year=1965,
    )
    service.add_memory(
        db,
        patient_id=patient.id,
        title="First grandchild",
        description="...",
        tone=EmotionalTone.JOYFUL,
        approximate_year=1973,
    )
    # Second 1970s memory with different tone so breakdown is non-trivial.
    service.add_memory(
        db,
        patient_id=patient.id,
        title="Illness in the family",
        description="...",
        tone=EmotionalTone.DIFFICULT,
        approximate_year=1978,
    )
    # Explicit label overrides decade inference.
    service.add_memory(
        db,
        patient_id=patient.id,
        title="Vacation photo",
        description="...",
        tone=EmotionalTone.BITTERSWEET,
        era_label="Italy trips",
    )
    # Undated bucket.
    service.add_memory(
        db,
        patient_id=patient.id,
        title="Unknown date",
        description="...",
    )

    summaries = service.era_overview(db, patient.id)
    eras = [s.era for s in summaries]
    # 1960s, 1970s, Italy trips, Undated — Undated always last
    assert eras[0] == "1960s"
    assert eras[1] == "1970s"
    assert eras[-1] == "Undated"
    assert "Italy trips" in eras

    seventies = next(s for s in summaries if s.era == "1970s")
    assert seventies.memory_count == 2
    assert seventies.tone_breakdown == {"joyful": 1, "difficult": 1}


def test_memories_in_era(db) -> None:
    patient = service.create_patient(db, "Eleanor")
    wedding = service.add_memory(
        db,
        patient_id=patient.id,
        title="Wedding",
        description="...",
        approximate_year=1965,
    )
    service.add_memory(
        db,
        patient_id=patient.id,
        title="Move",
        description="...",
        approximate_year=1975,
    )
    results = service.memories_in_era(db, patient.id, "1960s")
    assert [m.id for m in results] == [wedding.id]


def test_visit_report_warm_classification(db) -> None:
    patient = service.create_patient(db, "Eleanor")
    mem_a = service.add_memory(
        db, patient_id=patient.id, title="Roses", description="...",
        tone=EmotionalTone.JOYFUL,
    )
    mem_b = service.add_memory(
        db, patient_id=patient.id, title="Songs", description="...",
        tone=EmotionalTone.JOYFUL,
    )
    visit = service.start_session(db, patient.id, caregiver_name="Daughter")
    bridge = _bridge()
    for memory_id in (mem_a.id, mem_b.id):
        service.log_reaction(
            db,
            session_id=visit.id,
            memory_id=memory_id,
            kind=ReactionKind.RECOGNIZED_POSITIVE,
            bridge=bridge,
        )
    service.end_session(db, visit.id, summary="Nice afternoon.")

    report = service.build_visit_report(db, visit.id)
    assert report["overall_tone"] == "warm"
    assert report["positive_count"] == 2
    assert report["distress_count"] == 0
    assert report["memories_surfaced"] == 2
    assert len(report["highlights"]) == 2
    assert report["concerns"] == []
    assert report["duration_minutes"] is not None


def test_visit_report_concerning_on_distress(db) -> None:
    patient = service.create_patient(db, "Eleanor")
    mem = service.add_memory(
        db, patient_id=patient.id, title="Hospital year",
        description="Long illness.", tone=EmotionalTone.DIFFICULT,
    )
    mem_ok = service.add_memory(
        db, patient_id=patient.id, title="Tea",
        description="Her favourite.", tone=EmotionalTone.JOYFUL,
    )
    visit = service.start_session(db, patient.id)
    bridge = _bridge()
    service.log_reaction(
        db, session_id=visit.id, memory_id=mem.id,
        kind=ReactionKind.RECOGNIZED_DISTRESS, bridge=bridge,
    )
    service.log_reaction(
        db, session_id=visit.id, memory_id=mem_ok.id,
        kind=ReactionKind.RECOGNIZED_POSITIVE, bridge=bridge,
    )
    service.log_reaction(
        db, session_id=visit.id, memory_id=mem.id,
        kind=ReactionKind.RECOGNIZED_DISTRESS, bridge=bridge,
    )

    report = service.build_visit_report(db, visit.id)
    # 2 distress, 1 positive → concerning
    assert report["overall_tone"] == "concerning"
    assert report["distress_count"] == 2
    assert any(c["concern"] == "distress" for c in report["concerns"])


def test_visit_report_no_data_when_session_empty(db) -> None:
    patient = service.create_patient(db, "Eleanor")
    visit = service.start_session(db, patient.id)
    report = service.build_visit_report(db, visit.id)
    assert report["overall_tone"] == "no_data"
    assert report["reactions_logged"] == 0
    assert "No reactions were logged" in " ".join(report["follow_up_suggestions"])


def test_csv_import_round_trip(db, tmp_path) -> None:
    from memory_lane import importers

    patient = service.create_patient(db, "Eleanor")
    csv_path = tmp_path / "memories.csv"
    csv_path.write_text(
        "title,description,tone,approximate_year,era_label,"
        "valence_start,valence_peak,valence_end,tags\n"
        "Wedding,Sunny afternoon in 1965,joyful,1965,,,0.9,0.7,anniversary\n"
        "Dog Biscuit,Her childhood dog,joyful,1948,childhood,0.4,0.95,0.6,\n"
        "Invalid tone row,A description,badtone,,,,,,\n"
        ",No title row,,,,,,,,\n",
        encoding="utf-8",
    )

    report = importers.import_csv(db, patient.id, csv_path)
    assert report.imported == 3   # rows 1, 2, 3 (row 3 coerces tone to neutral)
    assert report.skipped == 1    # row 4 missing title
    assert any("unknown tone" in w for w in report.warnings)
    assert any("missing title" in w for w in report.warnings)

    memories = service.list_memories(db, patient.id)
    titles = {m.title for m in memories}
    assert "Wedding" in titles
    assert "Dog Biscuit" in titles
    # All imported memories start out vivid.
    assert all(m.status == MemoryStatus.VIVID for m in memories)
