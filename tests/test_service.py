"""Service-layer tests — the core product logic."""

from __future__ import annotations

from memory_lane import service
from memory_lane.lmd_bridge import LMDBridge
from memory_lane.models import EmotionalTone, MemoryStatus, ReactionKind


def _no_grounding_bridge() -> LMDBridge:
    # Skip the sentence-transformers load in tests; the hashing fallback
    # is deterministic and sufficient.
    return LMDBridge(use_language_grounding=False)


def test_create_patient_and_memory(db) -> None:
    patient = service.create_patient(db, "Eleanor Thomas", birth_year=1942)
    assert patient.id
    assert patient.display_name == "Eleanor Thomas"

    memory = service.add_memory(
        db,
        patient_id=patient.id,
        title="Wedding day",
        description="A sunny July afternoon in 1965.",
        tone=EmotionalTone.JOYFUL,
        valence_start=0.5,
        valence_peak=0.95,
        valence_end=0.7,
        approximate_year=1965,
    )
    assert memory.patient_id == patient.id
    assert memory.status == MemoryStatus.VIVID
    assert memory.energy == 1.0


def test_reaction_updates_energy_and_status(db) -> None:
    patient = service.create_patient(db, "Eleanor")
    memory = service.add_memory(
        db,
        patient_id=patient.id,
        title="Dog Biscuit",
        description="Her family dog from childhood.",
        tone=EmotionalTone.JOYFUL,
    )
    session = service.start_session(db, patient.id, caregiver_name="Daughter")

    bridge = _no_grounding_bridge()
    # A positive reaction nudges energy up.
    service.log_reaction(
        db,
        session_id=session.id,
        memory_id=memory.id,
        kind=ReactionKind.RECOGNIZED_POSITIVE,
        bridge=bridge,
    )
    db.refresh(memory)
    assert memory.energy > 1.0
    assert memory.status == MemoryStatus.VIVID
    assert memory.last_surfaced_at is not None


def test_not_recognized_pushes_status_down_over_time(db) -> None:
    patient = service.create_patient(db, "Eleanor")
    memory = service.add_memory(
        db,
        patient_id=patient.id,
        title="Old neighborhood",
        description="The street where she grew up.",
    )
    session = service.start_session(db, patient.id)
    bridge = _no_grounding_bridge()

    for _ in range(20):
        service.log_reaction(
            db,
            session_id=session.id,
            memory_id=memory.id,
            kind=ReactionKind.NOT_RECOGNIZED,
            bridge=bridge,
        )
    db.refresh(memory)
    assert memory.energy < 0.8
    assert memory.status in {
        MemoryStatus.DORMANT,
        MemoryStatus.FADING,
        MemoryStatus.GHOST,
    }


def test_repeated_distress_auto_flags_memory(db) -> None:
    patient = service.create_patient(db, "Eleanor")
    memory = service.add_memory(
        db,
        patient_id=patient.id,
        title="A difficult year",
        description="Loss in the family.",
        tone=EmotionalTone.DIFFICULT,
    )
    session = service.start_session(db, patient.id)
    bridge = _no_grounding_bridge()

    for _ in range(2):
        service.log_reaction(
            db,
            session_id=session.id,
            memory_id=memory.id,
            kind=ReactionKind.RECOGNIZED_DISTRESS,
            bridge=bridge,
        )
    db.refresh(memory)
    assert memory.flagged_distressing is True


def test_anchor_skips_flagged_and_difficult_memories(db) -> None:
    patient = service.create_patient(db, "Eleanor")
    joyful = service.add_memory(
        db,
        patient_id=patient.id,
        title="Garden roses",
        description="Her late-summer rose garden.",
        tone=EmotionalTone.JOYFUL,
    )
    difficult = service.add_memory(
        db,
        patient_id=patient.id,
        title="Hospital stay",
        description="A long illness.",
        tone=EmotionalTone.DIFFICULT,
    )
    service.flag_memory_distressing(db, difficult.id, note="Triggers sadness")

    anchors = service.suggest_anchor_for_patient(
        db,
        patient.id,
        bridge=_no_grounding_bridge(),
    )
    ids = {a.memory_id for a in anchors}
    assert joyful.id in ids
    assert difficult.id not in ids


def test_visit_plan_prioritizes_fading_but_positive(db) -> None:
    patient = service.create_patient(db, "Eleanor")
    bridge = _no_grounding_bridge()

    vivid = service.add_memory(
        db,
        patient_id=patient.id,
        title="Everyday favorite",
        description="Her favorite tea.",
        tone=EmotionalTone.JOYFUL,
    )
    dormant_joyful = service.add_memory(
        db,
        patient_id=patient.id,
        title="School bus route",
        description="The bus she took as a child.",
        tone=EmotionalTone.JOYFUL,
    )
    # Force dormant_joyful into DORMANT without logging sessions.
    dormant_joyful.energy = 0.5
    dormant_joyful.status = MemoryStatus.DORMANT
    db.commit()

    plan = service.suggest_visit_memories_for_patient(db, patient.id, bridge=bridge)
    assert len(plan) >= 2
    # Fading joyful memory should outrank the still-vivid one.
    plan_ids = [p.memory_id for p in plan]
    assert plan_ids.index(dormant_joyful.id) < plan_ids.index(vivid.id)


def test_fading_list(db) -> None:
    patient = service.create_patient(db, "Eleanor")
    fading_mem = service.add_memory(
        db,
        patient_id=patient.id,
        title="Old lullaby",
        description="The song her mother sang.",
        tone=EmotionalTone.BITTERSWEET,
    )
    fading_mem.energy = 0.2
    fading_mem.status = MemoryStatus.FADING
    db.commit()

    fading = service.fading_memories(db, patient.id)
    assert fading_mem.id in {m.id for m in fading}
