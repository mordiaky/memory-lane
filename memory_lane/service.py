"""Business-logic layer — the thing the API and CLI both call.

Each function here takes a SQLAlchemy Session plus inputs and returns
plain Python objects or ORM instances. No HTTP, no CLI concerns. Keep
all LMD interaction routed through `lmd_bridge.LMDBridge`.
"""

from __future__ import annotations

from typing import List

from sqlalchemy import select
from sqlalchemy.orm import Session

from .lmd_bridge import AnchorSuggestion, LMDBridge, VisitSuggestion
from .models import (
    EmotionalTone,
    Memory,
    MemoryStatus,
    Patient,
    Reaction,
    ReactionKind,
)
from .models import (
    Session as VisitSession,
)

# ---- Patients ----------------------------------------------------


def create_patient(
    db: Session,
    display_name: str,
    birth_year: int | None = None,
    notes: str | None = None,
) -> Patient:
    patient = Patient(display_name=display_name, birth_year=birth_year, notes=notes)
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient


def get_patient(db: Session, patient_id: str) -> Patient | None:
    return db.get(Patient, patient_id)


def list_patients(db: Session) -> List[Patient]:
    return list(db.scalars(select(Patient).order_by(Patient.created_at.desc())))


# ---- Memories ----------------------------------------------------


def add_memory(
    db: Session,
    *,
    patient_id: str,
    title: str,
    description: str,
    tone: EmotionalTone = EmotionalTone.NEUTRAL,
    valence_start: float = 0.0,
    valence_peak: float = 0.5,
    valence_end: float = 0.3,
    approximate_year: int | None = None,
    era_label: str | None = None,
    tags: str | None = None,
    photo_path: str | None = None,
    audio_path: str | None = None,
) -> Memory:
    patient = db.get(Patient, patient_id)
    if patient is None:
        raise ValueError(f"Patient not found: {patient_id}")

    memory = Memory(
        patient_id=patient_id,
        title=title,
        description=description,
        tone=tone,
        valence_start=valence_start,
        valence_peak=valence_peak,
        valence_end=valence_end,
        approximate_year=approximate_year,
        era_label=era_label,
        tags=tags,
        photo_path=photo_path,
        audio_path=audio_path,
        status=MemoryStatus.VIVID,
        energy=1.0,
    )
    db.add(memory)
    db.commit()
    db.refresh(memory)
    return memory


def list_memories(
    db: Session,
    patient_id: str,
    *,
    status: MemoryStatus | None = None,
    tone: EmotionalTone | None = None,
) -> List[Memory]:
    stmt = select(Memory).where(Memory.patient_id == patient_id)
    if status is not None:
        stmt = stmt.where(Memory.status == status)
    if tone is not None:
        stmt = stmt.where(Memory.tone == tone)
    stmt = stmt.order_by(Memory.updated_at.desc())
    return list(db.scalars(stmt))


def flag_memory_distressing(
    db: Session,
    memory_id: str,
    note: str | None = None,
) -> Memory:
    memory = db.get(Memory, memory_id)
    if memory is None:
        raise ValueError(f"Memory not found: {memory_id}")
    memory.flagged_distressing = True
    memory.distress_note = note
    db.commit()
    db.refresh(memory)
    return memory


def clear_distress_flag(db: Session, memory_id: str) -> Memory:
    memory = db.get(Memory, memory_id)
    if memory is None:
        raise ValueError(f"Memory not found: {memory_id}")
    memory.flagged_distressing = False
    memory.distress_note = None
    db.commit()
    db.refresh(memory)
    return memory


# ---- Sessions / Reactions ---------------------------------------


def start_session(
    db: Session,
    patient_id: str,
    caregiver_name: str | None = None,
) -> VisitSession:
    patient = db.get(Patient, patient_id)
    if patient is None:
        raise ValueError(f"Patient not found: {patient_id}")
    visit = VisitSession(patient_id=patient_id, caregiver_name=caregiver_name)
    db.add(visit)
    db.commit()
    db.refresh(visit)
    return visit


def end_session(db: Session, session_id: str, summary: str | None = None) -> VisitSession:
    from datetime import datetime, timezone

    visit = db.get(VisitSession, session_id)
    if visit is None:
        raise ValueError(f"Session not found: {session_id}")
    visit.ended_at = datetime.now(timezone.utc)
    if summary is not None:
        visit.summary = summary
    db.commit()
    db.refresh(visit)
    return visit


def log_reaction(
    db: Session,
    *,
    session_id: str,
    memory_id: str,
    kind: ReactionKind,
    notes: str | None = None,
    bridge: LMDBridge | None = None,
) -> Reaction:
    """Record a reaction and update the associated memory's energy/status.

    If `bridge` is not provided, a fresh one is constructed for this call.
    Caller can supply a long-lived bridge to amortize the grounding
    model-load cost across many reactions in one session.
    """
    visit = db.get(VisitSession, session_id)
    if visit is None:
        raise ValueError(f"Session not found: {session_id}")
    memory = db.get(Memory, memory_id)
    if memory is None:
        raise ValueError(f"Memory not found: {memory_id}")
    if memory.patient_id != visit.patient_id:
        raise ValueError("Memory and session belong to different patients.")

    reaction = Reaction(
        session_id=session_id,
        memory_id=memory_id,
        kind=kind,
        notes=notes,
    )
    db.add(reaction)
    # Give the reaction a created_at before using it below.
    db.flush()

    active_bridge = bridge or LMDBridge(use_language_grounding=False)
    active_bridge.apply_reaction(memory, reaction)

    # Auto-flag on repeated distress reactions — a single distress event
    # is informational; two or more pushes the memory to flagged so
    # suggest_anchor and suggest_visit_memories skip it next time.
    # Query directly rather than via memory.reactions to avoid relationship
    # cache staleness right after the flush above.
    if kind == ReactionKind.RECOGNIZED_DISTRESS:
        from sqlalchemy import func

        distress_count = db.scalar(
            select(func.count(Reaction.id)).where(
                Reaction.memory_id == memory.id,
                Reaction.kind == ReactionKind.RECOGNIZED_DISTRESS,
            )
        ) or 0
        if distress_count >= 2 and not memory.flagged_distressing:
            memory.flagged_distressing = True
            memory.distress_note = (
                memory.distress_note
                or "Auto-flagged after repeated distress reactions."
            )

    db.commit()
    db.refresh(reaction)
    return reaction


# ---- Recommendations -------------------------------------------


def suggest_anchor_for_patient(
    db: Session,
    patient_id: str,
    top_k: int = 3,
    bridge: LMDBridge | None = None,
) -> List[AnchorSuggestion]:
    memories = list_memories(db, patient_id)
    active_bridge = bridge or LMDBridge(use_language_grounding=False)
    return active_bridge.suggest_anchor(memories, top_k=top_k)


def suggest_visit_memories_for_patient(
    db: Session,
    patient_id: str,
    top_k: int = 5,
    bridge: LMDBridge | None = None,
) -> List[VisitSuggestion]:
    memories = list_memories(db, patient_id)
    active_bridge = bridge or LMDBridge(use_language_grounding=False)
    return active_bridge.suggest_visit_memories(memories, top_k=top_k)


def fading_memories(db: Session, patient_id: str) -> List[Memory]:
    """Memories whose status is DORMANT/FADING/GHOST — at-risk for loss."""
    at_risk = {MemoryStatus.DORMANT, MemoryStatus.FADING, MemoryStatus.GHOST}
    return [m for m in list_memories(db, patient_id) if m.status in at_risk]
