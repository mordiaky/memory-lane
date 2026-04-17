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


# ---- Era browsing -----------------------------------------------


def _era_key(memory: Memory) -> str:
    """Return the era label for a memory, derived from explicit label or year."""
    if memory.era_label:
        return memory.era_label
    if memory.approximate_year is not None:
        decade = (memory.approximate_year // 10) * 10
        return f"{decade}s"
    return "Undated"


class EraSummary:
    """A snapshot of a single era of the patient's life story."""

    __slots__ = (
        "era",
        "memory_count",
        "status_breakdown",
        "tone_breakdown",
        "average_energy",
        "fading_count",
        "vivid_count",
    )

    def __init__(
        self,
        era: str,
        memory_count: int,
        status_breakdown: dict[str, int],
        tone_breakdown: dict[str, int],
        average_energy: float,
        fading_count: int,
        vivid_count: int,
    ) -> None:
        self.era = era
        self.memory_count = memory_count
        self.status_breakdown = status_breakdown
        self.tone_breakdown = tone_breakdown
        self.average_energy = average_energy
        self.fading_count = fading_count
        self.vivid_count = vivid_count

    def to_dict(self) -> dict:
        return {
            "era": self.era,
            "memory_count": self.memory_count,
            "status_breakdown": self.status_breakdown,
            "tone_breakdown": self.tone_breakdown,
            "average_energy": round(self.average_energy, 3),
            "fading_count": self.fading_count,
            "vivid_count": self.vivid_count,
        }


def era_overview(db: Session, patient_id: str) -> List[EraSummary]:
    """Return a summary of the patient's life story grouped by era.

    Eras come from the explicit `era_label` when set, otherwise from the
    decade of `approximate_year`; memories missing both land in an
    "Undated" bucket. Eras are sorted chronologically when the label
    parses as a decade (e.g. "1960s"), otherwise alphabetically, with
    "Undated" always last.
    """
    memories = list_memories(db, patient_id)
    buckets: dict[str, list[Memory]] = {}
    for m in memories:
        buckets.setdefault(_era_key(m), []).append(m)

    summaries: list[EraSummary] = []
    for era, era_memories in buckets.items():
        statuses: dict[str, int] = {}
        tones: dict[str, int] = {}
        energy_sum = 0.0
        fading = 0
        vivid = 0
        at_risk = {MemoryStatus.DORMANT, MemoryStatus.FADING, MemoryStatus.GHOST}
        for m in era_memories:
            statuses[m.status.value] = statuses.get(m.status.value, 0) + 1
            tones[m.tone.value] = tones.get(m.tone.value, 0) + 1
            energy_sum += m.energy
            if m.status in at_risk:
                fading += 1
            if m.status == MemoryStatus.VIVID:
                vivid += 1
        summaries.append(
            EraSummary(
                era=era,
                memory_count=len(era_memories),
                status_breakdown=statuses,
                tone_breakdown=tones,
                average_energy=energy_sum / len(era_memories),
                fading_count=fading,
                vivid_count=vivid,
            )
        )

    def sort_key(s: EraSummary) -> tuple[int, str]:
        # Undated last, decade labels numeric, everything else alphabetical.
        if s.era == "Undated":
            return (2, "")
        if s.era.endswith("s") and s.era[:-1].isdigit():
            return (0, s.era[:-1].zfill(4))  # sort "1960s" before "1970s"
        return (1, s.era)

    summaries.sort(key=sort_key)
    return summaries


def memories_in_era(db: Session, patient_id: str, era: str) -> List[Memory]:
    """Return all memories in a given era label for a patient."""
    return [m for m in list_memories(db, patient_id) if _era_key(m) == era]


# ---- Visit reports ----------------------------------------------


def _overall_tone_for_session(
    positive: int,
    neutral: int,
    distress: int,
    not_recognized: int,
) -> str:
    """Classify the overall feel of a visit based on reaction mix.

    The category is deliberately conservative — never 'great' if
    there was any distress, never 'concerning' just from skipped
    memories. Caregivers should read these as a sanity check on their
    own intuition, not a medical judgement.
    """
    total = positive + neutral + distress + not_recognized
    if total == 0:
        return "no_data"
    if distress >= 2 or (distress >= 1 and distress > positive):
        return "concerning"
    if positive >= 2 and distress == 0 and not_recognized < positive:
        return "warm"
    if positive > 0 and distress == 0:
        return "steady"
    if not_recognized > positive and distress == 0:
        return "muted"
    return "mixed"


def build_visit_report(db: Session, session_id: str) -> dict:
    """Produce an end-of-visit report.

    The return shape is a plain dict so both the API and CLI can render
    it without pulling in the Pydantic layer here. It covers:
      - headline counts of each reaction kind
      - the memories that landed well (highlights)
      - the memories that caused distress or weren't recognized (concerns)
      - a handful of concrete follow-up suggestions (e.g., 'revisit X
        next time, skip Y')
    """
    visit = db.get(VisitSession, session_id)
    if visit is None:
        raise ValueError(f"Session not found: {session_id}")

    positive = neutral = distress = not_recognized = skipped = 0
    highlights: list[dict] = []
    concerns: list[dict] = []
    memories_touched: set[str] = set()

    for reaction in visit.reactions:
        memories_touched.add(reaction.memory_id)
        memory = reaction.memory
        entry = {
            "memory_id": memory.id,
            "title": memory.title,
            "kind": reaction.kind.value,
            "notes": reaction.notes,
        }
        if reaction.kind == ReactionKind.RECOGNIZED_POSITIVE:
            positive += 1
            highlights.append(entry)
        elif reaction.kind == ReactionKind.RECOGNIZED_NEUTRAL:
            neutral += 1
        elif reaction.kind == ReactionKind.RECOGNIZED_DISTRESS:
            distress += 1
            concerns.append(
                {**entry, "concern": "distress", "severity": "high"}
            )
        elif reaction.kind == ReactionKind.NOT_RECOGNIZED:
            not_recognized += 1
            concerns.append(
                {**entry, "concern": "not_recognized", "severity": "medium"}
            )
        elif reaction.kind == ReactionKind.SKIPPED:
            skipped += 1

    tone = _overall_tone_for_session(positive, neutral, distress, not_recognized)

    # Build follow-up suggestions from the data.
    follow_ups: list[str] = []
    if distress > 0:
        follow_ups.append(
            f"{distress} memory/memories caused distress this visit. Review them "
            "before next visit and consider flagging if distress recurs."
        )
    if not_recognized > 0:
        follow_ups.append(
            f"{not_recognized} memory/memories were not recognized. Their status "
            "has been adjusted; prioritize reinforcing still-vivid memories "
            "from the same era next visit."
        )
    if positive > 0 and distress == 0:
        follow_ups.append(
            f"{positive} memory/memories landed well — safe ground for future visits."
        )
    if not follow_ups:
        follow_ups.append("No reactions were logged in this session.")

    duration_minutes: float | None = None
    if visit.ended_at is not None:
        delta = visit.ended_at - visit.started_at
        duration_minutes = round(delta.total_seconds() / 60.0, 1)

    return {
        "session_id": visit.id,
        "patient_id": visit.patient_id,
        "caregiver_name": visit.caregiver_name,
        "started_at": visit.started_at,
        "ended_at": visit.ended_at,
        "duration_minutes": duration_minutes,
        "reactions_logged": positive + neutral + distress + not_recognized + skipped,
        "positive_count": positive,
        "neutral_count": neutral,
        "distress_count": distress,
        "not_recognized_count": not_recognized,
        "skipped_count": skipped,
        "memories_surfaced": len(memories_touched),
        "overall_tone": tone,
        "highlights": highlights,
        "concerns": concerns,
        "follow_up_suggestions": follow_ups,
    }


