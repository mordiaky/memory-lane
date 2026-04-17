"""SQLAlchemy ORM models for MemoryLane.

These models are the user-facing data shapes — the things a family or
caregiver reads and writes. LMD's embedding + dynamics state is stored
alongside (see `memory_lane.lmd_bridge`) but is never exposed directly
through the API; callers see labels, text, and scores.

Design principles:
  - Every Memory belongs to exactly one Patient.
  - Emotional tone is stored at two levels: the caregiver's human
    categorization (joyful / bittersweet / difficult / neutral) AND a
    3-point valence trajectory (start, peak, end) that maps to LMD's
    ValenceTrajectory without exposing it.
  - Each Session captures a caregiver visit. Reactions within a session
    link back to specific memories and are the primary signal that
    drives LMD's energy updates.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import List

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _new_id() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class EmotionalTone(str, enum.Enum):
    """Caregiver-provided emotional categorization of a memory.

    This is the coarse human label. The fine-grained valence trajectory
    lives in Memory.valence_start/peak/end.
    """

    JOYFUL = "joyful"
    BITTERSWEET = "bittersweet"
    DIFFICULT = "difficult"
    NEUTRAL = "neutral"


class MemoryStatus(str, enum.Enum):
    """Tracks the caregiver-visible liveness of a memory.

    Derived from the underlying LMD metabolic energy but exposed as a
    human-meaningful label for the caregiver UI.
    """

    VIVID = "vivid"        # Patient recognized recently
    ACTIVE = "active"      # Patient recognized somewhat recently
    DORMANT = "dormant"    # Not recently surfaced, may need reinforcement
    FADING = "fading"      # Showing sustained decline in recognition
    GHOST = "ghost"        # No recent recognition; family archive only


class ReactionKind(str, enum.Enum):
    """Caregiver-observed reaction when surfacing a memory in a session."""

    RECOGNIZED_POSITIVE = "recognized_positive"   # e.g. smile, calm, engagement
    RECOGNIZED_NEUTRAL = "recognized_neutral"     # e.g. acknowledgement, no affect
    RECOGNIZED_DISTRESS = "recognized_distress"   # e.g. agitation, sadness
    NOT_RECOGNIZED = "not_recognized"             # patient did not recognize
    SKIPPED = "skipped"                           # caregiver chose not to surface


class Patient(Base):
    """A person the family is supporting."""

    __tablename__ = "patients"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    display_name: Mapped[str] = mapped_column(String(200))
    birth_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    memories: Mapped[List[Memory]] = relationship(
        back_populates="patient",
        cascade="all, delete-orphan",
    )
    sessions: Mapped[List[Session]] = relationship(
        back_populates="patient",
        cascade="all, delete-orphan",
    )


class Memory(Base):
    """A single life-story memory a family wants to share with a patient."""

    __tablename__ = "memories"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    patient_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("patients.id", ondelete="CASCADE"),
        index=True,
    )

    # Human-readable content — this is what a caregiver types in.
    title: Mapped[str] = mapped_column(String(300))
    description: Mapped[str] = mapped_column(Text)
    approximate_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    era_label: Mapped[str | None] = mapped_column(String(100), nullable=True)
    tags: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON-encoded list

    # Optional media pointers.
    photo_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    audio_path: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Emotional shape. `tone` is the caregiver label; the three valence
    # floats map to LMD's ValenceTrajectory(points=[start, peak, end]).
    tone: Mapped[EmotionalTone] = mapped_column(
        Enum(EmotionalTone),
        default=EmotionalTone.NEUTRAL,
    )
    valence_start: Mapped[float] = mapped_column(Float, default=0.0)
    valence_peak: Mapped[float] = mapped_column(Float, default=0.5)
    valence_end: Mapped[float] = mapped_column(Float, default=0.3)

    # Status — derived from LMD energy, stored here so the UI can query
    # quickly without computing every time.
    status: Mapped[MemoryStatus] = mapped_column(
        Enum(MemoryStatus),
        default=MemoryStatus.VIVID,
    )
    energy: Mapped[float] = mapped_column(Float, default=1.0)
    last_surfaced_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Safety flag — caregiver can mark a memory that causes distress.
    # suggest_anchor() and suggest_visit_memories() must skip these.
    flagged_distressing: Mapped[bool] = mapped_column(Boolean, default=False)

    # Reverse link to any caregiver note about why a memory was flagged.
    distress_note: Mapped[str | None] = mapped_column(Text, nullable=True)

    # LMD coupling — populated by the bridge module. Nullable so we can
    # insert a memory before the LMD embedding run completes.
    embedding_cached: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_now,
        onupdate=_now,
    )

    patient: Mapped[Patient] = relationship(back_populates="memories")
    reactions: Mapped[List[Reaction]] = relationship(
        back_populates="memory",
        cascade="all, delete-orphan",
    )


class Session(Base):
    """A caregiver visit during which memories were surfaced."""

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    patient_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("patients.id", ondelete="CASCADE"),
        index=True,
    )
    caregiver_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    patient: Mapped[Patient] = relationship(back_populates="sessions")
    reactions: Mapped[List[Reaction]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )


class Reaction(Base):
    """One observed patient reaction to a surfaced memory in a session."""

    __tablename__ = "reactions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        index=True,
    )
    memory_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("memories.id", ondelete="CASCADE"),
        index=True,
    )
    kind: Mapped[ReactionKind] = mapped_column(Enum(ReactionKind))
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    session: Mapped[Session] = relationship(back_populates="reactions")
    memory: Mapped[Memory] = relationship(back_populates="reactions")
