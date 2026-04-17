"""Pydantic DTOs for the HTTP API.

Kept separate from SQLAlchemy models so the wire format can evolve
independently of the database schema, and so we don't leak internal
fields (embedding_cached, etc.) to API clients.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from .models import EmotionalTone, MemoryStatus, ReactionKind


class PatientCreate(BaseModel):
    display_name: str
    birth_year: int | None = None
    notes: str | None = None


class PatientRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    display_name: str
    birth_year: int | None
    notes: str | None
    created_at: datetime


class MemoryCreate(BaseModel):
    patient_id: str
    title: str = Field(min_length=1, max_length=300)
    description: str = Field(min_length=1)
    tone: EmotionalTone = EmotionalTone.NEUTRAL
    valence_start: float = 0.0
    valence_peak: float = 0.5
    valence_end: float = 0.3
    approximate_year: int | None = None
    era_label: str | None = None
    tags: str | None = None
    photo_path: str | None = None
    audio_path: str | None = None


class MemoryRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    patient_id: str
    title: str
    description: str
    tone: EmotionalTone
    valence_start: float
    valence_peak: float
    valence_end: float
    approximate_year: int | None
    era_label: str | None
    status: MemoryStatus
    energy: float
    last_surfaced_at: datetime | None
    flagged_distressing: bool
    distress_note: str | None
    created_at: datetime
    updated_at: datetime


class SessionStart(BaseModel):
    patient_id: str
    caregiver_name: str | None = None


class SessionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    patient_id: str
    caregiver_name: str | None
    started_at: datetime
    ended_at: datetime | None
    summary: str | None


class SessionEnd(BaseModel):
    summary: str | None = None


class ReactionCreate(BaseModel):
    session_id: str
    memory_id: str
    kind: ReactionKind
    notes: str | None = None


class ReactionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    session_id: str
    memory_id: str
    kind: ReactionKind
    notes: str | None
    created_at: datetime


class AnchorSuggestionOut(BaseModel):
    memory_id: str
    title: str
    reason: str
    score: float


class VisitSuggestionOut(BaseModel):
    memory_id: str
    title: str
    reason: str
    priority: float


class FlagRequest(BaseModel):
    note: str | None = None
