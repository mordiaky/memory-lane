"""FastAPI application exposing MemoryLane's service layer.

Run locally with:
    uvicorn memory_lane.api:app --reload
"""

from __future__ import annotations

from typing import List

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from . import exporters, service
from .lmd_bridge import AnchorSuggestion, VisitSuggestion
from .models import MemoryStatus
from .schemas import (
    AnchorSuggestionOut,
    EraSummaryOut,
    FlagRequest,
    MemoryCreate,
    MemoryRead,
    PatientCreate,
    PatientRead,
    ReactionCreate,
    ReactionRead,
    SessionEnd,
    SessionRead,
    SessionStart,
    VisitReportOut,
    VisitSuggestionOut,
)
from .storage import get_engine, init_db, iter_session, session_factory

_engine = get_engine()
init_db(_engine)
_session_factory = session_factory(_engine)


def get_db() -> Session:
    # Wrapper so FastAPI can inject; iter_session is a generator.
    yield from iter_session(_session_factory)


app = FastAPI(
    title="MemoryLane",
    version="0.1.0",
    description=(
        "Caregiver-facing life-story and reminiscence companion tool. "
        "Not a medical device; does not diagnose, treat, cure, or prevent "
        "any condition. See README.md for full disclosure."
    ),
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# ---- Patients ---------------------------------------------------


@app.post("/patients", response_model=PatientRead, status_code=status.HTTP_201_CREATED)
def create_patient(payload: PatientCreate, db: Session = Depends(get_db)) -> PatientRead:
    patient = service.create_patient(
        db,
        display_name=payload.display_name,
        birth_year=payload.birth_year,
        notes=payload.notes,
    )
    return PatientRead.model_validate(patient)


@app.get("/patients", response_model=List[PatientRead])
def list_patients(db: Session = Depends(get_db)) -> List[PatientRead]:
    return [PatientRead.model_validate(p) for p in service.list_patients(db)]


@app.get("/patients/{patient_id}", response_model=PatientRead)
def get_patient(patient_id: str, db: Session = Depends(get_db)) -> PatientRead:
    patient = service.get_patient(db, patient_id)
    if patient is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Patient not found")
    return PatientRead.model_validate(patient)


# ---- Memories ---------------------------------------------------


@app.post("/memories", response_model=MemoryRead, status_code=status.HTTP_201_CREATED)
def create_memory(payload: MemoryCreate, db: Session = Depends(get_db)) -> MemoryRead:
    try:
        memory = service.add_memory(
            db,
            patient_id=payload.patient_id,
            title=payload.title,
            description=payload.description,
            tone=payload.tone,
            valence_start=payload.valence_start,
            valence_peak=payload.valence_peak,
            valence_end=payload.valence_end,
            approximate_year=payload.approximate_year,
            era_label=payload.era_label,
            tags=payload.tags,
            photo_path=payload.photo_path,
            audio_path=payload.audio_path,
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return MemoryRead.model_validate(memory)


@app.get("/patients/{patient_id}/memories", response_model=List[MemoryRead])
def list_memories(
    patient_id: str,
    only_status: MemoryStatus | None = None,
    db: Session = Depends(get_db),
) -> List[MemoryRead]:
    return [
        MemoryRead.model_validate(m)
        for m in service.list_memories(db, patient_id, status=only_status)
    ]


@app.post("/memories/{memory_id}/flag", response_model=MemoryRead)
def flag_memory(
    memory_id: str,
    payload: FlagRequest,
    db: Session = Depends(get_db),
) -> MemoryRead:
    try:
        memory = service.flag_memory_distressing(db, memory_id, payload.note)
    except ValueError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return MemoryRead.model_validate(memory)


@app.post("/memories/{memory_id}/unflag", response_model=MemoryRead)
def unflag_memory(memory_id: str, db: Session = Depends(get_db)) -> MemoryRead:
    try:
        memory = service.clear_distress_flag(db, memory_id)
    except ValueError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return MemoryRead.model_validate(memory)


# ---- Sessions / Reactions --------------------------------------


@app.post("/sessions", response_model=SessionRead, status_code=status.HTTP_201_CREATED)
def start_session(payload: SessionStart, db: Session = Depends(get_db)) -> SessionRead:
    try:
        visit = service.start_session(db, payload.patient_id, payload.caregiver_name)
    except ValueError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return SessionRead.model_validate(visit)


@app.post("/sessions/{session_id}/end", response_model=SessionRead)
def end_session(
    session_id: str,
    payload: SessionEnd,
    db: Session = Depends(get_db),
) -> SessionRead:
    try:
        visit = service.end_session(db, session_id, payload.summary)
    except ValueError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return SessionRead.model_validate(visit)


@app.post("/reactions", response_model=ReactionRead, status_code=status.HTTP_201_CREATED)
def log_reaction(payload: ReactionCreate, db: Session = Depends(get_db)) -> ReactionRead:
    try:
        reaction = service.log_reaction(
            db,
            session_id=payload.session_id,
            memory_id=payload.memory_id,
            kind=payload.kind,
            notes=payload.notes,
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    return ReactionRead.model_validate(reaction)


# ---- Recommendations -------------------------------------------


@app.get(
    "/patients/{patient_id}/anchor",
    response_model=List[AnchorSuggestionOut],
)
def suggest_anchor(
    patient_id: str,
    top_k: int = 3,
    db: Session = Depends(get_db),
) -> List[AnchorSuggestionOut]:
    suggestions: List[AnchorSuggestion] = service.suggest_anchor_for_patient(
        db,
        patient_id,
        top_k=top_k,
    )
    return [AnchorSuggestionOut(**vars(s)) for s in suggestions]


@app.get(
    "/patients/{patient_id}/visit-plan",
    response_model=List[VisitSuggestionOut],
)
def suggest_visit_plan(
    patient_id: str,
    top_k: int = 5,
    db: Session = Depends(get_db),
) -> List[VisitSuggestionOut]:
    suggestions: List[VisitSuggestion] = service.suggest_visit_memories_for_patient(
        db,
        patient_id,
        top_k=top_k,
    )
    return [VisitSuggestionOut(**vars(s)) for s in suggestions]


@app.get("/patients/{patient_id}/fading", response_model=List[MemoryRead])
def list_fading(patient_id: str, db: Session = Depends(get_db)) -> List[MemoryRead]:
    return [MemoryRead.model_validate(m) for m in service.fading_memories(db, patient_id)]


@app.get(
    "/patients/{patient_id}/eras",
    response_model=List[EraSummaryOut],
)
def era_overview(patient_id: str, db: Session = Depends(get_db)) -> List[EraSummaryOut]:
    summaries = service.era_overview(db, patient_id)
    return [EraSummaryOut(**s.to_dict()) for s in summaries]


@app.get(
    "/patients/{patient_id}/eras/{era}/memories",
    response_model=List[MemoryRead],
)
def memories_in_era(
    patient_id: str,
    era: str,
    db: Session = Depends(get_db),
) -> List[MemoryRead]:
    return [
        MemoryRead.model_validate(m)
        for m in service.memories_in_era(db, patient_id, era)
    ]


@app.get(
    "/sessions/{session_id}/report",
    response_model=VisitReportOut,
)
def visit_report(session_id: str, db: Session = Depends(get_db)) -> VisitReportOut:
    try:
        return VisitReportOut(**service.build_visit_report(db, session_id))
    except ValueError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc


@app.get("/patients/{patient_id}/export/json")
def export_json(patient_id: str, db: Session = Depends(get_db)) -> dict:
    """Full patient archive as JSON (patient + memories + sessions + reactions)."""
    try:
        return exporters.export_patient_json(db, patient_id)
    except ValueError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc


@app.get(
    "/patients/{patient_id}/export/csv",
    response_class=PlainTextResponse,
)
def export_csv(patient_id: str, db: Session = Depends(get_db)) -> str:
    """Patient's memories as CSV — round-trips through the CSV importer."""
    try:
        return exporters.export_memories_csv_string(db, patient_id)
    except ValueError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
