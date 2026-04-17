"""Server-rendered HTML UI, mounted on the same FastAPI app as the JSON API.

This is deliberately small and templated — no SPA, no bundler. HTMX is
used on the session page so caregivers can log reactions without a
full page reload while still keeping the server authoritative.

The HTML and JSON routes share the same service layer. They MUST NOT
duplicate business logic — if you find yourself reimplementing
something from `service.py` here, stop and route through the service.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from . import media as media_module
from . import service
from .models import EmotionalTone, Memory, Patient, ReactionKind
from .models import Session as VisitSession

_PKG_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(_PKG_DIR / "templates"))
static_files = StaticFiles(directory=str(_PKG_DIR / "static"))

router = APIRouter()


def _get_db():
    # The api module owns the engine; import at call-time to avoid a
    # circular import between web.py and api.py.
    from .api import _session_factory

    with _session_factory() as session:
        yield session


def _status_counts(memories: list[Memory]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for m in memories:
        counts[m.status.value] = counts.get(m.status.value, 0) + 1
    return counts


# ---- Index -----------------------------------------------------


@router.get("/", response_class=HTMLResponse)
def index(request: Request, db: Session = Depends(_get_db)) -> HTMLResponse:
    patients = service.list_patients(db)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "patients": patients},
    )


@router.post("/patients")
def web_create_patient(
    display_name: str = Form(...),
    birth_year: int | None = Form(None),
    notes: str | None = Form(None),
    db: Session = Depends(_get_db),
) -> RedirectResponse:
    patient = service.create_patient(db, display_name, birth_year, notes)
    return RedirectResponse(
        f"/patients/{patient.id}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


# ---- Patient dashboard ----------------------------------------


@router.get("/patients/{patient_id}", response_class=HTMLResponse)
def web_patient(
    patient_id: str,
    request: Request,
    db: Session = Depends(_get_db),
) -> HTMLResponse:
    patient = service.get_patient(db, patient_id)
    if patient is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Patient not found")
    memories = service.list_memories(db, patient_id)
    eras = service.era_overview(db, patient_id)
    # Recent sessions, newest first, cap to 5.
    recent_sessions = sorted(
        patient.sessions, key=lambda s: s.started_at, reverse=True
    )[:5]
    return templates.TemplateResponse(
        "patient.html",
        {
            "request": request,
            "patient": patient,
            "memories": memories,
            "status_counts": _status_counts(memories),
            "eras": [e.to_dict() for e in eras],
            "recent_sessions": recent_sessions,
        },
    )


# ---- Memory pages ---------------------------------------------


@router.get("/patients/{patient_id}/memories/new", response_class=HTMLResponse)
def web_new_memory_form(
    patient_id: str,
    request: Request,
    db: Session = Depends(_get_db),
) -> HTMLResponse:
    patient = service.get_patient(db, patient_id)
    if patient is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Patient not found")
    return templates.TemplateResponse(
        "add_memory.html",
        {"request": request, "patient": patient},
    )


@router.post("/patients/{patient_id}/memories")
def web_create_memory(
    patient_id: str,
    title: str = Form(...),
    description: str = Form(...),
    tone: EmotionalTone = Form(EmotionalTone.NEUTRAL),
    valence_start: float = Form(0.0),
    valence_peak: float = Form(0.5),
    valence_end: float = Form(0.3),
    approximate_year: int | None = Form(None),
    era_label: str | None = Form(None),
    db: Session = Depends(_get_db),
) -> RedirectResponse:
    try:
        memory = service.add_memory(
            db,
            patient_id=patient_id,
            title=title,
            description=description,
            tone=tone,
            valence_start=valence_start,
            valence_peak=valence_peak,
            valence_end=valence_end,
            approximate_year=approximate_year,
            era_label=era_label,
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return RedirectResponse(
        f"/memories/{memory.id}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/memories/{memory_id}", response_class=HTMLResponse)
def web_memory(
    memory_id: str,
    request: Request,
    db: Session = Depends(_get_db),
) -> HTMLResponse:
    memory = db.get(Memory, memory_id)
    if memory is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Memory not found")
    patient = db.get(Patient, memory.patient_id)
    return templates.TemplateResponse(
        "memory.html",
        {"request": request, "memory": memory, "patient": patient},
    )


@router.post("/memories/{memory_id}/media")
def web_upload_media(
    memory_id: str,
    file: UploadFile,
    db: Session = Depends(_get_db),
) -> RedirectResponse:
    data = file.file.read()
    try:
        service.attach_media_to_memory(db, memory_id, file.filename or "upload", data)
    except ValueError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except media_module.UnsupportedMediaType as exc:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)
        ) from exc
    return RedirectResponse(
        f"/memories/{memory_id}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.post("/memories/{memory_id}/flag")
def web_flag(
    memory_id: str,
    note: str | None = Form(None),
    db: Session = Depends(_get_db),
) -> RedirectResponse:
    try:
        service.flag_memory_distressing(db, memory_id, note)
    except ValueError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return RedirectResponse(
        f"/memories/{memory_id}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.post("/memories/{memory_id}/unflag")
def web_unflag(memory_id: str, db: Session = Depends(_get_db)) -> RedirectResponse:
    try:
        service.clear_distress_flag(db, memory_id)
    except ValueError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return RedirectResponse(
        f"/memories/{memory_id}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


# ---- Sessions -------------------------------------------------


@router.post("/patients/{patient_id}/sessions")
def web_start_session(
    patient_id: str,
    caregiver_name: str | None = Form(None),
    db: Session = Depends(_get_db),
) -> RedirectResponse:
    try:
        visit = service.start_session(db, patient_id, caregiver_name)
    except ValueError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return RedirectResponse(
        f"/sessions/{visit.id}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/sessions/{session_id}", response_class=HTMLResponse)
def web_session(
    session_id: str,
    request: Request,
    db: Session = Depends(_get_db),
) -> HTMLResponse:
    visit = db.get(VisitSession, session_id)
    if visit is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
    patient = db.get(Patient, visit.patient_id)
    suggestions = service.suggest_visit_memories_for_patient(db, visit.patient_id)
    # Pull memory objects for each suggestion so the template has both.
    memories_by_id = {m.id: m for m in service.list_memories(db, visit.patient_id)}
    rendered_suggestions = []
    for s in suggestions:
        m = memories_by_id.get(s.memory_id)
        if m is None:
            continue
        rendered_suggestions.append({"suggestion": s, "memory": m})
    return templates.TemplateResponse(
        "session.html",
        {
            "request": request,
            "session": visit,
            "patient": patient,
            "suggestions": rendered_suggestions,
        },
    )


@router.post(
    "/sessions/{session_id}/reactions",
    response_class=HTMLResponse,
)
def web_log_reaction(
    session_id: str,
    request: Request,
    memory_id: str = Form(...),
    kind: ReactionKind = Form(...),
    db: Session = Depends(_get_db),
) -> HTMLResponse:
    """HTMX target — logs a reaction and returns the updated memory card."""
    try:
        service.log_reaction(
            db,
            session_id=session_id,
            memory_id=memory_id,
            kind=kind,
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    visit = db.get(VisitSession, session_id)
    memory = db.get(Memory, memory_id)
    suggestion = {"memory_id": memory_id, "title": memory.title, "reason": "", "priority": 0.0}
    return templates.TemplateResponse(
        "fragments/memory_card.html",
        {
            "request": request,
            "suggestion": suggestion,
            "memory": memory,
            "session": visit,
            "logged_reaction": kind.value,
        },
    )


@router.post("/sessions/{session_id}/end")
def web_end_session(
    session_id: str,
    summary: str | None = Form(None),
    db: Session = Depends(_get_db),
) -> RedirectResponse:
    try:
        service.end_session(db, session_id, summary or None)
    except ValueError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return RedirectResponse(
        f"/sessions/{session_id}/report",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/sessions/{session_id}/report", response_class=HTMLResponse)
def web_report(
    session_id: str,
    request: Request,
    db: Session = Depends(_get_db),
) -> HTMLResponse:
    try:
        report = service.build_visit_report(db, session_id)
    except ValueError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    patient = db.get(Patient, report["patient_id"])
    return templates.TemplateResponse(
        "report.html",
        {"request": request, "report": report, "patient": patient},
    )
