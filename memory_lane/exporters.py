"""Exporters — let families take their data with them in portable formats.

Two formats supported:
  - JSON: the complete archive (patient + all memories + all sessions +
    all reactions). Roundtrips losslessly back through the DB.
  - CSV: memories only, matching the columns the CSV importer accepts.

No family should be trapped in this tool. Export is a first-class
feature, not an afterthought.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from . import service
from .models import Memory, Patient


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _memory_to_dict(memory: Memory) -> dict[str, Any]:
    return {
        "id": memory.id,
        "title": memory.title,
        "description": memory.description,
        "tone": memory.tone.value,
        "valence_start": memory.valence_start,
        "valence_peak": memory.valence_peak,
        "valence_end": memory.valence_end,
        "approximate_year": memory.approximate_year,
        "era_label": memory.era_label,
        "tags": memory.tags,
        "photo_path": memory.photo_path,
        "audio_path": memory.audio_path,
        "status": memory.status.value,
        "energy": memory.energy,
        "flagged_distressing": memory.flagged_distressing,
        "distress_note": memory.distress_note,
        "last_surfaced_at": _iso(memory.last_surfaced_at),
        "created_at": _iso(memory.created_at),
        "updated_at": _iso(memory.updated_at),
    }


def export_patient_json(db: Session, patient_id: str) -> dict[str, Any]:
    """Return a dict representing the complete patient archive.

    Shape: {patient, memories, sessions (each with nested reactions)}.
    Timestamps are ISO-8601 strings.
    """
    patient = db.get(Patient, patient_id)
    if patient is None:
        raise ValueError(f"Patient not found: {patient_id}")

    sessions_out: list[dict[str, Any]] = []
    for s in patient.sessions:
        sessions_out.append(
            {
                "id": s.id,
                "caregiver_name": s.caregiver_name,
                "started_at": _iso(s.started_at),
                "ended_at": _iso(s.ended_at),
                "summary": s.summary,
                "reactions": [
                    {
                        "id": r.id,
                        "memory_id": r.memory_id,
                        "kind": r.kind.value,
                        "notes": r.notes,
                        "created_at": _iso(r.created_at),
                    }
                    for r in s.reactions
                ],
            }
        )

    return {
        "schema": "memory-lane-export/v1",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "patient": {
            "id": patient.id,
            "display_name": patient.display_name,
            "birth_year": patient.birth_year,
            "notes": patient.notes,
            "created_at": _iso(patient.created_at),
        },
        "memories": [_memory_to_dict(m) for m in patient.memories],
        "sessions": sessions_out,
    }


def export_patient_json_string(
    db: Session,
    patient_id: str,
    *,
    indent: int | None = 2,
) -> str:
    """Serialize the patient archive as a JSON string."""
    return json.dumps(export_patient_json(db, patient_id), indent=indent)


def export_memories_csv_string(db: Session, patient_id: str) -> str:
    """Serialize a patient's memories as CSV matching the importer's columns."""
    memories = service.list_memories(db, patient_id)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "title",
            "description",
            "tone",
            "approximate_year",
            "era_label",
            "valence_start",
            "valence_peak",
            "valence_end",
            "tags",
        ]
    )
    for m in memories:
        writer.writerow(
            [
                m.title,
                m.description,
                m.tone.value,
                m.approximate_year if m.approximate_year is not None else "",
                m.era_label or "",
                m.valence_start,
                m.valence_peak,
                m.valence_end,
                m.tags or "",
            ]
        )
    return buf.getvalue()
