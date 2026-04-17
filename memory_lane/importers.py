"""Bulk importers — let families bring existing memory lists in without
having to retype them through the CLI.

Currently supports CSV with the following columns (header row required):

    title, description, tone, approximate_year, era_label,
    valence_start, valence_peak, valence_end, tags

All columns except `title` and `description` are optional. Unknown tone
values are coerced to NEUTRAL and logged. Rows missing a title or
description are skipped.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sqlalchemy.orm import Session

from . import service
from .models import EmotionalTone


@dataclass
class ImportReport:
    imported: int
    skipped: int
    warnings: list[str]


def _coerce_tone(raw: str | None, warnings: list[str], row_num: int) -> EmotionalTone:
    if raw is None or raw.strip() == "":
        return EmotionalTone.NEUTRAL
    key = raw.strip().lower()
    for tone in EmotionalTone:
        if tone.value == key:
            return tone
    warnings.append(
        f"row {row_num}: unknown tone '{raw}', defaulting to 'neutral'"
    )
    return EmotionalTone.NEUTRAL


def _coerce_optional_int(raw: str | None) -> int | None:
    if raw is None:
        return None
    raw = raw.strip()
    if raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _coerce_optional_float(raw: str | None, default: float) -> float:
    if raw is None:
        return default
    raw = raw.strip()
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def import_csv(
    db: Session,
    patient_id: str,
    csv_path: str | Path,
) -> ImportReport:
    """Bulk-insert memories for a patient from a CSV file.

    Each successful row is committed individually so a bad row late in
    the file doesn't undo earlier progress. Returns an ImportReport
    the caller can display.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    imported = 0
    skipped = 0
    warnings: list[str] = []

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=2):  # row 1 is the header
            title = (row.get("title") or "").strip()
            description = (row.get("description") or "").strip()
            if not title or not description:
                skipped += 1
                warnings.append(
                    f"row {row_num}: missing title or description — skipped"
                )
                continue

            tone = _coerce_tone(row.get("tone"), warnings, row_num)
            approximate_year = _coerce_optional_int(row.get("approximate_year"))
            era_label = (row.get("era_label") or None)
            if isinstance(era_label, str) and era_label.strip() == "":
                era_label = None
            tags = (row.get("tags") or None)
            if isinstance(tags, str) and tags.strip() == "":
                tags = None

            valence_start = _coerce_optional_float(row.get("valence_start"), 0.0)
            valence_peak = _coerce_optional_float(row.get("valence_peak"), 0.5)
            valence_end = _coerce_optional_float(row.get("valence_end"), 0.3)

            try:
                service.add_memory(
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
                    tags=tags,
                )
                imported += 1
            except ValueError as exc:
                skipped += 1
                warnings.append(f"row {row_num}: {exc}")

    return ImportReport(imported=imported, skipped=skipped, warnings=warnings)


def iter_rows(csv_path: str | Path) -> Iterable[dict]:
    """Generator version — yields each parsed row without inserting.

    Useful for dry-run previews in the CLI.
    """
    with Path(csv_path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        yield from reader
