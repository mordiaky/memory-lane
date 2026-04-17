"""Command-line interface for MemoryLane.

Useful for local smoke testing, data entry before a UI exists, and
scripting. Calls the same service layer the API does.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import importers, service
from .lmd_bridge import LMDBridge
from .models import EmotionalTone, MemoryStatus, ReactionKind
from .storage import get_engine, init_db, session_factory

app = typer.Typer(help="MemoryLane — caregiver-facing life-story tool.")
console = Console()


def _session():
    engine = get_engine()
    init_db(engine)
    return session_factory(engine)()


# ---- Patients ---------------------------------------------------


@app.command("add-patient")
def add_patient(
    display_name: str,
    birth_year: int | None = typer.Option(None),
    notes: str | None = typer.Option(None),
) -> None:
    """Create a new patient record."""
    with _session() as db:
        patient = service.create_patient(db, display_name, birth_year, notes)
    console.print(f"[green]Created patient[/] [bold]{patient.display_name}[/] ({patient.id})")


@app.command("list-patients")
def list_patients() -> None:
    with _session() as db:
        patients = service.list_patients(db)
    if not patients:
        console.print("[yellow]No patients yet.[/]")
        return
    table = Table(title="Patients")
    table.add_column("id", style="cyan")
    table.add_column("name")
    table.add_column("birth_year")
    for p in patients:
        table.add_row(p.id, p.display_name, str(p.birth_year or ""))
    console.print(table)


# ---- Memories ---------------------------------------------------


@app.command("add-memory")
def add_memory(
    patient_id: str,
    title: str,
    description: str,
    tone: EmotionalTone = typer.Option(EmotionalTone.NEUTRAL),
    valence_start: float = typer.Option(0.0),
    valence_peak: float = typer.Option(0.5),
    valence_end: float = typer.Option(0.3),
    approximate_year: int | None = typer.Option(None),
    era_label: str | None = typer.Option(None),
) -> None:
    with _session() as db:
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
    console.print(f"[green]Added memory[/] [bold]{memory.title}[/] ({memory.id})")


@app.command("list-memories")
def list_memories(patient_id: str, only: MemoryStatus | None = None) -> None:
    with _session() as db:
        memories = service.list_memories(db, patient_id, status=only)
    if not memories:
        console.print("[yellow]No memories for that patient.[/]")
        return
    table = Table(title=f"Memories for {patient_id}")
    table.add_column("id", style="cyan", no_wrap=True)
    table.add_column("title")
    table.add_column("status")
    table.add_column("tone")
    table.add_column("energy")
    table.add_column("flagged", justify="center")
    for m in memories:
        table.add_row(
            m.id[:8],
            m.title,
            m.status.value,
            m.tone.value,
            f"{m.energy:.2f}",
            "✓" if m.flagged_distressing else "",
        )
    console.print(table)


@app.command("flag-memory")
def flag_memory(memory_id: str, note: str | None = typer.Option(None)) -> None:
    with _session() as db:
        memory = service.flag_memory_distressing(db, memory_id, note)
    console.print(
        f"[yellow]Flagged[/] '{memory.title}' — will be skipped in anchor and "
        "visit recommendations."
    )


# ---- Sessions / Reactions --------------------------------------


@app.command("start-session")
def start_session(patient_id: str, caregiver_name: str | None = None) -> None:
    with _session() as db:
        visit = service.start_session(db, patient_id, caregiver_name)
    console.print(f"[green]Session started[/]: {visit.id}")


@app.command("log-reaction")
def log_reaction(
    session_id: str,
    memory_id: str,
    kind: ReactionKind,
    notes: str | None = typer.Option(None),
) -> None:
    bridge = LMDBridge(use_language_grounding=False)
    with _session() as db:
        reaction = service.log_reaction(
            db,
            session_id=session_id,
            memory_id=memory_id,
            kind=kind,
            notes=notes,
            bridge=bridge,
        )
    console.print(f"[green]Logged reaction[/] {reaction.kind.value} ({reaction.id[:8]})")


@app.command("end-session")
def end_session(session_id: str, summary: str | None = typer.Option(None)) -> None:
    with _session() as db:
        visit = service.end_session(db, session_id, summary)
    console.print(f"[green]Session ended[/]: {visit.id}")


# ---- Recommendations -------------------------------------------


@app.command("anchor")
def anchor(patient_id: str, top_k: int = 3) -> None:
    """Top N memories to surface if the patient is upset or needs grounding."""
    bridge = LMDBridge(use_language_grounding=False)
    with _session() as db:
        suggestions = service.suggest_anchor_for_patient(
            db,
            patient_id,
            top_k=top_k,
            bridge=bridge,
        )
    if not suggestions:
        console.print("[yellow]No anchor memories available.[/]")
        return
    table = Table(title="Anchor memories (comfort / grounding)")
    table.add_column("score")
    table.add_column("title")
    table.add_column("reason", overflow="fold")
    for s in suggestions:
        table.add_row(f"{s.score:.2f}", s.title, s.reason)
    console.print(table)


@app.command("visit-plan")
def visit_plan(patient_id: str, top_k: int = 5) -> None:
    """Suggested memories to surface in the next caregiver visit."""
    bridge = LMDBridge(use_language_grounding=False)
    with _session() as db:
        suggestions = service.suggest_visit_memories_for_patient(
            db,
            patient_id,
            top_k=top_k,
            bridge=bridge,
        )
    if not suggestions:
        console.print("[yellow]No visit suggestions available.[/]")
        return
    table = Table(title="Visit plan")
    table.add_column("priority")
    table.add_column("title")
    table.add_column("reason", overflow="fold")
    for s in suggestions:
        table.add_row(f"{s.priority:.2f}", s.title, s.reason)
    console.print(table)


@app.command("fading")
def fading(patient_id: str) -> None:
    """Memories at risk of being lost — revisit before they fade further."""
    with _session() as db:
        memories = service.fading_memories(db, patient_id)
    if not memories:
        console.print("[green]No memories currently fading.[/]")
        return
    table = Table(title="At-risk memories")
    table.add_column("id")
    table.add_column("title")
    table.add_column("status")
    table.add_column("energy")
    for m in memories:
        table.add_row(m.id[:8], m.title, m.status.value, f"{m.energy:.2f}")
    console.print(table)


@app.command("eras")
def eras(patient_id: str) -> None:
    """Show a status-at-a-glance overview grouped by life era."""
    with _session() as db:
        summaries = service.era_overview(db, patient_id)
    if not summaries:
        console.print("[yellow]No memories yet for this patient.[/]")
        return
    table = Table(title="Life eras")
    table.add_column("era")
    table.add_column("memories", justify="right")
    table.add_column("vivid", justify="right")
    table.add_column("at-risk", justify="right")
    table.add_column("avg energy", justify="right")
    table.add_column("tones", overflow="fold")
    for s in summaries:
        tones_str = ", ".join(f"{k}:{v}" for k, v in s.tone_breakdown.items())
        table.add_row(
            s.era,
            str(s.memory_count),
            str(s.vivid_count),
            str(s.fading_count),
            f"{s.average_energy:.2f}",
            tones_str,
        )
    console.print(table)


@app.command("visit-report")
def visit_report(session_id: str) -> None:
    """Print an end-of-visit report for a caregiver session."""
    with _session() as db:
        try:
            report = service.build_visit_report(db, session_id)
        except ValueError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(code=1) from exc

    tone_color = {
        "warm": "green",
        "steady": "cyan",
        "muted": "yellow",
        "mixed": "yellow",
        "concerning": "red",
        "no_data": "white",
    }.get(report["overall_tone"], "white")

    header = (
        f"Session {report['session_id'][:8]} • "
        f"patient {report['patient_id'][:8]} • "
        f"caregiver {report['caregiver_name'] or 'unknown'}"
    )
    duration = (
        f"{report['duration_minutes']} min"
        if report["duration_minutes"] is not None
        else "session still open"
    )
    console.print(
        Panel(
            f"[{tone_color}]Overall tone: {report['overall_tone']}[/]\n"
            f"Reactions logged: {report['reactions_logged']} "
            f"(positive {report['positive_count']}, neutral {report['neutral_count']}, "
            f"distress {report['distress_count']}, not-recognized {report['not_recognized_count']}, "
            f"skipped {report['skipped_count']})\n"
            f"Memories surfaced: {report['memories_surfaced']}\n"
            f"Duration: {duration}",
            title=header,
        )
    )

    if report["highlights"]:
        hl_table = Table(title="Highlights (landed well)")
        hl_table.add_column("title")
        hl_table.add_column("notes", overflow="fold")
        for h in report["highlights"]:
            hl_table.add_row(h["title"], h["notes"] or "")
        console.print(hl_table)

    if report["concerns"]:
        co_table = Table(title="Concerns")
        co_table.add_column("title")
        co_table.add_column("concern")
        co_table.add_column("severity")
        co_table.add_column("notes", overflow="fold")
        for c in report["concerns"]:
            co_table.add_row(c["title"], c["concern"], c["severity"], c["notes"] or "")
        console.print(co_table)

    console.print("\n[bold]Follow-up suggestions:[/]")
    for s in report["follow_up_suggestions"]:
        console.print(f"  • {s}")


@app.command("import-csv")
def import_csv(patient_id: str, csv_path: str) -> None:
    """Bulk-import memories for a patient from a CSV file.

    Expected columns: title, description, tone, approximate_year,
    era_label, valence_start, valence_peak, valence_end, tags. Only
    title and description are required.
    """
    with _session() as db:
        try:
            report = importers.import_csv(db, patient_id, csv_path)
        except (FileNotFoundError, ValueError) as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(code=1) from exc

    console.print(
        f"[green]Imported[/] {report.imported}, "
        f"[yellow]skipped[/] {report.skipped}"
    )
    for w in report.warnings:
        console.print(f"  [yellow]warning:[/] {w}")


if __name__ == "__main__":
    app()
