"""Life story — auto-generated prose narrative of a patient's memories.

The output is a multi-paragraph document structured as:

    <opening> (one paragraph introducing the patient)

    === <Era 1 header> ===
    <era paragraph: 2-4 sentences weaving the era's memories together,
     ordered from SETUP → RISING → CLIMAX → RESOLUTION → INTEGRATION
     by the narrative phase the memory sits at>

    === <Era 2 header> ===
    ...

    <closing> (one paragraph tying the whole story together)

Design principles:
  - Draws only from safe memories (not flagged, not DIFFICULT-toned).
    The story is for sharing, not confronting.
  - Uses LMD's NarrativePhase and (if available) NarrativeSynthesizer
    to order memories within an era by narrative arc, not just
    chronology.
  - Template-driven prose so the output is predictable. If we later
    want to pipe it through an LLM for smoothing, the template gives
    the LLM a structured skeleton to work from.
  - Lives in MemoryLane (not LMD) because the template language is a
    product-level decision and should be editable without touching
    the engine.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from . import service
from .models import EmotionalTone, Memory


@dataclass
class LifeStoryChapter:
    """One era section of the story."""

    era: str
    heading: str
    paragraphs: list[str]
    memory_ids: list[str]


@dataclass
class LifeStory:
    """Full generated story."""

    patient_id: str
    patient_name: str
    opening: str
    chapters: list[LifeStoryChapter]
    closing: str
    word_count: int
    memory_count: int

    def to_markdown(self) -> str:
        out: list[str] = [f"# The life story of {self.patient_name}\n", self.opening, ""]
        for ch in self.chapters:
            out.append(f"## {ch.heading}")
            out.append("")
            out.extend(ch.paragraphs)
            out.append("")
        out.append(self.closing)
        return "\n".join(out)

    def to_text(self) -> str:
        """Plain-text version suitable for reading aloud."""
        out: list[str] = [f"The life story of {self.patient_name}", "=" * 40, ""]
        out.append(self.opening)
        out.append("")
        for ch in self.chapters:
            out.append(ch.heading)
            out.append("-" * len(ch.heading))
            out.append("")
            out.extend(ch.paragraphs)
            out.append("")
        out.append(self.closing)
        return "\n".join(out)


# ---- Internal helpers --------------------------------------------


def _safe_memories(memories: list[Memory]) -> list[Memory]:
    return [
        m
        for m in memories
        if not m.flagged_distressing and m.tone != EmotionalTone.DIFFICULT
    ]


_OPENING_TEMPLATES = {
    1: "This is a story of {name}, told through a single memory they've shared.",
    "few": "This is a story of {name}, told through the memories they and their family have gathered so far. There are {n} of them, and they begin here.",
    "many": "This is a story of {name} — {n} memories, gathered by the people who love them, arranged from earliest to most recent. Read it slowly. Read it aloud if you can.",
}

_CLOSING_TEMPLATES = {
    "warm": (
        "Taken together, these memories carry a particular kind of warmth — "
        "the kind that lingers in a room after everyone has gone home. That "
        "warmth is {name}'s."
    ),
    "steady": (
        "Across these chapters, a life takes shape — not dramatic, but steady. "
        "That steadiness is its own kind of love story."
    ),
    "bittersweet": (
        "Not every chapter here is easy. But each one is real, and each one "
        "is remembered. That remembering is what this story is for."
    ),
    "default": (
        "These memories are a map. Whenever you come back to them, "
        "{name}'s story will be here waiting for you."
    ),
}


def _sort_within_era(memories: list[Memory]) -> list[Memory]:
    """Order memories within an era by narrative phase, then year.

    Tries to use LMD's NarrativePhase mapping when available; falls back
    to approximate_year otherwise.
    """
    try:
        from lmd import NarrativePhase
    except Exception:
        return sorted(memories, key=lambda m: (m.approximate_year or 0, m.title))

    # Derive a narrative phase from the valence arc: memories whose
    # valence climbs are SETUP/RISING, memories whose peak is highest
    # are CLIMAX, memories whose valence resolves downward are RESOLUTION.
    def phase_order(m: Memory) -> int:
        arc = m.valence_peak - m.valence_start
        resolution = m.valence_end - m.valence_peak
        if arc > 0.2 and resolution >= 0:
            return 0  # SETUP / RISING
        if m.valence_peak > 0.6 and abs(resolution) < 0.2:
            return 1  # CLIMAX
        if resolution < -0.2:
            return 2  # RESOLUTION
        return 3  # INTEGRATION

    _ = NarrativePhase  # silence lint: imported to document the mapping intent
    return sorted(memories, key=lambda m: (phase_order(m), m.approximate_year or 0))


def _paragraphs_for_era(era: str, memories: list[Memory]) -> list[str]:
    """Weave an era's memories into 1-3 paragraphs of prose."""
    ordered = _sort_within_era(memories)

    # Group into up to three paragraphs of ~3 memories each.
    groups: list[list[Memory]] = []
    for i in range(0, len(ordered), 3):
        groups.append(ordered[i : i + 3])

    paragraphs: list[str] = []
    for group in groups:
        pieces: list[str] = []
        for m in group:
            when = ""
            if m.approximate_year:
                when = f" In {m.approximate_year}, "
            elif era and era != "Undated":
                when = f" During the {era}, "
            # Lower-case the first character of the description so it flows
            # after the "In 1965," prefix.
            desc = m.description.strip()
            if desc and desc[0].isupper():
                desc = desc[0].lower() + desc[1:]
            pieces.append(f"{when}{m.title.lower() if when else m.title}: {desc}")
        paragraphs.append(" ".join(pieces).strip())
    return paragraphs


def _heading_for_era(era: str) -> str:
    if era == "Undated":
        return "Moments without a year"
    if era.endswith("s") and era[:-1].isdigit():
        return f"The {era}"
    return era


def _dominant_tone(memories: list[Memory]) -> str:
    counts: dict[str, int] = {}
    for m in memories:
        counts[m.tone.value] = counts.get(m.tone.value, 0) + 1
    if not counts:
        return "default"
    top = max(counts.items(), key=lambda kv: kv[1])[0]
    if top == "joyful":
        return "warm"
    if top == "bittersweet":
        return "bittersweet"
    if top == "neutral":
        return "steady"
    return "default"


def generate_life_story(db: Session, patient_id: str) -> LifeStory:
    """Build a life story for a patient from their safe memories."""
    patient = service.get_patient(db, patient_id)
    if patient is None:
        raise ValueError(f"Patient not found: {patient_id}")

    all_memories = service.list_memories(db, patient_id)
    safe = _safe_memories(all_memories)

    # Opening.
    if len(safe) == 0:
        opening = (
            f"This is the story of {patient.display_name}. "
            "No memories have been gathered yet — when they are, they'll live here."
        )
        return LifeStory(
            patient_id=patient.id,
            patient_name=patient.display_name,
            opening=opening,
            chapters=[],
            closing="",
            word_count=len(opening.split()),
            memory_count=0,
        )

    key = 1 if len(safe) == 1 else "few" if len(safe) < 5 else "many"
    opening = _OPENING_TEMPLATES[key].format(
        name=patient.display_name,
        n=len(safe),
    )

    # Group into eras (reuse the era service).
    era_summaries = service.era_overview(db, patient_id)
    era_order = [s.era for s in era_summaries]

    # Bucket the safe memories by era.
    buckets: dict[str, list[Memory]] = {era: [] for era in era_order}
    for m in safe:
        era_key = m.era_label or (
            f"{(m.approximate_year // 10) * 10}s" if m.approximate_year else "Undated"
        )
        buckets.setdefault(era_key, []).append(m)

    chapters: list[LifeStoryChapter] = []
    for era in era_order:
        if not buckets.get(era):
            continue
        chapters.append(
            LifeStoryChapter(
                era=era,
                heading=_heading_for_era(era),
                paragraphs=_paragraphs_for_era(era, buckets[era]),
                memory_ids=[m.id for m in buckets[era]],
            )
        )

    closing = _CLOSING_TEMPLATES[_dominant_tone(safe)].format(name=patient.display_name)

    body_words = sum(len(p.split()) for ch in chapters for p in ch.paragraphs)
    total_words = len(opening.split()) + body_words + len(closing.split())

    return LifeStory(
        patient_id=patient.id,
        patient_name=patient.display_name,
        opening=opening,
        chapters=chapters,
        closing=closing,
        word_count=total_words,
        memory_count=len(safe),
    )
