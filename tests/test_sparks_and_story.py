"""Sparks, life-story, and dynamics tests."""

from __future__ import annotations

import pytest

from memory_lane import dynamics, life_story, service, sparks
from memory_lane.models import EmotionalTone, MemoryStatus


def _seed_eleanor(db):
    patient = service.create_patient(db, "Eleanor Thomas", birth_year=1942)
    service.add_memory(
        db, patient_id=patient.id, title="Wedding day",
        description="Sunny afternoon in 1965.", tone=EmotionalTone.JOYFUL,
        approximate_year=1965,
    )
    service.add_memory(
        db, patient_id=patient.id, title="Dog Biscuit",
        description="Her childhood dog.", tone=EmotionalTone.JOYFUL,
        approximate_year=1948,
    )
    service.add_memory(
        db, patient_id=patient.id, title="Garden roses",
        description="Late summer rose garden.", tone=EmotionalTone.JOYFUL,
        approximate_year=1985,
    )
    service.add_memory(
        db, patient_id=patient.id, title="Hospital year",
        description="A long illness.", tone=EmotionalTone.DIFFICULT,
        approximate_year=2001,
    )
    return patient


# ---- Sparks ----------------------------------------------------


def test_sparks_generate_from_safe_memories_only(db) -> None:
    patient = _seed_eleanor(db)
    results = sparks.generate_sparks(db, patient.id, n=5, rng_seed=42)
    # The DIFFICULT "Hospital year" should never appear in source titles.
    for s in results:
        assert "Hospital year" not in s.source_memory_titles
    assert 1 <= len(results) <= 5


def test_sparks_empty_when_too_few_memories(db) -> None:
    patient = service.create_patient(db, "Eleanor")
    service.add_memory(
        db, patient_id=patient.id, title="Only memory",
        description="Single data point.", tone=EmotionalTone.JOYFUL,
    )
    results = sparks.generate_sparks(db, patient.id, rng_seed=1)
    assert results == []


def test_sparks_are_deterministic_with_seed(db) -> None:
    patient = _seed_eleanor(db)
    a = sparks.generate_sparks(db, patient.id, n=3, rng_seed=7)
    b = sparks.generate_sparks(db, patient.id, n=3, rng_seed=7)
    assert [s.source_memory_ids for s in a] == [s.source_memory_ids for s in b]


def test_sparks_never_have_difficult_tone_in_sources(db) -> None:
    patient = service.create_patient(db, "Eleanor")
    joyful = service.add_memory(
        db, patient_id=patient.id, title="Summer",
        description="Warm.", tone=EmotionalTone.JOYFUL,
    )
    difficult = service.add_memory(
        db, patient_id=patient.id, title="Grief",
        description="A hard season.", tone=EmotionalTone.DIFFICULT,
    )
    service.add_memory(
        db, patient_id=patient.id, title="Neutral",
        description="Middle-ground.", tone=EmotionalTone.NEUTRAL,
    )
    results = sparks.generate_sparks(db, patient.id, n=5, rng_seed=3)
    # With only 2 non-difficult memories, every spark uses exactly those two.
    for s in results:
        assert difficult.id not in s.source_memory_ids
    assert any(joyful.id in s.source_memory_ids for s in results)


# ---- Life story -----------------------------------------------


def test_life_story_for_empty_patient(db) -> None:
    patient = service.create_patient(db, "Fresh")
    story = life_story.generate_life_story(db, patient.id)
    assert story.memory_count == 0
    assert story.chapters == []
    assert "Fresh" in story.opening


def test_life_story_basic_structure(db) -> None:
    patient = _seed_eleanor(db)
    story = life_story.generate_life_story(db, patient.id)
    # 4 memories total, 1 is difficult → 3 safe → at least 1 chapter.
    assert story.memory_count == 3
    assert len(story.chapters) >= 1
    # Chapters are chronological: 1940s era should come before 1980s.
    decades = [ch.era for ch in story.chapters if ch.era.endswith("s")]
    assert decades == sorted(decades)
    # The DIFFICULT memory's id is never referenced in the chapters.
    all_ids = {mid for ch in story.chapters for mid in ch.memory_ids}
    difficult_mem = next(m for m in service.list_memories(db, patient.id) if m.tone == EmotionalTone.DIFFICULT)
    assert difficult_mem.id not in all_ids
    # Opening mentions the patient by name.
    assert "Eleanor" in story.opening


def test_life_story_markdown_export(db) -> None:
    patient = _seed_eleanor(db)
    story = life_story.generate_life_story(db, patient.id)
    md = story.to_markdown()
    # Starts with the H1.
    assert md.startswith("# The life story of Eleanor Thomas")
    # Every chapter heading appears.
    for ch in story.chapters:
        assert f"## {ch.heading}" in md


def test_life_story_for_unknown_patient(db) -> None:
    with pytest.raises(ValueError):
        life_story.generate_life_story(db, "nope-nope-nope")


# ---- Dynamics --------------------------------------------------


def test_dynamics_tick_does_not_touch_flagged_memories(db) -> None:
    patient = _seed_eleanor(db)
    memories = service.list_memories(db, patient.id)
    difficult = next(m for m in memories if m.tone == EmotionalTone.DIFFICULT)
    initial_energy = difficult.energy

    result = dynamics.tick(db, patient.id, dt=1.0)
    assert result.patient_id == patient.id
    assert result.memories_unchanged >= 1

    db.refresh(difficult)
    assert difficult.energy == initial_energy


def test_dynamics_tick_with_too_few_memories(db) -> None:
    patient = service.create_patient(db, "Solo")
    service.add_memory(
        db, patient_id=patient.id, title="The only one",
        description="Just one.", tone=EmotionalTone.JOYFUL,
    )
    result = dynamics.tick(db, patient.id)
    assert result.memories_stepped == 0
    assert result.reason == "fewer-than-two-eligible-memories"


def test_dynamics_tick_updates_statuses_on_energy_change(db) -> None:
    patient = _seed_eleanor(db)
    # Force all eligible memories to fading energy.
    for m in service.list_memories(db, patient.id):
        if m.tone != EmotionalTone.DIFFICULT:
            m.energy = 0.3
            m.status = MemoryStatus.FADING
    db.commit()

    result = dynamics.tick(db, patient.id, dt=1.0)
    # Either stepped via LMD or fell through to natural_decay; either
    # way, at least one eligible memory was touched.
    assert result.memories_stepped >= 2
