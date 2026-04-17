"""LMD bridge unit tests.

The bridge is the one place MemoryLane touches LMD directly. We want to
be sure:
  - It works with *and without* sentence-transformers available.
  - Energy-to-status mapping is correct at every boundary.
  - Reaction deltas don't allow energy to drift out of [0.01, 2.0].
  - suggest_anchor skips distressing memories.
"""

from __future__ import annotations

import pytest

from memory_lane import service
from memory_lane.lmd_bridge import LMDBridge
from memory_lane.models import EmotionalTone, MemoryStatus, ReactionKind


def test_capabilities_without_language_grounding() -> None:
    bridge = LMDBridge(use_language_grounding=False)
    caps = bridge.capabilities
    assert caps.has_language_grounding is False
    assert caps.embedding_dim > 0


def test_hash_fallback_is_deterministic_and_unit_norm() -> None:
    bridge = LMDBridge(use_language_grounding=False)
    a = bridge.encode("fire-breathing dragon")
    b = bridge.encode("fire-breathing dragon")
    assert (a == b).all()
    assert abs(a.norm().item() - 1.0) < 1e-6


@pytest.mark.parametrize(
    "energy,expected",
    [
        (0.0, MemoryStatus.GHOST),
        (0.1, MemoryStatus.GHOST),
        (0.2, MemoryStatus.FADING),
        (0.39, MemoryStatus.FADING),
        (0.4, MemoryStatus.DORMANT),
        (0.79, MemoryStatus.DORMANT),
        (0.8, MemoryStatus.ACTIVE),
        (1.19, MemoryStatus.ACTIVE),
        (1.2, MemoryStatus.VIVID),
        (2.0, MemoryStatus.VIVID),
    ],
)
def test_energy_to_status_boundaries(energy: float, expected: MemoryStatus) -> None:
    assert LMDBridge.energy_to_status(energy) == expected


def test_apply_reaction_clamps_energy(db) -> None:
    bridge = LMDBridge(use_language_grounding=False)
    patient = service.create_patient(db, "Eleanor")
    memory = service.add_memory(
        db,
        patient_id=patient.id,
        title="Coffee mug",
        description="Her favorite mug.",
        tone=EmotionalTone.JOYFUL,
    )
    session = service.start_session(db, patient.id)

    # Flood with positive reactions and assert we never exceed the 2.0 cap.
    for _ in range(50):
        service.log_reaction(
            db,
            session_id=session.id,
            memory_id=memory.id,
            kind=ReactionKind.RECOGNIZED_POSITIVE,
            bridge=bridge,
        )
    db.refresh(memory)
    assert memory.energy <= 2.0

    # Flood with NOT_RECOGNIZED and assert we don't go below 0.01.
    for _ in range(200):
        service.log_reaction(
            db,
            session_id=session.id,
            memory_id=memory.id,
            kind=ReactionKind.NOT_RECOGNIZED,
            bridge=bridge,
        )
    db.refresh(memory)
    assert memory.energy >= 0.01


def test_natural_decay_reduces_energy(db) -> None:
    bridge = LMDBridge(use_language_grounding=False)
    patient = service.create_patient(db, "Eleanor")
    memory = service.add_memory(
        db,
        patient_id=patient.id,
        title="Decay test",
        description="Not visited in a month.",
    )
    initial = memory.energy
    bridge.natural_decay([memory], days_elapsed=30.0)
    db.commit()
    assert memory.energy < initial
    # ~30-day half-life means energy should be roughly halved.
    assert 0.3 < memory.energy < 0.7
