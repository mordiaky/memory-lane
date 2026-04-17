"""Live LMD dynamics — coupled decay driven by the real LMD engine.

MemoryLane's default energy update (inside `LMDBridge.apply_reaction`)
is a small hand-written rule: positive reaction boosts by 0.25, not-
recognized nudges down by 0.05. That rule is simple but it ignores
the structure of the patient's memory archive. Two memories about the
same summer couple strongly in LMD's embedding space; reinforcing one
SHOULD lift the other a little. That's what a real dynamics step
gives us.

This module exposes one service function, `tick(db, patient_id)`, that:

  1. Lifts every memory into an LMD `LivingMemory`.
  2. Runs one forward step of `LMDDynamics` — metabolism + coupling +
     narrative flow + resonance.
  3. Writes the updated energies and statuses back to the DB.

It's meant to be called periodically — once a day, or whenever the
caregiver asks MemoryLane to "take stock" of the archive. It is NOT
called implicitly on every reaction (that would amplify noise).

Guards:
  - Falls through with a no-op if LMD is not installed.
  - Skips patients with fewer than two memories (dynamics is pointless
    with a single-node system).
  - Never touches distress-flagged or DIFFICULT-toned memories; they
    stay under the caregiver's explicit control.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from . import service
from .lmd_bridge import LMDBridge
from .models import EmotionalTone


@dataclass
class DynamicsTickResult:
    """Summary of one dynamics tick."""

    patient_id: str
    memories_stepped: int
    memories_unchanged: int
    total_energy_before: float
    total_energy_after: float
    newly_vivid: int
    newly_faded: int
    used_lmd: bool
    reason: str | None = None


def _lmd_available() -> bool:
    try:
        from lmd import LMDConfig, LMDDynamics  # noqa: F401

        return True
    except Exception:
        return False


def tick(
    db: Session,
    patient_id: str,
    *,
    dt: float = 1.0,
    bridge: LMDBridge | None = None,
) -> DynamicsTickResult:
    """Advance LMD dynamics by one step over the patient's memory archive.

    Writes updated energies and statuses back to the database.
    """
    all_memories = service.list_memories(db, patient_id)
    eligible = [
        m
        for m in all_memories
        if not m.flagged_distressing and m.tone != EmotionalTone.DIFFICULT
    ]

    total_before = sum(m.energy for m in all_memories)

    if len(eligible) < 2:
        return DynamicsTickResult(
            patient_id=patient_id,
            memories_stepped=0,
            memories_unchanged=len(all_memories),
            total_energy_before=total_before,
            total_energy_after=total_before,
            newly_vivid=0,
            newly_faded=0,
            used_lmd=False,
            reason="fewer-than-two-eligible-memories",
        )

    if not _lmd_available():
        # Fall back to the bridge's natural_decay so the feature still
        # does SOMETHING useful when LMD itself isn't installed.
        active_bridge = bridge or LMDBridge(use_language_grounding=False)
        active_bridge.natural_decay(eligible, days_elapsed=dt)
        db.commit()
        total_after = sum(m.energy for m in all_memories)
        return DynamicsTickResult(
            patient_id=patient_id,
            memories_stepped=len(eligible),
            memories_unchanged=len(all_memories) - len(eligible),
            total_energy_before=total_before,
            total_energy_after=total_after,
            newly_vivid=0,
            newly_faded=0,
            used_lmd=False,
            reason="lmd-unavailable-used-natural-decay",
        )

    from lmd import LMDConfig, LMDDynamics

    active_bridge = bridge or LMDBridge(use_language_grounding=False)
    livings = active_bridge.to_living_batch(eligible)

    config = LMDConfig(content_dim=active_bridge.capabilities.embedding_dim)
    dynamics = LMDDynamics(config)

    # Step once. LMDDynamics mutates the LivingMemory objects in place.
    dynamics.step(livings, dt=dt)

    newly_vivid = 0
    newly_faded = 0

    from .models import MemoryStatus

    at_risk = {MemoryStatus.DORMANT, MemoryStatus.FADING, MemoryStatus.GHOST}

    for memory, living in zip(eligible, livings, strict=False):
        old_status = memory.status
        # Clamp the energy back into MemoryLane's expected range so the
        # status mapping behaves sensibly.
        new_energy = max(0.01, min(2.0, float(living.energy)))
        memory.energy = new_energy
        memory.status = active_bridge.energy_to_status(new_energy)
        if memory.status == MemoryStatus.VIVID and old_status != MemoryStatus.VIVID:
            newly_vivid += 1
        if memory.status in at_risk and old_status not in at_risk:
            newly_faded += 1

    db.commit()
    total_after = sum(m.energy for m in all_memories)

    return DynamicsTickResult(
        patient_id=patient_id,
        memories_stepped=len(eligible),
        memories_unchanged=len(all_memories) - len(eligible),
        total_energy_before=total_before,
        total_energy_after=total_after,
        newly_vivid=newly_vivid,
        newly_faded=newly_faded,
        used_lmd=True,
    )
