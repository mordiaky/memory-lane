"""Bridge between MemoryLane's user-facing models and LMD's engine.

This is the only module that touches LMD directly. Everything else in
MemoryLane operates on plain Memory/Patient/Session records and should
call into here when it needs:
  - a memory encoded as an embedding
  - the coupling graph between memories
  - an anchor memory recommendation
  - an updated energy/status after a reaction

Design notes:
  - LMD state is *derived*. The source of truth is the SQL database.
    Every bridge call rebuilds an LMD view from the current DB, applies
    logic, then writes changes back as scalars (energy, status).
  - Language grounding uses MiniLM when available; without it the
    bridge falls back to a hashing-based pseudo-embedding so the rest
    of MemoryLane still works for local development. This is made
    explicit by BridgeCapabilities.has_language_grounding.
  - Reactions drive energy:
      RECOGNIZED_POSITIVE -> boost
      RECOGNIZED_NEUTRAL  -> small hold
      RECOGNIZED_DISTRESS -> small boost but flag for review
      NOT_RECOGNIZED      -> no boost (natural decay wins)
      SKIPPED             -> no-op
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch

from .models import (
    EmotionalTone,
    Memory,
    MemoryStatus,
    Reaction,
    ReactionKind,
)

try:
    from lmd import (
        LivingMemory as LMDLivingMemory,
    )
    from lmd import (
        ValenceTrajectory,
        create_grounding,
    )

    _LMD_AVAILABLE = True
except Exception:  # pragma: no cover - optional at import time
    _LMD_AVAILABLE = False


# Embedding dimension for the hashing fallback. Must be fixed so memories
# remain comparable across calls.
FALLBACK_EMBEDDING_DIM = 384


@dataclass(frozen=True)
class BridgeCapabilities:
    """What this bridge instance can actually do at runtime."""

    has_lmd: bool
    has_language_grounding: bool
    embedding_dim: int


@dataclass
class AnchorSuggestion:
    """A memory recommended as an emotional anchor for the patient."""

    memory_id: str
    title: str
    reason: str
    score: float


@dataclass
class VisitSuggestion:
    """A memory recommended for surfacing in an upcoming visit."""

    memory_id: str
    title: str
    reason: str
    priority: float


class LMDBridge:
    """LMD adapter scoped to a single patient's memory set."""

    def __init__(self, use_language_grounding: bool = True):
        self._grounding = None
        if _LMD_AVAILABLE and use_language_grounding:
            try:
                self._grounding = create_grounding(encoder="minilm")
            except Exception:
                # sentence-transformers missing, offline, or model download
                # failed — fall back gracefully.
                self._grounding = None

    @property
    def capabilities(self) -> BridgeCapabilities:
        return BridgeCapabilities(
            has_lmd=_LMD_AVAILABLE,
            has_language_grounding=self._grounding is not None,
            embedding_dim=(
                self._grounding.embedding_dim
                if self._grounding is not None
                else FALLBACK_EMBEDDING_DIM
            ),
        )

    # ---- Encoding ----------------------------------------------------

    def encode(self, text: str) -> torch.Tensor:
        """Encode a text snippet into a unit-norm embedding tensor.

        Uses MiniLM when sentence-transformers is available, otherwise a
        deterministic hashing fallback so development works offline.
        """
        if self._grounding is not None:
            return self._grounding.encode(text).detach().cpu()
        return self._hash_embedding(text)

    @staticmethod
    def _hash_embedding(text: str) -> torch.Tensor:
        # Cheap deterministic pseudo-embedding. Not semantic — just stable
        # and unit-norm so the rest of the pipeline sees real tensors.
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Stretch 32 bytes into FALLBACK_EMBEDDING_DIM floats in [-1, 1].
        repeats = FALLBACK_EMBEDDING_DIM // len(digest) + 1
        raw = (digest * repeats)[:FALLBACK_EMBEDDING_DIM]
        tensor = torch.tensor([(b - 127.5) / 127.5 for b in raw], dtype=torch.float32)
        return tensor / (tensor.norm() + 1e-8)

    # ---- Lifting DB memories to LMD ---------------------------------

    def to_living(self, memory: Memory, numeric_id: int) -> LMDLivingMemory:
        """Materialize a MemoryLane Memory as an LMD LivingMemory.

        This does not persist anything. Call when you need LMD semantics
        for a short-lived computation (coupling, anchoring) and discard.
        """
        if not _LMD_AVAILABLE:
            raise RuntimeError(
                "LMD is not available in this environment. Install "
                "`living-memory-dynamics` to use living-memory operations."
            )
        content = self.encode(f"{memory.title}. {memory.description}")
        valence = ValenceTrajectory(
            points=torch.tensor(
                [memory.valence_start, memory.valence_peak, memory.valence_end],
                dtype=torch.float32,
            )
        )
        return LMDLivingMemory(
            id=numeric_id,
            content=content,
            valence=valence,
            energy=float(memory.energy),
            phase=0.0,
            label=memory.title[:80],
        )

    def to_living_batch(self, memories: Sequence[Memory]) -> List[LMDLivingMemory]:
        return [self.to_living(m, i) for i, m in enumerate(memories)]

    # ---- Reaction -> energy update ----------------------------------

    @staticmethod
    def energy_delta_for_reaction(kind: ReactionKind) -> float:
        """How much to move energy for one reaction.

        Kept small so a single session can't pin energy at ceiling;
        cumulative effect over many sessions is what makes a memory
        "stay vivid".
        """
        return {
            ReactionKind.RECOGNIZED_POSITIVE: +0.25,
            ReactionKind.RECOGNIZED_NEUTRAL: +0.05,
            ReactionKind.RECOGNIZED_DISTRESS: +0.10,
            ReactionKind.NOT_RECOGNIZED: -0.05,
            ReactionKind.SKIPPED: 0.0,
        }[kind]

    def apply_reaction(self, memory: Memory, reaction: Reaction) -> None:
        """Mutate a Memory in place based on a fresh Reaction.

        Caller owns persistence — this does not commit.
        """
        delta = self.energy_delta_for_reaction(reaction.kind)
        new_energy = max(0.01, min(2.0, memory.energy + delta))
        memory.energy = new_energy
        memory.last_surfaced_at = reaction.created_at
        memory.status = self.energy_to_status(new_energy)

    # ---- Energy -> human status -------------------------------------

    @staticmethod
    def energy_to_status(energy: float) -> MemoryStatus:
        if energy >= 1.2:
            return MemoryStatus.VIVID
        if energy >= 0.8:
            return MemoryStatus.ACTIVE
        if energy >= 0.4:
            return MemoryStatus.DORMANT
        if energy >= 0.15:
            return MemoryStatus.FADING
        return MemoryStatus.GHOST

    # ---- Recommendations --------------------------------------------

    def suggest_anchor(
        self,
        memories: Sequence[Memory],
        top_k: int = 3,
    ) -> List[AnchorSuggestion]:
        """Return memories best suited as emotional anchors.

        Preference order:
          1. Skip any memory flagged distressing.
          2. Skip memories with EmotionalTone.DIFFICULT — anchors are
             for comfort; a "difficult" memory has no business being
             recommended as a calming tool even if it's the only
             option available.
          3. Prefer positive emotional tone and valence arcs that rise
             and resolve (like ValenceTrajectory.climax or redemption).
          4. Prefer higher energy (still vivid/active).
          5. Prefer stronger coupling to other positive memories — a
             memory that activates neighbors when surfaced is more
             valuable than an isolated one.

        Falls back to a coupling-free ranking if LMD is not installed.
        """
        safe = [
            m for m in memories
            if not m.flagged_distressing and m.tone != EmotionalTone.DIFFICULT
        ]
        if not safe:
            return []

        scored: list[tuple[Memory, float, str]] = []

        # Shared per-memory "intrinsic" score.
        intrinsic: dict[str, float] = {}
        for m in safe:
            tone_bonus = {
                EmotionalTone.JOYFUL: 1.0,
                EmotionalTone.BITTERSWEET: 0.6,
                EmotionalTone.NEUTRAL: 0.3,
                EmotionalTone.DIFFICULT: 0.0,
            }[m.tone]
            arc_bonus = max(0.0, min(1.0, (m.valence_peak - m.valence_start) * 0.5 + 0.5))
            energy_bonus = min(1.0, m.energy / 1.5)
            intrinsic[m.id] = 0.45 * tone_bonus + 0.25 * arc_bonus + 0.30 * energy_bonus

        coupling_bonus: dict[str, float] = {m.id: 0.0 for m in safe}
        reason_suffix = ""
        if _LMD_AVAILABLE and len(safe) > 1:
            try:
                from lmd import CouplingField, LMDConfig

                config = LMDConfig(content_dim=self.capabilities.embedding_dim)
                coupling = CouplingField(config)
                livings = self.to_living_batch(safe)
                for i, mem in enumerate(safe):
                    total = 0.0
                    for j, other in enumerate(safe):
                        if i == j:
                            continue
                        if other.tone == EmotionalTone.DIFFICULT:
                            continue
                        total += coupling.get_coupling(livings[i], livings[j])
                    coupling_bonus[mem.id] = total / max(1, len(safe) - 1)
                reason_suffix = " with strong positive coupling"
            except Exception:
                # Any LMD failure degrades gracefully to intrinsic-only.
                reason_suffix = ""

        for m in safe:
            score = intrinsic[m.id] + 0.5 * coupling_bonus[m.id]
            reason = f"tone={m.tone.value}, status={m.status.value}{reason_suffix}"
            scored.append((m, score, reason))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            AnchorSuggestion(
                memory_id=mem.id,
                title=mem.title,
                reason=reason,
                score=round(score, 3),
            )
            for mem, score, reason in scored[:top_k]
        ]

    def suggest_visit_memories(
        self,
        memories: Sequence[Memory],
        top_k: int = 5,
    ) -> List[VisitSuggestion]:
        """Pick memories worth surfacing in an upcoming visit.

        Heuristic:
          - Skip flagged distressing memories.
          - Prefer memories whose status has slipped (DORMANT/FADING)
            but whose tone is still joyful or bittersweet — we want to
            reinforce them before they become GHOST.
          - Add a handful of still-VIVID joyful memories as safe ground.
        """
        candidates = [m for m in memories if not m.flagged_distressing]
        scored: list[VisitSuggestion] = []

        status_priority = {
            MemoryStatus.FADING: 1.0,
            MemoryStatus.DORMANT: 0.75,
            MemoryStatus.ACTIVE: 0.4,
            MemoryStatus.VIVID: 0.25,
            MemoryStatus.GHOST: 0.05,
        }
        tone_priority = {
            EmotionalTone.JOYFUL: 1.0,
            EmotionalTone.BITTERSWEET: 0.7,
            EmotionalTone.NEUTRAL: 0.4,
            EmotionalTone.DIFFICULT: 0.0,
        }

        for m in candidates:
            priority = status_priority[m.status] * 0.6 + tone_priority[m.tone] * 0.4
            reason = (
                f"status={m.status.value}, tone={m.tone.value}"
                f" — {'reinforce before it fades' if m.status in (MemoryStatus.FADING, MemoryStatus.DORMANT) else 'safe, familiar ground'}"
            )
            scored.append(
                VisitSuggestion(
                    memory_id=m.id,
                    title=m.title,
                    reason=reason,
                    priority=round(priority, 3),
                )
            )

        scored.sort(key=lambda s: s.priority, reverse=True)
        return scored[:top_k]

    def natural_decay(self, memories: Iterable[Memory], days_elapsed: float) -> None:
        """Apply a small natural-decay step to a set of memories.

        Intended to be called when a patient hasn't been visited recently
        — mirrors LMD's metabolism step without needing the full system.
        Caller owns persistence.
        """
        if days_elapsed <= 0:
            return
        # Half-life of ~30 days for an unvisited memory: 0.977^30 ≈ 0.5
        decay_per_day = 0.977
        factor = math.pow(decay_per_day, days_elapsed)
        for m in memories:
            new_energy = max(0.01, m.energy * factor)
            m.energy = new_energy
            m.status = self.energy_to_status(new_energy)
