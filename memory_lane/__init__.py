"""MemoryLane — a caregiver-facing life-story and reminiscence companion.

This package is **not a medical device**. It is a wellness tool intended to
help families organize and share memories with a loved one living with
dementia. It does not diagnose, treat, cure, or prevent any condition.
See README.md for the full non-medical-device disclosure.
"""

__version__ = "0.1.0"

from .models import EmotionalTone, Memory, MemoryStatus, Patient, Reaction, Session

__all__ = [
    "Patient",
    "Memory",
    "Session",
    "Reaction",
    "EmotionalTone",
    "MemoryStatus",
    "__version__",
]
