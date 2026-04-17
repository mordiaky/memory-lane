"""Shared pytest fixtures.

Every test file that needs a DB uses the `db` fixture, which hands out
a per-test SQLAlchemy session backed by a fresh in-memory SQLite. This
keeps tests fully isolated — no file on disk, no state leakage across
tests, no accidental cross-test interactions via SQLite's page cache.
"""

from __future__ import annotations

import os
from typing import Iterator

import pytest
from sqlalchemy.orm import Session

# Force in-memory DB before any memory_lane import triggers storage.
os.environ.setdefault("MEMORY_LANE_DB", "sqlite:///:memory:")

from memory_lane.storage import get_engine, init_db, session_factory  # noqa: E402


@pytest.fixture()
def db() -> Iterator[Session]:
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    factory = session_factory(engine)
    with factory() as session:
        yield session
