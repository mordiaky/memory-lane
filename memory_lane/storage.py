"""SQLAlchemy session / engine wiring.

The default storage is a local SQLite file under ~/.memory-lane/data.db.
Override with the MEMORY_LANE_DB env var for testing or alternate
deployments (set to "sqlite:///:memory:" for in-memory).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base


def _default_db_url() -> str:
    home = Path.home() / ".memory-lane"
    home.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{home / 'data.db'}"


def get_engine(url: str | None = None) -> Engine:
    """Build a SQLAlchemy engine. `url` wins over the env var wins over default."""
    db_url = url or os.environ.get("MEMORY_LANE_DB") or _default_db_url()
    engine = create_engine(db_url, future=True)
    # SQLite-specific: enforce FK constraints so cascade deletes work.
    if db_url.startswith("sqlite"):
        from sqlalchemy import event

        @event.listens_for(engine, "connect")
        def _enable_fk(dbapi_connection, _connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def init_db(engine: Engine) -> None:
    """Create tables if they don't exist. Safe to call repeatedly."""
    Base.metadata.create_all(engine)


def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def iter_session(factory: sessionmaker[Session]) -> Iterator[Session]:
    """FastAPI-style dependency generator."""
    with factory() as session:
        yield session
