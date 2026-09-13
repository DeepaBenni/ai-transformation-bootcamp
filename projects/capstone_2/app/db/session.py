"""Database engine and session factory.

Boundary: this module owns the single Engine for the process. Every other module
gets a Session from db_session() or asks ping() whether the database is up; none
of them create their own engine.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings

_engine: Engine = create_engine(
    get_settings().mysql_url,
    pool_pre_ping=True,  # silently reconnects after MySQL drops an idle connection
    pool_recycle=1800,  # recycle connections older than 30 min, before MySQL times them out
    future=True,
)

SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False, class_=Session)


@contextmanager
def db_session() -> Iterator[Session]:
    """Yield a session that commits on success and rolls back on any exception."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        # Re-raised after rollback: the caller still sees the failure, but the
        # transaction is guaranteed not to leak a half-written state.
        session.rollback()
        raise
    finally:
        session.close()


def ping() -> bool:
    """Return True if the database answers. Used by the health endpoint."""
    try:
        with _engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:  # health probe: any failure at all means "not reachable"
        return False
