"""
Database connection and session management.

Defaults to a local SQLite file so the app runs with no configuration at all;
set DATABASE_URL to point at Postgres (Supabase, Neon, ...) in production.

Note that Render's own free Postgres instances are removed after 30 days, so
for anything meant to outlive a month, prefer a provider whose free tier has no
expiry.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# ── URL ───────────────────────────────────────────────────────────────────────
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./recomai.db")

# Some providers still hand out the legacy "postgres://" scheme, which
# SQLAlchemy 2.x no longer recognises as a driver name.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# ── Engine ────────────────────────────────────────────────────────────────────
_connect_args = {}
_engine_kwargs = {}

if DATABASE_URL.startswith("sqlite"):
    # FastAPI serves sync endpoints from a threadpool, so a connection can be
    # touched by a different thread than the one that opened it.
    _connect_args["check_same_thread"] = False
else:
    # Free-tier Postgres (Supabase/Neon) suspends on idle and drops open
    # sockets. pool_pre_ping discards dead connections instead of surfacing
    # them to the first request after a wake-up.
    _engine_kwargs.update(pool_pre_ping=True, pool_recycle=300, pool_size=5,
                          max_overflow=2)

engine = create_engine(DATABASE_URL, connect_args=_connect_args, **_engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency — yields a session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create tables if they don't exist. Called once at startup."""
    from backend.db import models  # noqa: F401 — registers models on Base
    Base.metadata.create_all(bind=engine)
