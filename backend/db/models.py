"""
SQLAlchemy models.

Two tables:
  users      — credentials plus the genre picks that define their taste profile
  list_items — the "My List" entries, one row per saved title

Why users carry a rec_user_id
-----------------------------
The recommendation engine keeps cold-start profiles in an in-memory dict, so
every restart wipes them. Persisting the engine's user id here lets a returning
user's profile be rehydrated into the engine on demand, which is what makes
their selections survive a redeploy.
"""

from datetime import datetime, timezone

from sqlalchemy import (Column, Integer, String, DateTime, ForeignKey, JSON,
                        UniqueConstraint, Index)
from sqlalchemy.orm import relationship

from backend.db.database import Base


def _utcnow():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id       = Column(Integer, primary_key=True, index=True)
    username = Column(String(40), unique=True, nullable=False, index=True)

    # bcrypt output — never the password itself
    password_hash = Column(String(255), nullable=False)

    display_name = Column(String(60), nullable=False)

    # genre picks, e.g. ["Action", "Sci-Fi"]
    genres = Column(JSON, nullable=False, default=list)

    # id this user is known by inside the recommendation engine
    rec_user_id = Column(Integer, nullable=True, index=True)

    created_at    = Column(DateTime(timezone=True), default=_utcnow)
    last_login_at = Column(DateTime(timezone=True), default=_utcnow)

    list_items = relationship("ListItem", back_populates="user",
                              cascade="all, delete-orphan")


class ListItem(Base):
    """
    A title the user opened, most recent first.

    Kept deliberately denormalised: the engine excludes already-seen items from
    future recommendations, so a saved title may never appear in a response
    again. Storing the display fields means My List can render without needing
    the recommender to surface that movie a second time.
    """
    __tablename__ = "list_items"
    __table_args__ = (
        UniqueConstraint("user_id", "movie_idx", name="uq_user_movie"),
        Index("ix_list_items_user_saved", "user_id", "saved_at"),
    )

    id        = Column(Integer, primary_key=True, index=True)
    user_id   = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"),
                       nullable=False, index=True)
    movie_idx = Column(Integer, nullable=False)

    title      = Column(String(300), nullable=False)
    genres     = Column(String(300), default="")
    poster_url = Column(String(500), nullable=True)
    avg_rating = Column(String(16), nullable=True)   # string keeps None/format simple

    saved_at = Column(DateTime(timezone=True), default=_utcnow, index=True)

    user = relationship("User", back_populates="list_items")
