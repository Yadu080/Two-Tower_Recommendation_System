"""
"My List" endpoints — the titles a user has opened, newest first.

  GET    /my-list             most recent saved titles
  POST   /my-list             save a title (idempotent per movie)
  DELETE /my-list/{movie_idx} remove one
  DELETE /my-list             clear all

Why this exists
---------------
Opening a title logs an implicit signal, and the engine then excludes seen
items from later recommendations — so a movie disappears from the feed the
moment you click it. My List is the durable record of those picks, which is
why each row stores its own display fields rather than expecting the
recommender to surface that movie again.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import delete
from sqlalchemy.orm import Session

from backend.core.auth import get_current_user
from backend.db.database import get_db
from backend.db.models import ListItem, User

router = APIRouter(prefix="/my-list", tags=["my-list"])

# How many titles a user's list holds. Older entries are trimmed past this so
# the table can't grow without bound on a free-tier database.
LIST_CAP = 50
DEFAULT_LIMIT = 15


class SaveItemRequest(BaseModel):
    movie_idx : int
    title      : str = Field(max_length=300)
    genres     : str = Field(default="", max_length=300)
    poster_url : Optional[str] = Field(default=None, max_length=500)
    avg_rating : Optional[float] = None


class ListItemOut(BaseModel):
    movie_idx  : int
    title      : str
    genres     : str
    poster_url : Optional[str]
    avg_rating : Optional[float]
    saved_at   : Optional[str]


def _serialise(row: ListItem) -> ListItemOut:
    rating = None
    if row.avg_rating is not None:
        try:
            rating = float(row.avg_rating)
        except (TypeError, ValueError):
            rating = None
    return ListItemOut(
        movie_idx  = row.movie_idx,
        title      = row.title,
        genres     = row.genres or "",
        poster_url = row.poster_url,
        avg_rating = rating,
        saved_at   = row.saved_at.isoformat() if row.saved_at else None,
    )


@router.get("", response_model=dict)
def get_list(limit: int = Query(DEFAULT_LIMIT, ge=1, le=LIST_CAP),
             user: User = Depends(get_current_user),
             db: Session = Depends(get_db)):
    rows = (db.query(ListItem)
              .filter(ListItem.user_id == user.id)
              .order_by(ListItem.saved_at.desc(), ListItem.id.desc())
              .limit(limit)
              .all())
    return {"items": [_serialise(r).model_dump() for r in rows],
            "count": len(rows)}


@router.post("", response_model=dict, status_code=status.HTTP_201_CREATED)
def save_item(req: SaveItemRequest,
              user: User = Depends(get_current_user),
              db: Session = Depends(get_db)):
    existing = (db.query(ListItem)
                  .filter(ListItem.user_id == user.id,
                          ListItem.movie_idx == req.movie_idx)
                  .first())

    if existing is not None:
        # Re-opening a title moves it back to the top rather than duplicating.
        from datetime import datetime, timezone
        existing.saved_at   = datetime.now(timezone.utc)
        existing.title      = req.title
        existing.poster_url = req.poster_url
        db.commit()
        return {"status": "updated", "movie_idx": req.movie_idx}

    db.add(ListItem(
        user_id    = user.id,
        movie_idx  = req.movie_idx,
        title      = req.title,
        genres     = req.genres or "",
        poster_url = req.poster_url,
        avg_rating = None if req.avg_rating is None else str(req.avg_rating),
    ))
    db.commit()

    # trim anything past the cap, oldest first
    total = db.query(ListItem).filter(ListItem.user_id == user.id).count()
    if total > LIST_CAP:
        stale = (db.query(ListItem.id)
                   .filter(ListItem.user_id == user.id)
                   .order_by(ListItem.saved_at.asc(), ListItem.id.asc())
                   .limit(total - LIST_CAP)
                   .all())
        db.execute(delete(ListItem).where(ListItem.id.in_([s.id for s in stale])))
        db.commit()

    return {"status": "saved", "movie_idx": req.movie_idx}


@router.delete("/{movie_idx}", response_model=dict)
def remove_item(movie_idx: int,
                user: User = Depends(get_current_user),
                db: Session = Depends(get_db)):
    db.execute(delete(ListItem).where(ListItem.user_id == user.id,
                                      ListItem.movie_idx == movie_idx))
    db.commit()
    return {"status": "removed", "movie_idx": movie_idx}


@router.delete("", response_model=dict)
def clear_list(user: User = Depends(get_current_user),
               db: Session = Depends(get_db)):
    db.execute(delete(ListItem).where(ListItem.user_id == user.id))
    db.commit()
    return {"status": "cleared"}
