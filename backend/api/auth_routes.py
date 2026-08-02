"""
Authentication endpoints.

  POST /auth/register   create an account
  POST /auth/login      exchange credentials for a bearer token
  GET  /auth/me         current account + saved genre profile
  PUT  /auth/genres     replace the saved genre selection
"""

import re
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.core.auth import (create_access_token, get_current_user,
                               hash_password, validate_password,
                               verify_password)
from backend.core.recommender import engine
from backend.db.database import get_db
from backend.db.models import User

router = APIRouter(prefix="/auth", tags=["auth"])

USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,40}$")

# ── login throttling ──────────────────────────────────────────────────────────
# In-process and therefore per-instance — adequate for a single free-tier
# container, but a shared store would be needed if this ever scaled out.
_ATTEMPT_WINDOW_S = 300
_MAX_ATTEMPTS = 8
_attempts: dict = defaultdict(deque)


def _throttle(key: str) -> None:
    now = time.monotonic()
    bucket = _attempts[key]
    while bucket and now - bucket[0] > _ATTEMPT_WINDOW_S:
        bucket.popleft()
    if len(bucket) >= _MAX_ATTEMPTS:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "Too many login attempts. Try again in a few minutes.",
        )
    bucket.append(now)


def _clear_throttle(key: str) -> None:
    _attempts.pop(key, None)


# ── schemas ───────────────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=40)
    password: str = Field(min_length=8, max_length=200)
    display_name: Optional[str] = Field(default=None, max_length=60)
    genres: List[str] = Field(default_factory=list)


class LoginRequest(BaseModel):
    username: str = Field(max_length=40)
    password: str = Field(max_length=200)


class GenresRequest(BaseModel):
    genres: List[str]


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


def _valid_genres(genres: List[str]) -> List[str]:
    """Keep only genres the engine actually knows, preserving order."""
    if not engine._loaded:
        return list(genres)
    seen, out = set(), []
    for g in genres:
        if g in engine.genre_vocab and g not in seen:
            seen.add(g)
            out.append(g)
    return out


def _profile(user: User) -> dict:
    return {
        "id"          : user.id,
        "username"    : user.username,
        "name"        : user.display_name,
        "genres"      : user.genres or [],
        "rec_user_id" : user.rec_user_id,
        "is_new"      : True,
    }


def _sync_engine_profile(user: User) -> None:
    """Make sure the engine knows about this account's taste profile."""
    if user.rec_user_id is not None:
        engine.ensure_user(user.rec_user_id, user.display_name, user.genres or [])


# ── endpoints ─────────────────────────────────────────────────────────────────
@router.post("/register", response_model=AuthResponse,
             status_code=status.HTTP_201_CREATED)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    username = req.username.strip()
    if not USERNAME_RE.match(username):
        raise HTTPException(
            400, "Username must be 3–40 characters: letters, digits, . _ - only"
        )
    validate_password(req.password)

    genres = _valid_genres(req.genres)
    display_name = (req.display_name or username).strip()[:60]

    user = User(
        username      = username,
        password_hash = hash_password(req.password),
        display_name  = display_name,
        genres        = genres,
    )

    # A UNIQUE violation is the authoritative duplicate check — a pre-SELECT
    # would still race two concurrent registrations of the same name.
    db.add(user)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "Username already taken")

    if genres and engine._loaded:
        user.rec_user_id = engine.register_new_user(display_name, genres)

    db.commit()
    db.refresh(user)

    return AuthResponse(access_token=create_access_token(user.id),
                        user=_profile(user))


@router.post("/login", response_model=AuthResponse)
def login(req: LoginRequest, request: Request, db: Session = Depends(get_db)):
    client = request.client.host if request.client else "unknown"
    _throttle(f"{client}:{req.username.strip().lower()}")

    user = db.query(User).filter(User.username == req.username.strip()).first()

    # Same response whether the account is missing or the password is wrong, so
    # the endpoint can't be used to enumerate valid usernames.
    if user is None or not verify_password(req.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED,
                            "Incorrect username or password")

    _clear_throttle(f"{client}:{req.username.strip().lower()}")

    user.last_login_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(user)

    _sync_engine_profile(user)

    return AuthResponse(access_token=create_access_token(user.id),
                        user=_profile(user))


@router.get("/me")
def me(user: User = Depends(get_current_user)):
    _sync_engine_profile(user)
    return _profile(user)


@router.put("/genres")
def set_genres(req: GenresRequest,
               user: User = Depends(get_current_user),
               db: Session = Depends(get_db)):
    genres = _valid_genres(req.genres)
    if not genres:
        raise HTTPException(400, "At least one valid genre required")

    user.genres = genres

    if not engine._loaded:
        raise HTTPException(503, "Engine not loaded yet")

    if user.rec_user_id is None:
        user.rec_user_id = engine.register_new_user(user.display_name, genres)
    else:
        engine.update_user_genres(user.rec_user_id, user.display_name, genres)

    db.commit()
    db.refresh(user)
    return _profile(user)
