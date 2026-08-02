"""
Authentication: password hashing, JWT issue/verify, and the current-user
dependency.

Design notes
------------
* Passwords are hashed with bcrypt. Plaintext is never stored or logged, and
  the hash is never returned by any endpoint.
* Tokens are bearer JWTs sent in the Authorization header rather than cookies.
  The frontend and API live on different origins (Vercel and Render), and
  cross-site cookies bring SameSite/third-party-cookie problems that a header
  simply doesn't have.
* JWT_SECRET must be set in production. A missing secret falls back to a
  per-process random value, which is safe-by-default (tokens simply stop
  validating after a restart) but means everyone is logged out on redeploy.
"""

import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from backend.db.database import get_db
from backend.db.models import User

# ── config ────────────────────────────────────────────────────────────────────
_ENV_SECRET = os.environ.get("JWT_SECRET", "").strip()
JWT_SECRET = _ENV_SECRET or secrets.token_urlsafe(48)
JWT_ALGORITHM = "HS256"
TOKEN_TTL_HOURS = int(os.environ.get("JWT_TTL_HOURS", "168"))  # 7 days

if not _ENV_SECRET:
    print(
        "  ⚠ JWT_SECRET not set — using a random per-process secret. "
        "Sessions will not survive a restart. Set JWT_SECRET in production.",
        flush=True,
    )

# bcrypt caps input at 72 bytes and silently ignores the remainder, so a long
# password would be equivalent to its first 72 bytes. Reject instead.
MAX_PASSWORD_BYTES = 72
MIN_PASSWORD_LEN = 8

_bearer = HTTPBearer(auto_error=False)


# ── passwords ─────────────────────────────────────────────────────────────────
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"),
                              password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        # malformed hash in the row — treat as a failed login, never a 500
        return False


def validate_password(password: str) -> None:
    """Raise HTTPException if the password is unusable. Called on register."""
    if len(password) < MIN_PASSWORD_LEN:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Password must be at least {MIN_PASSWORD_LEN} characters",
        )
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Password must be at most {MAX_PASSWORD_BYTES} bytes",
        )


# ── tokens ────────────────────────────────────────────────────────────────────
def create_access_token(user_id: int) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + timedelta(hours=TOKEN_TTL_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _decode(token: str) -> Optional[int]:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return int(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        return None


# ── dependencies ──────────────────────────────────────────────────────────────
def get_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User:
    """Require a valid token. 401 otherwise."""
    unauthorized = HTTPException(
        status.HTTP_401_UNAUTHORIZED,
        "Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if creds is None or not creds.credentials:
        raise unauthorized

    user_id = _decode(creds.credentials)
    if user_id is None:
        raise unauthorized

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise unauthorized
    return user


def get_optional_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """Resolve a user when a token is present, but never reject the request."""
    if creds is None or not creds.credentials:
        return None
    user_id = _decode(creds.credentials)
    if user_id is None:
        return None
    return db.query(User).filter(User.id == user_id).first()
