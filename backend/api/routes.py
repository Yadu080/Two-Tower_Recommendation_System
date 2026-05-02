from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List
import random

from backend.core.recommender import engine

router = APIRouter()

_click_log: dict[int, set] = {}


class ClickEvent(BaseModel):
    user_id: int
    movie_idx: int
    rating: Optional[float] = None


class RegisterUserRequest(BaseModel):
    name: str
    genres: List[str]


@router.get("/genres")
def list_genres():
    if not engine._loaded:
        raise HTTPException(503, "Engine not loaded yet")
    return {"genres": engine.genre_names}


@router.post("/users/register")
def register_user(req: RegisterUserRequest):
    if not engine._loaded:
        raise HTTPException(503, "Engine not loaded yet")
    if not req.name.strip():
        raise HTTPException(400, "Name cannot be empty")
    valid = [g for g in req.genres if g in engine.genre_vocab]
    if not valid:
        raise HTTPException(400, "At least one valid genre required")
    user_id = engine.register_new_user(req.name.strip(), valid)
    return {"user_id": user_id, "name": req.name.strip(), "genres": valid}


@router.get("/recommend")
def recommend(
    user_id: int = Query(..., description="MovieLens userId or new user id (>= 1000000)"),
    n: int = Query(10, ge=1, le=50),
):
    if not engine._loaded:
        raise HTTPException(503, "Engine not loaded yet")

    seen = _click_log.get(user_id, set())

    # Existing MovieLens user
    user_idx = engine.user_id_map.get(user_id)
    if user_idx is not None:
        results = engine.recommend(user_idx, n_retrieve=500, n_return=n, exclude_seen=seen)
        display_name = f"User {user_id}"
        return {
            "user_id"    : user_id,
            "user_idx"   : user_idx,
            "display_name": display_name,
            "n_results"  : len(results),
            "cache_stats": engine.user_cache.stats(),
            "results"    : results,
        }

    # Cold-start new user
    profile = engine.new_users.get(user_id)
    if profile is None:
        raise HTTPException(404, f"user_id {user_id} not found")

    user_emb = engine.user_cache.get(user_id)
    if user_emb is None:
        user_emb = engine._genre_embedding(profile["genres"])
        engine.user_cache.put(user_id, user_emb)

    results = engine.recommend(
        user_idx=user_id,
        n_retrieve=500,
        n_return=n,
        exclude_seen=seen,
        user_emb_override=user_emb,
    )
    return {
        "user_id"     : user_id,
        "user_idx"    : user_id,
        "display_name": profile["name"],
        "n_results"   : len(results),
        "cache_stats" : engine.user_cache.stats(),
        "results"     : results,
    }


@router.post("/log_click")
def log_click(event: ClickEvent):
    if event.user_id not in _click_log:
        _click_log[event.user_id] = set()
    _click_log[event.user_id].add(event.movie_idx)
    # invalidate cached embedding so next request recomputes
    if event.user_id in engine.user_cache.cache:
        del engine.user_cache.cache[event.user_id]
    return {"status": "logged", "seen_count": len(_click_log[event.user_id])}


@router.get("/users")
def list_users(n: int = Query(20, ge=1, le=100)):
    if not engine._loaded:
        raise HTTPException(503, "Engine not loaded yet")
    sample = random.sample(list(engine.user_id_map.keys()), min(n, len(engine.user_id_map)))
    existing = [{"id": uid, "name": f"User {uid}", "is_new": False} for uid in sorted(sample)]
    new = [
        {"id": uid, "name": profile["name"], "genres": profile["genres"], "is_new": True}
        for uid, profile in engine.new_users.items()
    ]
    return {"users": existing + new}


@router.get("/health")
def health():
    return {
        "status"      : "ok",
        "engine_ready": engine._loaded,
        "cache"       : engine.user_cache.stats() if engine._loaded else {},
    }
