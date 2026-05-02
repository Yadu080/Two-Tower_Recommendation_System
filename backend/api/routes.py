"""
Phase 7 — FastAPI routes.

Endpoints:
  GET  /recommend?user_id=<int>&n=<int>   → top-N recommendations
  POST /log_click                          → record a click, update seen history
  GET  /users                              → list sample users for the UI
  GET  /health                             → liveness check + cache stats
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
import random

from backend.core.recommender import engine

router = APIRouter()

# In-memory click log: user_id → set of seen movie_idxs
# In production this would be Redis or a DB. For local dev, RAM is fine.
_click_log: dict[int, set] = {}


class ClickEvent(BaseModel):
    user_id: int
    movie_idx: int
    rating: Optional[float] = None


@router.get("/recommend")
def recommend(
    user_id: int = Query(..., description="Original MovieLens userId"),
    n: int = Query(10, ge=1, le=50),
):
    """
    Main recommendation endpoint.

    1. Translate user_id → user_idx (contiguous index)
    2. Run full pipeline: UserTower → FAISS → LightGBM → top-N
    3. Return results with explainability fields
    """
    if not engine._loaded:
        raise HTTPException(503, "Engine not loaded yet")

    user_idx = engine.user_id_map.get(user_id)
    if user_idx is None:
        raise HTTPException(404, f"user_id {user_id} not found in training data")

    seen = _click_log.get(user_id, set())
    results = engine.recommend(user_idx, n_retrieve=500, n_return=n, exclude_seen=seen)

    return {
        "user_id"    : user_id,
        "user_idx"   : user_idx,
        "n_results"  : len(results),
        "cache_stats": engine.user_cache.stats(),
        "results"    : results,
    }


@router.post("/log_click")
def log_click(event: ClickEvent):
    """
    Record that a user clicked/rated an item.
    Updates in-memory seen set so future recommendations exclude it.
    Also flushes the user's cached embedding so the next request
    re-runs the UserTower (picking up any fine-tuning if applicable).
    """
    if event.user_id not in _click_log:
        _click_log[event.user_id] = set()
    _click_log[event.user_id].add(event.movie_idx)

    # invalidate cached embedding — user's preference has shifted
    if event.user_id in engine.user_cache.cache:
        del engine.user_cache.cache[event.user_id]

    return {"status": "logged", "seen_count": len(_click_log[event.user_id])}


@router.get("/users")
def list_users(n: int = Query(20, ge=1, le=100)):
    """
    Return a sample of valid user_ids for the UI's user selector.
    Picks users with the most ratings (most informative recommendations).
    """
    if not engine._loaded:
        raise HTTPException(503, "Engine not loaded yet")
    sample = random.sample(list(engine.user_id_map.keys()), min(n, len(engine.user_id_map)))
    return {"users": sorted(sample)}


@router.get("/health")
def health():
    return {
        "status"      : "ok",
        "engine_ready": engine._loaded,
        "cache"       : engine.user_cache.stats() if engine._loaded else {},
    }
