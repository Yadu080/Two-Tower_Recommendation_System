import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router
from backend.api.auth_routes import router as auth_router
from backend.api.list_routes import router as list_router
from backend.core.recommender import engine
from backend.db.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    # create tables before the first request can touch them
    init_db()
    # load all models once at startup — not per request
    engine.load()
    yield
    # shutdown: nothing to clean up for local dev


app = FastAPI(title="RecomAI API", version="1.1.0", lifespan=lifespan)

# Allowed origins. Production domains come from ALLOWED_ORIGINS (comma
# separated) so a new deploy URL doesn't need a code change; localhost stays
# permitted so `npm run dev` works against a local backend.
_DEFAULT_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://two-tower-recommendation-system.vercel.app",
]
_extra = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_DEFAULT_ORIGINS + _extra,
    # Two patterns, both needed:
    #  * Vercel gives every preview build its own generated subdomain, so they
    #    can't be enumerated ahead of time.
    #  * Vite picks the next free port when 5173 is taken, so pinning the dev
    #    origin to one port silently breaks auth for anyone running a second
    #    instance — matching any localhost port avoids that trap.
    allow_origin_regex=r"https://.*\.vercel\.app|http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(auth_router)
app.include_router(list_router)


@app.get("/")
def root():
    """Render's health checker polls / — return something instead of a 404."""
    return {"service": "RecomAI API", "docs": "/docs", "health": "/health"}
