import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router
from backend.core.recommender import engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load all models once at startup — not per request
    engine.load()
    yield
    # shutdown: nothing to clean up for local dev


app = FastAPI(title="RecomAI API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # local dev — lock down to your domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
