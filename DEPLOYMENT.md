# Deployment Guide

Backend (FastAPI) deploys to **Render**, frontend (Vite + React) deploys to
**Vercel**. Deploy the backend first — the frontend needs its URL at build time.

---

## 0. Clone the repo

```bash
git clone https://github.com/Yadu080/Two-Tower_Recommendation_System.git
cd Two-Tower_Recommendation_System
```

To work on the fix branch:

```bash
git fetch origin claude/project-review-publish-69sa6s
git checkout claude/project-review-publish-69sa6s
```

### Run it locally first

```bash
# Terminal 1 — backend on :8000
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — frontend on :5173
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The frontend defaults to `http://localhost:8000`
for the API, so no env var is needed for local development.

---

## 1. Backend → Render

`render.yaml` at the repo root already declares the service, so Render picks
up the config automatically.

1. Push your branch to GitHub.
2. Go to [dashboard.render.com](https://dashboard.render.com) → **New** →
   **Blueprint**.
3. Connect the GitHub repo and select the branch. Render reads `render.yaml`
   and proposes a web service named `recomai-backend`.
4. Set the one secret it can't infer — in the service's **Environment** tab add:

   | Key            | Value                    |
   |----------------|--------------------------|
   | `TMDB_API_KEY` | your TMDB API key        |

   Without it the app still runs; posters fall back to generated colour tiles.
5. Click **Apply** / **Create**. First build takes a few minutes (it installs
   torch and friends).
6. Note the resulting URL, e.g. `https://recomai-backend.onrender.com`.
   Verify it with:

   ```bash
   curl https://recomai-backend.onrender.com/health
   ```

**Free-tier note:** Render spins free services down after ~15 minutes idle, so
the first request after a pause takes 30–60s while it wakes up. The frontend
shows its loading skeleton during this, but if you're demoing it live, hit the
`/health` URL once beforehand to warm it up.

---

## 2. Frontend → Vercel

`frontend/vercel.json` already handles SPA routing (all paths rewrite to
`index.html`).

1. Go to [vercel.com/new](https://vercel.com/new) and import the same repo.
2. Set **Root Directory** to `frontend`. Vercel then auto-detects Vite and
   fills in `npm run build` / `dist` correctly.
3. Under **Environment Variables**, add:

   | Key            | Value                                     |
   |----------------|-------------------------------------------|
   | `VITE_API_URL` | `https://recomai-backend.onrender.com`    |

   Use your actual Render URL, with no trailing slash. This is required —
   without it the deployed site calls `localhost:8000` and every request fails.
4. **Deploy.**

> `VITE_*` variables are baked in at build time, not read at runtime. If you
> change `VITE_API_URL` later you must redeploy for it to take effect.

---

## 3. Allow the Vercel origin through CORS

The backend needs to accept requests from the Vercel domain. Check the
`CORSMiddleware` block in `backend/main.py` — if `allow_origins` is not `["*"]`,
add your Vercel URL to it, commit, and let Render redeploy:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://your-app.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Frontend loads, recommendations never arrive | `VITE_API_URL` unset or wrong | Set it in Vercel → Settings → Environment Variables, then **redeploy** |
| Browser console shows a CORS error | Vercel origin not allowed | Add the domain to `allow_origins` in `backend/main.py` |
| First request hangs ~45s | Render free tier cold start | Expected; hit `/health` to warm it, or upgrade the plan |
| Render build fails on `pip install` | Heavy ML deps exceed free build limits | Check the build log; pin lighter versions or use a paid instance |
| Posters are plain colour tiles | `TMDB_API_KEY` missing | Add it in Render → Environment |
| 404 on refresh of a sub-path | SPA rewrite missing | Confirm Root Directory is `frontend` so `vercel.json` is picked up |
