# 🎬 RECOMAI — Two-Tower Movie Recommendation System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react&logoColor=black"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=flat-square&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Tailwind_CSS-3-06B6D4?style=flat-square&logo=tailwindcss&logoColor=white"/>
  <img src="https://img.shields.io/badge/Dataset-MovieLens_25M-FF6B6B?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-MIT-22C55E?style=flat-square"/>
</p>

<p align="center">
  A full-stack movie recommendation engine built end-to-end — from raw data to a Netflix-style web app.<br/>
  Neural retrieval → learning-to-rank → live personalisation, all in one deployable project.
</p>

---

## 📸 Screenshots

<!-- Drop your screenshots into docs/screenshots/ and update the paths below -->

| Landing | Genre Picker | Recommendations |
|:-------:|:------------:|:---------------:|
| ![Landing page](docs/screenshots/landing.png) | ![Genre picker](docs/screenshots/genres.png) | ![Recommendations](docs/screenshots/recs.png) |

<details>
<summary>How to add screenshots</summary>

1. Create the folder: `mkdir -p docs/screenshots`
2. Take screenshots of your running app
3. Save them as `landing.png`, `genres.png`, `recs.png` inside `docs/screenshots/`
4. Push to GitHub — the table above renders automatically

</details>

---

## ✨ Features

- **Netflix-style UI** — onboarding-first flow with a full-screen landing, genre picker, and personalised recommendation grid
- **New-user onboarding** — visitors enter their name and select genre preferences; a cold-start embedding is computed on the fly from matching item vectors — no login or password required
- **Real movie posters** — TMDb API integration fetches poster images for the entire catalogue
- **Two-Tower neural retrieval** — UserTower and ItemTower trained with InfoNCE loss on 25 M ratings
- **GBM re-ranker** — gradient-boosted re-scoring on top of retrieval candidates (Val AUC = 0.98)
- **Live click logging** — clicked items are excluded from subsequent recommendations
- **Explainability** — every card shows the top genre driving the recommendation
- **Demo profiles** — sample MovieLens users available behind a drawer for instant exploration

---

## 🏗️ System Architecture

```
Browser (React + Vite)
        │
        │  GET /recommend?user_id=X
        ▼
FastAPI Serving Layer
  ┌─────────────────────────────────────────────────┐
  │  1. user_id → user_idx  (hash map, O(1))        │
  │  2. UserTower inference  (PyTorch, 128-dim)      │
  │  3. LRU embedding cache  (OrderedDict, cap=2048) │
  │  4. ANN retrieval        (numpy dot-product)     │
  │     user_emb @ item_embs → top-500 candidates   │
  │  5. GBM re-ranker        (sklearn, 8 features)  │
  │  6. Top-K selection      (min-heap, O(N log K))  │
  │  7. Enrich + explain     (title, poster, why)    │
  └─────────────────────────────────────────────────┘
        │
        │  JSON results
        ▼
   Movie card grid
```

---

## 📐 Training Pipeline

| Phase | Script | Description |
|------:|--------|-------------|
| 1 | `ml/scripts/preprocess.py` | Raw CSVs → temporal train/val split, ID remapping, genre vectors |
| 2 | `ml/models/two_tower.py` | Model definition — UserTower, ItemTower, learnable temperature |
| 3 | `ml/scripts/train_two_tower.py` | InfoNCE training with in-batch negatives (MPS / CUDA / CPU) |
| 4 | `ml/scripts/generate_embeddings.py` | Export all item embeddings as `item_embeddings.npy` |
| 5 | `ml/scripts/build_faiss_index.py` | Build FAISS IVF index (4.5× faster than brute-force, 99.8% recall) |
| 6 | `ml/scripts/train_ranker.py` | Train GBM ranker on (user, item) feature pairs |
| 7 | `ml/scripts/evaluate.py` | Compute Recall@K and NDCG@K on the val set |
| 8 | `ml/scripts/fetch_posters.py` | Fetch TMDb poster URLs for the catalogue |

---

## 📊 Model Performance

### Two-Tower Retrieval — offline evaluation (2,000 val users)

| Metric | @10 | @50 | @100 |
|--------|----:|----:|-----:|
| Recall | 2.6% | 10.2% | 15.7% |
| NDCG   | 1.1% |  2.8% |  3.8% |

> Evaluated in a cold-start setting: no interaction history at serve time, 10 K item catalogue.

### GBM Re-ranker

| Metric | Value |
|--------|------:|
| Validation AUC | **0.9799** |
| Top feature | `popularity × similarity` (70%) |
| 2nd feature | `embedding_similarity` (23%) |

### FAISS Index Benchmark

| Index type | Recall@10 | Latency |
|------------|----------:|--------:|
| `IndexFlatIP` — exact brute-force | 100% | 1× |
| `IndexIVFFlat` nlist=100, nprobe=10 | **99.8%** | **4.5× faster** |

---

## 🚀 Local Setup

### Prerequisites

- Python 3.11+
- Node 18+
- MovieLens 25M dataset — [download here](https://grouplens.org/datasets/movielens/25m/), unzip into `data/`

### 1 — Clone & install

```bash
git clone https://github.com/Yadu080/Two-Tower_Recommendation_System.git
cd Two-Tower_Recommendation_System

# Python env
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Frontend
cd frontend && npm install && cd ..
```

### 2 — Train the models

Run each phase in order, or use the convenience script:

```bash
source venv/bin/activate

python ml/scripts/preprocess.py
python ml/scripts/train_two_tower.py        # ~15 min on Apple MPS / GPU
python ml/scripts/generate_embeddings.py
python ml/scripts/build_faiss_index.py
python ml/scripts/train_ranker.py
python ml/scripts/evaluate.py
```

Or all at once:

```bash
bash scripts/build_pipeline.sh
```

### 3 — Fetch movie posters (optional)

Create a `.env` file in the project root:

```
TMDB_API_KEY=your_tmdb_api_key_here
```

Then run:

```bash
python ml/scripts/fetch_posters.py
```

This generates `ml/data/poster_map.json`. Skip this step and the app will show genre-coloured gradient cards instead.

### 4 — Start the app

```bash
# Terminal 1 — backend
source venv/bin/activate
uvicorn backend.main:app --reload

# Terminal 2 — frontend
cd frontend
npm run dev
```

Open **http://localhost:5173**

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/recommend?user_id=42&n=10` | Get top-N personalised recommendations |
| `GET` | `/users?n=30` | List sample users |
| `GET` | `/genres` | List available genre tags |
| `POST` | `/users/register` | Register a new user with name + genres |
| `POST` | `/log_click` | Log a movie click (excludes it from future recs) |
| `GET` | `/health` | Health check + cache stats |

**Example — recommendation response:**

```json
{
  "user_id": 42,
  "display_name": "User 42",
  "results": [
    {
      "rank": 1,
      "movie_id": 318,
      "title": "Shawshank Redemption, The (1994)",
      "genres": "Drama",
      "avg_rating": 4.43,
      "embedding_sim": 0.912,
      "ranking_score": 0.971,
      "why_recommended": "Matches your taste in Drama",
      "poster_url": "https://image.tmdb.org/t/p/w500/...",
      "latency_ms": 18.3
    }
  ]
}
```

---

## 🌐 Deployment

### Backend → Render

1. Go to [render.com](https://render.com) → **New Web Service**
2. Connect your GitHub repo — Render auto-detects `render.yaml`
3. Add environment variable: `TMDB_API_KEY = your_key`
4. Click **Deploy** — first build takes ~3–5 min
5. Copy your service URL (e.g. `https://recomai-backend.onrender.com`)

### Frontend → Vercel

1. Go to [vercel.com](https://vercel.com) → **New Project** → import the same repo
2. Set **Root Directory** to `frontend`
3. Add environment variable: `VITE_API_URL = https://recomai-backend.onrender.com`
4. Click **Deploy** — done in ~1–2 min

> **Note:** Render free-tier services spin down after 15 min of inactivity. The first request after idle takes ~30 s to cold-start. Upgrade to a paid plan for always-on behaviour.

---

## 📁 Project Structure

```
Two-Tower_Recommendation_System/
├── ml/
│   ├── models/
│   │   ├── two_tower.py              # UserTower + ItemTower definition
│   │   ├── two_tower.pt              # Trained weights
│   │   ├── ranker.joblib             # Trained GBM ranker
│   │   └── ranker_features.json      # Feature names
│   ├── scripts/
│   │   ├── preprocess.py
│   │   ├── train_two_tower.py
│   │   ├── generate_embeddings.py
│   │   ├── build_faiss_index.py
│   │   ├── train_ranker.py
│   │   ├── evaluate.py
│   │   └── fetch_posters.py
│   ├── embeddings/
│   │   └── item_embeddings.npy
│   └── data/
│       ├── item_meta.csv
│       ├── user_features.csv
│       ├── item_features.csv
│       ├── genre_vocab.json
│       ├── user_id_map.json
│       ├── movie_id_map.json
│       └── poster_map.json           # generated by fetch_posters.py
├── backend/
│   ├── main.py                       # FastAPI app + CORS + lifespan
│   ├── api/routes.py                 # All route handlers
│   └── core/recommender.py           # Retrieval + ranking engine
├── frontend/
│   ├── src/
│   │   ├── App.jsx                   # View state machine (landing → genres → recs)
│   │   ├── api.js                    # Axios wrappers
│   │   └── components/
│   │       ├── LandingPage.jsx       # Full-screen hero + name input
│   │       ├── GenrePicker.jsx       # Genre selection grid
│   │       ├── MovieCard.jsx         # Poster card with hover overlay
│   │       └── DemoDrawer.jsx        # Sample user browser
│   ├── index.html
│   ├── tailwind.config.js
│   └── vercel.json
├── data/                             # Raw MovieLens CSVs (not in git)
├── docs/screenshots/                 # Add your app screenshots here
├── scripts/build_pipeline.sh
├── render.yaml
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Neural model | PyTorch — Two-Tower with InfoNCE loss |
| Re-ranker | scikit-learn GradientBoostingClassifier |
| Vector search | FAISS IVF / numpy dot-product |
| Backend API | FastAPI + Uvicorn |
| Frontend | React 18 + Vite + Tailwind CSS + Framer Motion |
| Movie posters | TMDb API |
| Dataset | MovieLens 25M (GroupLens) |
| Backend deploy | Render |
| Frontend deploy | Vercel |

---

## 📄 License

MIT — free to use, fork, and build on.

---

<p align="center">Built by <strong>Yadunandan M Nimbalkar</strong></p>
