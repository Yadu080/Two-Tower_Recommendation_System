# RECOMAI — Deep Dive & Interview Preparation

A complete technical walkthrough of this project: what it does, how every piece
works, why each technology was chosen over the alternatives, what every metric
actually means, and the questions an interviewer is likely to ask.

**Read the "Honest Weaknesses" section before any interview.** It contains two
issues an experienced interviewer will probably spot. Knowing them in advance
turns your biggest risk into your strongest answer.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [The Architecture, End to End](#2-the-architecture-end-to-end)
3. [The Tech Stack, From Scratch](#3-the-tech-stack-from-scratch)
4. [Alternatives Considered](#4-alternatives-considered)
5. [Every Metric, Explained](#5-every-metric-explained)
6. [Honest Weaknesses](#6-honest-weaknesses)
7. [Development Walkthrough](#7-development-walkthrough)
8. [Interview Questions & Answers](#8-interview-questions--answers)

---

## 1. What This Project Does

### The one-sentence version

A movie recommendation engine that learns 128-dimensional vector
representations of users and movies, retrieves candidates by vector similarity,
re-ranks them with a gradient-boosted classifier, and serves personalised
results through a REST API to a streaming-service-style web UI.

### The problem it solves

You have 10,523 movies and a user. Which 10 do you show them?

Scoring every (user, movie) pair with a heavyweight model is too slow — you
would run 10,523 model evaluations per page load. The industry-standard answer,
which this project implements, is **two-stage recommendation**:

| Stage | Job | Speed | Precision |
|-------|-----|-------|-----------|
| **Retrieval** | Narrow 10,523 → 500 | Very fast | Rough |
| **Ranking** | Order those 500 precisely | Slower per item | High |

You get the best of both: a cheap model looks at everything, an expensive model
looks at only the survivors. This is how YouTube, Netflix, and Pinterest
structure their recommenders.

### The user-facing flow

1. Visitor enters a name → picks genres → a cold-start profile is created
2. Backend synthesises an embedding for them from their genre choices
3. Retrieval + ranking produce 24 titles
4. UI renders a hero billboard, a Top 10 row, and themed carousels
5. Clicking a title logs implicit feedback and refreshes the recommendations

---

## 2. The Architecture, End to End

### The Two-Tower model

The core idea: **two separate neural networks that never see each other's
input**, producing vectors in the same shared space.

```
   user_idx                    item_idx + genre/rating/popularity features
      │                                        │
      ▼                                        ▼
 ┌──────────┐                            ┌──────────┐
 │ Embedding│ 64-dim                     │ Embedding│ 64-dim
 └────┬─────┘                            └────┬─────┘
      │                                       │ concat content features (21)
      │                                       ▼  → 85-dim
 ┌────▼─────┐                            ┌────▼─────┐
 │ Linear   │ 64 → 128                   │ Linear   │ 85 → 128
 │ ReLU     │                            │ ReLU     │
 │ Linear   │ 128 → 128                  │ Linear   │ 128 → 128
 └────┬─────┘                            └────┬─────┘
      ▼                                       ▼
 L2 normalise                            L2 normalise
      │                                       │
      └──────────────► dot product ◄──────────┘
                    = cosine similarity
```

**Why two separate towers?** Because item embeddings can be computed *once,
offline*, and cached. At serving time you only run the user tower (one forward
pass), then do a matrix multiply against the precomputed item matrix. If you
used a single model taking (user, item) jointly, you would need 10,523 forward
passes per request. This architectural split is the entire reason the system
serves in milliseconds.

**Why L2-normalise both outputs?** Once vectors are unit length, the dot
product *is* the cosine similarity. This means retrieval reduces to a single
matrix multiplication — the fastest operation available — instead of computing
cosine distances with explicit norm divisions.

Implementation: `ml/models/two_tower.py`

### Training with InfoNCE

The model is trained with **InfoNCE loss** (also called NT-Xent — the same loss
behind SimCLR and CLIP).

For a batch of B (user, liked-movie) pairs:

1. Compute all user embeddings → `(B, 128)`
2. Compute all item embeddings → `(B, 128)`
3. Similarity matrix `S = user_emb @ item_emb.T / temperature` → `(B, B)`
4. `S[i,i]` is a real (user, liked-movie) pair — a **positive**
5. `S[i,j]` where `i≠j` pairs user *i* with a movie liked by some *other* user
   — treated as a **negative**
6. Cross-entropy over each row, with the correct label being the diagonal

The elegance: from B training examples you get B² units of training signal, and
you never have to explicitly sample negative examples. These are called
**in-batch negatives**.

The loss is computed symmetrically (rows *and* columns), so the model learns
both "which movie fits this user" and "which user fits this movie."

**Learnable temperature.** `model.log_temp` is a trainable parameter, clamped
to [0.01, 1.0] after exponentiation. Temperature controls how sharply the
model distinguishes positives from negatives — low temperature means confident,
peaked distributions. Rather than hand-tuning it, the model learns the right
sharpness itself.

Implementation: `ml/scripts/train_two_tower.py`

### The serving pipeline

Every `/recommend` request runs these five steps
(`backend/core/recommender.py::recommend`):

```
1. user_idx → user_emb            UserTower forward pass, or LRU cache hit
2. user_emb → top-500 candidates  item_embs @ user_emb, then argpartition
3. candidates → scores            GBM predict_proba on 8 features per candidate
4. scores → top-K                 min-heap selection
5. format + explainability        title, genres, why_recommended, latency
```

**Step 2 detail — why `argpartition` and not `sort`?**
Sorting 10,523 similarity scores is O(N log N). But you don't need them sorted
— you only need the top 500. `np.argpartition` uses quickselect to partition in
**O(N)**, then you sort only those 500: O(K log K). For N=10,523 and K=500,
that is meaningfully faster than a full sort.

**Step 4 detail — why a heap?**
Same principle at a different scale. Selecting the top-10 from 500 scored
candidates via a min-heap of size K is O(N log K) rather than O(N log N).

**The LRU cache.** Running a user through the UserTower costs ~0.5ms. If the
same user requests recommendations repeatedly, that is wasted work. An
`OrderedDict`-backed LRU cache (capacity 2048) gives O(1) get/put — on access,
`move_to_end` marks a key most-recently-used; on overflow, `popitem(last=False)`
evicts the least-recently-used. This is precisely LeetCode 146.

### Cold-start handling

A brand new user has no learned embedding — their ID was never in the training
set. The solution (`_genre_embedding`):

1. Find all items whose genre vector matches any selected genre
2. Average those item embeddings (capped at 2,000 items for speed)
3. L2-normalise the result

The new user is placed in the same vector space, at the centroid of the content
they said they like. New user IDs start at 1,000,000 (`NEW_USER_BASE`) to stay
clear of real MovieLens IDs.

### The re-ranker

The GBM scores each of the 500 candidates using 8 features:

| # | Feature | Why it matters |
|---|---------|----------------|
| 1 | `emb_similarity` | Two-Tower's opinion — the core relevance signal |
| 2 | `item_avg_rating` | Is this film actually good? |
| 3 | `item_num_ratings` | Confidence in that rating |
| 4 | `item_popularity` | Normalised popularity prior |
| 5 | `user_avg_rating` | Is this user generous or harsh? |
| 6 | `user_num_ratings` | How much do we know about them? |
| 7 | `genre_match` | Cosine between user genre prefs and item genres |
| 8 | `popularity × similarity` | Interaction term: relevant **and** popular |

Feature 8 is the interesting one. Neither popularity nor similarity alone is
sufficient — a blockbuster you'd hate is bad, and an obscure perfect match is
risky. Their product captures "relevant *and* broadly liked," and the trained
model weights it most heavily.

---

## 3. The Tech Stack, From Scratch

### PyTorch

**What it is.** A Python library for building and training neural networks. It
provides tensors (multi-dimensional arrays with GPU support) and automatic
differentiation — you write the forward computation, and PyTorch derives the
gradients needed to train.

**What it does here.** Defines and trains the Two-Tower model. `nn.Embedding`
provides the learnable ID→vector lookup tables; `nn.Linear` the fully-connected
layers; `F.normalize` the L2 normalisation; autograd handles backpropagation
through the InfoNCE loss.

**Key concepts you should be able to explain:**
- *Tensor* — an n-dimensional array that tracks operations for autograd
- *Autograd* — automatic computation of gradients via the chain rule
- *`nn.Module`* — base class; subclass it, define `forward()`, get parameter
  tracking and device movement for free
- *`model.eval()` / `torch.no_grad()`* — disable dropout/batchnorm training
  behaviour and stop building the gradient graph, for inference

### FastAPI

**What it is.** A modern Python web framework for building APIs. Its
distinguishing feature is that it derives request validation, serialisation,
and OpenAPI documentation directly from Python type hints.

**What it does here.** Exposes `/recommend`, `/genres`, `/users`,
`/users/register`, `/log_click`, `/health`. Pydantic models
(`RegisterUserRequest`, `ClickEvent`) declare the expected request shapes, and
FastAPI rejects malformed input automatically with a 422.

**Key concepts:**
- *ASGI* — the async server interface FastAPI targets (vs. WSGI for Flask/Django)
- *Dependency injection* — shared logic declared as function parameters
- *`lifespan`* — startup/shutdown hooks; this project uses it to load all ML
  artifacts **once** at boot rather than per request

### Uvicorn

**What it is.** The ASGI server that actually runs the FastAPI app — it handles
the network socket, HTTP parsing, and the async event loop.

**What it does here.** `uvicorn backend.main:app` — the production entry point
on Render.

### NumPy

**What it is.** The foundational numerical computing library for Python.
Provides contiguous typed arrays and vectorised operations implemented in C.

**What it does here.** The entire retrieval step. `item_embs @ user_emb` is a
single matrix-vector multiply over a (10523, 128) matrix — roughly 0.6ms.
`argpartition` provides the O(N) top-K selection.

**Why it's fast:** operations run as compiled C loops over contiguous memory,
avoiding Python's per-element interpreter overhead entirely.

### scikit-learn

**What it is.** The general-purpose classical (non-deep) machine learning
library for Python — a consistent `fit()`/`predict()` interface across dozens of
algorithms.

**What it does here.** `GradientBoostingClassifier` is the re-ranker:
200 trees, depth 4, learning rate 0.05, subsample 0.8.

**Gradient boosting explained:** train a shallow decision tree; look at what it
got wrong; train a second tree specifically to correct those errors; repeat.
Each tree is weak, but the additive ensemble is strong. "Gradient" refers to
each tree fitting the gradient of the loss with respect to the current
predictions. Contrast with Random Forest, where trees are trained independently
and averaged — boosting is sequential and error-correcting.

### pandas

**What it is.** Tabular data manipulation — DataFrames with labelled columns,
grouping, joining, and I/O.

**What it does here.** All preprocessing: loading the 25M-row ratings CSV,
filtering users and movies by activity thresholds, computing per-user and
per-item aggregate statistics, and the temporal train/val split.

### FAISS

**What it is.** Facebook AI Similarity Search — a C++/Python library for
efficient nearest-neighbour search over dense vectors, including approximate
methods that trade a little accuracy for large speedups.

**What it does here.** Benchmarked in `build_faiss_index.py`, but **not used at
serving time** — see the note in [Honest Weaknesses](#6-honest-weaknesses).
An `IndexIVFFlat` partitions vectors into clusters; at query time only the
nearest `nprobe` clusters are searched, reducing the work from O(N) to roughly
O(√N).

### React + Vite + Tailwind + Framer Motion

- **React** — component-based UI library. State changes trigger re-renders of
  only the affected components via virtual-DOM diffing.
- **Vite** — build tool and dev server. Serves native ES modules in development
  (near-instant startup, no bundling), bundles with Rollup for production.
- **Tailwind CSS** — utility-first CSS. Instead of writing custom classes, you
  compose pre-defined atomic utilities (`flex`, `gap-2`, `text-white`) directly
  in markup.
- **Framer Motion** — declarative animation for React. Powers the card hover
  scaling, modal transitions, and view changes.

### MovieLens 25M

The dataset: 25 million ratings from 162,000 users across 62,000 movies,
published by GroupLens Research at the University of Minnesota. The standard
academic benchmark for recommender systems.

After filtering (users with ≥20 ratings, movies with ≥50 ratings, capped at 2M
interactions), this project trains on **10,523 movies**.

---

## 4. Alternatives Considered

This section matters in interviews. "Why did you choose X?" is really asking
"do you understand the trade-off space, or did you just follow a tutorial?"

### Retrieval model

| Alternative | How it works | Why not chosen |
|---|---|---|
| **Matrix Factorisation (ALS/SVD)** | Factor the rating matrix into user × item latent factors | Simpler and often a strong baseline, but cannot use content features. Genre/rating/popularity data would be wasted, and cold-start would be impossible — which is the entire onboarding flow here. |
| **Neural Collaborative Filtering (NCF)** | One MLP over concatenated user+item features | More expressive per pair, but there is no separable item tower, so you cannot precompute item embeddings. Serving would need 10,523 forward passes per request instead of one. |
| **Sequential models (SASRec, BERT4Rec)** | Transformer over the user's interaction *sequence* | Genuinely stronger — order matters in real behaviour. But requires per-user interaction histories at inference, far more training compute, and does not fit a "pick genres, get recommendations immediately" flow. |
| **Graph models (LightGCN, PinSage)** | Message-passing over the user-item bipartite graph | Excellent accuracy, but heavy infrastructure and slow to iterate on for a solo project. |
| **✅ Two-Tower** | Separate encoders into a shared space | Precomputable item side, content features supported, natural cold-start via content averaging, and the industry-standard retrieval architecture. |

### Loss function

| Alternative | Why not chosen |
|---|---|
| **BPR (Bayesian Personalised Ranking)** | Pairwise: one positive vs one sampled negative per step. Works, but far less sample-efficient — InfoNCE gets B² signal from B examples. |
| **BCE with sampled negatives** | Requires an explicit negative sampling strategy, which becomes its own tuning problem. |
| **✅ InfoNCE / in-batch negatives** | No explicit sampling needed, B² signal density, and a learnable temperature. Same loss family as CLIP. |

### Re-ranker

| Alternative | Why not chosen |
|---|---|
| **LightGBM / XGBoost** | Genuinely better at scale — faster training, histogram-based splits. **This was the original choice**; the code still says "LightGBM" in places. It was replaced because LightGBM's OpenMP runtime segfaults when loaded in the same process as PyTorch on macOS. At 8 features, sklearn's accuracy is equivalent. |
| **Logistic regression** | Fast and interpretable, but cannot learn feature interactions without manual engineering. |
| **Neural re-ranker** | Overkill for 8 tabular features. Gradient-boosted trees dominate on small tabular problems. |
| **✅ sklearn GradientBoostingClassifier** | Handles non-linear interactions, no OpenMP conflict, `joblib` loads cleanly alongside torch. |

### Vector search

| Alternative | Why not chosen |
|---|---|
| **FAISS at serving time** | Benchmarked at 4.5× faster, but same macOS OpenMP conflict with PyTorch. At 10,523 items, brute-force NumPy is already ~0.6ms — the added complexity buys nothing at this scale. |
| **Managed vector DBs (Pinecone, Weaviate, Milvus)** | Network hop per query, plus cost and operational overhead. Justified at 100M+ vectors, not 10K. |
| **✅ NumPy brute force** | O(N·D) exact search, no dependencies, no conflicts. **The honest engineering answer: at this scale, approximate search is premature optimisation.** |

### Backend framework

| Alternative | Why not chosen |
|---|---|
| **Flask** | Synchronous WSGI, no built-in validation or type-driven docs. |
| **Django REST Framework** | Batteries-included ORM/admin/auth — none needed here; this app has no database. |
| **✅ FastAPI** | Async, Pydantic validation from type hints, auto-generated OpenAPI docs, and the de-facto standard for Python ML serving. |

### Frontend

| Alternative | Why not chosen |
|---|---|
| **Next.js** | SSR and file-based routing add value for SEO and multi-page apps. This is a single-view SPA hitting a separate API — the extra framework surface buys little. |
| **Vue / Svelte** | Both fine. React chosen for ecosystem size and hiring relevance. |
| **Plain CSS / styled-components** | Tailwind keeps styling colocated with markup and avoids naming-convention overhead in a solo project. |

---

## 5. Every Metric, Explained

For each metric: what it means, how it is computed, what counts as good, and
**where this project actually lands**.

---

### Recall@K

**What it measures.** Of the items the user actually liked, what fraction did
we surface in our top-K?

**How it's calculated** (`ml/scripts/evaluate.py::recall_at_k`):

```
Recall@K = |retrieved_top_K ∩ relevant| / min(|relevant|, K)
```

Note the denominator is `min(|relevant|, K)`, not `|relevant|`. This is a
deliberate correction: if a user liked 200 films and K=10, you *cannot*
retrieve more than 10, so dividing by 200 would unfairly cap the score at 5%.

**Range.** 0.0 to 1.0. Higher is better.

**What's good?** Entirely dependent on catalogue size and evaluation protocol —
raw numbers are meaningless without context. The honest benchmark is the
**random baseline**: with 10,523 items, randomly picking 10 gives
`10/10523 = 0.095%`.

**This project's results:**

| K | Recall | Random baseline | Lift over random |
|---|--------|-----------------|------------------|
| 10 | 2.57% | 0.095% | **27×** |
| 50 | 10.22% | 0.475% | **21×** |
| 100 | 15.72% | 0.95% | **17×** |

**Verdict: the model has clearly learned real signal — 27× random is not
noise — but the absolute numbers are weak.** Published Two-Tower results on
MovieLens-scale data typically reach Recall@10 in the 5–15% range. This model
underperforms that, for identifiable reasons (see
[Honest Weaknesses](#6-honest-weaknesses)).

Do not oversell this number in an interview. Say: *"27× better than random,
which confirms the architecture works, but below what I'd expect from a tuned
Two-Tower — I know why, and here's what I'd fix first."* That answer is far
stronger than pretending 2.57% is good.

---

### NDCG@K (Normalised Discounted Cumulative Gain)

**What it measures.** Recall only asks *whether* a relevant item appeared in
the top-K. NDCG asks *how highly it was ranked*. A hit at position 1 is worth
more than a hit at position 10.

**How it's calculated** (`ml/scripts/evaluate.py`):

```
DCG@K  = Σ  relevance_i / log₂(i + 2)        for i in [0, K)
IDCG@K = DCG of the ideal ordering (all relevant items first)
NDCG@K = DCG@K / IDCG@K
```

The `log₂(i+2)` discount is the key: position 0 divides by log₂(2)=1.00,
position 1 by log₂(3)=1.58, position 9 by log₂(11)=3.46. Later positions
contribute progressively less. Normalising by the ideal DCG bounds the result
to [0, 1] and makes it comparable across users with different numbers of
relevant items.

**Range.** 0.0 to 1.0, where 1.0 means perfect ranking.

**What's good?** NDCG is always lower than Recall for the same K (it is Recall
weighted by a discount ≤ 1). Strong production recommenders report NDCG@10 in
the 0.10–0.40 range depending on domain.

**This project's results:**

| K | NDCG |
|---|------|
| 10 | 0.0111 |
| 50 | 0.0281 |
| 100 | 0.0376 |

**Verdict: weak, and notably low even relative to this project's own Recall.**
NDCG@10 (1.1%) is less than half of Recall@10 (2.57%), which tells you
something specific: **when the model does find a relevant item, it tends to
rank it near the bottom of the top-10 rather than the top.** The retrieval is
finding signal but ordering it poorly — exactly what you would expect from a
model whose user tower is a pure ID embedding with no behavioural features.

---

### ROC AUC (the re-ranker's 0.9799)

**What it measures.** Given a random positive example and a random negative
example, the probability that the classifier scores the positive higher.

**How it's calculated.** Plot the true-positive rate against the false-positive
rate across every possible decision threshold; AUC is the area under that
curve. Computed by `sklearn.metrics.roc_auc_score`.

**Range.** 0.5 = random guessing. 1.0 = perfect separation. Below 0.5 means
your model is inverted.

**Rough interpretation guide:**

| AUC | Reading |
|-----|---------|
| 0.5 | No signal |
| 0.6–0.7 | Weak |
| 0.7–0.8 | Acceptable |
| 0.8–0.9 | Strong |
| 0.9+ | Excellent — **or leakage** |

**This project's result: 0.9799.**

**Verdict: this number is inflated by label leakage and should not be quoted
without qualification.** See the next section — this is the single most
important thing to understand before an interview.

---

### FAISS benchmark (99.8% recall, 4.5× speedup)

**What it measures.** How much accuracy an *approximate* index loses relative
to exact brute-force search, and how much speed it gains.

**How it's calculated** (`ml/scripts/build_faiss_index.py`): run the same 200
queries through `IndexFlatIP` (exact) and `IndexIVFFlat` (approximate,
nlist=100, nprobe=10); measure the overlap of their top-10 results and compare
per-query latency.

**What's good?** For approximate nearest neighbour, >95% recall at a meaningful
speedup is generally considered a good operating point.

**This project's result: 99.8% recall at 4.5× speedup — a genuinely good
trade-off curve.**

**Important caveat: this index is not used in production.** Serving uses NumPy
brute force. The benchmark demonstrates that you understand the ANN trade-off
space; it does not describe the deployed system. Be precise about this
distinction if asked — claiming FAISS is in your serving path when it isn't
would be a serious credibility problem.

---

### Latency (~20ms end-to-end)

**What it measures.** Wall-clock time from request arrival to formatted
response, measured in `recommend()` via `time.perf_counter()` and returned in
each result as `latency_ms`.

**Rough budget:**

| Step | Cost |
|------|------|
| UserTower forward pass | ~0.5ms (0ms on cache hit) |
| NumPy retrieval (10,523 × 128 matmul + argpartition) | ~0.6ms |
| Building 500 feature rows (Python loop) | several ms |
| GBM `predict_proba` on 500 × 8 | a few ms |
| Heap top-K + formatting | <1ms |

**What's good?** For interactive UI, <100ms feels instant; <20ms is
comfortably fast. **Verdict: genuinely good, and the least qualified claim in
the project.** Note this is local latency excluding network — on Render's free
tier, cold starts add 30–60s, which is an infrastructure characteristic rather
than a model one.

---

### Model size

- **Parameters:** dominated by the two embedding tables — 162K users × 64 +
  10.5K items × 64 ≈ 11M parameters, plus small MLP heads.
- **Checkpoint:** 36MB (`two_tower.pt`)
- **Item embeddings:** 5.2MB (10,523 × 128 float32)
- **Ranker:** 472KB

**Verdict: small.** Fits comfortably in memory on a free-tier instance, which
is why this deploys at all.

---

## 6. Honest Weaknesses

**Read this section carefully.** Two of these are the kind of thing a sharp
interviewer finds by reading your code. Raising them yourself, with a fix in
mind, reads as senior. Being caught unaware does not.

> **Status update:** issues 1, 2 and 3 below have since been **fixed** in the
> code. They are kept here in full because the reasoning is the valuable part —
> and because "I found this in my own code and fixed it" is a stronger
> interview answer than either hiding it or never having had it. What follows
> describes the original defect; the fix is noted at the end of each.

### ⚠️ 1. Label leakage in the re-ranker (the 0.98 AUC was not real) — FIXED

In `ml/scripts/train_ranker.py`, the user embedding used to build training
features was computed like this:

```python
pos_set  = user_pos_items[user_idx]              # items the user liked
pos_embs = [item_embs[idx] for idx in pos_set]
user_emb = np.mean(pos_embs, axis=0)             # user = mean of their positives
user_emb /= np.linalg.norm(user_emb)
```

Then, immediately afterwards, those **same** positive items are used as the
positive training rows:

```python
for item_idx in pos_set:                          # ← the identical set
    rows.append(build_feature_row(..., user_emb, ...))
    labels.append(1)
```

The first feature in every row is `emb_sim = dot(user_emb, item_emb)`. But
`user_emb` was *constructed by averaging those very item embeddings*. So
positives are guaranteed high similarity essentially by definition, while
negatives are randomly sampled items that had no hand in building the vector.

The classifier isn't learning "what does this user like" — it is substantially
learning "was this item one of the vectors I was just averaged from."

**Corroborating evidence:** the README reports `popularity × similarity` at 70%
feature importance and `embedding_similarity` at 23% — **93% of the model's
decision rests on the two features derived from the leaked quantity.**

**The fix (applied).** `train_ranker.py` now builds the user representation
from `UserTower` — the same module the backend serves with, fit on the *train*
split — and labels against *validation* positives. Nothing about the labels
touches the feature side any more. Negatives are sampled from items the user
rated in neither split.

**Expected effect:** AUC will drop substantially — likely into the 0.70–0.85
range. That is not a regression; that is the honest number appearing. The new
value is written to `ml/models/ranker_metrics.json` on each training run.

**How to talk about it:** *"The 0.98 AUC is inflated — I have label leakage in
how I construct the user embedding for ranker training. I'd fix it with a
disjoint split and expect the real number around 0.75–0.85."* This is one of
the strongest things you can say in an ML interview. It demonstrates you can
audit your own work.

### ⚠️ 2. Train/serve skew in the ranker — FIXED

Related but distinct. The user embedding was computed **differently** in the
two settings:

| Setting | How `user_emb` was produced |
|---------|----------------------------|
| Ranker training | Mean of the user's positive item embeddings |
| Serving | `UserTower(user_idx)` forward pass |

The ranker learned a relationship between `emb_sim` and relevance under one
definition, then encountered a different definition in production. The feature
distributions did not match, so learned thresholds were miscalibrated.

**The fix (applied).** Ranker training now calls the same `UserTower` the
backend uses, so both paths see an identical feature distribution. The single
change that fixed the leakage fixed this too.

### ⚠️ 3. The README's "cold-start evaluation" claim — FIXED

The README stated the retrieval metrics were *"Evaluated cold-start — no
interaction history at inference time."* That is not what the code does.
`evaluate.py` calls:

```python
user_emb = model.user_tower(idx_t).numpy()
```

That is the **learned** embedding for a user who *was* in the training set —
the opposite of cold-start. The reported Recall/NDCG describe warm-start
performance on known users.

**The fix (applied).** The README now describes these as warm-start numbers,
states the real catalogue size (10,523, not 62K), gives the random baseline for
context, and notes separately that cold-start is supported at serving time but
is not what the numbers measure.

### 4. The user tower had no content features — FIXED

The item tower combines a learned ID embedding with genre, rating, and
popularity features. The user tower was a **bare `nn.Embedding` lookup** — no
demographics, no aggregate behaviour, no genre preferences.

Two consequences: the model could only memorise per-user patterns rather than
generalise across users with similar taste, and this was likely a primary
driver of the weak NDCG.

**The fix (applied).** `UserTower` now optionally concatenates the normalised
19-dim `genre_pref` vector — data that was already being computed and stored in
`user_features.csv`, just unused. `ml/models/user_features.py` centralises how
that vector is built so training and serving cannot drift apart.

Because this changes the tower's input width, **the checkpoint must be
regenerated** — see [Retraining](#retraining-after-these-fixes). The checkpoint
now records `num_user_features`; loaders default it to 0 when absent, so
checkpoints saved before this change still load into their original
architecture rather than crashing on a shape mismatch.

### Retraining after these fixes

Fixes 1, 2 and 4 change the *code*, not the saved artifacts. Until you retrain,
the deployed model is still the old one and the metrics are still the old ones.

Requires the raw MovieLens 25M data in `data/`.

```bash
source venv/bin/activate

python ml/scripts/preprocess.py           # regenerate train.csv / val.csv
python ml/scripts/train_two_tower.py      # new UserTower — the slow step
python ml/scripts/generate_embeddings.py  # item embeddings from new weights
python ml/scripts/train_ranker.py         # leakage-free ranker
python ml/scripts/evaluate.py             # honest Recall / NDCG
```

Order matters — each step consumes the previous step's output. Then commit the
regenerated artifacts:

```bash
git add ml/models/two_tower.pt ml/models/ranker.joblib \
        ml/models/ranker_metrics.json ml/models/eval_results.json \
        ml/embeddings/item_embeddings.npy
```

**What to expect.** Ranker AUC should drop from 0.9799 into roughly 0.70–0.85 —
that is the leakage being removed, not a regression. Recall and NDCG should
*improve* somewhat from the user-tower content features, though how much is an
empirical question. Update the README with whatever you actually measure.

**Deploy safety.** You can push the code changes before retraining without
breaking production: the old checkpoint lacks `num_user_features`, loaders
default it to 0, and the backend rebuilds the original architecture. Verified —
`/health`, warm-start `/recommend`, registration, and cold-start
recommendations all work unchanged against the pre-fix checkpoint.

### 5. In-memory state is lost on restart

`engine.new_users` and `_click_log` live in Python dictionaries. Every Render
redeploy or free-tier spin-down wipes every profile users created. There is a
`new_users.json` persistence path, but on an ephemeral filesystem it does not
survive either. A real deployment needs Postgres or Redis.

### 6. Popularity bias

`popularity × similarity` at 70% importance means the ranker leans heavily on
what is broadly popular. This produces safe, agreeable recommendations and
systematically buries niche content — the classic recommender failure mode.
Diversity-aware re-ranking (MMR, or a popularity-penalty term) would address it.

### 7. No A/B testing or online metrics

Everything reported is offline. Offline metrics correlate imperfectly with real
engagement. Click logging exists (`/log_click`) but nothing analyses it.

---

## 7. Development Walkthrough

The order in which this was built, and why each step depends on the previous.

### Phase 1 — Preprocessing (`preprocess.py`)

1. Load 25M ratings and 62K movies
2. **Filter:** users with ≥20 ratings, movies with ≥50 ratings, cap at 2M
   interactions. This removes noise (users with 2 ratings teach nothing) and
   keeps local training tractable.
3. **Build ID maps.** Raw MovieLens IDs are sparse (movie 193609 exists but
   most integers below it do not). `nn.Embedding` needs contiguous indices
   `[0, N)`, so `movieId → movie_idx` and `userId → user_idx` maps are built
   and saved as JSON.
4. **Build features.** Per-item: 19-dim genre multi-hot, avg rating,
   popularity. Per-user: genre preference vector, activity stats.
5. **Temporal split.** For each user, their most recent 10% of ratings become
   validation.

**Why temporal and not random?** A random split lets the model train on a
user's *future* behaviour and predict their *past* — leaking information that
would never exist in production. Splitting by time simulates reality: train on
what you knew then, evaluate on what happened next. This is the correct
protocol for recommenders, and worth calling out in an interview.

### Phase 2 — Train the Two-Tower (`train_two_tower.py`)

Batch 512, 10 epochs, Adam at 3e-4, cosine LR annealing, gradient clipping at
1.0. Only ratings ≥3.5 count as positives. Checkpoints on best validation loss.

The item feature tensor is pre-built as a single `(num_items, 21)` array so
`__getitem__` is one O(1) tensor index rather than an O(N) DataFrame lookup —
a roughly 100× speedup in data loading.

### Phase 3 — Generate item embeddings (`generate_embeddings.py`)

Run all 10,523 items through the trained ItemTower once, save as
`item_embeddings.npy`. **This is the step that makes serving fast** — the item
side is now a static matrix requiring no model evaluation at request time.

### Phase 4 — FAISS benchmark (`build_faiss_index.py`)

Build exact and approximate indices, measure recall and speedup. Informative;
not used in production.

### Phase 5 — Train the re-ranker (`train_ranker.py`)

Sample up to 5,000 validation users, build positive rows from their liked items
and 4× as many sampled negatives, compute the 8 features per pair, train the
GBM, report AUC and feature importances.

*(This is the phase containing the leakage described above.)*

### Phase 6 — Evaluate (`evaluate.py`)

Recall@K and NDCG@K at K ∈ {10, 50, 100} over 2,000 sampled validation users.

### Phase 7 — Fetch posters (`fetch_posters.py`)

One-time TMDB API pull mapping `movieId → poster URL`, cached to
`poster_map.json`. Achieved 97.7% coverage (10,282 / 10,523).

### Phase 8 — Backend (`backend/`)

FastAPI app; artifacts loaded once in the `lifespan` startup hook; the
recommendation engine as a module-level singleton.

### Phase 9 — Frontend (`frontend/`)

React SPA: landing → genre picker → browse. Billboard, carousels, hover-expand
cards, detail modal.

### Phase 10 — Deployment

Backend on Render (CPU-only torch to keep the image small), frontend on Vercel
with `VITE_API_URL` baked in at build time.

---

## 8. Interview Questions & Answers

### Architecture

**Q: Why two towers instead of one model taking user and item together?**

Because the towers are independent, item embeddings can be computed offline and
cached as a static matrix. At serving time only the user tower runs — one
forward pass — then retrieval is a single matmul. A joint model would need one
forward pass *per candidate*: 10,523 per request. The separation is what makes
sub-millisecond retrieval possible. The trade-off is expressiveness: a joint
model can learn fine-grained user-item interactions that two independent
encoders cannot, which is precisely why a second-stage re-ranker exists.

**Q: Why L2-normalise the outputs?**

It makes the dot product equal to cosine similarity, so retrieval is a plain
matrix multiply with no norm divisions. It also bounds similarities to [-1, 1],
which stabilises the InfoNCE loss — without normalisation the model could
inflate vector magnitudes to reduce loss rather than learning better directions.

**Q: Explain InfoNCE.**

For a batch of B (user, liked-item) pairs, compute the B×B similarity matrix.
The diagonal holds real pairs; everything off-diagonal pairs a user with a
different user's liked item, treated as a negative. Apply cross-entropy per row
with the diagonal as the target. You extract B² training signal from B examples
and never sample negatives explicitly. The temperature parameter — learnable
here — controls the sharpness of the resulting distribution.

**Q: Why a two-stage pipeline?**

Cost asymmetry. Retrieval must touch all 10,523 items, so it must be cheap — a
dot product. Ranking touches only 500 survivors, so it can afford richer
features and a heavier model. One stage alone forces you to choose between
being too slow or too imprecise.

### Metrics

**Q: Your Recall@10 is 2.6%. Isn't that terrible?**

In absolute terms it is weak, and I would not defend it as good. In context:
random selection from 10,523 items yields 0.095%, so this is about 27× random
— the architecture is clearly learning. But published Two-Tower results on
comparable data reach 5–15%, so I am underperforming. The main reason is that
my user tower is a pure ID embedding with no content features, so it can only
memorise per-user patterns rather than generalise. Adding the genre-preference
vector — which I already compute and store — is my first fix.

**Q: What's the difference between Recall@K and NDCG@K?**

Recall asks *whether* relevant items appeared in the top-K. NDCG asks *where*.
NDCG applies a `1/log₂(rank+2)` discount, so a hit at position 1 counts roughly
3.5× a hit at position 10, then normalises by the ideal ordering. In my
results, NDCG@10 (1.1%) is under half of Recall@10 (2.6%), which specifically
tells me that when the model finds relevant items it ranks them near the bottom
of the list — a ranking-quality problem distinct from a retrieval problem.

**Q: Your ranker had 0.98 AUC. Isn't that suspiciously high?**

It was, and it was wrong — I found it auditing my own code. The original
`train_ranker.py` built each user's embedding by averaging their positive item
embeddings, then used those same positives as label-1 rows. The primary feature
is the dot product between that averaged vector and each item, so positives
scored high essentially by construction, and the two features derived from it
accounted for 93% of feature importance.

I fixed it two ways at once. The user representation now comes from the
`UserTower` rather than from averaging label items, and features are built from
train-split history while labels come from val-split positives — so the item
being scored is never part of what produced the embedding scoring it. That also
removed a train/serve skew I had, since the ranker now sees the same
distribution in training that the backend produces in production. The honest
AUC is materially lower, which is the point.

**Q: Why is your denominator `min(|relevant|, K)` in Recall?**

Because a user with 200 relevant items cannot possibly have more than K of them
retrieved in a top-K list. Dividing by 200 at K=10 caps the achievable score at
5% regardless of model quality, which measures the evaluation protocol rather
than the model. Capping the denominator makes the metric comparable across
users with very different history lengths.

### Data & training

**Q: Why a temporal split instead of random?**

A random split lets the model see a user's future ratings during training and
be evaluated on earlier ones — information that cannot exist at serving time,
so results are optimistically biased. A temporal split (most recent 10% per
user held out) mirrors production: train on the past, predict the future.

**Q: How do you handle cold-start users?**

Content-based averaging. A new user picks genres; I collect all items matching
those genres, average their embeddings, and L2-normalise. That places them at
the centroid of the content they said they like, in the same space as trained
users, so the identical retrieval path works with no special-casing. New IDs
start at 1,000,000 to avoid colliding with real MovieLens IDs.

**Q: Why threshold ratings at 3.5?**

To convert explicit ratings into implicit binary feedback. On a 5-point scale,
3.5 separates genuine enjoyment from indifference — a 3 is lukewarm. The
architecture and loss are designed around "did they like it," not rating
prediction.

### Engineering

**Q: Why `argpartition` instead of `sort`?**

Sorting all 10,523 scores is O(N log N), but I only need the top 500.
`argpartition` uses quickselect to partition in O(N), and I sort only the 500
survivors at O(K log K).

**Q: Explain your caching strategy.**

An LRU cache over user embeddings, capacity 2048, backed by `OrderedDict`.
`get` promotes via `move_to_end`; overflow evicts via `popitem(last=False)`.
Both O(1). It saves the ~0.5ms UserTower pass on repeat requests. Cached
entries are invalidated on click so the next request reflects new feedback.

**Q: You mention FAISS. Is it in your serving path?**

No, and that distinction matters. I benchmarked it — `IndexIVFFlat` gave 99.8%
recall at 4.5× speedup versus exact search — but serving uses NumPy brute
force. Two reasons: FAISS's OpenMP runtime segfaults alongside PyTorch in one
process on macOS, and at 10,523 items an exact matmul is already ~0.6ms.
Approximate search would be premature optimisation. I would revisit above
roughly a million vectors.

**Q: What breaks if this gets 1000× more traffic?**

Several things, in order. In-memory state (`new_users`, `_click_log`) is
per-process and lost on restart — that needs Postgres or Redis first. Retrieval
is O(N·D) per query and single-threaded; at 10M items I would move to FAISS or
a vector DB. Ranker feature construction is a Python loop over 500 candidates —
that should be vectorised. And the model would need periodic retraining with an
embedding-refresh pipeline, which does not exist today.

### Reflective

**Q: What would you do differently?**

Two of the things I'd have flagged, I've since fixed: the ranker leakage that
invalidated my headline metric, and the bare ID-embedding user tower that I
believe was the main cause of weak NDCG — it now takes the genre-preference
vector I was already computing but never using.

What's still open: in-memory state should be a real datastore, since every
redeploy wipes user profiles. And I'd add diversity-aware re-ranking — 70%
feature importance on `popularity × similarity` means the system is strongly
popularity-biased and buries niche content. Longer term, the honest gap is that
everything I measure is offline; I log clicks but have never run an A/B test.

**Q: What was the hardest part?**

The library conflicts were the most instructive. FAISS and LightGBM both ship
their own OpenMP runtimes, which segfault when loaded alongside PyTorch in a
single process on macOS. Rather than fight it, I evaluated what each actually
bought me at this scale: NumPy retrieval was already fast enough, and sklearn's
GBM matched LightGBM on 8 features. Choosing the simpler dependency was the
right engineering call, not a compromise.

**Q: How do you know your recommendations are any good?**

Honestly — I do not, fully. I have offline metrics (Recall, NDCG, AUC), and one
of them I now know is inflated. Offline metrics are a weak proxy for
satisfaction; the only real answer is an online A/B test measuring engagement,
which I have not run. I log clicks but do not yet analyse them. That is the
biggest gap between this project and a production system.

---

## Quick Reference

| Claim | Number | Honest verdict |
|-------|--------|----------------|
| Recall@10 | 2.57% | Weak absolute, 27× random — real but underperforming. Pre-fix; expect improvement after retraining |
| Recall@100 | 15.72% | Weak-moderate. Pre-fix |
| NDCG@10 | 1.11% | Weak — ranking quality is the bottleneck. Pre-fix |
| Ranker AUC | ~~0.9799~~ | **Was inflated by leakage. Retrain and quote the new number from `ranker_metrics.json`** |
| FAISS recall | 99.8% @ 4.5× | Good, but not in the serving path |
| Latency | ~20ms | Genuinely good |
| Poster coverage | 97.7% | Good |
| Catalogue | 10,523 films | After filtering from 62K |

All retrieval numbers above predate the user-tower content features and will
change once you retrain. Do not quote them post-retrain without re-running
`evaluate.py`.

**The single most valuable thing you can do in an interview on this project:**
volunteer the leakage finding before you are asked. Engineers who audit their
own results and report them accurately are rarer, and more valuable, than
engineers with better numbers.
