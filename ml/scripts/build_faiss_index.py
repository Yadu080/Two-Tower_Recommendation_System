"""
Phase 4 — Build FAISS index from precomputed item embeddings.

Builds two indexes for comparison:
  1. IndexFlatIP  — exact brute-force (baseline, O(N) per query)
  2. IndexIVFFlat — approximate, partitioned (O(sqrt(N)) per query)

Both use inner product (dot product == cosine similarity since embeddings are L2-normalised).

DSA note: IVF = Inverted File Index — same naming as inverted index in search engines.
  The "inverted" part: instead of item→clusters, you store cluster→items (inverted mapping).
"""

import os, sys, time, json
import numpy as np
import faiss

EMB_DIR   = os.path.join(os.path.dirname(__file__), "../embeddings")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "../embeddings")


def build_flat_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Brute-force exact search. No training required.
    O(N × D) per query — fine for N < 100K, too slow for production scale.
    """
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)   # IP = inner product
    index.add(embeddings)
    return index


def build_ivf_index(embeddings: np.ndarray,
                    n_clusters: int = 100,
                    n_probe: int = 10) -> faiss.IndexIVFFlat:
    """
    IVF: partition space into n_clusters Voronoi cells via k-means.
    At query time, only probe n_probe nearest cells.

    Speed vs recall tradeoff:
      n_probe=1   → fastest, lowest recall
      n_probe=100 → same as brute force, slowest

    Rule of thumb: n_clusters = sqrt(N), n_probe = sqrt(n_clusters)
    """
    d = embeddings.shape[1]
    quantizer = faiss.IndexFlatIP(d)        # used to assign vectors to clusters
    index = faiss.IndexIVFFlat(quantizer, d, n_clusters, faiss.METRIC_INNER_PRODUCT)

    print(f"  Training IVF index (k-means, {n_clusters} clusters) …", flush=True)
    index.train(embeddings)                 # learns the cluster centroids
    index.add(embeddings)
    index.nprobe = n_probe                  # how many clusters to search at query time
    return index


def benchmark(index, embeddings: np.ndarray,
              query_idx: int = 0, k: int = 10, n_runs: int = 100) -> float:
    """
    Run n_runs queries and return average latency in ms.
    Uses a single query vector (simulates one user lookup).
    """
    query = embeddings[query_idx:query_idx + 1]   # (1, D)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        index.search(query, k)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


def recall_at_k(flat_index, approx_index,
                embeddings: np.ndarray,
                k: int = 10, n_queries: int = 200) -> float:
    """
    Recall@K: fraction of true top-K (from brute force) that ANN also returns.
    This is the standard way to measure ANN index quality.
    """
    queries = embeddings[:n_queries]
    _, true_ids  = flat_index.search(queries, k)
    _, approx_ids = approx_index.search(queries, k)

    hits = 0
    for t, a in zip(true_ids, approx_ids):
        hits += len(set(t) & set(a))
    return hits / (n_queries * k)


def main():
    print("Loading item embeddings …", flush=True)
    embeddings = np.load(os.path.join(EMB_DIR, "item_embeddings.npy"))
    print(f"  Shape: {embeddings.shape}  dtype: {embeddings.dtype}", flush=True)

    # FAISS requires float32 and C-contiguous layout
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    N, D = embeddings.shape
    n_clusters = max(10, int(np.sqrt(N)))   # rule of thumb: sqrt(N)
    n_probe    = max(1,  int(np.sqrt(n_clusters)))

    print(f"  N={N}  D={D}  clusters={n_clusters}  probe={n_probe}", flush=True)

    # ── Build indexes ─────────────────────────────────────────────────────────
    print("\nBuilding IndexFlatIP (brute-force) …", flush=True)
    flat_index = build_flat_index(embeddings)
    print(f"  Vectors indexed: {flat_index.ntotal}", flush=True)

    print("\nBuilding IndexIVFFlat (approximate) …", flush=True)
    ivf_index = build_ivf_index(embeddings, n_clusters, n_probe)
    print(f"  Vectors indexed: {ivf_index.ntotal}", flush=True)

    # ── Benchmark ─────────────────────────────────────────────────────────────
    print("\nBenchmarking query latency (100 queries each) …", flush=True)
    flat_ms = benchmark(flat_index, embeddings, k=500)
    ivf_ms  = benchmark(ivf_index,  embeddings, k=500)
    speedup = flat_ms / ivf_ms if ivf_ms > 0 else 0

    print(f"  IndexFlatIP  : {flat_ms:.3f} ms/query")
    print(f"  IndexIVFFlat : {ivf_ms:.3f} ms/query  ({speedup:.1f}× speedup)")

    # ── Recall@10 comparison ──────────────────────────────────────────────────
    rec = recall_at_k(flat_index, ivf_index, embeddings, k=10, n_queries=200)
    print(f"\n  ANN Recall@10 vs brute-force: {rec:.4f} ({rec*100:.1f}%)")
    print("  (1.0 = ANN returns exactly the same results as brute force)")

    # ── Save indexes ─────────────────────────────────────────────────────────
    flat_path = os.path.join(INDEX_DIR, "faiss_flat.index")
    ivf_path  = os.path.join(INDEX_DIR, "faiss_ivf.index")
    faiss.write_index(flat_index, flat_path)
    faiss.write_index(ivf_index,  ivf_path)

    # save index metadata for the API to load correct index
    meta = {
        "n_items"   : N,
        "dim"       : D,
        "n_clusters": n_clusters,
        "n_probe"   : n_probe,
        "flat_path" : flat_path,
        "ivf_path"  : ivf_path,
    }
    with open(os.path.join(INDEX_DIR, "faiss_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ Saved faiss_flat.index  ({os.path.getsize(flat_path)//1024} KB)")
    print(f"✓ Saved faiss_ivf.index   ({os.path.getsize(ivf_path)//1024} KB)")

    # ── Sanity check: top-10 for item 0 ──────────────────────────────────────
    D_scores, I_indices = flat_index.search(embeddings[0:1], 11)
    with open(os.path.join(EMB_DIR, "item_meta.csv")) as f:
        import csv
        reader = csv.DictReader(f)
        idx_to_title = {int(r["movie_idx"]): r["title"] for r in reader}

    print(f"\nTop-10 neighbours of '{idx_to_title.get(0, 'item_0')}':")
    for rank, (score, idx) in enumerate(zip(D_scores[0][1:], I_indices[0][1:]), 1):
        print(f"  {rank:2d}. {idx_to_title.get(idx, '?'):<45s}  sim={score:.4f}")


if __name__ == "__main__":
    main()
