#!/bin/bash
# Run the full post-training pipeline in order.
# Prerequisites: training complete (ml/models/two_tower.pt must exist)

set -e  # exit on any error
cd "$(dirname "$0")/.."

source venv/bin/activate

echo "═══════════════════════════════════════════"
echo " RecomAI — Post-Training Pipeline"
echo "═══════════════════════════════════════════"

echo ""
echo "Step 1/4 — Generate item embeddings"
PYTHONUNBUFFERED=1 python ml/scripts/generate_embeddings.py

echo ""
echo "Step 2/4 — Build FAISS index"
PYTHONUNBUFFERED=1 python ml/scripts/build_faiss_index.py

echo ""
echo "Step 3/4 — Train LightGBM ranker"
PYTHONUNBUFFERED=1 python ml/scripts/train_ranker.py

echo ""
echo "Step 4/4 — Evaluate (Recall@K, NDCG@K)"
PYTHONUNBUFFERED=1 python ml/scripts/evaluate.py

echo ""
echo "═══════════════════════════════════════════"
echo " ✓ Pipeline complete. Start the system:"
echo ""
echo "   Terminal 1: uvicorn backend.main:app --reload"
echo "   Terminal 2: cd frontend && npm run dev"
echo "═══════════════════════════════════════════"
