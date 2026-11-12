"""FastAPI app: search, facets, and a side-by-side 2014 vs 2026 comparison."""

from __future__ import annotations

import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .data import Wands
from .evaluate import RESULTS_DIR, ndcg_at_k

STATIC = Path(__file__).resolve().parents[2] / "app" / "static"
MODES = {
    "title_bm25_2014": "2014: keyword search on titles",
    "fields_bm25": "Keyword search (BM25)",
    "dense": "Semantic search",
    "hybrid": "Hybrid (best)",
    "hybrid_rerank": "Hybrid + rerank",
}

app = FastAPI(title="Product search, 2014 vs 2026")


@lru_cache(maxsize=1)
def _state():
    from .cli import build_all

    w = Wands()
    retrievers = build_all(w.products)
    query_ids = {q.strip().lower(): int(i) for i, q in zip(w.queries["query_id"], w.queries["query"], strict=True)}
    return w, retrievers, query_ids


def _card(p, qrel: dict[int, float] | None) -> dict:
    rating = p["average_rating"]
    label = None
    if qrel is not None:
        g = qrel.get(int(p["product_id"]))
        label = (
            {1.0: "Exact", 0.5: "Partial", 0.0: "Irrelevant"}.get(g, "Unlabelled") if g is not None else "Unlabelled"
        )
    return {
        "id": int(p["product_id"]),
        "name": p["product_name"],
        "product_class": p["product_class"],
        "category": p["category"].split(" / ")[-1] if p["category"] else "",
        "rating": None if rating != rating else round(float(rating), 1),  # NaN check
        "reviews": int(p["review_count"]),
        "snippet": re.sub(r"\s+([,.!?’'])", r"\1", p["product_description"][:180]).strip(),
        "label": label,
    }


def _run(mode: str, q: str, k: int = 200):
    w, retrievers, query_ids = _state()
    if mode not in retrievers:
        raise HTTPException(400, f"unknown mode {mode}")
    hits = retrievers[mode].search(q, k)
    qid = query_ids.get(q.strip().lower())
    qrel = w.qrels.get(qid) if qid is not None else None
    return w, hits, qrel


@app.get("/api/search")
def search(
    q: str = Query(..., min_length=1, max_length=200),
    mode: str = "hybrid",
    product_class: str | None = None,
    min_rating: float = 0.0,
    limit: int = Query(20, le=50),
):
    w, hits, qrel = _run(mode, q)
    rows = [w.products.loc[pid] for pid, _ in hits]
    facets = Counter(r["product_class"] for r in rows if r["product_class"]).most_common(12)
    if product_class:
        rows = [r for r in rows if r["product_class"] == product_class]
    if min_rating:
        rows = [r for r in rows if r["average_rating"] == r["average_rating"] and r["average_rating"] >= min_rating]
    return {
        "query": q,
        "mode": mode,
        "labelled": qrel is not None,
        "facets": [{"value": v, "count": c} for v, c in facets],
        "results": [_card(r, qrel) for r in rows[:limit]],
    }


@app.get("/api/compare")
def compare(q: str = Query(..., min_length=1, max_length=200), limit: int = Query(10, le=20)):
    out = {}
    for mode in ("title_bm25_2014", "hybrid"):
        w, hits, qrel = _run(mode, q, 100)
        ranked = [pid for pid, _ in hits]
        out[mode] = {
            "label": MODES[mode],
            "ndcg@10": round(ndcg_at_k(ranked, qrel), 3) if qrel else None,
            "results": [_card(w.products.loc[pid], qrel) for pid in ranked[:limit]],
        }
    return {"query": q, "labelled": qrel is not None, **out}


@app.get("/api/meta")
def meta():
    """Modes, overall scores and example queries where the 2026 stack wins most."""
    path = RESULTS_DIR / "results.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    scores = {k: round(v["ndcg@10"], 3) for k, v in results.items()}
    examples = []
    if "title_bm25_2014" in results and "hybrid" in results:
        old, new = results["title_bm25_2014"]["per_query_ndcg"], results["hybrid"]["per_query_ndcg"]
        gains = sorted(((new[q] - old[q], q) for q in new if q in old), reverse=True)[:8]
        w, _, _ = _state()
        examples = [w.queries.loc[int(q), "query"] for _, q in gains]
    return {"modes": MODES, "ndcg": scores, "examples": examples}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
