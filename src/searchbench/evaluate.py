"""Offline evaluation against the WANDS human labels.

Metrics, averaged over the 480 queries:
* NDCG@10    graded relevance (Exact = 1, Partial = 0.5), the headline metric
* Precision@10 counting Exact matches only: "how many of the first page are right"
* MRR        reciprocal rank of the first Exact match

Unlabelled products count as irrelevant. WANDS labels were collected from
pooled results of several systems, so a retriever that surfaces many
unlabelled products is slightly penalised; the report says how many top-10
results were unlabelled so this is visible.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

RESULTS_DIR = Path("results")


def ndcg_at_k(ranked: list[int], qrel: dict[int, float], k: int = 10) -> float:
    dcg = sum(qrel.get(pid, 0.0) / math.log2(i + 2) for i, pid in enumerate(ranked[:k]))
    ideal = sorted(qrel.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def precision_exact_at_k(ranked: list[int], qrel: dict[int, float], k: int = 10) -> float:
    return sum(qrel.get(pid, 0.0) == 1.0 for pid in ranked[:k]) / k


def mrr_exact(ranked: list[int], qrel: dict[int, float]) -> float:
    for i, pid in enumerate(ranked, start=1):
        if qrel.get(pid, 0.0) == 1.0:
            return 1.0 / i
    return 0.0


def evaluate(retriever, queries, qrels: dict[int, dict[int, float]], k: int = 10) -> dict:
    ndcg, p_at_k, mrr, unlabelled, per_query = [], [], [], [], {}
    t0 = time.time()
    for qid, text in zip(queries["query_id"], queries["query"], strict=True):
        qrel = qrels.get(int(qid), {})
        ranked = [pid for pid, _ in retriever.search(text, 100)]
        n = ndcg_at_k(ranked, qrel, k)
        ndcg.append(n)
        p_at_k.append(precision_exact_at_k(ranked, qrel, k))
        mrr.append(mrr_exact(ranked, qrel))
        unlabelled.append(sum(pid not in qrel for pid in ranked[:k]) / k)
        per_query[int(qid)] = round(n, 4)
    elapsed = time.time() - t0
    return {
        "name": retriever.name,
        "ndcg@10": float(np.mean(ndcg)),
        "precision_exact@10": float(np.mean(p_at_k)),
        "mrr_exact": float(np.mean(mrr)),
        "unlabelled@10": float(np.mean(unlabelled)),
        "ms_per_query": 1000 * elapsed / len(ndcg),
        "per_query_ndcg": per_query,
    }


def save(results: dict[str, dict], path: Path = RESULTS_DIR / "results.json") -> None:
    path.parent.mkdir(exist_ok=True)
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing.update(results)
    path.write_text(json.dumps(existing, indent=1))
