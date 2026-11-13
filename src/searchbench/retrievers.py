"""Retrievers, from the 2014 approach to a 2026 hybrid with reranking.

Every retriever returns a ranked list of (product_id, score) for a query.

* TitleBM25     the 2014 setup: the Haystack index template held only the
                product title (and URL), searched by Elasticsearch, i.e. BM25
* FieldsBM25    BM25 over name, class, category, description and features
* Dense         sentence embeddings (bge-small-en-v1.5), cosine similarity
* Hybrid        reciprocal rank fusion of FieldsBM25 and Dense
* Reranked      a cross-encoder rescores the hybrid top-k
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .data import doc_text

CACHE = Path(".cache")
DENSE_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

Hits = list[tuple[int, float]]


class _BM25:
    def __init__(self, ids: np.ndarray, texts: list[str]):
        import bm25s
        import Stemmer

        self._bm25s = bm25s
        self.stemmer = Stemmer.Stemmer("english")
        self.ids = ids
        self.index = bm25s.BM25()
        self.index.index(
            bm25s.tokenize(texts, stopwords="en", stemmer=self.stemmer, show_progress=False), show_progress=False
        )

    def search(self, query: str, k: int = 100) -> Hits:
        tokens = self._bm25s.tokenize([query], stopwords="en", stemmer=self.stemmer, show_progress=False)
        if not tokens.vocab:
            return []
        k = min(k, len(self.ids))
        docs, scores = self.index.retrieve(tokens, k=k, show_progress=False)
        return [(int(self.ids[d]), float(s)) for d, s in zip(docs[0], scores[0], strict=True) if s > 0]


class TitleBM25(_BM25):
    name = "2014: BM25 on title only"

    def __init__(self, products: pd.DataFrame):
        super().__init__(products["product_id"].to_numpy(), products["product_name"].tolist())


class FieldsBM25(_BM25):
    name = "BM25 on all fields"

    def __init__(self, products: pd.DataFrame):
        # The name is repeated so it counts for more than a long description.
        texts = [
            f"{n} {n} {c} {cat} {d} {f}"
            for n, c, cat, d, f in zip(
                products["product_name"],
                products["product_class"],
                products["category"],
                products["product_description"],
                products["product_features"].str.replace("|", " "),
                strict=True,
            )
        ]
        super().__init__(products["product_id"].to_numpy(), texts)


class Dense:
    name = "Semantic: sentence embeddings"

    def __init__(self, products: pd.DataFrame, model_name: str = DENSE_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.ids = products["product_id"].to_numpy()
        texts = [doc_text(p) for _, p in products.iterrows()]
        key = hashlib.sha256(("\n".join(texts) + model_name).encode()).hexdigest()[:16]
        path = CACHE / f"dense_{key}.npy"
        if path.exists():
            self.vecs = np.load(path)
        else:
            self.vecs = self.model.encode(
                texts, batch_size=128, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True
            ).astype(np.float32)
            path.parent.mkdir(exist_ok=True)
            np.save(path, self.vecs)

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode([QUERY_PREFIX + query], normalize_embeddings=True, convert_to_numpy=True)[0]

    def search(self, query: str, k: int = 100) -> Hits:
        sims = self.vecs @ self.encode_query(query)
        top = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
        top = top[np.argsort(-sims[top])]
        return [(int(self.ids[i]), float(sims[i])) for i in top]


def rrf(*rankings: Hits, k: int = 60, depth: int = 100) -> Hits:
    """Reciprocal rank fusion: score = sum over rankings of 1 / (k + rank)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, (pid, _) in enumerate(ranking[:depth], start=1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


class Hybrid:
    name = "Hybrid: BM25 + semantic (RRF)"

    def __init__(self, bm25: FieldsBM25, dense: Dense):
        self.bm25, self.dense = bm25, dense

    def search(self, query: str, k: int = 100) -> Hits:
        return rrf(self.bm25.search(query, 100), self.dense.search(query, 100))[:k]


class Reranked:
    name = "Hybrid + cross-encoder rerank"

    def __init__(self, first_stage, products: pd.DataFrame, top_k: int = 50, model_name: str = RERANK_MODEL):
        from sentence_transformers import CrossEncoder

        self.first, self.top_k = first_stage, top_k
        self.model = CrossEncoder(model_name)
        self.texts = {int(pid): doc_text(p, description_chars=300) for pid, p in products.iterrows()}

    def search(self, query: str, k: int = 100) -> Hits:
        cands = self.first.search(query, max(k, self.top_k))
        head, tail = cands[: self.top_k], cands[self.top_k :]
        if not head:
            return []
        scores = self.model.predict([(query, self.texts[pid]) for pid, _ in head], show_progress_bar=False)
        head = sorted(zip([pid for pid, _ in head], map(float, scores), strict=True), key=lambda kv: -kv[1])
        return (head + tail)[:k]
