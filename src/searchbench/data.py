"""WANDS: the Wayfair ANnotation DataSet (MIT licence).

42,994 products, 480 real search queries and 233,448 human relevance labels
(Exact / Partial / Irrelevant). https://github.com/wayfair/WANDS
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/wands")
BASE_URL = "https://raw.githubusercontent.com/wayfair/WANDS/main/dataset"
FILES = ("product", "query", "label")

# Graded relevance used by the WANDS paper.
GAIN = {"Exact": 1.0, "Partial": 0.5, "Irrelevant": 0.0}


def download(data_dir: Path = DATA_DIR) -> None:
    """Fetch the three tab-separated files (~96 MB)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = data_dir / f"{name}.csv"
        if not dest.exists():
            print(f"Downloading {name}.csv ...")
            urllib.request.urlretrieve(f"{BASE_URL}/{name}.csv", dest)


def _read(name: str, data_dir: Path) -> pd.DataFrame:
    path = data_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `searchbench download`")
    return pd.read_csv(path, sep="\t")


@dataclass
class Wands:
    data_dir: Path = DATA_DIR

    @cached_property
    def products(self) -> pd.DataFrame:
        df = _read("product", self.data_dir)
        df = df.rename(columns={"category hierarchy": "category"})
        for col in ("product_name", "product_class", "category", "product_description", "product_features"):
            df[col] = df[col].fillna("").astype(str)
        df["average_rating"] = pd.to_numeric(df["average_rating"], errors="coerce")
        df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce").fillna(0).astype(int)
        return df.set_index("product_id", drop=False).sort_index()

    @cached_property
    def queries(self) -> pd.DataFrame:
        return _read("query", self.data_dir).set_index("query_id", drop=False)

    @cached_property
    def labels(self) -> pd.DataFrame:
        df = _read("label", self.data_dir)
        df["gain"] = df["label"].map(GAIN).fillna(0.0)
        return df

    @cached_property
    def qrels(self) -> dict[int, dict[int, float]]:
        """query_id -> {product_id: gain} for every labelled pair."""
        out: dict[int, dict[int, float]] = {}
        for q, p, g in zip(self.labels["query_id"], self.labels["product_id"], self.labels["gain"], strict=True):
            out.setdefault(int(q), {})[int(p)] = float(g)
        return out


def doc_text(p: pd.Series | dict, description_chars: int = 400) -> str:
    """The text a 2026 retriever sees for one product."""
    desc = p["product_description"][:description_chars]
    feats = p["product_features"].replace("|", ". ")[:400]
    return f"{p['product_name']}. {p['product_class']}. {p['category']}. {desc} {feats}".strip()
