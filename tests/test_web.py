"""End-to-end API tests. They build the real indexes, so they need the WANDS
data (`searchbench download`) and take ~30 s on first run."""

from pathlib import Path

import pytest

if not Path("data/wands/product.csv").exists():
    pytest.skip("WANDS data not downloaded", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

from searchbench.web import app  # noqa: E402

client = TestClient(app)


def test_search_with_filters():
    r = client.get("/api/search", params={"q": "hardwood beds", "product_class": "Beds", "min_rating": 4})
    assert r.status_code == 200
    data = r.json()
    assert data["labelled"] is True
    assert data["results"] and all(x["product_class"] == "Beds" and x["rating"] >= 4 for x in data["results"])
    assert data["facets"][0]["value"] == "Beds"


def test_compare_labelled_query_scores_both_columns():
    data = client.get("/api/compare", params={"q": "hardwood beds"}).json()
    old, new = data["title_bm25_2014"], data["hybrid"]
    assert new["ndcg@10"] > old["ndcg@10"]
    assert all(x["label"] in {"Exact", "Partial", "Irrelevant", "Unlabelled"} for x in new["results"])


def test_unlabelled_query_has_no_labels():
    data = client.get("/api/search", params={"q": "comfy chair for a small apartment"}).json()
    assert data["labelled"] is False and data["results"]
    assert all(x["label"] is None for x in data["results"])


def test_bad_mode_is_rejected():
    assert client.get("/api/search", params={"q": "sofa", "mode": "nope"}).status_code == 400
