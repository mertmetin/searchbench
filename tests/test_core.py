import math

import pandas as pd
import pytest

from prodsearch.data import doc_text
from prodsearch.evaluate import mrr_exact, ndcg_at_k, precision_exact_at_k
from prodsearch.retrievers import FieldsBM25, TitleBM25, rrf


def _products():
    rows = [
        (0, "solid wood platform bed", "Beds", "Furniture / Bedroom / Beds", "acacia frame, sturdy", "material: wood"),
        (1, "velvet accent chair", "Accent Chairs", "Furniture / Living Room / Chairs", "soft and cozy armchair", ""),
        (2, "outdoor patio sofa", "Sofas", "Outdoor / Patio Furniture", "weather resistant wicker", "seats: 3"),
    ]
    df = pd.DataFrame(
        rows,
        columns=["product_id", "product_name", "product_class", "category", "product_description", "product_features"],
    )
    return df.set_index("product_id", drop=False)


# --------------------------------------------------------------------------- metrics


def test_ndcg_perfect_and_reversed():
    qrel = {1: 1.0, 2: 0.5, 3: 0.0}
    assert ndcg_at_k([1, 2, 3], qrel) == pytest.approx(1.0)
    worse = ndcg_at_k([3, 2, 1], qrel)
    expected = (0.5 / math.log2(3) + 1.0 / math.log2(4)) / (1.0 + 0.5 / math.log2(3))
    assert worse == pytest.approx(expected)


def test_unlabelled_counts_as_irrelevant_and_empty_qrel_is_zero():
    assert ndcg_at_k([99, 1], {1: 1.0}) == pytest.approx(1 / math.log2(3))
    assert ndcg_at_k([1, 2], {}) == 0.0


def test_precision_and_mrr_count_exact_only():
    qrel = {1: 0.5, 2: 1.0}
    assert precision_exact_at_k([1, 2, 3], qrel, k=2) == 0.5
    assert mrr_exact([1, 2, 3], qrel) == 0.5
    assert mrr_exact([1], qrel) == 0.0


# --------------------------------------------------------------------------- retrieval


def test_rrf_rewards_agreement():
    fused = rrf([(1, 9.0), (2, 8.0)], [(2, 0.9), (3, 0.8)])
    assert [pid for pid, _ in fused][0] == 2


def test_title_bm25_only_sees_titles():
    r = TitleBM25(_products())
    assert [pid for pid, _ in r.search("chair")] == [1]
    assert r.search("armchair") == []  # only in the description


def test_fields_bm25_uses_description_and_stems():
    r = FieldsBM25(_products())
    assert [pid for pid, _ in r.search("armchairs")][0] == 1
    assert [pid for pid, _ in r.search("patio")][0] == 2


def test_stopword_only_query_returns_nothing():
    assert TitleBM25(_products()).search("the and of") == []


def test_doc_text_includes_every_field():
    text = doc_text(_products().loc[0])
    for part in ("solid wood platform bed", "Beds", "Furniture / Bedroom / Beds", "acacia", "material: wood"):
        assert part in text
