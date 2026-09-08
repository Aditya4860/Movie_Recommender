"""Unit tests for src/evaluation.py

Covers:
  - precision_at_k normal cases
  - recall_at_k normal cases
  - edge cases: empty lists, k=0, k > len(recommended), no overlap
  - invalid / boundary inputs
  - measure_latency basic sanity
"""

from __future__ import annotations

import time

import pytest

from src.evaluation import measure_latency, precision_at_k, recall_at_k


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------

class TestPrecisionAtK:

    def test_perfect_precision(self):
        rec = [1, 2, 3, 4, 5]
        rel = [1, 2, 3, 4, 5]
        assert precision_at_k(rec, rel, k=5) == 1.0

    def test_zero_precision_no_overlap(self):
        rec = [1, 2, 3]
        rel = [4, 5, 6]
        assert precision_at_k(rec, rel, k=3) == 0.0

    def test_partial_precision(self):
        rec = [1, 2, 3, 4, 5]
        rel = [1, 3]
        # 2 hits in top-5
        assert precision_at_k(rec, rel, k=5) == pytest.approx(2 / 5)

    def test_k_smaller_than_rec_list(self):
        rec = [1, 2, 3, 4, 5]
        rel = [1, 2, 3, 4, 5]
        # Only look at top-3; all 3 are relevant
        assert precision_at_k(rec, rel, k=3) == 1.0

    def test_k_larger_than_rec_list(self):
        rec = [1, 2]
        rel = [1, 2, 3, 4, 5]
        # 2 relevant in the 2 returned; precision = 2/2 = 1.0
        assert precision_at_k(rec, rel, k=10) == 1.0

    def test_k_zero_returns_zero(self):
        assert precision_at_k([1, 2, 3], [1, 2], k=0) == 0.0

    def test_empty_recommended_returns_zero(self):
        assert precision_at_k([], [1, 2, 3], k=5) == 0.0

    def test_empty_relevant_returns_zero(self):
        assert precision_at_k([1, 2, 3], [], k=5) == 0.0

    def test_both_empty_returns_zero(self):
        assert precision_at_k([], [], k=5) == 0.0

    def test_single_hit(self):
        rec = [99, 1, 2, 3, 4]
        rel = [99]
        assert precision_at_k(rec, rel, k=5) == pytest.approx(1 / 5)

    def test_hit_outside_k_window_not_counted(self):
        rec = [10, 20, 30, 1, 2]
        rel = [1, 2]
        # 1 and 2 are outside top-3 window
        assert precision_at_k(rec, rel, k=3) == 0.0

    def test_duplicate_recommended_ids(self):
        """Duplicates in recommended should still be evaluated by position."""
        rec = [1, 1, 1, 1, 1]
        rel = [1]
        # 5 positions all hit rel[0]; precision = 5/5 = 1.0
        assert precision_at_k(rec, rel, k=5) == 1.0

    def test_k_equals_one(self):
        rec = [7, 8, 9]
        rel = [7]
        assert precision_at_k(rec, rel, k=1) == 1.0

    def test_k_equals_one_miss(self):
        rec = [7, 8, 9]
        rel = [8]
        assert precision_at_k(rec, rel, k=1) == 0.0


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------

class TestRecallAtK:

    def test_perfect_recall(self):
        rec = [1, 2, 3]
        rel = [1, 2, 3]
        assert recall_at_k(rec, rel, k=3) == 1.0

    def test_zero_recall_no_overlap(self):
        rec = [1, 2, 3]
        rel = [4, 5, 6]
        assert recall_at_k(rec, rel, k=3) == 0.0

    def test_partial_recall(self):
        rec = [1, 2, 3, 4, 5]
        rel = [1, 3, 7, 8]
        # 2 of 4 relevant items in top-5
        assert recall_at_k(rec, rel, k=5) == pytest.approx(2 / 4)

    def test_k_smaller_misses_relevant(self):
        rec = [1, 2, 3, 4, 5]
        rel = [1, 5]
        # At k=3, only 1 is captured; 5 is at position 5
        assert recall_at_k(rec, rel, k=3) == pytest.approx(1 / 2)

    def test_k_zero_returns_zero(self):
        assert recall_at_k([1, 2, 3], [1, 2], k=0) == 0.0

    def test_empty_recommended_returns_zero(self):
        assert recall_at_k([], [1, 2, 3], k=5) == 0.0

    def test_empty_relevant_returns_zero(self):
        assert recall_at_k([1, 2, 3], [], k=5) == 0.0

    def test_both_empty_returns_zero(self):
        assert recall_at_k([], [], k=5) == 0.0

    def test_more_k_than_relevant(self):
        rec = [1, 2, 10, 11, 12]
        rel = [1, 2]
        assert recall_at_k(rec, rel, k=10) == 1.0

    def test_k_equals_one_hit(self):
        rec = [5, 6, 7]
        rel = [5, 6, 7]
        assert recall_at_k(rec, rel, k=1) == pytest.approx(1 / 3)

    def test_relevant_superset_of_recommended(self):
        rec = [1, 2]
        rel = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert recall_at_k(rec, rel, k=2) == pytest.approx(2 / 10)


# ---------------------------------------------------------------------------
# measure_latency
# ---------------------------------------------------------------------------

class TestMeasureLatency:

    def test_returns_float(self):
        result = measure_latency(lambda: time.sleep(0), runs=1)
        assert isinstance(result, float)

    def test_positive_latency(self):
        result = measure_latency(lambda: time.sleep(0.001), runs=2)
        assert result >= 0.0

    def test_invalid_runs_raises(self):
        with pytest.raises(ValueError):
            measure_latency(lambda: None, runs=0)

    def test_single_run(self):
        counter = {"n": 0}
        def fn():
            counter["n"] += 1
        measure_latency(fn, runs=1)
        assert counter["n"] == 1

    def test_multiple_runs_averaged(self):
        counter = {"n": 0}
        def fn():
            counter["n"] += 1
        measure_latency(fn, runs=5)
        assert counter["n"] == 5
