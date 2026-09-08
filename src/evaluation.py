"""Evaluation metrics for CineMatch content-based recommendation models.

Implements information-retrieval style metrics appropriate for a top-N
content-based recommender where no explicit user-interaction log exists.

Methodology
-----------
Because this project has no real user-interaction dataset, we use a
manually curated ground-truth set (data/evaluation_set.json).  For each
seed movie a human curator listed films that a domain expert would
consider genuinely similar.  These are used as the positive set.

Metrics
-------
Precision@K  — fraction of the top-K recommendations that appear in the
               relevant set for the seed movie.

Recall@K     — fraction of the relevant set that appears in the top-K
               recommendations.

Latency      — wall-clock time (seconds) to call recommender.recommend()
               for a single query, measured with time.perf_counter.

All functions accept plain Python collections and are fully testable
without a running recommender instance.
"""

from __future__ import annotations

import time
from typing import Callable, Sequence


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def precision_at_k(
    recommended: Sequence[int],
    relevant: Sequence[int],
    k: int,
) -> float:
    """Precision@K: fraction of top-K recommendations that are relevant.

    Parameters
    ----------
    recommended:
        Ordered list of recommended movie IDs (best first).
    relevant:
        Collection of IDs considered relevant for the query.
    k:
        Cut-off depth.

    Returns
    -------
    float in [0.0, 1.0].  Returns 0.0 when k <= 0 or recommended is empty.
    """
    if k <= 0 or len(recommended) == 0:
        return 0.0
    if len(relevant) == 0:
        return 0.0

    top_k = list(recommended)[:k]
    relevant_set = set(relevant)
    hits = sum(1 for movie_id in top_k if movie_id in relevant_set)
    return hits / min(k, len(top_k))


def recall_at_k(
    recommended: Sequence[int],
    relevant: Sequence[int],
    k: int,
) -> float:
    """Recall@K: fraction of relevant items that appear in the top-K.

    Parameters
    ----------
    recommended:
        Ordered list of recommended movie IDs (best first).
    relevant:
        Collection of IDs considered relevant for the query.
    k:
        Cut-off depth.

    Returns
    -------
    float in [0.0, 1.0].  Returns 0.0 when relevant is empty.
    """
    if len(relevant) == 0:
        return 0.0
    if k <= 0 or len(recommended) == 0:
        return 0.0

    top_k = set(list(recommended)[:k])
    relevant_set = set(relevant)
    hits = len(top_k & relevant_set)
    return hits / len(relevant_set)


def measure_latency(
    fn: Callable[[], object],
    runs: int = 3,
) -> float:
    """Return the average wall-clock seconds for ``fn()`` over ``runs`` calls.

    Parameters
    ----------
    fn:
        Zero-argument callable to time (e.g. ``lambda: recommender.recommend(id, 10)``).
    runs:
        Number of repetitions to average over.  Defaults to 3.

    Returns
    -------
    Average elapsed seconds as a float.
    """
    if runs < 1:
        raise ValueError("runs must be >= 1")
    total = 0.0
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        total += time.perf_counter() - t0
    return total / runs


# ---------------------------------------------------------------------------
# Aggregate evaluation helper
# ---------------------------------------------------------------------------

def evaluate_recommender(
    recommender,
    evaluation_set: list[dict],
    k_values: list[int] | None = None,
    latency_runs: int = 3,
) -> dict:
    """Run the full evaluation suite against a curated ground-truth set.

    Parameters
    ----------
    recommender:
        A fitted ``MovieRecommender`` instance.
    evaluation_set:
        List of dicts loaded from ``data/evaluation_set.json``.
        Each dict must have:
          - ``seed_id``   (int)   — TMDB ID of the query movie
          - ``seed_title`` (str)  — human-readable title (for display only)
          - ``relevant_ids`` (list[int]) — IDs of genuinely similar movies
    k_values:
        List of cut-off depths to evaluate.  Defaults to [5, 10].
    latency_runs:
        Number of timing repetitions per seed.

    Returns
    -------
    dict with keys:
        ``results``  — list of per-seed result dicts
        ``summary``  — macro-averaged metrics across all seeds
        ``skipped``  — list of seed titles not found in the recommender corpus
    """
    if k_values is None:
        k_values = [5, 10]

    results = []
    skipped = []

    for entry in evaluation_set:
        seed_id: int = int(entry["seed_id"])
        seed_title: str = entry["seed_title"]
        relevant_ids: list[int] = [int(x) for x in entry["relevant_ids"]]

        # Skip if this seed is not in the recommender corpus (e.g. data mismatch)
        corpus_ids = set(recommender.movies["id"].tolist())
        if seed_id not in corpus_ids:
            skipped.append(seed_title)
            continue

        max_k = max(k_values)
        latency = measure_latency(
            lambda sid=seed_id, mk=max_k: recommender.recommend(sid, mk),
            runs=latency_runs,
        )
        recommended_df = recommender.recommend(seed_id, max_k)
        recommended_ids = recommended_df["id"].tolist()

        row: dict = {
            "seed_id": seed_id,
            "seed_title": seed_title,
            "relevant_count": len(relevant_ids),
            "latency_s": round(latency, 4),
        }
        for k in k_values:
            row[f"precision@{k}"] = round(precision_at_k(recommended_ids, relevant_ids, k), 4)
            row[f"recall@{k}"] = round(recall_at_k(recommended_ids, relevant_ids, k), 4)

        results.append(row)

    # Macro-average summary
    summary: dict = {"n_evaluated": len(results), "n_skipped": len(skipped)}
    if results:
        summary["avg_latency_s"] = round(
            sum(r["latency_s"] for r in results) / len(results), 4
        )
        for k in k_values:
            summary[f"avg_precision@{k}"] = round(
                sum(r[f"precision@{k}"] for r in results) / len(results), 4
            )
            summary[f"avg_recall@{k}"] = round(
                sum(r[f"recall@{k}"] for r in results) / len(results), 4
            )

    return {"results": results, "summary": summary, "skipped": skipped}
