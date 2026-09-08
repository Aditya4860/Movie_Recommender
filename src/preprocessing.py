"""Reusable preprocessing pipeline for the CineMatch recommendation engine.

This module is responsible for ALL feature engineering:
  - loading / merging the TMDB movies and credits datasets
  - parsing JSON-like metadata columns (genres, keywords, cast, crew)
  - extracting structured fields (genre names, keywords, top cast, director)
  - handling missing values
  - constructing the final weighted 'tags' string used by vectorizers

recommender.py imports build_feature_frame() from here and has NO
direct knowledge of the dataset schema.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data import load_base_datasets, parse_json_list


# ---------------------------------------------------------------------------
# Default per-field weights.
# A weight of N causes the token to be repeated N times in the tags string,
# giving it proportionally more influence in the bag-of-words / TF-IDF space.
#
# Rationale:
#   - overview text is long (~30–60 words) and full of common verbs/nouns.
#     Keeping its weight at 1 avoids it overwhelming the structured signals.
#   - genres, keywords, cast, director are short but highly discriminative.
#     Boosting them ensures the structured metadata drives similarity.
# These defaults will be systematically evaluated and tuned in the next phase.
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS: dict[str, int] = {
    "overview": 1,
    "genres": 3,
    "keywords": 3,
    "cast": 2,
    "director": 3,
}


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _extract_names(value: object, limit: int | None = None) -> list[str]:
    """Extract 'name' fields from a JSON-list column value.

    Spaces are removed from each name so that multi-word names (e.g.
    'Science Fiction', 'Christopher Nolan') become a single token and are
    not fragmented by the vectorizer.
    """
    items = parse_json_list(value)
    names = [item.get("name", "") for item in items if isinstance(item, dict)]
    names = [n.replace(" ", "") for n in names if n]
    return names[:limit] if limit is not None else names


def _extract_director(crew_value: object) -> list[str]:
    """Return a one-element list containing the director's name, or []."""
    for person in parse_json_list(crew_value):
        if isinstance(person, dict) and person.get("job") == "Director":
            return [str(person.get("name", "")).replace(" ", "")]
    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_feature_frame(
    data_dir: str | Path = "data",
    weights: dict[str, int] | None = None,
    cast_limit: int = 3,
) -> pd.DataFrame:
    """Load, merge, and preprocess the TMDB datasets.

    Returns a DataFrame with columns:
        id, title, genres (list[str]), tags (str)

    Parameters
    ----------
    data_dir:
        Directory containing the TMDB CSV files.
    weights:
        Per-field repetition weights.  Keys: overview, genres, keywords,
        cast, director.  Missing keys fall back to DEFAULT_WEIGHTS.
    cast_limit:
        Maximum number of cast members to include.  Defaults to 3.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    movies, credits = load_base_datasets(data_dir)

    # ---- validate required columns ----------------------------------------
    required_movies = {"id", "title", "overview", "genres", "keywords"}
    required_credits = {"title", "cast", "crew"}
    missing_m = required_movies - set(movies.columns)
    missing_c = required_credits - set(credits.columns)
    if missing_m or missing_c:
        raise ValueError(
            "Dataset columns are incomplete. "
            f"movies missing: {sorted(missing_m)}; "
            f"credits missing: {sorted(missing_c)}"
        )

    # ---- merge on title (inner join to keep only matched rows) -------------
    credits = credits[["title", "cast", "crew"]].copy()
    merged = movies.merge(credits, on="title", how="inner")
    frame = merged[
        ["id", "title", "overview", "genres", "keywords", "cast", "crew"]
    ].copy()

    # ---- fill missing values -----------------------------------------------
    frame = frame.dropna(subset=["title"])
    frame["overview"] = frame["overview"].fillna("").astype(str)
    frame["genres"] = frame["genres"].fillna("[]")
    frame["keywords"] = frame["keywords"].fillna("[]")
    frame["cast"] = frame["cast"].fillna("[]")
    frame["crew"] = frame["crew"].fillna("[]")

    # ---- extract structured features ---------------------------------------
    frame["genres_tokens"] = frame["genres"].map(_extract_names)
    frame["keywords_tokens"] = frame["keywords"].map(_extract_names)
    frame["cast_tokens"] = frame["cast"].map(
        lambda v: _extract_names(v, limit=cast_limit)
    )
    frame["director_tokens"] = frame["crew"].map(_extract_director)
    frame["overview_tokens"] = frame["overview"].map(str.split)

    # ---- construct weighted tags string ------------------------------------
    def _make_tags(row: pd.Series) -> str:
        parts: list[str] = []
        parts += row["overview_tokens"] * w["overview"]
        parts += row["genres_tokens"] * w["genres"]
        parts += row["keywords_tokens"] * w["keywords"]
        parts += row["cast_tokens"] * w["cast"]
        parts += row["director_tokens"] * w["director"]
        return " ".join(parts).lower()

    frame["tags"] = frame.apply(_make_tags, axis=1)

    # ---- deduplicate and reset index --------------------------------------
    frame = frame.drop_duplicates(subset="title").reset_index(drop=True)

    # Return only the columns needed by the recommender / display layer
    return frame[["id", "title", "genres_tokens", "tags"]].rename(
        columns={"genres_tokens": "genres"}
    )
