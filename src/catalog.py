"""Cached-friendly metadata helpers for the CineMatch Streamlit interface."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.recommender import MOVIE_FILE_NAMES, _find_data_file, _parse_json_list


def _genre_names(value: object) -> list[str]:
    return [item["name"] for item in _parse_json_list(value) if item.get("name")]


def load_catalog(data_dir: str | Path = "data") -> pd.DataFrame:
    """Load display metadata from the TMDB movie CSV without model retraining."""
    movies = pd.read_csv(_find_data_file(data_dir, MOVIE_FILE_NAMES))
    catalog = movies[
        ["id", "title", "overview", "genres", "release_date", "vote_average", "vote_count", "popularity", "runtime"]
    ].copy()
    catalog["genres"] = catalog["genres"].map(_genre_names)
    catalog["genre_label"] = catalog["genres"].map(lambda items: " • ".join(items) if items else "Unclassified")
    catalog["release_year"] = pd.to_datetime(catalog["release_date"], errors="coerce").dt.year.astype("Int64")
    catalog["overview"] = catalog["overview"].fillna("No overview available.")
    catalog["vote_average"] = catalog["vote_average"].fillna(0.0)
    catalog["vote_count"] = catalog["vote_count"].fillna(0).astype(int)
    catalog["popularity"] = catalog["popularity"].fillna(0.0)
    catalog["runtime"] = catalog["runtime"].fillna(0).astype(int)
    return catalog.drop_duplicates(subset="title").reset_index(drop=True)


def top_movies(catalog: pd.DataFrame, count: int = 10) -> pd.DataFrame:
    """Rank titles with enough votes by a weighted rating score."""
    eligible = catalog[catalog["vote_count"] >= 100].copy()
    mean_rating = eligible["vote_average"].mean()
    vote_threshold = eligible["vote_count"].quantile(0.60)
    eligible["weighted_rating"] = (
        eligible["vote_count"] / (eligible["vote_count"] + vote_threshold) * eligible["vote_average"]
        + vote_threshold / (eligible["vote_count"] + vote_threshold) * mean_rating
    )
    return eligible.nlargest(count, "weighted_rating")


def genre_counts(catalog: pd.DataFrame) -> pd.DataFrame:
    """Return a chart-ready genre frequency table."""
    return (
        catalog.explode("genres")
        .dropna(subset=["genres"])
        .groupby("genres")
        .size()
        .sort_values(ascending=False)
        .head(12)
        .rename_axis("genre")
        .reset_index(name="movies")
    )
