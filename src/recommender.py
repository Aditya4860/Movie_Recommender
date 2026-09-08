"""Content-based movie recommendation engine for the TMDB 5000 dataset.

This module is responsible ONLY for model logic:
  - selecting a vectorization method (CountVectorizer or TfidfVectorizer)
  - fitting on the pre-processed tags produced by src.preprocessing
  - computing cosine similarity
  - returning ranked recommendations

All feature engineering lives in src/preprocessing.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import DEFAULT_WEIGHTS, build_feature_frame


# Supported vectorization methods
VectorizerMethod = Literal["count", "tfidf"]

# Maximum vocabulary size shared across both methods for consistency
MAX_FEATURES = 5000


def _build_vectorizer(method: VectorizerMethod):
    """Return a configured, unfitted vectorizer for the requested method."""
    common_kwargs = dict(stop_words="english", max_features=MAX_FEATURES)
    if method == "tfidf":
        return TfidfVectorizer(**common_kwargs)
    # Default / "count"
    return CountVectorizer(**common_kwargs)


@dataclass
class MovieRecommender:
    """Fit and query a cosine-similarity recommender.

    Attributes
    ----------
    movies:
        DataFrame with columns [id, title, genres, tags] — one row per movie.
    similarity:
        Square float32 cosine-similarity matrix (n_movies × n_movies).
    method:
        The vectorization method used to build this instance.
    """

    movies: pd.DataFrame
    similarity: np.ndarray
    method: VectorizerMethod = "count"

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(
        cls,
        data_dir: str | Path = "data",
        method: VectorizerMethod = "count",
        weights: dict[str, int] | None = None,
    ) -> "MovieRecommender":
        """Build a recommender from the TMDB CSV files.

        Parameters
        ----------
        data_dir:
            Directory containing the TMDB CSV files.
        method:
            Vectorization method: ``"count"`` (CountVectorizer, default)
            or ``"tfidf"`` (TfidfVectorizer).
        weights:
            Per-field importance weights passed through to preprocessing.
            Keys: overview, genres, keywords, cast, director.
            See ``src.preprocessing.DEFAULT_WEIGHTS`` for defaults.
        """
        frame = build_feature_frame(data_dir=data_dir, weights=weights)
        return cls._fit(frame, method=method)

    @classmethod
    def from_frames(
        cls,
        movies: pd.DataFrame,
        credits: pd.DataFrame,
        method: VectorizerMethod = "count",
        weights: dict[str, int] | None = None,
    ) -> "MovieRecommender":
        """Build a recommender from already-loaded DataFrames (mainly for tests)."""
        # Delegate to build_feature_frame via a temp path workaround is not
        # possible without writing files.  Instead we replicate the minimal
        # validation check here and call the private fit helper directly after
        # constructing the feature frame inline.
        from src.preprocessing import (
            DEFAULT_WEIGHTS,
            _extract_director,
            _extract_names,
        )

        w = {**DEFAULT_WEIGHTS, **(weights or {})}

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

        credits = credits[["title", "cast", "crew"]].copy()
        merged = movies.merge(credits, on="title", how="inner")
        frame = merged[
            ["id", "title", "overview", "genres", "keywords", "cast", "crew"]
        ].copy()
        frame = frame.dropna(subset=["title"])
        frame["overview"] = frame["overview"].fillna("").astype(str)
        for col in ["genres", "keywords", "cast", "crew"]:
            frame[col] = frame[col].fillna("[]")

        frame["genres_tokens"] = frame["genres"].map(_extract_names)
        frame["keywords_tokens"] = frame["keywords"].map(_extract_names)
        frame["cast_tokens"] = frame["cast"].map(
            lambda v: _extract_names(v, limit=3)
        )
        frame["director_tokens"] = frame["crew"].map(_extract_director)
        frame["overview_tokens"] = frame["overview"].map(str.split)

        def _make_tags(row: pd.Series) -> str:
            parts: list[str] = []
            parts += row["overview_tokens"] * w["overview"]
            parts += row["genres_tokens"] * w["genres"]
            parts += row["keywords_tokens"] * w["keywords"]
            parts += row["cast_tokens"] * w["cast"]
            parts += row["director_tokens"] * w["director"]
            return " ".join(parts).lower()

        frame["tags"] = frame.apply(_make_tags, axis=1)
        frame = frame.drop_duplicates(subset="title").reset_index(drop=True)
        frame = frame[["id", "title", "genres_tokens", "tags"]].rename(
            columns={"genres_tokens": "genres"}
        )

        return cls._fit(frame, method=method)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @classmethod
    def _fit(
        cls,
        frame: pd.DataFrame,
        method: VectorizerMethod = "count",
    ) -> "MovieRecommender":
        """Vectorize ``frame['tags']`` and compute the similarity matrix."""
        vectorizer = _build_vectorizer(method)
        vectors = vectorizer.fit_transform(frame["tags"])
        # Compute dense similarity; store as float32 to halve memory usage
        sim = cosine_similarity(vectors).astype(np.float32)
        return cls(movies=frame, similarity=sim, method=method)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def titles(self) -> list[str]:
        """Sorted list of all movie titles in the corpus."""
        return self.movies["title"].sort_values().tolist()

    def recommend(self, movie_id: int, count: int = 5) -> pd.DataFrame:
        """Return the top-N most similar movies for a given TMDB movie ID.

        Parameters
        ----------
        movie_id:
            The TMDB integer ID of the seed movie.
        count:
            Number of recommendations to return.

        Returns
        -------
        DataFrame with columns [id (int), similarity (float)].
        Titles can be joined from ``self.movies`` via the ``id`` column.
        """
        matches = self.movies.index[self.movies["id"] == movie_id].tolist()
        if not matches:
            raise ValueError(f"Movie ID '{movie_id}' was not found in the dataset.")

        idx = matches[0]
        row = self.similarity[idx]
        # argsort ascending, then reverse; skip position 0 (the movie itself)
        ranked_indices = np.argsort(row)[::-1][1 : count + 1]
        recommendations = [
            {
                "id": int(self.movies.iloc[i]["id"]),
                "similarity": round(float(row[i]) * 100, 1),
            }
            for i in ranked_indices
        ]
        return pd.DataFrame(recommendations)
