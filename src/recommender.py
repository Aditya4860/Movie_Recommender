"""Content-based movie recommendation engine for the TMDB 5000 dataset."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


MOVIE_FILE_NAMES = ("tmdb_5000_movies.csv", "movies.csv")
CREDITS_FILE_NAMES = ("tmdb_5000_credits.csv", "credits.csv")


def _find_data_file(data_dir: str | Path, candidates: Iterable[str]) -> Path:
    """Return the first matching dataset file or raise a helpful error."""
    folder = Path(data_dir)
    for name in candidates:
        path = folder / name
        if path.exists():
            return path
    choices = ", ".join(candidates)
    raise FileNotFoundError(
        f"Could not find a dataset in '{folder}'. Expected one of: {choices}."
    )


def _parse_json_list(value: object) -> list[dict]:
    """Safely parse TMDB's Python-list-like JSON columns."""
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def _names(value: object, limit: int | None = None) -> list[str]:
    items = _parse_json_list(value)
    names = [item.get("name", "") for item in items if isinstance(item, dict)]
    names = [name.replace(" ", "") for name in names if name]
    return names[:limit] if limit else names


def _director(value: object) -> list[str]:
    for person in _parse_json_list(value):
        if isinstance(person, dict) and person.get("job") == "Director":
            return [str(person.get("name", "")).replace(" ", "")]
    return []


@dataclass
class MovieRecommender:
    """Train and query a CountVectorizer + cosine-similarity recommender."""

    movies: pd.DataFrame
    similarity: object

    @classmethod
    def from_csv(cls, data_dir: str | Path = "data") -> "MovieRecommender":
        movie_path = _find_data_file(data_dir, MOVIE_FILE_NAMES)
        credits_path = _find_data_file(data_dir, CREDITS_FILE_NAMES)

        movies = pd.read_csv(movie_path)
        credits = pd.read_csv(credits_path)
        return cls.from_frames(movies, credits)

    @classmethod
    def from_frames(cls, movies: pd.DataFrame, credits: pd.DataFrame) -> "MovieRecommender":
        required_movies = {"id", "title", "overview", "genres", "keywords"}
        required_credits = {"title", "cast", "crew"}
        missing_movies = required_movies - set(movies.columns)
        missing_credits = required_credits - set(credits.columns)
        if missing_movies or missing_credits:
            raise ValueError(
                "Dataset columns are incomplete. "
                f"movies missing: {sorted(missing_movies)}; "
                f"credits missing: {sorted(missing_credits)}"
            )

        credits = credits[["title", "cast", "crew"]].copy()
        merged = movies.merge(credits, on="title", how="inner")
        frame = merged[["id", "title", "overview", "genres", "keywords", "cast", "crew"]].copy()
        frame = frame.dropna(subset=["title"]).fillna({"overview": ""})

        frame["genres"] = frame["genres"].map(_names)
        frame["keywords"] = frame["keywords"].map(_names)
        frame["cast"] = frame["cast"].map(lambda value: _names(value, limit=3))
        frame["crew"] = frame["crew"].map(_director)
        frame["overview"] = frame["overview"].map(lambda text: str(text).split())
        frame["tags"] = frame.apply(
            lambda row: " ".join(
                row["overview"] + row["genres"] + row["keywords"] + row["cast"] + row["crew"]
            ).lower(),
            axis=1,
        )
        frame = frame.drop_duplicates(subset="title").reset_index(drop=True)

        vectorizer = CountVectorizer(stop_words="english", max_features=5000)
        vectors = vectorizer.fit_transform(frame["tags"])
        similarity = cosine_similarity(vectors)
        return cls(movies=frame[["id", "title", "genres", "tags"]], similarity=similarity)

    @property
    def titles(self) -> list[str]:
        return self.movies["title"].sort_values().tolist()

    def recommend(self, title: str, count: int = 5) -> pd.DataFrame:
        """Return the most similar movies, excluding the selected movie."""
        query = title.casefold().strip()
        matches = self.movies.index[self.movies["title"].str.casefold() == query].tolist()
        if not matches:
            raise ValueError(f"Movie '{title}' was not found in the dataset.")

        index = matches[0]
        ranked = sorted(enumerate(self.similarity[index]), key=lambda item: item[1], reverse=True)
        recommendations = [
            {"title": self.movies.iloc[i]["title"], "similarity": round(float(score) * 100, 1)}
            for i, score in ranked[1 : count + 1]
        ]
        return pd.DataFrame(recommendations)
