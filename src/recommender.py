"""Content-based movie recommendation engine for the TMDB 5000 dataset."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data import load_base_datasets, parse_json_list


def _names(value: object, limit: int | None = None) -> list[str]:
    items = parse_json_list(value)
    names = [item.get("name", "") for item in items if isinstance(item, dict)]
    names = [name.replace(" ", "") for name in names if name]
    return names[:limit] if limit else names


def _director(value: object) -> list[str]:
    for person in parse_json_list(value):
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
        movies, credits = load_base_datasets(data_dir)
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

    def recommend(self, movie_id: int, count: int = 5) -> pd.DataFrame:
        """Return the most similar movies, excluding the selected movie."""
        matches = self.movies.index[self.movies["id"] == movie_id].tolist()
        if not matches:
            raise ValueError(f"Movie ID '{movie_id}' was not found in the dataset.")

        index = matches[0]
        ranked = sorted(enumerate(self.similarity[index]), key=lambda item: item[1], reverse=True)
        recommendations = [
            {"id": self.movies.iloc[i]["id"], "similarity": round(float(score) * 100, 1)}
            for i, score in ranked[1 : count + 1]
        ]
        return pd.DataFrame(recommendations)
