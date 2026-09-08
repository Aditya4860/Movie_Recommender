"""Centralized data loading and cleaning utilities."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st


MOVIE_FILE_NAMES = ("tmdb_5000_movies.csv", "movies.csv")
CREDITS_FILE_NAMES = ("tmdb_5000_credits.csv", "credits.csv")


def find_data_file(data_dir: str | Path, candidates: Iterable[str]) -> Path:
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


def parse_json_list(value: object) -> list[dict]:
    """Safely parse TMDB's Python-list-like JSON columns."""
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


@st.cache_data(show_spinner=False)
def load_base_datasets(data_dir: str | Path = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and return the raw movies and credits datasets from disk."""
    movie_path = find_data_file(data_dir, MOVIE_FILE_NAMES)
    credits_path = find_data_file(data_dir, CREDITS_FILE_NAMES)

    movies = pd.read_csv(movie_path)
    credits = pd.read_csv(credits_path)
    return movies, credits
