"""Centralized data loading and cleaning utilities.

Designed to be importable from both Streamlit and FastAPI/CLI contexts.
The @st.cache_data decorator is applied conditionally so that FastAPI,
scripts, and tests can import this module without a running Streamlit runtime.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

import pandas as pd


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


def _load_base_datasets_plain(data_dir: str | Path = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load movies and credits from disk without any caching layer."""
    movie_path = find_data_file(data_dir, MOVIE_FILE_NAMES)
    credits_path = find_data_file(data_dir, CREDITS_FILE_NAMES)
    movies = pd.read_csv(movie_path)
    credits = pd.read_csv(credits_path)
    return movies, credits


def _make_cached_loader():
    """Return a Streamlit-cached version of _load_base_datasets_plain if
    Streamlit is available and has a runtime, otherwise return the plain
    version.  This prevents ImportError / runtime errors when the module
    is used outside of a Streamlit process (FastAPI, tests, scripts).
    """
    try:
        import streamlit as st
        cached = st.cache_data(show_spinner=False)(_load_base_datasets_plain)
        return cached
    except Exception:
        return _load_base_datasets_plain


load_base_datasets = _make_cached_loader()
