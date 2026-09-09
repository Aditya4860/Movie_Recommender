"""Shared Streamlit loaders and small rendering helpers."""

from __future__ import annotations
from pathlib import Path

import pandas as pd
import streamlit as st

from src.api import fetch_poster_url, fetch_trailer_url
from src.catalog import load_catalog
from src.model_store import load_model
from src.recommender import MovieRecommender, VectorizerMethod

_MODELS_DIR = Path("models")
_ARTIFACTS_PRESENT = (_MODELS_DIR / "recommender.joblib").exists()


@st.cache_resource(show_spinner="Loading recommendation engine…")
def get_recommender(method: VectorizerMethod = "tfidf") -> MovieRecommender | None:
    """Return the production MovieRecommender.

    Load strategy (fastest first):
      1. If ``models/recommender.joblib`` exists, load the persisted TF-IDF
         model instantly (~< 1 s) — this is the production path.
      2. Otherwise train from the raw CSV files (~9 s) — development fallback.

    A separate cached instance is kept per method when training live so that
    switching between CountVectorizer and TF-IDF is non-destructive.
    The persisted model always uses TF-IDF (the selected production model).
    """
    if _ARTIFACTS_PRESENT:
        try:
            return load_model(_MODELS_DIR)
        except Exception as exc:
            st.warning(f"Could not load persisted model ({exc}); training from CSV…")
    # Fallback: train from raw CSVs
    try:
        return MovieRecommender.from_csv("data", method=method)
    except FileNotFoundError:
        st.error(
            "Dataset not found. Please download `tmdb_5000_movies.csv` and "
            "`tmdb_5000_credits.csv` and place them in the `data/` directory."
        )
        st.stop()


@st.cache_data(show_spinner=False)
def get_catalog() -> pd.DataFrame | None:
    try:
        return load_catalog("data")
    except FileNotFoundError:
        st.error("Dataset not found. Please download `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` and place them in the `data/` directory.")
        st.stop()


def movie_row(movie: pd.Series, show_add: bool = True) -> None:
    """Render a compact, data-rich movie card."""
    with st.container(border=True):
        poster_url = fetch_poster_url(movie["id"])
        
        if poster_url:
            poster_col, left, right = st.columns([1, 4, 1], vertical_alignment="center")
            with poster_col:
                st.image(poster_url, use_container_width=True)
        else:
            left, right = st.columns([5, 1], vertical_alignment="center")

        with left:
            year = "—" if pd.isna(movie.get("release_year")) else str(int(movie["release_year"]))
            st.markdown(f"#### {movie['title']}")
            st.caption(f"{year}  ·  {movie.get('genre_label', 'Unclassified')}")
            st.write(str(movie.get("overview", "No overview available."))[:260])
            st.caption(
                f":material/star: {float(movie.get('vote_average', 0)):.1f}/10 "
                f"from {int(movie.get('vote_count', 0)):,} votes"
            )
            
            # Interactive rating
            rating_key = f"rating_{movie['id']}"
            if rating_key not in st.session_state:
                st.session_state[rating_key] = st.session_state.ratings.get(movie["id"], None)
                
            st.write("Your rating:")
            rating = st.feedback("stars", key=rating_key)
            if rating is not None:
                st.session_state.ratings[movie["id"]] = rating

        if show_add:
            with right:
                saved = movie["id"] in st.session_state.favourites
                if st.button(
                    "Saved" if saved else "Save",
                    icon=":material/bookmark_added:" if saved else ":material/bookmark_add:",
                    key=f"save_{movie['id']}",
                    disabled=saved,
                    use_container_width=True,
                ):
                    st.session_state.favourites.append(movie["id"])
                    st.toast(f"Added {movie['title']} to My list", icon=":material/bookmark_added:")
                    st.rerun()

                trailer_url = fetch_trailer_url(movie["id"])
                if trailer_url:
                    st.link_button(
                        "Trailer", 
                        url=trailer_url, 
                        icon=":material/play_circle:", 
                        use_container_width=True
                    )
