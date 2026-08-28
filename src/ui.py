"""Shared Streamlit loaders and small rendering helpers."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.api import fetch_poster_url
from src.catalog import load_catalog
from src.recommender import MovieRecommender


@st.cache_resource(show_spinner="Preparing the recommendation engine...")
def get_recommender() -> MovieRecommender:
    return MovieRecommender.from_csv("data")


@st.cache_data(show_spinner=False)
def get_catalog() -> pd.DataFrame:
    return load_catalog("data")


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
        if show_add:
            with right:
                saved = movie["title"] in st.session_state.favourites
                if st.button(
                    "Saved" if saved else "Save",
                    icon=":material/bookmark_added:" if saved else ":material/bookmark_add:",
                    key=f"save_{movie['id']}",
                    disabled=saved,
                ):
                    st.session_state.favourites.append(movie["title"])
                    st.toast(f"Added {movie['title']} to My list", icon=":material/bookmark_added:")
                    st.rerun()
