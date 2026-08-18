"""CineMatch: a multi-page content-based movie discovery app."""

from __future__ import annotations

import streamlit as st


st.set_page_config(
    page_title="CineMatch",
    page_icon=":material/movie:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.session_state.setdefault("favourites", [])

with st.sidebar:
    st.title("CineMatch")
    st.caption("TMDB movie discovery studio")
    st.badge("Content-based model", icon=":material/auto_awesome:", color="violet")
    st.space("small")
    st.caption("Built from the TMDB 5000 Movie Dataset")

pages = st.navigation(
    {
        "Discover": [
            st.Page("app_pages/dashboard.py", title="Dashboard", icon=":material/dashboard:"),
            st.Page("app_pages/recommend.py", title="Recommend", icon=":material/auto_awesome:"),
            st.Page("app_pages/explore.py", title="Explore", icon=":material/explore:"),
        ],
        "Library": [
            st.Page("app_pages/favourites.py", title="My list", icon=":material/bookmark:"),
            st.Page("app_pages/about.py", title="About the model", icon=":material/psychology:"),
        ],
    }
)

pages.run()
