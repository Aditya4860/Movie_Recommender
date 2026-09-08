"""Saved titles page."""

import streamlit as st

from src.ui import get_catalog, movie_row


st.title("My list")
st.caption("Keep interesting titles here while you explore.")
catalog = get_catalog()
saved = st.session_state.favourites

if not saved:
    st.info("Your list is empty. Save films from the Dashboard, Explore, or Recommendation studio.", icon=":material/bookmark:")
    st.stop()

st.metric("Saved movies", len(saved), border=True)
for movie_id in saved:
    # Handle legacy title-based favourites gracefully if they exist
    if isinstance(movie_id, str):
        movie_matches = catalog.loc[catalog["title"].eq(movie_id)]
        if movie_matches.empty:
            continue
        movie = movie_matches.iloc[0]
        actual_id = movie["id"]
    else:
        movie = catalog.loc[catalog["id"].eq(movie_id)].iloc[0]
        actual_id = movie_id
        
    movie_row(movie, show_add=False)
    if st.button("Remove", icon=":material/bookmark_remove:", key=f"remove_{actual_id}"):
        st.session_state.favourites.remove(movie_id)
        st.rerun()
