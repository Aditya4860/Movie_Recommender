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
for title in saved:
    movie = catalog.loc[catalog["title"].eq(title)].iloc[0]
    movie_row(movie, show_add=False)
    if st.button("Remove", icon=":material/bookmark_remove:", key=f"remove_{movie['id']}"):
        st.session_state.favourites.remove(title)
        st.rerun()
