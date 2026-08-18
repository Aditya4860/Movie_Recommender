"""Movie explorer with in-library filtering."""

import streamlit as st

from src.ui import get_catalog, movie_row


st.title("Explore the library")
st.caption("Filter 5,000 TMDB films by genre, era, rating, and title.")
catalog = get_catalog()
all_genres = sorted(catalog.explode("genres")["genres"].dropna().unique())
years = catalog["release_year"].dropna().astype(int)

with st.form("explore_filters", border=True):
    query = st.text_input("Search titles", placeholder="Try: Batman, Matrix, Avatar…")
    selected_genres = st.multiselect("Genres", all_genres)
    year_range = st.slider("Release years", int(years.min()), int(years.max()), (int(years.min()), int(years.max())))
    minimum_rating = st.slider("Minimum TMDB rating", 0.0, 10.0, 6.5, 0.5)
    apply = st.form_submit_button("Apply filters", type="primary", icon=":material/filter_alt:")

if apply or "explore_results" not in st.session_state:
    filtered = catalog.copy()
    if query:
        filtered = filtered[filtered["title"].str.contains(query, case=False, na=False)]
    if selected_genres:
        filtered = filtered[filtered["genres"].map(lambda items: bool(set(items) & set(selected_genres)))]
    filtered = filtered[filtered["release_year"].between(year_range[0], year_range[1])]
    filtered = filtered[filtered["vote_average"] >= minimum_rating]
    st.session_state.explore_results = filtered.sort_values(["vote_average", "vote_count"], ascending=False)

results = st.session_state.explore_results
st.caption(f"{len(results):,} matching movies")
if results.empty:
    st.info("No films match these filters. Try broadening your search.", icon=":material/search_off:")
else:
    for _, movie in results.head(12).iterrows():
        movie_row(movie)
