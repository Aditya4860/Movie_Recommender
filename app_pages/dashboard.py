"""Home dashboard for CineMatch."""

import streamlit as st

from src.catalog import genre_counts, top_movies
from src.ui import get_catalog, get_recommender, movie_row


st.title("Find your next great movie")
st.caption("Explore a 5,000-title TMDB library with a content-based similarity engine.")

catalog_slot = st.container()
with catalog_slot.skeleton(height=260):
    catalog = get_catalog()
    recommender = get_recommender()

with st.container(horizontal=True):
    st.metric("Movies in library", f"{len(catalog):,}", border=True, chart_data=[4100, 4350, 4580, 4803], chart_type="line")
    st.metric("Genres indexed", str(catalog.explode("genres")["genres"].nunique()), border=True, chart_data=[15, 17, 19, 20], chart_type="line")
    st.metric("Features compared", "5 signals", "+ cast · crew · keywords", border=True)
    st.metric("Saved to my list", str(len(st.session_state.favourites)), border=True)

left, right = st.columns([1.2, 1])
with left:
    with st.container(border=True):
        st.subheader("Library by genre")
        st.caption("The most represented genres in the current dataset")
        st.bar_chart(genre_counts(catalog), x="genre", y="movies", horizontal=True)

with right:
    with st.container(border=True):
        st.subheader("Start with a favourite")
        st.write("Pick one film and let the model find titles with similar story, genre, cast, keywords, and director.")
        movie_options = catalog.set_index('id')['title'].to_dict()
        selected_id = st.selectbox(
            "Choose a title", 
            options=list(movie_options.keys()),
            format_func=lambda x: movie_options[x],
            index=None, 
            placeholder="Search the movie library…", 
            key="dashboard_movie_id"
        )
        if st.button("Open recommendations", type="primary", icon=":material/auto_awesome:"):
            if st.session_state.dashboard_movie_id:
                st.session_state.recommendation_seed_id = st.session_state.dashboard_movie_id
                st.switch_page("app_pages/recommend.py")
            else:
                st.toast("Choose a movie first", icon=":material/info:")

st.header("Critically loved picks")
st.caption("Weighted rating balances review score with the size of the audience.")
for _, movie in top_movies(catalog, 5).iterrows():
    movie_row(movie)
