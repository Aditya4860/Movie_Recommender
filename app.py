"""Streamlit interface for the TMDB content-based movie recommender."""

from pathlib import Path

import streamlit as st

from src.recommender import MovieRecommender


st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")


@st.cache_resource(show_spinner="Building movie similarity index...")
def load_recommender() -> MovieRecommender:
    return MovieRecommender.from_csv(Path("data"))


st.title("🎬 Movie Recommender")
st.caption("Content-based recommendations from genres, plot keywords, cast, and director.")

try:
    recommender = load_recommender()
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.info(
        "Download the TMDB 5000 Movie Dataset and place the movie and credits CSV files in the `data/` folder. "
        "See the README for accepted filenames."
    )
    st.stop()

selected_title = st.selectbox("Choose a movie", recommender.titles, index=None, placeholder="Search for a movie...")
number_of_results = st.slider("Recommendations", min_value=3, max_value=10, value=5)

if selected_title:
    results = recommender.recommend(selected_title, number_of_results)
    st.subheader(f"Because you liked {selected_title}")
    for rank, movie in results.reset_index(drop=True).iterrows():
        st.write(f"**{rank + 1}. {movie['title']}**  ·  {movie['similarity']}% similar")
