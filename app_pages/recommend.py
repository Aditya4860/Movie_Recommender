"""Recommendation studio page."""

import streamlit as st

from src.ui import get_catalog, get_recommender, movie_row


st.title("Recommendation studio")
st.caption("Find films that share meaningful metadata—not just the same genre.")

catalog = get_catalog()
recommender = get_recommender()
movie_options = catalog.set_index('id')['title'].to_dict()
options_list = list(movie_options.keys())

default_id = st.session_state.get("recommendation_seed_id")
default_index = options_list.index(default_id) if default_id in options_list else None

with st.form("recommendation_form", border=True):
    selected_id = st.selectbox("What did you enjoy?", options_list, format_func=lambda x: movie_options[x], index=default_index, placeholder="Search by movie title…")
    result_count = st.segmented_control("Number of results", [5, 8, 10], default=5)
    submitted = st.form_submit_button("Build recommendations", type="primary", icon=":material/auto_awesome:")

if submitted and selected_id:
    st.session_state.recommendation_seed_id = selected_id

chosen_id = st.session_state.get("recommendation_seed_id")
if not chosen_id:
    st.info("Choose a title and build recommendations to begin.", icon=":material/lightbulb:")
    st.stop()

source = catalog.loc[catalog["id"].eq(chosen_id)].iloc[0]
with st.container(border=True):
    st.markdown(f"### Starting from: {source['title']}")
    st.caption(f"{source['genre_label']}  ·  :material/star: {source['vote_average']:.1f}/10")
    st.write(source["overview"])

result_count = int(result_count or 5)
with st.status("Comparing story, genres, cast, keywords, and director…", expanded=False) as status:
    results = recommender.recommend(chosen_id, result_count)
    status.update(label="Recommendations ready", state="complete", expanded=False)

st.header("Your matches")
for rank, result in results.reset_index(drop=True).iterrows():
    movie = catalog.loc[catalog["id"].eq(result["id"])].iloc[0].copy()
    movie["overview"] = f"{movie['overview']}\n\nSimilarity match: {result['similarity']}%"
    st.caption(f"MATCH {rank + 1}")
    movie_row(movie)
